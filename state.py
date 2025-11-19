"""
All application state and communication queues are managed here.
"""
import queue
import threading
import multiprocessing
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

import config
from utils import calculate_fundamental_matrices
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from video_cache import VideoCacheReader
    CameraParameters = Dict[str, Union[float, np.ndarray]]
    CalibrationDict = Dict[str, CameraParameters]


@dataclass
class Queues:
    """Centralised queue management for threads communication."""

    # Main thread queues
    command: queue.Queue = field(default_factory=queue.Queue)
    results: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))

    # Worker input queues
    frames_for_tracking: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    frames_for_rendering: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))

    # Worker command queues
    tracking_command: queue.Queue = field(default_factory=queue.Queue)
    ga_command: multiprocessing.Queue = field(default_factory=multiprocessing.Queue)
    ba_command: multiprocessing.Queue = field(default_factory=multiprocessing.Queue)

    ba_results: multiprocessing.Queue = field(default_factory=multiprocessing.Queue)

    # Worker progress queues
    tracking_progress: queue.Queue = field(default_factory=queue.Queue)
    ga_progress: multiprocessing.Queue = field(default_factory=multiprocessing.Queue)
    cache_progress: queue.Queue = field(default_factory=queue.Queue)

    # Cancellation signals
    stop_batch_track: threading.Event = field(default_factory=threading.Event)
    stop_bundle_adjustment: multiprocessing.Event = field(default_factory=multiprocessing.Event)

    def shutdown_all(self):
        """Send shutdown commands to all workers."""

        self.command.put({"action": "shutdown"})
        self.frames_for_tracking.put({"action": "shutdown"})
        self.frames_for_rendering.put({"action": "shutdown"})
        self.ga_command.put({"action": "shutdown"})
        self.ba_command.put({"action": "shutdown"})
        self.stop_batch_track.set()
        self.stop_bundle_adjustment.set()


@dataclass
class CalibrationState:
    """Owns all camera calibration and related data."""

    # Calibration frames (frame indices)
    calibration_frames: List[int] = field(default_factory=list)

    best_individual: Optional['CalibrationDict'] = None
    best_fitness: float = float('inf')
    fundamental_matrices: Optional[Dict[Tuple[int, int], np.ndarray]] = None

    def set_calibration(self, individual: 'CalibrationDict', cam_names: List[str]):
        """Apply camera calibration and update dependent state."""

        self.best_individual = individual
        self.fundamental_matrices = calculate_fundamental_matrices(individual, cam_names)

        # A newly set calibration has 0 fitness for tracking purposes (= we trust it)
        self.best_fitness = 0.0
        print("Successfully applied camera calibration.")


class AppState:
    """Thread-safe container for application state."""

    def __init__(self, video_metadata: Dict[str, Any], skeleton_config: Dict[str, Any]):

        self.lock = threading.RLock()

        # Video metadata (read-only after init)
        self.video_metadata = video_metadata
        self.video_names: List[str] = []
        self.camera_names: List[str] = []
        self.cache_reader: Optional['VideoCacheReader'] = None

        # Skeleton configuration (read-only after init)
        self.point_names = skeleton_config['point_names']
        self.skeleton = skeleton_config['skeleton']
        self.point_colors = skeleton_config['point_colors']
        self.camera_colors = config.CAMERA_COLORS
        self.num_points = len(self.point_names)

        # Playback state
        self.frame_idx: int = 0
        self.paused: bool = True
        self.is_seeking: bool = False

        # UI state
        self.selected_point_idx: int = 0
        self.focus_selected_point: bool = False
        self.show_cameras_in_3d: bool = True
        self.drag_state: Dict[str, Any] = {}
        self.show_reprojection_error: bool = True
        self.show_all_labels: bool = False

        # Feature flags
        self.keypoint_tracking_enabled: bool = False
        self.needs_3d_reconstruction: bool = True

        # Annotation data
        num_frames = video_metadata['num_frames']
        num_videos = video_metadata['num_videos']

        # Enforce (x, y, confidence) storage (shape (F, C, P, 3))
        self.annotations = np.full(
            (num_frames, num_videos, self.num_points, 3),
            np.nan,
            dtype=np.float32
        )
        self.human_annotated = np.zeros(
            (num_frames, num_videos, self.num_points),
            dtype=bool
        )
        self.reconstructed_3d_points = np.full(
            (num_frames, self.num_points, 3),
            np.nan,
            dtype=np.float32
        )

        # Frame cache for UI zoom
        self.current_video_frames: Optional[List[np.ndarray]] = None

        # Camera calibration data lives in this class
        self.calibration_state = CalibrationState()

        self.scene_centre = np.zeros(3)

    def set_calibration_state(self, individual: 'CalibrationDict'):
        """Helper to call set_calibration on the nested state."""

        with self.lock:
            self.calibration_state.set_calibration(individual, self.camera_names)
            self.needs_3d_reconstruction = True # ensure 3D reconstruction is rerun with new calibration

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of state needed by the GA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration_state.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_fitness": self.calibration_state.best_fitness,
                "best_individual": self.calibration_state.best_individual,
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of state needed by the BA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration_state.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_individual": self.calibration_state.best_individual,
            }

    def save_to_disk(self, folder: Path):
        """Save all persistent state to disk."""

        print(f"Saving state to: '{folder}'")
        with self.lock:
            try:
                np.save(folder / 'annotations.npy', self.annotations)
                np.save(folder / 'human_annotated.npy', self.human_annotated)
                np.save(folder / 'reconstructed_3d_points.npy', self.reconstructed_3d_points)

                with open(folder / 'calibration_frames.json', 'w') as f:
                    json.dump(self.calibration_state.calibration_frames, f)

                if self.calibration_state.best_individual is not None:
                    with open(folder / 'best_individual.pkl', 'wb') as f:
                        pickle.dump(self.calibration_state.best_individual, f)

                print("State saved successfully.")
            except Exception as e:
                print(f"Error saving state: {e}")

    def load_from_disk(self, folder: Path):
        """Load persistent state from disk."""

        print(f"Loading state from: '{folder}'")

        files_to_load = {
            'annotations.npy': ('numpy', 'annotations'),
            'human_annotated.npy': ('numpy', 'human_annotated'),
            'reconstructed_3d_points.npy': ('numpy', 'reconstructed_3d_points'),
            'best_individual.pkl': ('pickle', 'best_individual'),
            'calibration_frames.json': ('json', 'calibration_frames'),
        }

        loaded_data = {}
        for filename, (file_type, attr_name) in files_to_load.items():
            file_path = folder / filename
            if not file_path.exists():
                continue

            try:
                if file_type == 'numpy':
                    loaded_data[attr_name] = np.load(file_path)
                elif file_type == 'pickle':
                    with file_path.open('rb') as f:
                        loaded_data[attr_name] = pickle.load(f)
                elif file_type == 'json':
                    with file_path.open('r') as f:
                        loaded_data[attr_name] = json.load(f)
                print(f"  - Loaded '{filename}'")
            except Exception as e:
                print(f"  - WARNING: Could not load '{filename}': {e}")

        if not loaded_data:
            print("No saved state found. Starting fresh.")
            return

        with self.lock:
            # Load simple attributes
            if 'human_annotated' in loaded_data:
                self.human_annotated = loaded_data['human_annotated']
            if 'reconstructed_3d_points' in loaded_data:
                self.reconstructed_3d_points = loaded_data['reconstructed_3d_points']

            # Load and fix annotations shape if necessary (for legacy saves)
            if 'annotations' in loaded_data:
                annots = loaded_data['annotations']
                if annots.shape[-1] == 2:
                    # Pad old (x, y) data with a confidence score of 1.0
                    F, C, P, _ = annots.shape
                    new_annots = np.full((F, C, P, 3), np.nan, dtype=np.float32)
                    new_annots[..., :2] = annots

                    # Only fill confidence where x, y are valid
                    is_valid = ~np.isnan(annots[..., 0])
                    new_annots[is_valid, 2] = 1.0
                    self.annotations = new_annots
                elif annots.shape[-1] == 3:
                    self.annotations = annots
                else:
                    print("  - WARNING: Loaded annotations have unsupported shape.")

            # Load calibration state
            if 'calibration_frames' in loaded_data:
                self.calibration_state.calibration_frames = loaded_data['calibration_frames']

            if 'best_individual' in loaded_data:
                loaded_calib = loaded_data['best_individual']

                # Load legacy CATAR calibration format (List[Dict] with CATAR keys)
                # TODO: This will be removed
                if isinstance(loaded_calib, list):
                    if not self.camera_names:
                        print(
                            "  - WARNING: Cannot convert legacy calibration, camera names are not yet defined. Skipping.")
                        loaded_calib = None
                    elif len(loaded_calib) == len(self.camera_names):
                        new_calib_dict: CalibrationDict = {}
                        for i, cam_name in enumerate(self.camera_names):
                            old_params = loaded_calib[i]

                            # Create K matrix from fx, fy, cx, cy (legacy keys)
                            K = np.array([
                                [old_params['fx'], 0.0, old_params['cx']],
                                [0.0, old_params['fy'], old_params['cy']],
                                [0.0, 0.0, 1.0]
                            ], dtype=np.float32)

                            # Map old keys to mokap keys
                            new_params = {
                                'camera_matrix': K,
                                'dist_coeffs': old_params['dist'],
                                'rvec': old_params['rvec'],
                                'tvec': old_params['tvec'],
                            }
                            # Ensure all values are numpy arrays
                            new_calib_dict[cam_name] = {k: np.asarray(v) if isinstance(v, (list, tuple)) else v for k, v
                                                        in new_params.items()}

                        loaded_calib = new_calib_dict
                        print("  - Converted legacy calibration format (List[Dict] -> Dict[str, Dict]).")
                    else:
                        print(
                            f"  - WARNING: Loaded calibration list length ({len(loaded_calib)}) mismatch with video count ({len(self.camera_names)}). Skipping calibration load.")
                        loaded_calib = None

                # loaded_calib is now either the new Dict[str, Dict] (mokap format) or None
                if loaded_calib is not None:
                    self.calibration_state.best_individual = loaded_calib
                    # Recalculate F matrices if camera names are available
                    if self.camera_names:
                        self.calibration_state.set_calibration(
                            self.calibration_state.best_individual,
                            self.camera_names
                        )

        print("State loading complete.")