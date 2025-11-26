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
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

import config
from state.data_manager import DataManager
from mokap.utils.fileio import probe_video

if TYPE_CHECKING:
    from video import DiskCacheReader
    from state.calibration_state import CalibrationState
    from utils import CalibrationDict


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


class AppState:
    """
    Thread-safe container for application state.
    """

    def __init__(
        self,
        data_folder: Union[Path, str],
        camera_names: List[str],
        video_paths: List[Union[Path, str]],
        calib_state: 'CalibrationState',
        skeleton_config: Dict[str, Any]
    ):
        if len(camera_names) != len(video_paths):
            raise ValueError("Mismatch between number of camera names and video paths.")

        self.data_folder = Path(data_folder)
        if hasattr(config, 'VIDEO_CACHE_FOLDER'):
            self.video_cache_dir = Path(config.VIDEO_CACHE_FOLDER)
        else:
            self.video_cache_dir = self.data_folder / 'video_cache'

        self.video_backend = None

        # Keep top-level lock for app-wide state coordination
        # (DataManager has its own lock for data arrays)
        self.lock = threading.RLock()

        # Cameras order and lookup accessors
        self.camera_names = tuple(camera_names)
        self.num_cameras = len(camera_names)
        self.camera_nti = {name: i for i, name in enumerate(self.camera_names)}
        self.camera_itn = {i: name for name, i in self.camera_nti.items()}

        # Video information (constant during runtime)
        self.video_paths: List[Path] = [Path(p).resolve() for p in video_paths]
        self.video_filenames: List[str] = [Path(p).name for p in video_paths]
        self.num_videos = len(self.video_paths)

        # Probe videos for metadata
        self._video_metadata: Dict[str, Dict[str, Any]] = {}
        for name, path in zip(camera_names, self.video_paths):
            self._video_metadata[name] = probe_video(path)

        # Use first video as session reference
        first_metadata = self._video_metadata[camera_names[0]]
        self.video_metadata = first_metadata.copy()
        self.video_metadata['num_videos'] = len(video_paths)

        # Video metadata
        self.frame_width: int = first_metadata['width']
        self.frame_height: int = first_metadata['height']
        self.num_frames: int = first_metadata['num_frames']
        self.video_duration: float = first_metadata['duration']
        self.fps: float = first_metadata['fps'] or 30.0     # default to 30 if fps is missing

        half_life_frames = config.TRACKER_HALF_LIFE_CONFIDENCE_DECAY * self.fps
        if half_life_frames > 0:
            self.tracker_decay_rate = 0.5 ** (1.0 / half_life_frames)
        else:
            self.tracker_decay_rate = 0.0  # instant decay if half-life is 0
        print(f"Tracker: Half-life {config.TRACKER_HALF_LIFE_CONFIDENCE_DECAY:.1f}s @ {self.fps:.2f} FPS "
              f"-> Decay rate per frame: {self.tracker_decay_rate:.4f}")

        # State Objects
        self.calibration: 'CalibrationState' = calib_state
        self.cache_reader: Optional['DiskCacheReader'] = None

        # Keypoints order and lookup accessors (read-only after init)
        self.point_names = tuple(skeleton_config['point_names'])
        self.point_nti = {name: i for i, name in enumerate(self.point_names)}
        self.point_itn = {i: name for name, i in self.point_nti.items()}

        self.skeleton = skeleton_config['skeleton']

        # TODO: Use these sets everywhere possible
        self.all_points_set = set(self.point_names)
        self.skeleton_points_set = {'s_small', 's_large'}
        # Points that are tracked but NOT part of the skeleton graph (objects, props, etc)
        # TODO: Move this to config, and add a GUI way to edit them
        self.non_skeleton_points_set = {'s_small', 's_large'}

        self.point_colors = skeleton_config['point_colors']
        self.camera_colors = config.CAMERA_COLORS
        self.num_points = len(self.point_names)

        # Centralised data manager
        self.data = DataManager(
            n_frames=self.num_frames,
            n_cameras=len(camera_names),
            n_points=self.num_points
        )

        # Playback State
        self.frame_idx: int = 0
        self.paused: bool = True
        self.is_seeking: bool = False

        # UI State
        self.current_grid_cols: int = 0
        self.selected_point_idx: int = 0
        self.focus_selected_point: bool = False
        self.show_cameras_in_3d: bool = True
        self.drag_state: Dict[str, Any] = {}
        self.show_reprojection_error: bool = True
        self.show_all_labels: bool = False
        self.show_epipolar_lines: bool = True
        self.temp_hide_overlays: bool = False

        # Feature Flags
        self.keypoint_tracking_enabled: bool = False
        self.needs_3d_reconstruction: bool = True
        self.tracker_collision_stop: bool = True

        # Transient UI Data

        # Frame cache for UI zoom (raw frames from current playback position)
        self.current_video_frames: Optional[List[np.ndarray]] = None

        # Scene centre for 3D visualization
        self.scene_centre = np.zeros(3)

    def get_video_metadata(self, camera_name: str) -> Dict[str, Any]:
        """Get metadata for a specific camera's video."""
        return self._video_metadata[camera_name]

    # Snapshot Methods (for GA and BA workers)

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """
        Create snapshot of state needed by the GA worker.
        """
        with self.lock:
            with self.data.bulk_lock():
                annotations_copy = self.data.annotations.copy()

            return {
                "annotations": annotations_copy,
                "calibration_frames": list(self.calibration.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_fitness": self.calibration.best_fitness,
                "best_individual": self.calibration.best_calibration,
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of state needed by the BA worker.
        """
        with self.lock:
            with self.data.bulk_lock():
                annotations_copy = self.data.annotations.copy()

            return {
                "annotations": annotations_copy,
                "calibration_frames": list(self.calibration.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_individual": self.calibration.best_calibration,
            }

    # Persistence (Save / Load)

    def save_to_disk(self, folder: Path):
        """
        Save all persistent state to disk.
        """
        print(f"Saving state to: '{folder}'")

        with self.lock:
            try:
                with self.data._lock:
                    np.save(folder / 'annotations.npy', self.data.annotations)
                    np.save(folder / 'human_annotated.npy', self.data.human_annotated)
                    np.save(folder / 'reconstructed_3d.npy', self.data.reconstructed_3d)

                # Save calibration state
                with open(folder / 'calibration_frames.json', 'w') as f:
                    json.dump(self.calibration.calibration_frames, f)

                if self.calibration.best_calibration is not None:
                    with open(folder / 'best_individual.pkl', 'wb') as f:
                        pickle.dump(self.calibration.best_calibration, f)

                print("State saved successfully.")
            except Exception as e:
                print(f"Error saving state: {e}")

    def load_from_disk(self, folder: Path):
        """
        Load persistent state from disk.
        """
        print(f"Loading state from: '{folder}'")

        files_to_load = {
            'annotations.npy': ('numpy', 'annotations'),
            'human_annotated.npy': ('numpy', 'human_annotated'),
            'reconstructed_3d.npy': ('numpy', 'reconstructed_3d'),
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
            # Load into DataManager's arrays
            with self.data._lock:

                # Load human annotation flags
                if 'human_annotated' in loaded_data:
                    self.data.human_annotated = loaded_data['human_annotated']

                # Load 3D reconstruction data
                if 'reconstructed_3d' in loaded_data:
                    loaded_pts = loaded_data['reconstructed_3d']

                    # Handle legacy 3-channel data (old format was (N, 3))
                    if loaded_pts.shape[-1] == 3:
                        print("  - Converting legacy 3D points (N, 3) to (N, 4)...")
                        F, P, _ = loaded_pts.shape
                        new_pts = np.full((F, P, 4), np.nan, dtype=np.float32)
                        new_pts[..., :3] = loaded_pts
                        new_pts[..., 3] = np.nan  # No scores in legacy format
                        self.data.reconstructed_3d = new_pts
                    else:
                        self.data.reconstructed_3d = loaded_pts

                # Load annotation data
                if 'annotations' in loaded_data:
                    annots = loaded_data['annotations']

                    # Handle legacy 2-channel data (old format was (x, y) only)
                    if annots.ndim == 4 and annots.shape[-1] == 2:
                        print("  - Converting legacy 2D annotations to 3D (x, y, confidence)...")
                        F, C, P, _ = annots.shape
                        new_annots = np.full((F, C, P, 3), np.nan, dtype=np.float32)
                        new_annots[..., :2] = annots

                        # Add default confidence of 1.0 where points exist
                        is_valid = ~np.isnan(annots[..., 0])
                        new_annots[is_valid, 2] = 1.0

                        self.data.annotations = new_annots

                    # Modern format with confidence
                    elif annots.ndim == 4 and annots.shape[-1] == 3:
                        self.data.annotations = annots
                    else:
                        print(f"  - WARNING: Loaded annotations have unsupported shape {annots.shape}. Skipping.")

            # Load calibration data
            if 'calibration_frames' in loaded_data:
                self.calibration.calibration_frames = loaded_data['calibration_frames']

            if 'best_individual' in loaded_data:
                loaded_calib = loaded_data['best_individual']

                # Handle legacy CATAR calibration format (List[Dict] with old keys)
                if isinstance(loaded_calib, list):
                    if not self.camera_names:
                        print("  - WARNING: Cannot convert legacy calibration without camera names. Skipping.")
                        new_calib_dict = None

                    elif len(loaded_calib) == len(self.camera_names):
                        new_calib_dict: 'CalibrationDict' = {}

                        for i, cam_name in enumerate(self.camera_names):
                            old_params = loaded_calib[i]

                            # Create K matrix from fx, fy, cx, cy (legacy keys)
                            K = np.array([
                                [old_params['fx'], 0.0, old_params['cx']],
                                [0.0, old_params['fy'], old_params['cy']],
                                [0.0, 0.0, 1.0]
                            ], dtype=np.float32)

                            # Map old keys to new keys
                            new_params = {
                                'camera_matrix': K,
                                'dist_coeffs': old_params['dist'],
                                'rvec': old_params['rvec'],
                                'tvec': old_params['tvec'],
                            }

                            # Ensure all values are numpy arrays
                            new_calib_dict[cam_name] = {
                                k: np.asarray(v) if isinstance(v, (list, tuple)) else v
                                for k, v in new_params.items()
                            }

                        print("  - Converted legacy calibration format (List[Dict] -> Dict[str, Dict]).")
                    else:
                        print(
                            f"  - WARNING: Loaded calibration list length ({len(loaded_calib)}) "
                            f"mismatch with video count ({len(self.camera_names)}). Skipping."
                        )
                        new_calib_dict = None

                else:
                    # Modern format
                    new_calib_dict = loaded_calib

                if new_calib_dict is not None:
                    self.calibration.update_calibration(new_calib_dict)

        print("State loading complete.")
