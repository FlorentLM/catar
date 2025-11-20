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
from utils import calculate_fundamental_matrices, probe_video, get_projection_matrix
from mokap.utils.geometry import transforms

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cache_utils import DiskCacheReader
    CameraParameters = Dict[str, Union[float, np.ndarray]]
    CalibrationDict = Dict[str, CameraParameters]


@dataclass
class VideoInfo:
    """Container for a single video's data."""

    path: Path
    filename: str
    width: int
    height: int
    num_frames: int
    fps: float
    fourcc: str
    duration: float


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


class CalibrationState:
    """Owns all camera calibration and provides efficient, cached access."""

    def __init__(self, initial_calibration: 'CalibrationDict', camera_names: List[str]):
        self._calibrations: 'CalibrationDict' = initial_calibration
        self._camera_names: List[str] = camera_names
        self._cache: Dict[str, np.ndarray] = {}
        self._f_mats_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None

        self.best_fitness: float = float('inf')
        self.calibration_frames: List[int] = []

    def _invalidate_cache(self):
        """Clears cached arrays when calibration changes."""

        self._cache.clear()
        self._f_mats_cache = None

    def update_calibration(self, new_calibration: 'CalibrationDict'):
        """Updates the entire calibration state and invalidates caches."""

        self._calibrations = new_calibration

        # GA's first evaluation of a new population will set a real fitness baseline
        self.best_fitness = float('inf')

        self._invalidate_cache()
        # TODO: Might be better to only invalidate what's actually changed

        print("Successfully applied new camera calibration.")

    def get(self, camera_name: str) -> 'CameraParameters':
        """Returns the parameter dictionary for a single camera."""
        return self._calibrations[camera_name]

    @property
    def camera_names(self) -> List[str]:
        return self._camera_names

    @property
    def best_calibration(self) -> Optional['CalibrationDict']:
        """Read-only access to the underlying dictionary for saving/snapshots."""
        return self._calibrations

    def _get_or_compute(self, key: str, dtype=np.float32) -> np.ndarray:
        """Helper to get an array from cache or compute and store it."""

        if key not in self._cache:
            # example for key 'camera_matrix', stack all 'camera_matrix' arrays
            data_list = [self._calibrations[name][key] for name in self._camera_names]
            self._cache[key] = np.array(data_list, dtype=dtype)

        return self._cache[key]

    @property
    def K_mats(self) -> np.ndarray:
        """Returns (C, 3, 3) array of camera matrices."""
        return self._get_or_compute('camera_matrix')

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Returns (C, D) array of distortion coefficients."""
        return self._get_or_compute('dist_coeffs')

    @property
    def rvecs_c2w(self) -> np.ndarray:
        """Returns (C, 3) array of rotation vectors (camera-to-world)."""
        return self._get_or_compute('rvec')

    @property
    def tvecs_c2w(self) -> np.ndarray:
        """Returns (C, 3) array of translation vectors (camera-to-world)."""
        return self._get_or_compute('tvec')

    @property
    def rvecs_w2c(self) -> np.ndarray:
        """Returns (C, 3) array of rotation vectors (world-to-camera)."""
        if 'rvecs_w2c' not in self._cache:
            r_inv, _ = transforms.invert_rtvecs(self.rvecs_c2w, self.tvecs_c2w)
            self._cache['rvecs_w2c'] = np.asarray(r_inv)
        return self._cache['rvecs_w2c']

    @property
    def tvecs_w2c(self) -> np.ndarray:
        """Returns (C, 3) array of translation vectors (world-to-camera)."""
        if 'tvecs_w2c' not in self._cache:
            _, t_inv = transforms.invert_rtvecs(self.rvecs_c2w, self.tvecs_c2w)
            self._cache['tvecs_w2c'] = np.asarray(t_inv)
        return self._cache['tvecs_w2c']

    @property
    def P_mats(self) -> np.ndarray:
        """Returns (C, 3, 4) array of projection matrices."""
        if 'P_mats' not in self._cache:
            proj_matrices = [get_projection_matrix(self._calibrations[name]) for name in self._camera_names]
            self._cache['P_mats'] = np.array(proj_matrices)
        return self._cache['P_mats']

    @property
    def F_mats(self) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
        """Calculates and caches fundamental matrices for all camera pairs."""
        if self._f_mats_cache is None and self._calibrations:
            self._f_mats_cache = calculate_fundamental_matrices(self._calibrations, self._camera_names)
        return self._f_mats_cache


class VideoState:
    """Manages video sources and their metadata."""

    def __init__(self, camera_names: List[str], video_paths: List[Union[Path, str]]):
        if len(camera_names) != len(video_paths):
            raise ValueError("Mismatch between number of camera names and video paths.")

        self._camera_names: List[str] = camera_names
        self._videos: Dict[str, VideoInfo] = {}

        # The first video's metadata is used as the reference
        self.session_metadata = probe_video(video_paths[0])
        self.session_metadata['num_videos'] = len(video_paths)

        for name, path_str in zip(camera_names, video_paths):
            path = Path(path_str)
            metadata = probe_video(path)

            self._videos[name] = VideoInfo(
                path=path.resolve(),
                filename=path.name,
                width=metadata['width'],
                height=metadata['height'],
                num_frames=metadata['num_frames'],
                fps=metadata['fps'],
                fourcc=metadata['fourcc'],
                duration=metadata['duration']
            )

    @property
    def camera_names(self) -> List[str]:
        """Returns the canonical ordered list of camera names."""
        return self._camera_names

    @property
    def filepaths(self) -> List[str]:
        """Returns the ordered list of video paths."""
        return [str(self._videos[name].path) for name in self._camera_names]

    @property
    def filenames(self) -> List[str]:
        """Returns the ordered list of video filenames."""
        return [self._videos[name].filename for name in self._camera_names]

    def get(self, camera_name: str) -> Optional[VideoInfo]:
        """Get VideoInfo for a specific camera by name."""
        return self._videos.get(camera_name)

    def __getitem__(self, camera_name: str) -> VideoInfo:
        """Dictionary-style access to VideoInfo."""
        return self._videos[camera_name]


class AppState:
    """
    Thread-safe container for application state.

    This refactored version delegates the management of video and calibration
    data to dedicated state objects (VideoState and CalibrationState) for
    improved robustness and clarity.
    """

    def __init__(self, video_state: 'VideoState', calib_state: 'CalibrationState', skeleton_config: Dict[str, Any]):

        self.lock = threading.RLock()

        # Dedicated state objects
        self.videos: 'VideoState' = video_state
        self.calibration: 'CalibrationState' = calib_state
        self.cache_reader: Optional['DiskCacheReader'] = None

        # Read-only metadata derived from state objects for convenience
        self.video_metadata = self.videos.session_metadata
        self.camera_names: List[str] = self.videos.camera_names
        self.video_names: List[str] = self.videos.filenames

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
        self.current_grid_cols: int = 0
        self.selected_point_idx: int = 0
        self.focus_selected_point: bool = False
        self.show_cameras_in_3d: bool = True
        self.drag_state: Dict[str, Any] = {}
        self.show_reprojection_error: bool = True
        self.show_all_labels: bool = False
        self.show_epipolar_lines: bool = True
        self.temp_hide_overlays: bool = False

        # Feature flags
        self.keypoint_tracking_enabled: bool = False
        self.needs_3d_reconstruction: bool = True

        #Annotation data
        num_frames = self.video_metadata['num_frames']
        num_videos = self.video_metadata['num_videos']

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
        self.scene_centre = np.zeros(3)

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of state needed by the GA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_fitness": self.calibration.best_fitness,
                "best_individual": self.calibration.best_calibration,
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of state needed by the BA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "camera_names": list(self.camera_names),
                "best_individual": self.calibration.best_calibration,
            }

    def save_to_disk(self, folder: Path):
        """Save all persistent state to disk."""

        print(f"Saving state to: '{folder}'")
        with self.lock:
            try:
                np.save(folder / 'annotations.npy', self.annotations)
                np.save(folder / 'human_annotated.npy', self.human_annotated)
                np.save(folder / 'reconstructed_3d_points.npy', self.reconstructed_3d_points)

                # Save data from the calibration state object
                with open(folder / 'calibration_frames.json', 'w') as f:
                    json.dump(self.calibration.calibration_frames, f)

                if self.calibration.best_calibration is not None:
                    with open(folder / 'best_individual.pkl', 'wb') as f:
                        pickle.dump(self.calibration.best_calibration, f)

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
            # Load simple attributes directly into self
            if 'human_annotated' in loaded_data:
                self.human_annotated = loaded_data['human_annotated']

            if 'reconstructed_3d_points' in loaded_data:
                self.reconstructed_3d_points = loaded_data['reconstructed_3d_points']

            if 'annotations' in loaded_data:
                annots = loaded_data['annotations']

                # Fix annotations shape if necessary (for legacy saves)
                # TODO: Will be removed eventually

                # Check if the loaded data is the old (x, y) format
                if annots.ndim == 4 and annots.shape[-1] == 2:
                    print("  - Converting legacy 2D annotations to 3D (x, y, confidence)...")
                    F, C, P, _ = annots.shape
                    # Create a new, correctly shaped array
                    new_annots = np.full((F, C, P, 3), np.nan, dtype=np.float32)
                    # Copy the old (x, y) data
                    new_annots[..., :2] = annots

                    # Add a default confidence of 1.0 where points exist
                    is_valid = ~np.isnan(annots[..., 0])
                    new_annots[is_valid, 2] = 1.0

                    # Assign the newly converted array
                    self.annotations = new_annots

                # If it's already the correct shape, just assign it
                elif annots.ndim == 4 and annots.shape[-1] == 3:
                    self.annotations = annots
                else:
                    print(f"  - WARNING: Loaded annotations have an unsupported shape {annots.shape}. Skipping.")

            # Load calibration data into the dedicated state object
            if 'calibration_frames' in loaded_data:
                self.calibration.calibration_frames = loaded_data['calibration_frames']

            if 'best_individual' in loaded_data:
                loaded_calib = loaded_data['best_individual']

                # Load legacy CATAR calibration format (List[Dict] with CATAR keys)
                # TODO: This will be removed
                if isinstance(loaded_calib, list):
                    if not self.camera_names:
                        print(
                            "  - WARNING: Cannot convert legacy calibration, camera names are not yet defined. Skipping.")
                        new_calib_dict = None

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

                        print("  - Converted legacy calibration format (List[Dict] -> Dict[str, Dict]).")
                    else:
                        print(
                            f"  - WARNING: Loaded calibration list length ({len(loaded_calib)}) mismatch with video count ({len(self.camera_names)}). Skipping calibration load.")

                else:
                    new_calib_dict = loaded_calib

                if new_calib_dict is not None:
                    self.calibration.update_calibration(new_calib_dict)

        print("State loading complete.")