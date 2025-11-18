"""
All application state and communication queues are managed here.
"""
import queue
import threading
import multiprocessing
import tomllib
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import config
from utils import calculate_fundamental_matrices
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from video_cache import VideoCacheReader


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

    def shutdown_all(self):
        """Send shutdown commands to all workers."""

        self.command.put({"action": "shutdown"})
        self.frames_for_tracking.put({"action": "shutdown"})
        self.frames_for_rendering.put({"action": "shutdown"})
        self.ga_command.put({"action": "shutdown"})
        self.ba_command.put({"action": "shutdown"})


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
        self.show_all_labels: bool = False

        # Feature flags
        self.keypoint_tracking_enabled: bool = False
        self.needs_3d_reconstruction: bool = True

        # Batch tracking control
        self.stop_batch_track = threading.Event()

        # Annotation data
        num_frames = video_metadata['num_frames']
        num_videos = video_metadata['num_videos']
        self.annotations = np.full(
            (num_frames, num_videos, self.num_points, 2),
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

        # Calibration state
        self.calibration_frames: List[int] = []
        self.best_individual: Optional[List[Dict[str, Any]]] = None
        self.best_fitness: float = float('inf')
        self.fundamental_matrices: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self.is_ga_running: bool = False

        self.scene_centre = np.zeros(3)

    def set_calibration(self, individual: List[Dict[str, Any]]):
        """Apply camera calibration and update dependent state."""

        with self.lock:
            self.best_individual = individual
            self.fundamental_matrices = calculate_fundamental_matrices(individual)
            self.best_fitness = 0.0  # Loaded calibrations have fitness 0
            self.needs_3d_reconstruction = True
        print("Successfully applied camera calibration.")

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of state needed by the GA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "best_fitness": self.best_fitness,
                "best_individual": self.best_individual,
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of state needed by the BA worker."""

        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "best_individual": self.best_individual,
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
                    json.dump(self.calibration_frames, f)

                if self.best_individual is not None:
                    with open(folder / 'best_individual.pkl', 'wb') as f:
                        pickle.dump(self.best_individual, f)

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
            for attr_name, value in loaded_data.items():
                setattr(self, attr_name, value)

        print("State loading complete.")

    def load_calibration_from_toml(self, file_path: Path):
        """Load camera calibration from TOML file."""

        print(f"Loading calibration from '{file_path}'")
        if not file_path.exists():
            print("  - ERROR: File not found.")
            return

        try:
            with file_path.open("rb") as f:
                calib_data = tomllib.load(f)
        except Exception as e:
            print(f"  - ERROR: Could not parse TOML file: {e}")
            return

        # Assume sorted order matches video order
        sorted_camera_names = sorted(calib_data.keys())

        individual = []
        for cam_name in sorted_camera_names:
            cam = calib_data[cam_name]
            individual.append({
                'fx': cam['camera_matrix'][0][0],
                'fy': cam['camera_matrix'][1][1],
                'cx': cam['camera_matrix'][0][2],
                'cy': cam['camera_matrix'][1][2],
                'dist': np.array(cam['dist_coeffs'], dtype=np.float32),
                'rvec': np.array(cam['rvec'], dtype=np.float32),
                'tvec': np.array(cam['tvec'], dtype=np.float32),
            })

        self.set_calibration(individual)