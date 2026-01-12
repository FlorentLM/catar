"""
All application state and communication queues are managed here.
"""
import queue
import threading
import multiprocessing
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

import config
from state.data_manager import DataManager
from lucida.utils import probe_video
from lucida import CameraRig


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
        video_paths: List[Union[Path, str]],
        rig: CameraRig,
        skeleton_config: Dict[str, Any]
    ):
        if len(rig) != len(video_paths):
            raise ValueError("Mismatch between number of cameras in rig and video paths.")

        self.data_folder = Path(data_folder)
        if hasattr(config, 'VIDEO_CACHE_FOLDER'):
            self.video_cache_dir = Path(config.VIDEO_CACHE_FOLDER)
        else:
            self.video_cache_dir = self.data_folder / 'video_cache'

        self.video_backend = None

        # Keep top-level lock for app-wide state coordination
        self.lock = threading.RLock()

        # Video information (constant during runtime)
        self.video_paths: List[Path] = [Path(p).resolve() for p in video_paths]
        self.video_filenames: List[str] = [Path(p).name for p in video_paths]

        # Probe videos for metadata
        self._video_metadata: Dict[str, Dict[str, Any]] = {}
        for name, path in zip(rig.names, self.video_paths):
            self._video_metadata[name] = probe_video(path)

            # Update rig image sizes and scale K if mismatch (e.g. using proxies)
            cam = rig[name]
            vid_w = self._video_metadata[name]['width']
            vid_h = self._video_metadata[name]['height']

            # Check for explicit resolution mismatch vs what was loaded
            if cam.intrinsics.width != vid_w or cam.intrinsics.height != vid_h:
                print(f"Resizing camera '{name}' intrinsics from {cam.intrinsics.image_size} to {(vid_w, vid_h)}")
                rig[name] = cam.resize((vid_w, vid_h))

        # Use first video as session reference
        first_metadata = self._video_metadata[rig.names[0]]
        self.video_metadata = first_metadata.copy()
        self.video_metadata['num_videos'] = len(video_paths)

        # Video metadata
        self.frame_width: int = first_metadata['width']
        self.frame_height: int = first_metadata['height']
        self.num_frames: int = first_metadata['num_frames']
        self.video_duration: float = first_metadata['duration']
        self.fps: float = first_metadata['fps'] or 30.0

        half_life_frames = config.TRACKER_HALF_LIFE_CONFIDENCE_DECAY * self.fps
        if half_life_frames > 0:
            self.tracker_decay_rate = 0.5 ** (1.0 / half_life_frames)
        else:
            self.tracker_decay_rate = 0.0
        print(f"Tracker: Half-life {config.TRACKER_HALF_LIFE_CONFIDENCE_DECAY:.1f}s @ {self.fps:.2f} FPS "
              f"-> Decay rate per frame: {self.tracker_decay_rate:.4f}")

        # State objects
        self.rig: CameraRig = rig

        # Additional calibration state not held in Lucida
        self.calibration_frames: List[int] = []
        self.best_fitness: float = float('inf')

        # Default scene bounds and centre
        self.volume_bounds = {'x': (-1e9, 1e9), 'y': (-1e9, 1e9), 'z': (-1e9, 1e9)}
        self.scene_centre = np.zeros(3)

        # Keypoints order and lookup accessors
        self.point_names = tuple(skeleton_config['point_names'])
        self.point_nti = {name: i for i, name in enumerate(self.point_names)}
        self.point_itn = {i: name for name, i in self.point_nti.items()}

        self.skeleton = skeleton_config['skeleton']
        self.all_points_set = set(self.point_names)

        # TODO: Derive skeleton_points_set from skeleton config (points that are part of the skeleton graph)
        # TODO: Add GUI functionality to add/remove non-skeleton points for object tracking and scaffolding
        self.skeleton_points_set = set(self.point_names) - {'s_small', 's_large'}
        self.non_skeleton_points_set = {'s_small', 's_large'}

        self.point_colors = skeleton_config['point_colors']
        self.camera_colors = config.CAMERA_COLORS
        self.num_points = len(self.point_names)

        # Centralised data manager
        self.data = DataManager(
            n_frames=self.num_frames,
            n_cameras=len(rig),
            n_points=self.num_points
        )

        # Playback State
        self.frame_idx: int = 0
        self.paused: bool = True
        self.is_seeking: bool = False

        # UI State
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

        # Transient UI data
        # Frame cache for UI zoom (raw frames from current playback position)
        self.current_video_frames: Optional[List[np.ndarray]] = None

    def get_video_metadata(self, camera_name: str) -> Dict[str, Any]:
        """Get metadata for a specific camera's video."""
        return self._video_metadata[camera_name]

    # Snapshot Methods (for GA and BA workers)

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """
        Create snapshot of state needed by the GA worker.
        Camera names and image sizes are derived from the rig.
        """
        with self.lock:
            with self.data.bulk_lock():
                annotations_copy = self.data.annotations.copy()

            return {
                "annotations": annotations_copy,
                "calibration_frames": list(self.calibration_frames),
                "best_fitness": self.best_fitness,
                "best_individual": self.rig.to_dict(),
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of state needed by the BA worker.
        Camera names and image sizes are derived from the rig.
        """
        with self.lock:
            with self.data.bulk_lock():
                annotations_copy = self.data.annotations.copy()

            return {
                "annotations": annotations_copy,
                "calibration_frames": list(self.calibration_frames),
                "best_individual": self.rig.to_dict(),
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
                    json.dump(self.calibration_frames, f)

                self.rig.save(folder / 'rig.toml')

                print("State saved successfully.")
            except Exception as e:
                print(f"Error saving state: {e}")

    def load_from_disk(self, folder: Path):
        """
        Load persistent state from disk.
        """
        print(f"Loading state from: '{folder}'")

        files_to_load = [
            ('annotations.npy', 'numpy', 'annotations'),
            ('human_annotated.npy', 'numpy', 'human_annotated'),
            ('reconstructed_3d.npy', 'numpy', 'reconstructed_3d'),
            ('calibration_frames.json', 'json', 'calibration_frames'),
            # Rig is loaded via CameraRig.load separately, checked below
        ]

        loaded_data = {}

        for filename, file_type, attr_name in files_to_load:
            file_path = folder / filename
            if not file_path.exists():
                continue

            try:
                if file_type == 'numpy':
                    loaded_data[attr_name] = np.load(file_path)
                elif file_type == 'json':
                    with file_path.open('r') as f:
                        loaded_data[attr_name] = json.load(f)
                print(f"  - Loaded '{filename}'")
            except Exception as e:
                print(f"  - WARNING: Could not load '{filename}': {e}")

        # Load Rig
        rig_path = folder / 'rig.toml'
        if rig_path.exists():
            try:
                loaded_rig = CameraRig.load(rig_path)

                # Verify camera names match
                if set(loaded_rig.names) == set(self.rig.names):
                    self.rig = loaded_rig
                    print("  - Loaded 'rig.toml'")
                else:
                    print("  - WARNING: Loaded rig camera names do not match current session.")
            except Exception as e:
                print(f"  - WARNING: Could not load rig.toml: {e}")

        with self.lock:
            # Load into DataManager arrays
            with self.data._lock:
                if 'human_annotated' in loaded_data:
                    data = loaded_data['human_annotated']
                    if data.shape == self.data.human_annotated.shape:
                        self.data.human_annotated = data

                if 'reconstructed_3d' in loaded_data:
                    data = loaded_data['reconstructed_3d']
                    if data.shape[0] == self.num_frames and data.shape[1] == self.num_points:
                         self.data.reconstructed_3d = data

                if 'annotations' in loaded_data:
                    annots = loaded_data['annotations']
                    if annots.shape[0] == self.num_frames and annots.shape[1] == len(self.rig):
                        self.data.annotations = annots

            if 'calibration_frames' in loaded_data:
                self.calibration_frames = loaded_data['calibration_frames']

        print("State loading complete.")