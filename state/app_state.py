"""
All application state and communication queues are managed here.
"""
import queue
import sys
import threading
import multiprocessing

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

import config
from gui.rendering import Viewer3D

from state.data_manager import DataManager

from lucida.utils import probe_video
from lucida import CameraRig

from utils import load_and_match_videos

from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonStats

from video import DiskCacheBuilder, create_video_backend


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    frame_count: int
    fps: float


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

    def __init__(self):

        self.lock = threading.RLock()

        # Load config values # TODO: Use an actual config file
        data_folder = Path(config.DATA_FOLDER) if hasattr(config, 'DATA_FOLDER') else Path.cwd() / 'data'
        rig_path = Path(config.CAMERA_RIG_TOML) if hasattr(config, 'CAMERA_RIG_TOML') else data_folder / 'rig.toml'
        skel_path = Path(config.SKELETON_FILE) if hasattr(config, 'SKELETON_FILE') else data_folder / 'messor_skeleton.toml'
        skel_stats_path = Path(config.SKELETON_STATS_FILE) if hasattr(config, 'SKELETON_STATS_FILE') else data_folder / 'skeleton_stats.json'
        vid_folder = Path(config.VIDEOS_PATH) if hasattr(config, 'VIDEOS_PATH') else data_folder
        vid_fmt = config.VIDEO_FORMAT if hasattr(config, 'VIDEO_FORMAT') else 'mp4'
        vid_cache = Path(config.VIDEO_CACHE_FOLDER) if hasattr(config, 'VIDEO_CACHE_FOLDER') else data_folder / 'video_cache'
        disable_viwer_3d = config.DISABLE_3D_VIEW if hasattr(config, 'DISABLE_3D_VIEW') else False


        # Create data folder if first launch
        if not data_folder.is_dir():
            data_folder.mkdir(parents=True)
            print(f"Created '{data_folder}' directory. Please add videos and calibration.")
            sys.exit(0)
        else:
            self.data_folder = data_folder

        # Load camera rig and match video files
        camera_rig, ordered_video_paths = load_and_match_videos(
            rig_path=rig_path,
            videos_folder=vid_folder,
            video_format=vid_fmt
        )
        self.rig = camera_rig
        self.video_paths: Dict[str, Path] = dict(zip(camera_rig.names, ordered_video_paths))

        # Videos information
        self.video_info: Dict[str, VideoInfo] = {}
        for cam_name, path in self.video_paths.items():
            m = probe_video(path)
            self.video_info[cam_name] = VideoInfo(
                path=path,
                width=m['width'],
                height=m['height'],
                frame_count=m['num_frames'],
                fps=m['fps']
            )

        fps_vals = [v.fps for v in self.video_info.values()]
        nbframes_vals = [m.frame_count for m in self.video_info.values()]

        # All framerates should be identidal (whithin smol tolerance)
        for fps in fps_vals:
            if abs(fps - fps_vals[0]) > 0.01:
                raise ValueError(f"Inconsistent framerates detected. All cameras must have synchronized framerates.")

        self.fps = fps_vals[0]  # all the same, take any
        self.frame_count = min(nbframes_vals)  # take the one with fewer frames just in case # TODO: could alternatively get the readers to retrn blakc frames for shorted videos

        print("Initialising video backend...")
        self.video_cache_dir = vid_cache
        builder = DiskCacheBuilder(
            video_paths=self.video_paths,
            cache_dir=self.video_cache_dir
        )
        cache_exists, cache_metadata = builder.check_cache_exists()

        video_backend = create_video_backend(
            video_paths=self.video_paths,
            video_metadata=self.video_info,
            cache_dir=builder.cache_dir if cache_exists else None,
            backend_type='auto',
            ram_budget_gb=config.RAM_MAX_BUDGET_GB
        )
        print(f"Using backend: {type(video_backend).__name__}")
        self.video_backend = video_backend


        # Tracking params
        half_life_frames = config.TRACKER_HALF_LIFE_CONFIDENCE_DECAY * self.fps
        if half_life_frames > 0:
            self.tracker_decay_rate = 0.5 ** (1.0 / half_life_frames)
        else:
            self.tracker_decay_rate = 0.0

        # Load skeleton definition and stats
        self.skeleton = Skeleton.load(skel_path)
        self.skeleton_stats = SkeletonStats.load(skel_stats_path, self.skeleton)

        if self.skeleton_stats.reference_bone:
            print(f"  Reference bone: {self.skeleton_stats.reference_bone}")
            print(
                f"  Reference length: {self.skeleton_stats.reference_length_world:.2f}"
                f" {self.skeleton.metadata.units if self.skeleton.metadata else 'units'}")
        else:
            print("  No reference bone set (stats will be learned online)")

        # Keypoints order and lookup accessors

        self.point_nti = {name: i for i, name in enumerate(self.skeleton.keypoints)}
        self.point_itn = {i: name for name, i in self.point_nti.items()}

        # TODO: Derive skeleton_points_set from skeleton config (points that are part of the skeleton graph)
        # TODO: Add GUI functionality to add/remove non-skeleton points for object tracking and scaffolding
        self.non_skeleton_points = ('s_small', 's_large')

        # Additional calibration state not held in Lucida
        self.calibration_frames: List[int] = []
        self.ga_best_fitness: float = float('inf')

        # Default scene bounds and centre
        self.volume_bounds = {'x': (-1e9, 1e9), 'y': (-1e9, 1e9), 'z': (-1e9, 1e9)}
        self.scene_centre = np.zeros(3)

        self.camera_colors = config.CAMERA_COLORS

        # Initialise 2D and 3D data manager
        self.data = DataManager(
            nb_frames=self.frame_count,
            camera_names=self.rig.names,
            keypoint_names=self.skeleton.keypoints,
        )

        # Initialise viewer 3D
        self.viewer_3d = Viewer3D() if not disable_viwer_3d else None

        # UI State
        self.frame_idx: int = 0
        self.selected_keypoint: str = self.point_names[0]
        self.focus_mode: bool = False

        self.paused: bool = True
        self.is_seeking: bool = False
        self.drag_state: Dict[str, Any] = {}

        self.show_cameras_in_3d: bool = True
        self.show_reprojection_error: bool = True
        self.show_all_labels: bool = False
        self.show_epipolar_lines: bool = True
        self.temp_hide_overlays: bool = False

        # Feature flags
        self.live_tracking_enabled: bool = False
        self.needs_reconstruction: bool = True
        self.tracker_collision_stop: bool = True

        # Transient UI data
        # Frame cache for UI zoom (raw frames from current playback position)
        self.current_frames_data: Optional[List[np.ndarray]] = None     # TODO: this needs to be a dict

    @property
    def point_names(self):
        return list(self.skeleton.keypoints) + list(self.non_skeleton_points)

    @property
    def num_points(self):
        return len(self.point_names)

    # Snapshot Methods (for GA and BA workers)

    def get_ga_snapshot(self) -> Dict[str, Any]:
        """
        Create snapshot of state needed by the GA worker.
        """
        with self.lock:
            return {
                "annotations": self.data.get_2d(copy=True),
                "calibration_frames": list(self.calibration_frames),
                "best_fitness": self.ga_best_fitness,
                "best_calibration": self.rig.to_dict(),
                "generation": 0,
                "scene_centre": self.scene_centre.copy()
            }

    def get_ba_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of state needed by the BA worker.
        """
        with self.lock:
            return {
                "annotations": self.data.get_2d(copy=True),
                "calibration_frames": list(self.calibration_frames),
                "best_calibration": self.rig.to_dict(),
            }

    # Persistence

    def save(self, folder: Optional[Union[str, Path]] = None):
        """
        Save all persistent state to disk.
        """
        if folder is None:
            folder = self.data_folder
        else:
            folder = Path(folder)
        print(f"Saving state to: '{folder}'")

        with self.lock:
            try:
                self.data.save(folder, prefix='catar')
                print("  - Saved annotations and 3D points (Parquet format)")


                with open(folder / 'calibration_frames.json', 'w') as f:
                    json.dump(self.calibration_frames, f)
                print("  - Saved 'calibration_frames.json'")

                self.rig.save(folder / 'rig.toml')
                print("  - Saved 'rig.toml'")

                print("State saved successfully.")
            except Exception as e:
                print(f"Error saving state: {e}")
                import traceback
                traceback.print_exc()

    def load(self, folder: Optional[Union[str, Path]] = None):
        """
        Load persistent state from disk.
        """
        if folder is None:
            folder = self.data_folder
        else:
           folder = Path(folder)
        print(f"Loading state from: '{folder}'")

        data_files_exist = (
            (folder / 'catar_points2d.parquet').exists() or
            (folder / 'catar_points3d.parquet').exists()
        )

        # Load point data
        if data_files_exist:
            print("  Loading data (Polars/Parquet format)...")
            try:
                self.data.load(folder, prefix='catar')
                print("  - Loaded annotations and 3D points")
            except Exception as e:
                print(f"  - WARNING: Could not load Parquet data: {e}")
        else:
            print("  No saved annotation data found.")

        # Load calibration frames
        calib_path = folder / 'calibration_frames.json'
        if calib_path.exists():
            try:
                with calib_path.open('r') as f:
                    self.calibration_frames = json.load(f)
                print(f"  - Loaded 'calibration_frames.json' ({len(self.calibration_frames)} frames)")
            except Exception as e:
                print(f"  - WARNING: Could not load calibration_frames.json: {e}")

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

        print("State loading complete.")