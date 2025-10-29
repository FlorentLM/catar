import threading
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class AppState:
    """thread-safe container for the application's shared state"""

    def __init__(self, video_metadata: Dict[str, Any], skeleton_config: Dict[str, Any]):
        self.lock = threading.Lock()

        # Metadata (read-only after init)
        self.video_metadata = video_metadata
        self.POINT_NAMES = skeleton_config['point_names']
        self.SKELETON = skeleton_config['skeleton']
        self.point_colors = skeleton_config['point_colors']
        self.num_points = len(self.POINT_NAMES)
        self.video_names: List[str] = []

        # UI & Control state
        self.frame_idx: int = 0
        self.paused: bool = True
        self.selected_point_idx: int = 0
        self.focus_selected_point: bool = False
        self.show_cameras_in_3d: bool = True
        self.keypoint_tracking_enabled: bool = False
        self.save_output_video: bool = False
        self.needs_3d_reconstruction: bool = True

        # Core data
        num_frames = self.video_metadata['num_frames']
        num_videos = self.video_metadata['num_videos']
        self.annotations: np.ndarray = np.full((num_frames, num_videos, self.num_points, 2), np.nan, dtype=np.float32)
        self.human_annotated: np.ndarray = np.zeros((num_frames, num_videos, self.num_points), dtype=bool)
        self.reconstructed_3d_points: np.ndarray = np.full((num_frames, self.num_points, 3), np.nan, dtype=np.float32)

        # Calibration & GA state
        self.calibration_frames: List[int] = []
        self.best_individual: Optional[List[Dict[str, Any]]] = None
        self.best_fitness: float = float('inf')
        self.is_ga_running: bool = False

    def get_ga_state_snapshot(self) -> Dict[str, Any]:
        """Returns a copy of all data needed by the GA worker."""
        with self.lock:
            return {
                "annotations": self.annotations.copy(),
                "calibration_frames": list(self.calibration_frames),
                "video_metadata": self.video_metadata.copy(),
                "best_fitness": self.best_fitness,
                "best_individual": self.best_individual, # This can be large... consider sending only on start?
                "generation": 0, # GA worker manages its own generation count
            }

    def load_data_from_files(self, data_folder: Path):
        """Loads all persistent data from files into the state object."""

        print(f"\nAttempting to load saved state from: '{data_folder}'")
        # TODO: Simplify this, no need for a thousand files of different file formats
        files_to_load = {
            'numpy': ['annotations.npy', 'human_annotated.npy', 'reconstructed_3d_points.npy'],
            'pickle': ['best_individual.pkl'],
            'json': ['calibration_frames.json']
        }

        loaded_data = {}
        for file_type, filenames in files_to_load.items():
            for filename in filenames:
                file_path = data_folder / filename
                if file_path.exists():
                    try:
                        if file_type == 'numpy': data = np.load(file_path)
                        elif file_type == 'pickle':
                            with file_path.open('rb') as f: data = pickle.load(f)
                        elif file_type == 'json':
                            with file_path.open('r') as f: data = json.load(f)

                        loaded_data[filename.split('.')[0]] = data
                        print(f"  - Successfully loaded '{filename}'")
                    except Exception as e:
                        print(f"  - WARNING: Could not load '{filename}'. Error: {e}")

        if not loaded_data:
            print("No saved state files found. Starting with a fresh state.")
            return

        with self.lock:
            for key, value in loaded_data.items():
                setattr(self, key, value)
        print("\nState loading complete.")

    def save_data_to_files(self, data_folder: Path):
        """Saves all persistent data from the state object to files."""

        print(f"\nAttempting to save state to: '{data_folder}'")
        with self.lock:
            try:
                np.save(data_folder / 'annotations.npy', self.annotations)
                np.save(data_folder / 'human_annotated.npy', self.human_annotated)
                np.save(data_folder / 'reconstructed_3d_points.npy', self.reconstructed_3d_points)

                with open(data_folder / 'calibration_frames.json', 'w') as f:
                    json.dump(self.calibration_frames, f)

                if self.best_individual is not None:
                    with open(data_folder / 'best_individual.pkl', 'wb') as f:
                        pickle.dump(self.best_individual, f)

                print("State saved successfully.")
            except Exception as e:
                print(f"Error saving state: {e}")