from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseTracker(ABC):
    def __init__(self, app_state):
        self.app_state = app_state

    @abstractmethod
    def initialize_tracks(self, frame_idx: int, direction: int = 1):
        """
        Prepare tracker state (Kalman filters, etc) starting from a specific frame.
        """
        pass

    @abstractmethod
    def track_frame(self,
                    frame_idx: int,
                    prev_frame_idx: int,
                    source_frames: List[np.ndarray],
                    dest_frames: List[np.ndarray],
                    active_point_indices: Optional[List[int]] = None) -> bool:
        """
        Process a single frame step.
        Returns True if successful, False if tracking failed/collided.
        """
        pass

    @abstractmethod
    def get_debug_info(self) -> dict:
        """Returns internal state stats (e.g. 'Walking', 'Stationary') for UI."""
        return {}
