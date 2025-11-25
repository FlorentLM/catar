"""
Centralized data access layer for annotations and 3D reconstructions.
"""
import threading
import numpy as np
from typing import Optional


class DataManager:
    """
    Centralized manager for all annotation and 3D point data.
    All array indexing is validated.
    """

    def __init__(self, n_frames: int, n_cameras: int, n_points: int):
        """
        Initialize data manager with empty arrays.

        Args:
            n_frames: Total number of frames in videos
            n_cameras: Number of camera views
            n_points: Number of keypoints per skeleton
        """
        self.n_frames = n_frames
        self.n_cameras = n_cameras
        self.n_points = n_points

        # Core data arrays
        # Shape (F, C, P, 3) where last dim is (x, y, confidence)
        self.annotations = np.full(
            (n_frames, n_cameras, n_points, 3),
            np.nan,
            dtype=np.float32
        )

        # Shape (F, P, 4) where last dim is (x, y, z, score/confidence)
        self.reconstructed_3d = np.full(
            (n_frames, n_points, 4),
            np.nan,
            dtype=np.float32
        )

        # Shape (F, C, P), boolean flags indicating human annotation
        self.human_annotated = np.zeros(
            (n_frames, n_cameras, n_points),
            dtype=bool
        )

        self._lock = threading.RLock()

    # Context managers for lock control

    class _ReadLock:
        """Context manager for read operations."""
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *args):
            self._lock.release()

    class _BulkLock:
        """Context manager for bulk operations."""
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *args):
            self._lock.release()

    def read_lock(self):
        """
        Context manager to use when multiple reads need to be consistent with each other.

        Example:
            with data.read_lock():
                annots = data.get_frame_annotations(frame_idx)
                points = data.get_3d_points(frame_idx)
                # These are guaranteed to be from the same moment
        """
        return self._ReadLock(self._lock)

    def bulk_lock(self):
        """
        Context manager for bulk operations (multiple writes or direct access).

        Example:
            with data.bulk_lock():
                for i in range(10):
                    data.set_annotation_2d_unsafe(frame, cam, i, xy, conf)
                # Lock acquired once for all operations
        """
        return self._BulkLock(self._lock)

    # Validation

    def _validate_frame_idx(self, frame_idx: int):
        """Validate frame index is in bounds."""
        if not (0 <= frame_idx < self.n_frames):
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self.n_frames})"
            )

    def _validate_camera_idx(self, cam_idx: int):
        """Validate camera index is in bounds."""
        if not (0 <= cam_idx < self.n_cameras):
            raise IndexError(
                f"Camera index {cam_idx} out of range [0, {self.n_cameras})"
            )

    def _validate_point_idx(self, point_idx: int):
        """Validate point index is in bounds."""
        if not (0 <= point_idx < self.n_points):
            raise IndexError(
                f"Point index {point_idx} out of range [0, {self.n_points})"
            )

    # Read operations - 2D annotations

    def get_annotation(
        self,
        frame_idx: int,
        cam_idx: int,
        point_idx: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get a single 2D annotation point.

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            point_idx: Point/keypoint index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (3,) containing (x, y, confidence)
            By default returns a view (zero-copy) for performance
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)
        self._validate_point_idx(point_idx)

        with self._lock:
            result = self.annotations[frame_idx, cam_idx, point_idx]
            return result.copy() if copy else result

    def get_frame_annotations(self, frame_idx: int, copy: bool = False) -> np.ndarray:
        """
        Get all annotations for a single frame across all cameras.

        Args:
            frame_idx: Frame index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (C, P, 3), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)

        with self._lock:
            result = self.annotations[frame_idx]
            return result.copy() if copy else result

    def get_camera_annotations(
        self,
        frame_idx: int,
        cam_idx: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get all annotations for a single camera view at a frame.

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (P, 3), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)

        with self._lock:
            result = self.annotations[frame_idx, cam_idx]
            return result.copy() if copy else result

    def get_point_annotations(
        self,
        frame_idx: int,
        point_idx: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get annotations for a single point across all cameras.

        Args:
            frame_idx: Frame index
            point_idx: Point/keypoint index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (C, 3), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)
        self._validate_point_idx(point_idx)

        with self._lock:
            result = self.annotations[frame_idx, :, point_idx]
            return result.copy() if copy else result

    def get_annotation_range(
        self,
        start_frame: int,
        end_frame: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get annotations for a range of frames.

        Args:
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (F, C, P, 3), view into data array (zero-copy)
        """
        self._validate_frame_idx(start_frame)
        self._validate_frame_idx(end_frame)

        if start_frame > end_frame:
            raise ValueError(f"start_frame {start_frame} > end_frame {end_frame}")

        with self._lock:
            result = self.annotations[start_frame:end_frame+1]
            return result.copy() if copy else result

    # Read operations - 3D points

    def get_point3d(
        self,
        frame_idx: int,
        point_idx: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get a single 3D reconstructed point.

        Args:
            frame_idx: Frame index
            point_idx: Point/keypoint index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (4,), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)
        self._validate_point_idx(point_idx)

        with self._lock:
            result = self.reconstructed_3d[frame_idx, point_idx]
            return result.copy() if copy else result

    def get_frame_points3d(self, frame_idx: int, copy: bool = False) -> np.ndarray:
        """
        Get all 3D reconstructed points for a frame.

        Args:
            frame_idx: Frame index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (P, 4), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)

        with self._lock:
            result = self.reconstructed_3d[frame_idx]
            return result.copy() if copy else result

    def get_points3d_range(
        self,
        start_frame: int,
        end_frame: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get 3D points for a range of frames.

        Args:
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (F, P, 4), view into data array (zero-copy)
        """
        self._validate_frame_idx(start_frame)
        self._validate_frame_idx(end_frame)

        if start_frame > end_frame:
            raise ValueError(f"start_frame {start_frame} > end_frame {end_frame}")

        with self._lock:
            result = self.reconstructed_3d[start_frame:end_frame+1]
            return result.copy() if copy else result

    # Read operations - Human annotation flags

    def is_human_annotated(
        self,
        frame_idx: int,
        cam_idx: int,
        point_idx: int
    ) -> bool:
        """
        Check if a specific annotation was created/modified by a human.

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            point_idx: Point/keypoint index

        Returns:
            True if this annotation is human-generated
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)
        self._validate_point_idx(point_idx)

        with self._lock:
            return bool(self.human_annotated[frame_idx, cam_idx, point_idx])

    def get_human_annotated_flags(
        self,
        frame_idx: int,
        copy: bool = False
    ) -> np.ndarray:
        """
        Get all human annotation flags for a frame.

        Args:
            frame_idx: Frame index
            copy: If True, return a copy instead of view (default: False)

        Returns:
            Array of shape (C, P), view into data array (zero-copy)
        """
        self._validate_frame_idx(frame_idx)

        with self._lock:
            result = self.human_annotated[frame_idx]
            return result.copy() if copy else result

    # Write operations - 2D annotations

    def set_annotation(
        self,
        frame_idx: int,
        cam_idx: int,
        point_idx: int,
        xy: np.ndarray,
        confidence: float,
        is_human: bool = False
    ):
        """
        Set a single 2D annotation.

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            point_idx: Point/keypoint index
            xy: Coordinates as array of shape (2,)
            confidence: Confidence value (typically 0.0 to 1.0)
            is_human: Whether this is a human annotation
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)
        self._validate_point_idx(point_idx)

        if xy.shape != (2,):
            raise ValueError(f"xy must have shape (2,), got {xy.shape}")

        with self._lock:
            self.annotations[frame_idx, cam_idx, point_idx, :2] = xy
            self.annotations[frame_idx, cam_idx, point_idx, 2] = confidence
            self.human_annotated[frame_idx, cam_idx, point_idx] = is_human

    def set_annotation_unsafe(
        self,
        frame_idx: int,
        cam_idx: int,
        point_idx: int,
        xy: np.ndarray,
        confidence: float,
        is_human: bool = False
    ):
        """
        Set annotation WITHOUT acquiring lock.
        Only call from within a bulk_lock() context!

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            point_idx: Point/keypoint index
            xy: Coordinates as array of shape (2,)
            confidence: Confidence value
            is_human: Whether this is a human annotation
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)
        self._validate_point_idx(point_idx)

        if xy.shape != (2,):
            raise ValueError(f"xy must have shape (2,), got {xy.shape}")

        # NO LOCK
        self.annotations[frame_idx, cam_idx, point_idx, :2] = xy
        self.annotations[frame_idx, cam_idx, point_idx, 2] = confidence
        self.human_annotated[frame_idx, cam_idx, point_idx] = is_human

    def clear_annotation(
        self,
        frame_idx: int,
        cam_idx: int,
        point_idx: int
    ):
        """
        Clear a single 2D annotation (set to NaN).

        Args:
            frame_idx: Frame index
            cam_idx: Camera index
            point_idx: Point/keypoint index
        """
        self._validate_frame_idx(frame_idx)
        self._validate_camera_idx(cam_idx)
        self._validate_point_idx(point_idx)

        with self._lock:
            self.annotations[frame_idx, cam_idx, point_idx] = np.nan
            self.human_annotated[frame_idx, cam_idx, point_idx] = False

    def set_frame_annotations(
        self,
        frame_idx: int,
        annotations: np.ndarray,
        human_flags: Optional[np.ndarray] = None
    ):
        """
        Set all annotations for a frame (in-place update).

        Args:
            frame_idx: Frame index
            annotations: Array of shape (C, P, 3) with annotations
            human_flags: Optional array of shape (C, P) with human flags
        """
        self._validate_frame_idx(frame_idx)

        expected_shape = (self.n_cameras, self.n_points, 3)
        if annotations.shape != expected_shape:
            raise ValueError(
                f"Annotations must have shape {expected_shape}, got {annotations.shape}"
            )

        if human_flags is not None:
            expected_flags_shape = (self.n_cameras, self.n_points)
            if human_flags.shape != expected_flags_shape:
                raise ValueError(
                    f"Human flags must have shape {expected_flags_shape}, "
                    f"got {human_flags.shape}"
                )

        with self._lock:
            np.copyto(self.annotations[frame_idx], annotations)
            if human_flags is not None:
                np.copyto(self.human_annotated[frame_idx], human_flags)

    # Write operations - 3D points

    def set_point3d(
        self,
        frame_idx: int,
        point_idx: int,
        xyz_score: np.ndarray
    ):
        """
        Set a single 3D reconstructed point.

        Args:
            frame_idx: Frame index
            point_idx: Point/keypoint index
            xyz_score: Array of shape (4,) containing (x, y, z, score)
        """
        self._validate_frame_idx(frame_idx)
        self._validate_point_idx(point_idx)

        if xyz_score.shape != (4,):
            raise ValueError(f"xyz_score must have shape (4,), got {xyz_score.shape}")

        with self._lock:
            self.reconstructed_3d[frame_idx, point_idx] = xyz_score

    def set_frame_points3d(self, frame_idx: int, points_3d: np.ndarray):
        """
        Set all 3D reconstructed points for a frame (in-place update).

        Args:
            frame_idx: Frame index
            points_3d: Array of shape (P, 4) containing (x, y, z, score)
        """
        self._validate_frame_idx(frame_idx)

        expected_shape = (self.n_points, 4)
        if points_3d.shape != expected_shape:
            raise ValueError(
                f"points_3d must have shape {expected_shape}, got {points_3d.shape}"
            )

        with self._lock:
            # In-place copy for performance
            np.copyto(self.reconstructed_3d[frame_idx], points_3d)

    def set_points3d_range(
        self,
        start_frame: int,
        end_frame: int,
        points_3d: np.ndarray
    ):
        """
        Set 3D points for a range of frames (in-place update).

        Args:
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            points_3d: Array of shape (F, P, 4) for the frame range
        """
        self._validate_frame_idx(start_frame)
        self._validate_frame_idx(end_frame)

        if start_frame > end_frame:
            raise ValueError(f"start_frame {start_frame} > end_frame {end_frame}")

        num_frames_in_range = end_frame - start_frame + 1
        expected_shape = (num_frames_in_range, self.n_points, 4)
        if points_3d.shape != expected_shape:
            raise ValueError(
                f"points_3d must have shape {expected_shape}, got {points_3d.shape}"
            )

        with self._lock:
            # In-place copy for performance
            np.copyto(self.reconstructed_3d[start_frame:end_frame+1], points_3d)

    # Bulk operations

    def clear_frame(self, frame_idx: int):
        """Clear all data for a frame (set to NaN/False)."""
        self._validate_frame_idx(frame_idx)

        with self._lock:
            self.annotations[frame_idx] = np.nan
            self.reconstructed_3d[frame_idx] = np.nan
            self.human_annotated[frame_idx] = False

    def clear_all(self):
        with self._lock:
            self.annotations.fill(np.nan)
            self.reconstructed_3d.fill(np.nan)
            self.human_annotated.fill(False)