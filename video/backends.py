"""
Video backend for video frame access:
- CachedBackend: Uses compressed disk cache for fast random access
- DirectBackend: Uses cv2.VideoCapture with in-memory caching
- HybridBackend: Combines both strategies based on access patterns
"""
import collections
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, TYPE_CHECKING, Union

import cv2
import numpy as np

if TYPE_CHECKING:
    from video.disk_cache import DiskCacheReader


class VideoBackend(ABC):
    """
    Abstract base class for video frame access.
    """

    def __init__(
        self,
        video_paths: Dict[str, Union[Path, str]],
        video_metadata: Dict
    ):
        """
        Initialise backend with video paths and metadata.

        Args:
            video_paths: Dict mapping camera_name -> video path
            video_metadata: Dict with keys: fps, nb_frames, video_dims (per-camera)
        """
        self.video_paths: Dict[str, Path] = {
            name: Path(p) for name, p in video_paths.items()
        }
        # Canonical ordering: sorted camera names
        self.camera_names: tuple[str, ...] = tuple(sorted(self.video_paths.keys()))
        self.metadata = video_metadata
        self.video_count = len(self.video_paths)
        self.lock = threading.RLock()

    @abstractmethod
    def get_frame(
        self,
        frame_idx: int,
        cameras: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get a specific frame from selected cameras.

        Args:
            frame_idx: Frame index (0 to num_frames-1)
            cameras: List of camera names, or None for all cameras

        Returns:
            Dict mapping camera_name -> BGR numpy array (H, W, 3)
        """
        pass

    @abstractmethod
    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """
        Hint that these frames will be needed soon.

        Args:
            frame_indices: List of frame indices to prefetch
            priority: Higher priority = prefetch sooner (0 = normal)
        """
        pass

    @abstractmethod
    def clear_cache(self):
        """Clear any in-memory caches to free RAM."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get backend statistics for debugging/monitoring.

        Returns:
            Dict with backend-specific stats (cache hits, memory usage, etc.)
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources (file handles, threads, etc.)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DirectBackend(VideoBackend):
    """
    Direct video file access using cv2.VideoCapture with in-memory LRU cache.
    """

    def __init__(
        self,
        video_paths: Dict[str, Union[Path, str]],
        video_metadata: Dict,
        ram_budget_gb: float = 1.5
    ):
        """
        Initialise direct backend.

        Args:
            video_paths: Dict mapping camera_name -> video path
            video_metadata: Video metadata dictionary
            ram_budget_gb: Maximum RAM to use for frame cache (default: 1.5 GB)
        """
        super().__init__(video_paths, video_metadata)

        # Get dimensions from metadata (use first camera as reference)
        video_dims = video_metadata.get('video_dims', {})
        if video_dims:
            first_cam = self.camera_names[0]
            width, height = video_dims.get(first_cam, (1920, 1080))
        else:
            # Fallback to legacy format
            width = video_metadata.get('width', 1920)
            height = video_metadata.get('height', 1080)

        # Calculate cache capacity based on RAM budget
        frame_bytes = width * height * 3
        total_capacity_frames = int((ram_budget_gb * 1024 ** 3) // frame_bytes)
        self.cache_capacity = max(10, total_capacity_frames // self.video_count)

        # Per-camera caches (LRU) and VideoCapture objects
        self.caches: Dict[str, collections.OrderedDict] = {
            cam: collections.OrderedDict() for cam in self.camera_names
        }
        self.captures: Dict[str, cv2.VideoCapture] = {
            cam: cv2.VideoCapture(self.video_paths[cam].as_posix())
            for cam in self.camera_names
        }

        # Track current file pointer position for each capture
        self.file_pointers: Dict[str, int] = {cam: -1 for cam in self.camera_names}

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'sequential_reads': 0,
            'seek_reads': 0
        }

        # Verify all captures opened successfully
        failed = [cam for cam, cap in self.captures.items() if not cap.isOpened()]
        if failed:
            raise RuntimeError(f"Failed to open video files for cameras: {failed}")

        print(f"DirectBackend initialised: {self.cache_capacity} frames/camera, "
              f"{total_capacity_frames} total frames, "
              f"{(total_capacity_frames * frame_bytes) / 1024 ** 3:.2f} GB budget")

    def get_frame(
        self,
        frame_idx: int,
        cameras: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get frame from cache or video file."""

        num_frames = self.metadata.get('nb_frames', self.metadata.get('num_frames', float('inf')))
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range")

        if cameras is None:
            cameras = list(self.camera_names)

        frames = {}
        with self.lock:
            for cam_name in cameras:
                if cam_name not in self.camera_names:
                    raise ValueError(f"Unknown camera: '{cam_name}'")
                frame = self._get_single_frame(cam_name, frame_idx)
                if frame is None:
                    raise RuntimeError(f"Failed to read frame {frame_idx} from camera '{cam_name}'")
                frames[cam_name] = frame

        return frames

    def _get_single_frame(self, camera_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get a single frame from one camera (assumes lock is held)."""

        cache = self.caches[camera_name]

        # Check cache first
        if frame_idx in cache:
            self.stats['cache_hits'] += 1
            cache.move_to_end(frame_idx)
            return cache[frame_idx]

        # Cache miss - read from file
        self.stats['cache_misses'] += 1
        cap = self.captures[camera_name]

        # Optimise for sequential access
        if frame_idx == self.file_pointers[camera_name]:
            ret, frame = cap.read()
            self.stats['sequential_reads'] += 1
            if ret:
                self.file_pointers[camera_name] += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            self.stats['seek_reads'] += 1
            if ret:
                self.file_pointers[camera_name] = frame_idx + 1

        if not ret:
            return None

        # Add to cache
        cache[frame_idx] = frame
        cache.move_to_end(frame_idx)

        # Enforce cache size limit
        while len(cache) > self.cache_capacity:
            cache.popitem(last=False)

        return frame

    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """Prefetch frames (TODO: background thread loading)."""
        pass

    def clear_cache(self):
        """Clear all in-memory frame caches."""

        with self.lock:
            for cache in self.caches.values():
                cache.clear()
            print("DirectBackend: RAM cache cleared")

    def get_stats(self) -> Dict:
        """Get backend statistics."""

        with self.lock:
            total_cached = sum(len(cache) for cache in self.caches.values())
            hit_rate = (self.stats['cache_hits'] /
                        max(1, self.stats['cache_hits'] + self.stats['cache_misses']))

            return {
                'backend_type': 'direct',
                'camera_names': list(self.camera_names),
                'frames_in_cache': total_cached,
                'cache_capacity': self.cache_capacity * self.video_count,
                'cache_hit_rate': hit_rate,
                'sequential_reads': self.stats['sequential_reads'],
                'seek_reads': self.stats['seek_reads'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses']
            }

    def close(self):
        """Release all video captures."""

        with self.lock:
            for cap in self.captures.values():
                cap.release()
            self.captures.clear()
            self.caches.clear()


class CachedBackend(VideoBackend):
    """
    Cached video backend using compressed disk chunks.
    Maintains an LRU cache of decompressed chunks in RAM.
    """

    def __init__(
        self,
        video_paths: Dict[str, Union[Path, str]],
        video_metadata: Dict,
        cache_dir: Optional[str] = None,
        cache_reader: Optional['DiskCacheReader'] = None,
        ram_budget_gb: float = 2.0
    ):
        """
        Initialise cached backend.

        Args:
            video_paths: Dict mapping camera_name -> video path (used for metadata only)
            video_metadata: Video metadata dictionary
            cache_dir: Path to cache directory (will create DiskCacheReader)
            cache_reader: Pre-initialized DiskCacheReader instance (alternative)
            ram_budget_gb: RAM budget for chunk cache
        """
        super().__init__(video_paths, video_metadata)

        if cache_reader is not None:
            self.cache_reader = cache_reader
        elif cache_dir is not None:
            from video.disk_cache import DiskCacheReader
            self.cache_reader = DiskCacheReader(
                cache_dir=cache_dir,
                ram_budget_gb=ram_budget_gb
            )
        else:
            raise ValueError("Must provide either cache_dir or cache_reader")

        print(f"CachedBackend initialised: {self.cache_reader}")

    def get_frame(
        self,
        frame_idx: int,
        cameras: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get frame from disk cache."""

        num_frames = self.metadata.get('nb_frames', self.metadata.get('num_frames', float('inf')))
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range")

        with self.lock:
            return self.cache_reader.get_frame(frame_idx, cameras)

    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """Trigger prefetch for upcoming frames (handled by DiskCacheReader)."""
        pass

    def clear_cache(self):
        """Clear the in-memory chunk cache."""
        with self.lock:
            self.cache_reader.clear_cache()
            print("CachedBackend: RAM chunk cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            cache_info = self.cache_reader.get_cache_info()
            return {
                'backend_type': 'cached',
                'camera_names': cache_info.get('camera_names', list(self.camera_names)),
                'cache_dir': cache_info['cache_dir'],
                'chunks_in_ram': cache_info['chunks_in_ram'],
                'ram_limit_chunks': cache_info['ram_limit_chunks'],
                'total_size_gb': cache_info['total_size_gb'],
                'frames_per_chunk': cache_info['frames_per_chunk']
            }

    def close(self):
        """Clean up cache resources."""

        with self.lock:
            if hasattr(self.cache_reader, '_executor'):
                self.cache_reader._executor.shutdown(wait=False)


class HybridBackend(VideoBackend):
    """
    Hybrid backend that uses cache when available, falls back to direct access.

    Useful during cache building or when cache is partially available.
    """

    def __init__(
        self,
        video_paths: Dict[str, Union[Path, str]],
        video_metadata: Dict,
        cache_reader: Optional['DiskCacheReader'] = None,
        ram_budget_gb: float = 1.0
    ):
        """
        Initialise hybrid backend.

        Args:
            video_paths: Dict mapping camera_name -> video path
            video_metadata: Video metadata dictionary
            cache_reader: Optional DiskCacheReader instance
            ram_budget_gb: RAM budget for direct backend fallback
        """
        super().__init__(video_paths, video_metadata)

        self.cache_reader = cache_reader
        self.direct_backend = DirectBackend(video_paths, video_metadata, ram_budget_gb)

        self.stats = {
            'cache_reads': 0,
            'direct_reads': 0
        }

        print(f"HybridBackend initialised with {'cache' if cache_reader else 'no cache'}")

    def get_frame(
        self,
        frame_idx: int,
        cameras: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get frame from cache if available, otherwise use direct backend."""

        num_frames = self.metadata.get('nb_frames', self.metadata.get('num_frames', float('inf')))
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range")

        with self.lock:
            if self.cache_reader is not None:
                try:
                    self.stats['cache_reads'] += 1
                    return self.cache_reader.get_frame(frame_idx, cameras)
                except Exception as e:
                    print(f"Cache read failed, falling back to direct: {e}")

            self.stats['direct_reads'] += 1
            return self.direct_backend.get_frame(frame_idx, cameras)

    def update_cache_reader(self, cache_reader: Optional['DiskCacheReader']):
        """Hot-swap the cache reader."""
        with self.lock:
            self.cache_reader = cache_reader
            print(f"HybridBackend: Cache reader updated ({'enabled' if cache_reader else 'disabled'})")

    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """Prefetch from active backend."""

        with self.lock:
            if self.cache_reader is None:
                self.direct_backend.prefetch(frame_indices, priority)

    def clear_cache(self):
        """Clear caches from both backends."""

        with self.lock:
            if self.cache_reader is not None:
                self.cache_reader.clear_cache()
            self.direct_backend.clear_cache()

    def get_stats(self) -> Dict:
        """Get combined statistics."""

        with self.lock:
            base_stats = {
                'backend_type': 'hybrid',
                'camera_names': list(self.camera_names),
                'cache_enabled': self.cache_reader is not None,
                'cache_reads': self.stats['cache_reads'],
                'direct_reads': self.stats['direct_reads']
            }

            if self.cache_reader is not None:
                cache_info = self.cache_reader.get_cache_info()
                base_stats.update({
                    'chunks_in_ram': cache_info['chunks_in_ram'],
                    'cache_size_gb': cache_info['total_size_gb']
                })

            direct_stats = self.direct_backend.get_stats()
            base_stats.update({
                'direct_frames_in_cache': direct_stats['frames_in_cache'],
                'direct_cache_hit_rate': direct_stats['cache_hit_rate']
            })

            return base_stats

    def close(self):
        """Clean up both backends."""

        with self.lock:
            if self.cache_reader is not None and hasattr(self.cache_reader, '_executor'):
                self.cache_reader._executor.shutdown(wait=False)
            self.direct_backend.close()


def create_video_backend(
    video_paths: Dict[str, Union[Path, str]],
    video_metadata: Dict,
    cache_dir: Optional[str] = None,
    cache_reader: Optional['DiskCacheReader'] = None,
    backend_type: str = 'auto',
    ram_budget_gb: float = 1.5
) -> VideoBackend:
    """
    Factory function to create the appropriate video backend.

    Args:
        video_paths: Dict mapping camera_name -> video path
        video_metadata: Video metadata dictionary
        cache_dir: Path to cache directory (for CachedBackend)
        cache_reader: Optional DiskCacheReader instance (alternative to cache_dir)
        backend_type: 'auto', 'direct', 'cached', or 'hybrid'
        ram_budget_gb: RAM budget for direct backend or cache
    """
    if backend_type == 'auto':
        if cache_reader is not None or cache_dir is not None:
            return CachedBackend(video_paths, video_metadata, cache_dir, cache_reader, ram_budget_gb)
        else:
            return DirectBackend(video_paths, video_metadata, ram_budget_gb)

    elif backend_type == 'direct':
        return DirectBackend(video_paths, video_metadata, ram_budget_gb)

    elif backend_type == 'cached':
        if cache_reader is None and cache_dir is None:
            raise ValueError("backend_type='cached' requires cache_reader or cache_dir")
        return CachedBackend(video_paths, video_metadata, cache_dir, cache_reader, ram_budget_gb)

    elif backend_type == 'hybrid':
        return HybridBackend(video_paths, video_metadata, cache_reader, ram_budget_gb)

    else:
        raise ValueError(f"Invalid backend_type: {backend_type}")