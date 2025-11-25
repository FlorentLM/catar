"""
Video backend for video frame access:
- CachedBackend: Uses compressed disk cache for fast random access
- DirectBackend: Uses cv2.VideoCapture with in-memory caching
- HybridBackend: Combines both strategies based on access patterns

All backends share the same interface and can be swapped.
"""
import collections
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, TYPE_CHECKING, Any, Union
import cv2
import numpy as np

if TYPE_CHECKING:
    from video.disk_cache import DiskCacheReader


class VideoBackend(ABC):
    """
    Abstract base class for video frame access.
    All implementations must provide thread-safe frame access for multiple synchronized videos.
    """

    def __init__(self, video_paths: List[Union[Path, str]], video_metadata: Dict):
        """
        Initialise backend with video paths and metadata.

        Args:
            video_paths: List of paths to video files (order must match calibration)
            video_metadata: Dict with keys: width, height, num_frames, fps, num_videos
        """
        self.video_paths = tuple([Path(v) for v in video_paths])
        self.metadata = video_metadata
        self.num_views = len(video_paths)
        self.lock = threading.RLock()

    @abstractmethod
    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Get a specific frame from selected views.

        Args:
            frame_idx: Frame index (0 to num_frames-1)
            views: List of view indices, or None for all views

        Returns:
            List of BGR numpy arrays (H, W, 3), one per requested view

        Raises:
            ValueError: If frame_idx is out of range
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

    Optimized for sequential access with efficient seeking for random access.
    Memory budget is configurable to prevent excessive RAM usage.
    """

    def __init__(
            self,
            video_paths: List[Union[Path, str]],
            video_metadata: Dict,
            ram_budget_gb: float = 1.5
    ):
        """
        Initialise direct backend.

        Args:
            video_paths: List of video file paths
            video_metadata: Video metadata dictionary
            ram_budget_gb: Maximum RAM to use for frame cache (default: 1.5 GB)
        """
        super().__init__(video_paths, video_metadata)

        # Calculate cache capacity based on RAM budget
        frame_bytes = video_metadata['width'] * video_metadata['height'] * 3
        total_capacity_frames = int((ram_budget_gb * 1024 ** 3) // frame_bytes)
        self.cache_capacity = max(10, total_capacity_frames // self.num_views)

        # Per-view caches (LRU) and VideoCapture objects
        self.caches = [collections.OrderedDict() for _ in range(self.num_views)]
        self.captures = [cv2.VideoCapture(path.as_posix()) for path in video_paths]

        # Track current file pointer position for each capture (for sequential optimization)
        self.file_pointers = [-1] * self.num_views

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'sequential_reads': 0,
            'seek_reads': 0
        }

        # Verify all captures opened successfully
        if not all(cap.isOpened() for cap in self.captures):
            raise RuntimeError("Failed to open one or more video files")

        print(f"DirectBackend initialised: {self.cache_capacity} frames/video, "
              f"{total_capacity_frames} total frames, "
              f"{(total_capacity_frames * frame_bytes) / 1024 ** 3:.2f} GB budget")

    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """Get frame from cache or video file."""

        if frame_idx < 0 or frame_idx >= self.metadata['num_frames']:
            raise ValueError(f"Frame index {frame_idx} out of range")

        if views is None:
            views = list(range(self.num_views))

        frames = []
        with self.lock:
            for view_idx in views:
                frame = self._get_single_frame(view_idx, frame_idx)
                if frame is None:
                    raise RuntimeError(f"Failed to read frame {frame_idx} from view {view_idx}")
                frames.append(frame)

        return frames

    def _get_single_frame(self, view_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """Get a single frame from one view (assumes lock is held)."""

        cache = self.caches[view_idx]

        # Check cache first
        if frame_idx in cache:
            self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            cache.move_to_end(frame_idx)
            return cache[frame_idx]

        # Cache miss - read from file
        self.stats['cache_misses'] += 1
        cap = self.captures[view_idx]

        # Optimize for sequential access
        if frame_idx == self.file_pointers[view_idx]:
            # File pointer is already at the right position
            ret, frame = cap.read()
            self.stats['sequential_reads'] += 1
            if ret:
                self.file_pointers[view_idx] += 1
        else:
            # Need to seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            self.stats['seek_reads'] += 1
            if ret:
                self.file_pointers[view_idx] = frame_idx + 1

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
        """
        Prefetch frames.
        """
        # TODO: implement background thread loading
        pass

    def clear_cache(self):
        """Clear all in-memory frame caches."""

        with self.lock:
            for cache in self.caches:
                cache.clear()
            print("DirectBackend: RAM cache cleared")

    def get_stats(self) -> Dict:
        """Get backend statistics."""

        with self.lock:
            total_cached = sum(len(cache) for cache in self.caches)
            hit_rate = (self.stats['cache_hits'] /
                        max(1, self.stats['cache_hits'] + self.stats['cache_misses']))

            return {
                'backend_type': 'direct',
                'frames_in_cache': total_cached,
                'cache_capacity': self.cache_capacity * self.num_views,
                'cache_hit_rate': hit_rate,
                'sequential_reads': self.stats['sequential_reads'],
                'seek_reads': self.stats['seek_reads'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses']
            }

    def close(self):
        """Release all video captures."""

        with self.lock:
            for cap in self.captures:
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
            video_paths: List[Union[Path, str]],
            video_metadata: Dict,
            cache_dir: Optional[str] = None,
            cache_reader: Optional['DiskCacheReader'] = None,
            ram_budget_gb: float = 2.0
    ):
        """
        Initialise cached backend.

        Args:
            video_paths: List of video file paths (used for metadata only)
            video_metadata: Video metadata dictionary
            cache_dir: Path to cache directory (will create DiskCacheReader)
            cache_reader: Pre-initialized DiskCacheReader instance (alternative)
            ram_budget_gb: RAM budget for chunk cache
        """
        super().__init__(video_paths, video_metadata)

        # Accept either cache_dir or cache_reader
        if cache_reader is not None:
            self.cache_reader = cache_reader
        elif cache_dir is not None:
            # import at runtime to avoid circular dependency
            from video.disk_cache import DiskCacheReader
            self.cache_reader = DiskCacheReader(
                cache_dir=cache_dir,
                ram_budget_gb=ram_budget_gb
            )
        else:
            raise ValueError("Must provide either cache_dir or cache_reader")

        print(f"CachedBackend initialised: {self.cache_reader}")

    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """Get frame from disk cache."""

        if frame_idx < 0 or frame_idx >= self.metadata['num_frames']:
            raise ValueError(f"Frame index {frame_idx} out of range")

        # DiskCacheReader.get_frame() already handles views parameter
        with self.lock:
            return self.cache_reader.get_frame(frame_idx, views)

    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """
        Trigger prefetch for upcoming frames.

        The DiskCacheReader has built-in prefetching via _trigger_prefetch,
        which is called automatically during get_frame.
        """

        # TODO: implement prefetch
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
                'cache_dir': cache_info['cache_dir'],
                'chunks_in_ram': cache_info['chunks_in_ram'],
                'ram_limit_chunks': cache_info['ram_limit_chunks'],
                'total_size_gb': cache_info['total_size_gb'],
                'frames_per_chunk': cache_info['frames_per_chunk']
            }

    def close(self):
        """Clean up cache resources."""

        with self.lock:
            if hasattr(self.cache_reader, 'delete_cache'):
                # Don't actually delete the cache, just clean up
                pass

            # Shutdown the thread executor
            if hasattr(self.cache_reader, '_executor'):
                self.cache_reader._executor.shutdown(wait=False)


class HybridBackend(VideoBackend):
    """
    Hybrid backend that uses cache when available, falls back to direct access.

    Useful during cache building or when cache is partially available.
    Automatically switches between backends based on availability.
    """

    def __init__(
            self,
            video_paths: List[Union[Path, str]],
            video_metadata: Dict,
            cache_reader=None,
            ram_budget_gb: float = 1.0
    ):
        """
        Initialise hybrid backend.

        Args:
            video_paths: List of video file paths
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

    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """Get frame from cache if available, otherwise use direct backend."""

        if frame_idx < 0 or frame_idx >= self.metadata['num_frames']:
            raise ValueError(f"Frame index {frame_idx} out of range")

        with self.lock:
            if self.cache_reader is not None:
                try:
                    self.stats['cache_reads'] += 1
                    return self.cache_reader.get_frame(frame_idx, views)

                except Exception as e:
                    print(f"Cache read failed, falling back to direct: {e}")
                    # Fall through to direct backend

            self.stats['direct_reads'] += 1
            return self.direct_backend.get_frame(frame_idx, views)

    def update_cache_reader(self, cache_reader):
        """
        Hot-swap the cache reader.

        Args:
            cache_reader: New DiskCacheReader instance or None
        """
        with self.lock:
            self.cache_reader = cache_reader
            print(f"HybridBackend: Cache reader updated ({'enabled' if cache_reader else 'disabled'})")

    def prefetch(self, frame_indices: List[int], priority: int = 0):
        """Prefetch from active backend."""

        with self.lock:
            if self.cache_reader is not None:
                # Cache backend handles prefetching automatically
                pass
            else:
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
        video_paths: List[Path],
        video_metadata: Dict,
        cache_dir: Optional[str] = None,
        cache_reader: Optional['DiskCacheReader'] = None,
        backend_type: str = 'auto',
        ram_budget_gb: float = 1.5
) -> VideoBackend:
    """
    Factory function to create the appropriate video backend.

    Args:
        video_paths: List of video file paths
        video_metadata: Video metadata dictionary
        cache_dir: Path to cache directory (for CachedBackend)
        cache_reader: Optional DiskCacheReader instance (alternative to cache_dir)
        backend_type: 'auto', 'direct', 'cached', or 'hybrid'
        ram_budget_gb: RAM budget for direct backend or cache

    Returns:
        Initialised VideoBackend instance

    Raises:
        ValueError: If backend_type is invalid
    """
    if backend_type == 'auto':
        # Auto-select based on cache availability
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