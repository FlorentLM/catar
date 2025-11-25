"""
Video caching system for fast random access to multi-view video frames.
"""
import os
import json
import time
import hashlib
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import cv2
import numpy as np
import blosc
from multiprocessing import Pool, cpu_count, Event
from concurrent.futures import ThreadPoolExecutor, Future

import config


class DiskCacheBuilder:
    """Builds compressed frame cache for multi-view videos."""

    def __init__(
        self,
        video_paths: List[Union[Path, str]],
        cache_dir: Optional[Union[Path, str]] = None,
        ram_budget_gb: float = 0.5
    ):
        # Preserve order!!! order must match calibration
        self.video_paths = tuple([Path(v) for v in video_paths])

        if cache_dir is None:
            if hasattr(config, 'VIDEO_CACHE_FOLDER'):
                self.cache_dir = Path(config.VIDEO_CACHE_FOLDER)
            elif hasattr(config, 'DATA_FOLDER'):
                self.cache_dir = Path(config.DATA_FOLDER) / 'video_cache'
            else:
                self.cache_dir = Path.cwd() / 'data' / 'video_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.ram_budget_gb = ram_budget_gb
        self.metadata_file = self.cache_dir / 'cache_metadata.json'

    def compute_video_set_hash(self) -> str:
        """Create a hash from video paths and their modification times."""

        hash_input = []
        for vp in self.video_paths:
            mtime = os.path.getmtime(vp)
            size = os.path.getsize(vp)
            hash_input.append(f"{vp}:{mtime}:{size}")
        hash_str = "|".join(hash_input)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def check_cache_exists(self) -> Tuple[bool, Optional[Dict]]:
        """Check if valid cache exists for the given video set."""

        if not self.metadata_file.is_file():
            return False, None

        try:
            with self.metadata_file.open(mode='r') as f:
                metadata = json.load(f)

            if not self.video_paths:
                print("Warning: checking cache existence without video paths. Hash will not be verified.")
            else:
                current_hash = self.compute_video_set_hash()
                if metadata.get('video_set_hash') != current_hash:
                    return False, None

            # Verify all chunk files exist
            for video in metadata['videos']:
                for chunk_file in video['chunk_files']:
                    if not Path(chunk_file).is_file():
                        return False, None

            return True, metadata

        except Exception as e:
            print(f"Error checking cache existence: {e}")
            return False, None

    def gather_video_info(self) -> List[Dict]:
        """Extract metadata from all videos."""

        video_info = []

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path.as_posix())
            info = {
                'path': video_path,
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
            }
            cap.release()
            video_info.append(info)

        # Verify consistency
        ref = video_info[0]
        for i, info in enumerate(video_info[1:], 1):
            if (info['frame_count'] != ref['frame_count'] or
                    info['width'] != ref['width'] or
                    info['height'] != ref['height']):
                print(f"WARNING: Video {i} has different properties than video 0")

        return video_info

    @staticmethod
    def _chunk_video_worker(args) -> Dict:
        """Worker function to chunk a single video."""

        video_idx, video_path, frame_count, width, height, frames_per_chunk, chunk_dir, progress_q, cancel_event = args

        # Just to be super safe
        chunk_dir = Path(chunk_dir)
        video_path = Path(video_path)

        cap = cv2.VideoCapture(video_path.as_posix())
        chunk_files = []
        compressed_sizes = []
        for chunk_idx, start in enumerate(range(0, frame_count, frames_per_chunk)):

            if cancel_event and cancel_event.is_set():
                cap.release()
                return {'cancelled': True, 'video_idx': video_idx}

            frames_in_chunk = min(frames_per_chunk, frame_count - start)
            chunk_array = np.empty((frames_in_chunk, height, width, 3), dtype=np.uint8)

            # Read frames
            for fi in range(frames_in_chunk):
                ret, frame = cap.read()
                if not ret:
                    frames_in_chunk = fi
                    break
                chunk_array[fi] = frame

            if frames_in_chunk < len(chunk_array):
                chunk_array = chunk_array[:frames_in_chunk]

            # Compress chunk
            compressed = blosc.compress(
                chunk_array.tobytes(), typesize=1, cname='lz4', clevel=5, shuffle=blosc.SHUFFLE
            )

            chunk_filepath = chunk_dir / f'video{video_idx}_chunk_{chunk_idx}.blosc'
            with chunk_filepath.open(mode='wb') as f:
                f.write(compressed)

            chunk_files.append(str(chunk_filepath))     # as str because it's only returned for dumping into json
            compressed_sizes.append(len(compressed))

            # Report progress
            if progress_q:
                progress_pct = ((start + frames_in_chunk) / frame_count) * 100
                progress_q.put((video_idx, progress_pct))

        cap.release()
        if progress_q:
            progress_q.put((video_idx, 100.0))
        return {
            'video_idx': video_idx,
            'chunk_files': chunk_files,
            'compressed_sizes': compressed_sizes,
            'total_compressed_bytes': sum(compressed_sizes),
        }

    def build_cache(self, progress_callback=None, video_progress_callback=None, cancel_event: Optional[Event] = None,
                    manager=None) -> Dict:
        """Build the video cache."""

        if manager is None:

            from multiprocessing import Manager
            with Manager() as managed_context:
                return self._build_cache_internal(
                    progress_callback, video_progress_callback, cancel_event, managed_context
                )
        else:
            # if a manager is provided use it directly
            return self._build_cache_internal(
                progress_callback, video_progress_callback, cancel_event, manager
            )

    def _build_cache_internal(self, progress_callback, video_progress_callback, cancel_event, manager) -> Dict:
        """Internal cache building logic that assumes a manager is present."""

        video_info = self.gather_video_info()
        if not video_info:
            raise ValueError("No videos found!")

        self.delete_cache()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        frame_count = video_info[0]['frame_count']
        width = video_info[0]['width']
        height = video_info[0]['height']
        fps = video_info[0]['fps']

        frame_size_bytes = width * height * 3
        # Default builder RAM budget is 0.5GB per chunk (uncompressed)
        frames_per_chunk = max(1, int((self.ram_budget_gb * (1024 ** 3)) // frame_size_bytes))

        print(f"Frames per chunk: {frames_per_chunk}")

        # Use the passed-in manager to create the queue
        video_progress_q = manager.Queue()

        args_list = [
            (idx, info['path'], info['frame_count'], info['width'], info['height'],
             frames_per_chunk, self.cache_dir, video_progress_q, cancel_event)
            for idx, info in enumerate(video_info)
        ]

        num_workers = min(len(self.video_paths), cpu_count())

        time_start = time.time()
        total_frames = frame_count * len(video_info)
        video_progress = {i: 0.0 for i in range(len(video_info))}

        with Pool(num_workers) as pool:
            async_results = pool.map_async(self._chunk_video_worker, args_list)

            while not async_results.ready():
                if cancel_event and cancel_event.is_set():
                    pool.terminate()
                    pool.join()
                    raise InterruptedError("Cache build was cancelled.")
                try:
                    v_idx, pct = video_progress_q.get(timeout=0.1)
                    video_progress[v_idx] = pct

                    if video_progress_callback:
                        video_progress_callback(v_idx, pct)

                    current_total_pct = sum(video_progress.values()) / len(video_info)
                    completed_frames = int(total_frames * (current_total_pct / 100.0))
                    if progress_callback:
                        progress_callback(completed_frames, total_frames)

                except Exception:
                    pass

            results = async_results.get()

        if any(r.get('cancelled') for r in results):
            raise InterruptedError("Cache build was cancelled by a worker.")

        elapsed = time.time() - time_start
        if progress_callback:
            progress_callback(total_frames, total_frames)

        total_compressed = sum(r.get('total_compressed_bytes', 0) for r in results)
        print("Cache build complete!")
        print(f"  Build time: {elapsed:.1f}s, Cache size: {total_compressed / 1024 ** 3:.1f} GB")

        metadata = {
            'version': '1.0',
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_set_hash': self.compute_video_set_hash(),
            'frame_count': frame_count, 'width': width, 'height': height, 'fps': fps,
            'frames_per_chunk': frames_per_chunk,
            'compression': {'algorithm': 'blosc-lz4', 'clevel': 5, 'shuffle': True},
            'videos': [
                {
                    'index': r['video_idx'],
                    'path': str(video_info[r['video_idx']]['path']),
                    'chunk_files': r['chunk_files'],
                    'num_chunks': len(r['chunk_files']),
                    'total_compressed_bytes': r['total_compressed_bytes']
                } for r in sorted(results, key=lambda x: x['video_idx'])
            ]
        }

        with self.metadata_file.open(mode='w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Cache saved to: {self.cache_dir}")
        return metadata

    def delete_cache(self):
        """Delete all cache files from disk."""

        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir)


class DiskCacheReader:
    """
    Fast random-access reader for cached multi-view video data.
    Thread-safe for concurrent reads with background prefetching.
    """

    def __init__(self,
                 cache_dir: Optional[Union[Path, str]] = None,
                 ram_budget_gb: float = 2.0
        ):
        if cache_dir is None:
            if hasattr(config, 'VIDEO_CACHE_FOLDER'):
                self.cache_dir = Path(config.VIDEO_CACHE_FOLDER)
            elif hasattr(config, 'DATA_FOLDER'):
                self.cache_dir = Path(config.DATA_FOLDER) / 'video_cache'
            else:
                self.cache_dir = Path.cwd() / 'data' / 'video_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self._lock = threading.Lock()

        # Background loader
        # Blosc releases GIL, so thread pool works well for parallel decompression
        self._executor = ThreadPoolExecutor(max_workers=min(4, cpu_count()))
        self._loading_futures: Dict[Tuple[int, int], Future] = {}

        # Load metadata
        if not self.metadata_file.is_file():
            raise FileNotFoundError(
                f"Cache not found: {self.metadata_file}\n"
                f"Please build the cache first."
            )

        with self.metadata_file.open(mode='r') as f:
            self.metadata = json.load(f)

        # Extract properties
        self.frame_count = self.metadata['frame_count']
        self.width = self.metadata['width']
        self.height = self.metadata['height']
        self.fps = self.metadata['fps']
        self.frames_per_chunk = self.metadata['frames_per_chunk']
        self.num_views = len(self.metadata['videos'])

        # Build chunk index
        self._chunk_index: Dict[Tuple[int, int], Path] = {}

        for video in self.metadata['videos']:
            for chunk_idx, chunk_file in enumerate(video['chunk_files']):

                self._chunk_index[(video['index'], chunk_idx)] = Path(chunk_file)

        # LRU cache for chunks
        self._chunk_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Calculate size of one decompressed chunk in RAM
        # Frame bytes = W * H * 3
        chunk_bytes = self.width * self.height * 3 * self.frames_per_chunk

        # Calculate how many chunks fit in the memory budget
        budget_bytes = ram_budget_gb * (1024 ** 3)
        calculated_limit = int(budget_bytes // chunk_bytes)

        # Ensure we have at least 2 chunks per video in memory for smooth scrolling
        self._cache_size_limit = max(self.num_views * 2, calculated_limit)

        print(f"VideoCacheReader initialized: {self.num_views} views, "
              f"{self.frame_count} frames, {self.width}x{self.height}")

        print(f"Cache RAM Budget: {ram_budget_gb:.1f} GB. "
              f"Chunk size: {chunk_bytes/1024**2:.1f} MB. "
              f"Keeping max {self._cache_size_limit} chunks in RAM.")

        self._cache_access_order: List[Tuple[int, int]] = []

    def _get_chunk_index(self, frame_idx: int) -> int:
        """Convert frame index to chunk index."""
        return frame_idx // self.frames_per_chunk

    def _get_frame_in_chunk(self, frame_idx: int) -> int:
        """Get frame position within its chunk."""
        return frame_idx % self.frames_per_chunk

    def _load_chunk_from_disk_internal(self, video_idx: int, chunk_idx: int) -> np.ndarray:
        """Actual disk I/O and decompression logic (stateless)."""

        cache_key = (video_idx, chunk_idx)
        chunk_file = self._chunk_index.get(cache_key)

        if chunk_file is None:
            raise ValueError(f"Chunk not found: video {video_idx}, chunk {chunk_idx}")

        if not chunk_file.is_file():
            raise FileNotFoundError(f"Chunk file missing: {chunk_file}")

        with chunk_file.open(mode='rb') as f:
            compressed = f.read()

        # Decompress (expensive operation)
        decompressed = blosc.decompress(compressed)

        # Determine chunk size
        start_frame = chunk_idx * self.frames_per_chunk
        frames_in_chunk = min(self.frames_per_chunk, self.frame_count - start_frame)

        chunk_array = np.frombuffer(
            decompressed,
            dtype=np.uint8
        ).reshape(frames_in_chunk, self.height, self.width, 3)

        return chunk_array

    def _load_chunk(self, video_idx: int, chunk_idx: int) -> np.ndarray:
        """
        Get a chunk from cache, or load it.
        If a background load is pending for this chunk, wait for it.
        """
        cache_key = (video_idx, chunk_idx)

        # Check existing cache (fast)
        with self._lock:
            if cache_key in self._chunk_cache:
                # Update LRU
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                return self._chunk_cache[cache_key]

            # Check if currently loading in background
            future = self._loading_futures.get(cache_key)

        # If loading, wait for it (outside lock)
        if future:
            try:
                chunk_array = future.result()
                # Cleanup future
                with self._lock:
                    if cache_key in self._loading_futures:
                        del self._loading_futures[cache_key]

                    # Cache the result
                    self._add_to_cache_safe(cache_key, chunk_array)
                return chunk_array
            except Exception as e:
                print(f"Error in background chunk load: {e}")
                # Fallthrough to synchronous load on error

        # Synchronous load (slow fallback)
        chunk_array = self._load_chunk_from_disk_internal(video_idx, chunk_idx)

        with self._lock:
            self._add_to_cache_safe(cache_key, chunk_array)

        return chunk_array

    def _add_to_cache_safe(self, key: Tuple[int, int], data: np.ndarray):
        """Internal helper to add to cache and enforce limits."""
        # Assumes lock is held by caller
        self._chunk_cache[key] = data
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
        self._cache_access_order.append(key)

        # Enforce size limit
        while len(self._cache_access_order) > self._cache_size_limit:
            oldest_key = self._cache_access_order.pop(0)
            # Don't evict if it's the one we just added (edge case with tiny cache)
            if oldest_key == key and len(self._cache_access_order) > 0:
                self._cache_access_order.append(key)
                oldest_key = self._cache_access_order.pop(0)

            if oldest_key in self._chunk_cache:
                del self._chunk_cache[oldest_key]

            # Also cancel any pending future for evicted key to save CPU
            if oldest_key in self._loading_futures:
                self._loading_futures[oldest_key].cancel()
                del self._loading_futures[oldest_key]

    def _trigger_prefetch(self, frame_idx: int):
        """Check if we need to preload the next chunk."""

        chunk_idx = self._get_chunk_index(frame_idx)
        frame_in_chunk = self._get_frame_in_chunk(frame_idx)

        # If we are past 60% of the chunk, verify next chunk is loading
        if frame_in_chunk > (self.frames_per_chunk * 0.6):
            next_chunk_idx = chunk_idx + 1

            # Determine max chunks
            # We assume all videos have same chunks
            total_chunks = self.metadata['videos'][0]['num_chunks']

            if next_chunk_idx >= total_chunks:
                return

            with self._lock:
                for view_idx in range(self.num_views):
                    next_key = (view_idx, next_chunk_idx)

                    # Skip if already cached or already loading
                    if next_key in self._chunk_cache or next_key in self._loading_futures:
                        continue

                    # Submit to background thread
                    future = self._executor.submit(self._load_chunk_from_disk_internal, view_idx, next_chunk_idx)
                    self._loading_futures[next_key] = future

    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Get a single frame from all views (or specified views).

        Args:
            frame_idx: Frame index (0 to frame_count-1)
            views: List of view indices. If None, get all views.

        Returns:
            List of numpy arrays (H, W, 3) BGR format
        """

        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index {frame_idx} out of range")

        # Trigger background loading of next chunk if needed
        self._trigger_prefetch(frame_idx)

        if views is None:
            views = list(range(self.num_views))

        chunk_idx = self._get_chunk_index(frame_idx)
        frame_in_chunk = self._get_frame_in_chunk(frame_idx)

        frames = []
        for view_idx in views:
            chunk = self._load_chunk(view_idx, chunk_idx)
            # frames.append(chunk[frame_in_chunk].copy())
            frames.append(chunk[frame_in_chunk]) # much faster without copy, but consumers should never touch the data

        return frames

    def clear_cache(self):
        """Clear the chunk cache to free memory."""

        with self._lock:
            self._chunk_cache.clear()
            self._cache_access_order.clear()
            self._loading_futures.clear()

    def get_cache_info(self) -> Dict:
        """Get info about the cache."""

        total_size = sum(
            video['total_compressed_bytes']
            for video in self.metadata['videos']
        )
        return {
            'cache_dir': self.cache_dir,
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024 ** 3),
            'num_videos': self.num_views,
            'frame_count': self.frame_count,
            'frames_per_chunk': self.frames_per_chunk,
            'total_chunks': sum(video['num_chunks'] for video in self.metadata['videos']),
            'chunks_in_ram': len(self._chunk_cache),
            'ram_limit_chunks': self._cache_size_limit,
            'creation_time': self.metadata.get('creation_time', 'Unknown'),
        }

    def get_loaded_chunk_ranges(self) -> List[Tuple[int, int]]:
        """
        Get list of frame ranges currently loaded in RAM.
        """
        with self._lock:
            if not self._chunk_cache:
                return []

            # Get all chunk indices currently in cache (just first view for simplicity)
            chunk_indices = sorted(set(
                chunk_idx for (view_idx, chunk_idx) in self._chunk_cache.keys()
                if view_idx == 0
            ))

            if not chunk_indices:
                return []

            ranges = []
            for chunk_idx in chunk_indices:
                start_frame = chunk_idx * self.frames_per_chunk
                end_frame = min(start_frame + self.frames_per_chunk - 1, self.frame_count - 1)
                ranges.append((start_frame, end_frame))

            return ranges

    def delete_cache(self):
        """Delete all cache files from disk."""

        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir)
            print(f"Cache deleted: {self.cache_dir}")
        self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"VideoCacheReader(views={self.num_views}, frames={self.frame_count}, "
            f"resolution={self.width}x{self.height})"
        )