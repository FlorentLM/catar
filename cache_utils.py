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
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
import blosc
from multiprocessing import Pool, cpu_count, Event


class DiskCacheBuilder:
    """Builds compressed frame cache for multi-view videos."""

    def __init__(
        self,
        video_paths: List[str],
        cache_dir: str = 'multiview_chunks_per_video',
        ram_budget_gb: float = 0.5
    ):
        # Preserve order!!! order must match calibration
        self.video_paths = video_paths
        self.cache_dir = cache_dir
        self.ram_budget_gb = ram_budget_gb
        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')

    def compute_video_set_hash(self) -> str:
        """Create a hash from video paths and their modification times."""

        hash_input = []
        for vp in self.video_paths:
            mtime = os.path.getmtime(vp)
            size = os.path.getsize(vp)
            hash_input.append(f"{vp}:{mtime}:{size}")
        hash_str = "|".join(hash_input)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def check_cache_exists(self) -> Tuple[bool, Optional[dict]]:
        """Check if valid cache exists for the given video set."""

        if not os.path.exists(self.metadata_file):
            return False, None

        try:
            with open(self.metadata_file, 'r') as f:
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
                    if not os.path.exists(chunk_file):
                        return False, None

            return True, metadata
        except Exception as e:
            print(f"Error checking cache existence: {e}")
            return False, None

    def gather_video_info(self) -> List[dict]:
        """Extract metadata from all videos."""

        video_info = []

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
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
    def _chunk_video_worker(args) -> dict:
        """Worker function to chunk a single video."""

        video_idx, video_path, frame_count, width, height, frames_per_chunk, chunk_dir, progress_q, cancel_event = args

        cap = cv2.VideoCapture(video_path)
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
            filename = os.path.join(chunk_dir, f'video{video_idx}_chunk_{chunk_idx}.blosc')
            with open(filename, 'wb') as f:
                f.write(compressed)

            chunk_files.append(filename)
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
                    manager=None) -> dict:
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

    def _build_cache_internal(self, progress_callback, video_progress_callback, cancel_event, manager) -> dict:
        """Internal cache building logic that assumes a manager is present."""

        video_info = self.gather_video_info()
        if not video_info:
            raise ValueError("No videos found!")

        self.delete_cache()
        os.makedirs(self.cache_dir, exist_ok=True)

        frame_count = video_info[0]['frame_count']
        width = video_info[0]['width']
        height = video_info[0]['height']
        fps = video_info[0]['fps']

        frame_size_bytes = width * height * 3
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
                {'index': r['video_idx'], 'path': video_info[r['video_idx']]['path'], 'chunk_files': r['chunk_files'],
                 'num_chunks': len(r['chunk_files']), 'total_compressed_bytes': r['total_compressed_bytes']}
                for r in sorted(results, key=lambda x: x['video_idx'])
            ]
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Cache saved to: {self.cache_dir}")
        return metadata

    def delete_cache(self):
        """Delete all cache files from disk."""
        if Path(self.cache_dir).exists():
            shutil.rmtree(self.cache_dir)


class DiskCacheReader:
    """
    Fast random-access reader for cached multi-view video data.
    Thread-safe for concurrent reads.
    """

    def __init__(self, cache_dir: str = 'multiview_chunks_per_video'):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
        self._lock = threading.Lock()

        # Load metadata
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(
                f"Cache not found: {self.metadata_file}\n"
                f"Please build the cache first."
            )

        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # Extract properties
        self.frame_count = self.metadata['frame_count']
        self.width = self.metadata['width']
        self.height = self.metadata['height']
        self.fps = self.metadata['fps']
        self.frames_per_chunk = self.metadata['frames_per_chunk']
        self.num_views = len(self.metadata['videos'])

        # Build chunk index
        self._chunk_index: Dict[Tuple[int, int], str] = {}
        for video in self.metadata['videos']:
            for chunk_idx, chunk_file in enumerate(video['chunk_files']):
                self._chunk_index[(video['index'], chunk_idx)] = chunk_file

        # LRU cache for chunks
        self._chunk_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cache_size_limit = 20  # keep last N chunks in memory
        self._cache_access_order: List[Tuple[int, int]] = []

        print(f"VideoCacheReader initialized: {self.num_views} views, "
              f"{self.frame_count} frames, {self.width}x{self.height}")

    def _get_chunk_index(self, frame_idx: int) -> int:
        """Convert frame index to chunk index."""
        return frame_idx // self.frames_per_chunk

    def _get_frame_in_chunk(self, frame_idx: int) -> int:
        """Get frame position within its chunk."""
        return frame_idx % self.frames_per_chunk

    def _load_chunk(self, video_idx: int, chunk_idx: int) -> np.ndarray:
        """Load a chunk from disk, using cache if available."""

        cache_key = (video_idx, chunk_idx)

        with self._lock:
            # Check cache
            if cache_key in self._chunk_cache:
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                return self._chunk_cache[cache_key]

        # Load from disk (outside lock for parallelism)
        chunk_file = self._chunk_index.get(cache_key)
        if chunk_file is None:
            raise ValueError(f"Chunk not found: video {video_idx}, chunk {chunk_idx}")

        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file missing: {chunk_file}")

        with open(chunk_file, 'rb') as f:
            compressed = f.read()

        decompressed = blosc.decompress(compressed)

        # Determine chunk size
        start_frame = chunk_idx * self.frames_per_chunk
        frames_in_chunk = min(self.frames_per_chunk, self.frame_count - start_frame)

        chunk_array = np.frombuffer(
            decompressed,
            dtype=np.uint8
        ).reshape(frames_in_chunk, self.height, self.width, 3)

        # Add to cache
        with self._lock:
            self._chunk_cache[cache_key] = chunk_array
            self._cache_access_order.append(cache_key)

            # Limit cache size
            while len(self._cache_access_order) > self._cache_size_limit:
                oldest_key = self._cache_access_order.pop(0)
                if oldest_key in self._chunk_cache:
                    del self._chunk_cache[oldest_key]

        return chunk_array

    def get_frame(self, frame_idx: int, views: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Get a single frame from all views (or specified views).

        Args:
            frame_idx: Frame index (0 to frame_count-1)
            views: List of view indices. If None, get all views.

        Returns:
            List of numpy arrays (H, W, 3) BGR format (like cv2.videocapture)
        """

        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index {frame_idx} out of range")

        if views is None:
            views = list(range(self.num_views))

        chunk_idx = self._get_chunk_index(frame_idx)
        frame_in_chunk = self._get_frame_in_chunk(frame_idx)

        frames = []
        for view_idx in views:
            chunk = self._load_chunk(view_idx, chunk_idx)
            frames.append(chunk[frame_in_chunk].copy())

        return frames

    def preload_range(self, start_frame: int, end_frame: int):
        """Preload chunks covering a frame range."""

        start_chunk = self._get_chunk_index(start_frame)
        end_chunk = self._get_chunk_index(end_frame - 1)

        for chunk_idx in range(start_chunk, end_chunk + 1):
            for view_idx in range(self.num_views):
                self._load_chunk(view_idx, chunk_idx)

    def clear_cache(self):
        """Clear the chunk cache to free memory."""

        with self._lock:
            self._chunk_cache.clear()
            self._cache_access_order.clear()

    def get_cache_info(self) -> dict:
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
            'creation_time': self.metadata.get('creation_time', 'Unknown'),
        }

    def get_loaded_chunk_ranges(self) -> List[Tuple[int, int]]:
        """
        Get list of frame ranges currently loaded in RAM.
        Returns list of tuples (start_frame, end_frame)
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

            # Convert to frame ranges
            ranges = []
            for chunk_idx in chunk_indices:
                start_frame = chunk_idx * self.frames_per_chunk
                end_frame = min(start_frame + self.frames_per_chunk - 1, self.frame_count - 1)
                ranges.append((start_frame, end_frame))

            return ranges

    def delete_cache(self):
        """Delete all cache files from disk."""
        import shutil
        if Path(self.cache_dir).exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cache deleted: {self.cache_dir}")

    def __repr__(self) -> str:
        return (
            f"VideoCacheReader(views={self.num_views}, frames={self.frame_count}, "
            f"resolution={self.width}x{self.height})"
        )