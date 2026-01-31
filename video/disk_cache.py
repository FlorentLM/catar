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
    """
    Builds compressed frame cache for multi-view videos.
    """

    def __init__(
        self,
        video_paths: Dict[str, Union[Path, str]],
        cache_dir: Optional[Union[Path, str]] = None,
        ram_budget_gb: float = 0.5
    ):
        """
        Initialise cache builder.

        Args:
            video_paths: Dict mapping camera_name -> video path
            cache_dir: Directory to store cache files
            ram_budget_gb: RAM budget per chunk during building
        """
        self.video_paths: Dict[str, Path] = {name: Path(p) for name, p in video_paths.items()}
        self.camera_names: Tuple[str, ...] = tuple(sorted(self.video_paths.keys()))

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
        """
        Create a hash from video paths and their modification times.
        """
        hash_input = []
        for cam_name in self.camera_names:
            vp = self.video_paths[cam_name]
            mtime = os.path.getmtime(vp)
            size = os.path.getsize(vp)
            hash_input.append(f"{cam_name}:{vp}:{mtime}:{size}")
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

            # Verify camera names match
            cached_cameras = set(metadata.get('camera_names', []))
            if cached_cameras != set(self.camera_names):
                print(f"Cache camera mismatch: cached={cached_cameras}, current={set(self.camera_names)}")
                return False, None

            # Verify all chunk files exist
            for cam_name, cam_data in metadata['videos'].items():
                for chunk_file in cam_data['chunk_files']:
                    if not Path(chunk_file).is_file():
                        return False, None

            return True, metadata

        except Exception as e:
            print(f"Error checking cache existence: {e}")
            return False, None

    def gather_video_info(self) -> Dict[str, Dict]:
        """
        Extract metadata from all videos.
        """
        # TODO: redundant with probe_video, but maybe safer to keep here

        video_info = {}

        for cam_name in self.camera_names:
            video_path = self.video_paths[cam_name]
            cap = cv2.VideoCapture(video_path.as_posix())
            video_info[cam_name] = {
                'camera_name': cam_name,
                'path': video_path,
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
            }
            cap.release()

        # Verify consistency
        ref_cam = self.camera_names[0]
        ref = video_info[ref_cam]
        for cam_name in self.camera_names[1:]:
            info = video_info[cam_name]
            if (info['frame_count'] != ref['frame_count'] or
                    info['width'] != ref['width'] or
                    info['height'] != ref['height']):
                print(f"WARNING: Video '{cam_name}' has different properties than '{ref_cam}'")

        return video_info

    @staticmethod
    def _chunk_video_worker(args_list) -> Dict:
        """Worker function to chunk a single video."""

        (camera_name, video_path, frame_count, width, height,
         frames_per_chunk, chunk_dir, progress_q, cancel_event) = args_list

        chunk_dir = Path(chunk_dir)

        cap = cv2.VideoCapture(str(video_path))
        chunk_files = []
        compressed_sizes = []

        for chunk_idx, start in enumerate(range(0, frame_count, frames_per_chunk)):

            if cancel_event and cancel_event.is_set():
                cap.release()
                return {'cancelled': True, 'camera_name': camera_name}

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

            safe_cam_name = camera_name.replace('/', '_').replace('\\', '_')
            chunk_filepath = chunk_dir / f'{safe_cam_name}_chunk_{chunk_idx}.blosc'
            with chunk_filepath.open(mode='wb') as f:
                f.write(compressed)

            chunk_files.append(str(chunk_filepath))
            compressed_sizes.append(len(compressed))

            # Report progress
            if progress_q:
                progress_pct = ((start + frames_in_chunk) / frame_count) * 100
                progress_q.put((camera_name, progress_pct))

        cap.release()
        if progress_q:
            progress_q.put((camera_name, 100.0))

        return {
            'camera_name': camera_name,
            'chunk_files': chunk_files,
            'compressed_sizes': compressed_sizes,
            'total_compressed_bytes': sum(compressed_sizes),
        }

    def build_cache(self, progress_callback=None, video_progress_callback=None, cancel_event: Optional[Event] = None, manager=None) -> Dict:
        """Build the video cache."""

        if manager is None:
            from multiprocessing import Manager
            with Manager() as managed_context:
                return self._build_cache_internal(
                    progress_callback, video_progress_callback, cancel_event, managed_context
                )
        else:
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

        # Use first camera as reference for dimensions
        ref_cam = self.camera_names[0]
        frame_count = video_info[ref_cam]['frame_count']
        width = video_info[ref_cam]['width']
        height = video_info[ref_cam]['height']
        fps = video_info[ref_cam]['fps']

        frame_size_bytes = width * height * 3
        frames_per_chunk = max(1, int((self.ram_budget_gb * (1024 ** 3)) // frame_size_bytes))

        print(f"Frames per chunk: {frames_per_chunk}")

        video_progress_q = manager.Queue()

        args_list = [
            (cam_name, info['path'], info['frame_count'], info['width'], info['height'],
             frames_per_chunk, self.cache_dir, video_progress_q, cancel_event)
            for cam_name, info in video_info.items()
        ]

        num_workers = min(len(self.camera_names), cpu_count())

        time_start = time.time()
        total_frames = frame_count * len(video_info)
        video_progress = {cam_name: 0.0 for cam_name in self.camera_names}

        with Pool(num_workers) as pool:
            async_results = pool.map_async(self._chunk_video_worker, args_list)

            while not async_results.ready():
                if cancel_event and cancel_event.is_set():
                    pool.terminate()
                    pool.join()
                    raise InterruptedError("Cache build was cancelled.")
                try:
                    cam_name, pct = video_progress_q.get(timeout=0.1)
                    video_progress[cam_name] = pct

                    if video_progress_callback:
                        video_progress_callback(cam_name, pct)

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

        # Build metadata keyed by camera name
        videos_metadata = {}
        for r in results:
            cam_name = r['camera_name']
            videos_metadata[cam_name] = {
                'camera_name': cam_name,
                'path': str(video_info[cam_name]['path']),
                'chunk_files': r['chunk_files'],
                'num_chunks': len(r['chunk_files']),
                'total_compressed_bytes': r['total_compressed_bytes']
            }

        metadata = {
            'version': '2.0',  # Bumped version for name-based format
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_set_hash': self.compute_video_set_hash(),
            'camera_names': list(self.camera_names),  # Canonical order
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'fps': fps,
            'frames_per_chunk': frames_per_chunk,
            'compression': {'algorithm': 'blosc-lz4', 'clevel': 5, 'shuffle': True},
            'videos': videos_metadata
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

    Frames are accessed by camera name, not index.
    """

    def __init__(
        self,
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
        self._executor = ThreadPoolExecutor(max_workers=min(4, cpu_count()))
        self._loading_futures: Dict[Tuple[str, int], Future] = {}

        # Load metadata
        if not self.metadata_file.is_file():
            raise FileNotFoundError(
                f"Cache not found: {self.metadata_file}\n"
                f"Please build the cache first."
            )

        with self.metadata_file.open(mode='r') as f:
            self.metadata = json.load(f)

        # Handle version differences
        version = self.metadata.get('version', '1.0')
        if version.startswith('1.'):
            raise ValueError(
                f"Cache version {version} uses index-based format. "
                f"Please rebuild cache with the new name-based builder."
            )

        # Extract properties
        self.frame_count = self.metadata['frame_count']
        self.width = self.metadata['width']
        self.height = self.metadata['height']
        self.fps = self.metadata['fps']
        self.frames_per_chunk = self.metadata['frames_per_chunk']
        self.camera_names: Tuple[str, ...] = tuple(self.metadata['camera_names'])
        self.num_views = len(self.camera_names)

        # Build chunk index: (camera_name, chunk_idx) -> Path
        self._chunk_index: Dict[Tuple[str, int], Path] = {}

        for cam_name, cam_data in self.metadata['videos'].items():
            for chunk_idx, chunk_file in enumerate(cam_data['chunk_files']):
                self._chunk_index[(cam_name, chunk_idx)] = Path(chunk_file)

        # LRU cache for chunks
        self._chunk_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._cache_access_order: List[Tuple[str, int]] = []

        # Calculate cache size limit
        chunk_bytes = self.width * self.height * 3 * self.frames_per_chunk
        budget_bytes = ram_budget_gb * (1024 ** 3)
        calculated_limit = int(budget_bytes // chunk_bytes)
        self._cache_size_limit = max(self.num_views * 2, calculated_limit)

        print(f"DiskCacheReader initialized: {self.num_views} views ({', '.join(self.camera_names)}), "
              f"{self.frame_count} frames, {self.width}x{self.height}")
        print(f"Cache RAM Budget: {ram_budget_gb:.1f} GB. "
              f"Chunk size: {chunk_bytes/1024**2:.1f} MB. "
              f"Keeping max {self._cache_size_limit} chunks in RAM.")

    def _get_chunk_index(self, frame_idx: int) -> int:
        """Convert frame index to chunk index."""
        return frame_idx // self.frames_per_chunk

    def _get_frame_in_chunk(self, frame_idx: int) -> int:
        """Get frame position within its chunk."""
        return frame_idx % self.frames_per_chunk

    def _load_chunk_from_disk_internal(self, camera_name: str, chunk_idx: int) -> np.ndarray:
        """Actual disk I/O and decompression logic (stateless)."""

        cache_key = (camera_name, chunk_idx)
        chunk_file = self._chunk_index.get(cache_key)

        if chunk_file is None:
            raise ValueError(f"Chunk not found: camera '{camera_name}', chunk {chunk_idx}")

        if not chunk_file.is_file():
            raise FileNotFoundError(f"Chunk file missing: {chunk_file}")

        with chunk_file.open(mode='rb') as f:
            compressed = f.read()

        decompressed = blosc.decompress(compressed)

        start_frame = chunk_idx * self.frames_per_chunk
        frames_in_chunk = min(self.frames_per_chunk, self.frame_count - start_frame)

        chunk_array = np.frombuffer(
            decompressed,
            dtype=np.uint8
        ).reshape(frames_in_chunk, self.height, self.width, 3)

        return chunk_array

    def _load_chunk(self, camera_name: str, chunk_idx: int) -> np.ndarray:
        """
        Get a chunk from cache, or load it.
        If a background load is pending for this chunk, wait for it.
        """
        cache_key = (camera_name, chunk_idx)

        # Check existing cache
        with self._lock:
            if cache_key in self._chunk_cache:
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                return self._chunk_cache[cache_key]

            future = self._loading_futures.get(cache_key)

        # If loading, wait for it
        if future:
            try:
                chunk_array = future.result()
                with self._lock:
                    if cache_key in self._loading_futures:
                        del self._loading_futures[cache_key]
                    self._add_to_cache_safe(cache_key, chunk_array)
                return chunk_array
            except Exception as e:
                print(f"Error in background chunk load: {e}")

        # Synchronous load
        chunk_array = self._load_chunk_from_disk_internal(camera_name, chunk_idx)

        with self._lock:
            self._add_to_cache_safe(cache_key, chunk_array)

        return chunk_array

    def _add_to_cache_safe(self, key: Tuple[str, int], data: np.ndarray):
        """Internal helper to add to cache and enforce limits. Assumes lock is held."""

        self._chunk_cache[key] = data
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
        self._cache_access_order.append(key)

        while len(self._cache_access_order) > self._cache_size_limit:
            oldest_key = self._cache_access_order.pop(0)
            if oldest_key == key and len(self._cache_access_order) > 0:
                self._cache_access_order.append(key)
                oldest_key = self._cache_access_order.pop(0)

            if oldest_key in self._chunk_cache:
                del self._chunk_cache[oldest_key]

            if oldest_key in self._loading_futures:
                self._loading_futures[oldest_key].cancel()
                del self._loading_futures[oldest_key]

    def _trigger_prefetch(self, frame_idx: int):
        """Check if we need to preload the next chunk."""

        chunk_idx = self._get_chunk_index(frame_idx)
        frame_in_chunk = self._get_frame_in_chunk(frame_idx)

        if frame_in_chunk > (self.frames_per_chunk * 0.6):
            next_chunk_idx = chunk_idx + 1

            # Use first camera to determine total chunks
            first_cam = self.camera_names[0]
            total_chunks = self.metadata['videos'][first_cam]['num_chunks']

            if next_chunk_idx >= total_chunks:
                return

            with self._lock:
                for cam_name in self.camera_names:
                    next_key = (cam_name, next_chunk_idx)

                    if next_key in self._chunk_cache or next_key in self._loading_futures:
                        continue

                    future = self._executor.submit(self._load_chunk_from_disk_internal, cam_name, next_chunk_idx)
                    self._loading_futures[next_key] = future

    def get_frame(
        self,
        frame_idx: int,
        cameras: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get a single frame from specified cameras.

        Args:
            frame_idx: Frame index (0 to frame_count-1)
            cameras: List of camera names. If None, get all cameras.

        Returns:
            Dict mapping camera_name -> numpy array (H, W, 3) BGR format
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index {frame_idx} out of range")

        self._trigger_prefetch(frame_idx)

        if cameras is None:
            cameras = list(self.camera_names)

        chunk_idx = self._get_chunk_index(frame_idx)
        frame_in_chunk = self._get_frame_in_chunk(frame_idx)

        frames = {}
        for cam_name in cameras:
            if cam_name not in self.camera_names:
                raise ValueError(f"Unknown camera: '{cam_name}'")
            chunk = self._load_chunk(cam_name, chunk_idx)
            frames[cam_name] = chunk[frame_in_chunk]

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
            cam_data['total_compressed_bytes']
            for cam_data in self.metadata['videos'].values()
        )
        return {
            'cache_dir': str(self.cache_dir),
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024 ** 3),
            'video_count': self.num_views,
            'camera_names': list(self.camera_names),
            'frame_count': self.frame_count,
            'frames_per_chunk': self.frames_per_chunk,
            'total_chunks': sum(cam_data['num_chunks'] for cam_data in self.metadata['videos'].values()),
            'chunks_in_ram': len(self._chunk_cache),
            'ram_limit_chunks': self._cache_size_limit,
            'creation_time': self.metadata.get('creation_time', 'Unknown'),
        }

    def get_loaded_chunk_ranges(self) -> List[Tuple[int, int]]:
        """Get list of frame ranges currently loaded in RAM."""

        with self._lock:
            if not self._chunk_cache:
                return []

            # Get chunk indices from first camera for simplicity
            first_cam = self.camera_names[0]
            chunk_indices = sorted(set(
                chunk_idx for (cam_name, chunk_idx) in self._chunk_cache.keys()
                if cam_name == first_cam
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
            f"DiskCacheReader(cameras={list(self.camera_names)}, frames={self.frame_count}, "
            f"resolution={self.width}x{self.height})"
        )