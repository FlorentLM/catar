import collections
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from state import AppState
    from cache_utils import DiskCacheReader


class VideoReaderWorker(threading.Thread):
    """Reads video frames and distributes them to processing workers."""

    def __init__(
        self,
        app_state: 'AppState',
        command_queue: queue.Queue,
        output_queues: List[queue.Queue],
        diskcache_reader: Optional['DiskCacheReader'] = None,
    ):
        super().__init__(daemon=True, name="VideoReaderWorker")
        self.app_state = app_state
        self.video_paths = app_state.video_paths
        self.command_queue = command_queue
        self.output_queues = output_queues
        self.diskcache_reader = diskcache_reader
        self.shutdown_event = threading.Event()
        self.video_captures = []

        # We only need the on-the-fly cache if not using the disk cache reader
        if self.diskcache_reader is None:
            self.onthefly_cache = [collections.OrderedDict() for _ in self.video_paths]

            # Target budget: 1.5 GB total for all videos combined in this worker.
            target_budget_bytes = 1.5 * 1024**3

            # Frame size = W * H * 3 (bytes)
            meta = self.app_state.video_metadata
            frame_bytes = meta['width'] * meta['height'] * 3

            total_capacity_frames = target_budget_bytes // frame_bytes
            num_videos = len(self.video_paths)
            per_video_capacity = int(total_capacity_frames // num_videos)

            # Ensure a minimum of 10 frames per video, but cap at the calculated max
            self.onthefly_cache_size = max(10, per_video_capacity)

            print(f"VideoReaderWorker: RAM Budget 1.5GB. Frame size: {frame_bytes/1024**2:.2f}MB.")
            print(f"VideoReaderWorker: On-the-fly cache set to {self.onthefly_cache_size} frames per video.")

        else:
            self.onthefly_cache = None

    def run(self):
        print("Video reader worker started.")
        try:
            self._initialize_captures()
            self._read_loop()
        finally:
            self._cleanup()

    def _initialize_captures(self):
        """Open all video files (if not using cache)."""

        if self.diskcache_reader is not None:
            print("Video reader using cached frames.")
            return

        self.video_captures = [cv2.VideoCapture(path) for path in self.video_paths]
        if not all(cap.isOpened() for cap in self.video_captures):
            print("ERROR: Video reader failed to open one or more videos.")
            self.shutdown_event.set()

    def _get_frames(self, frame_idx: int, is_sequential: bool) -> List[np.ndarray]:
        """
        Gets frames for a specific index.
        Uses cache reader if available, otherwise falls back to cv2.VideoCapture.
        """

        # Fast path: use disk cache reader if available
        if self.diskcache_reader is not None:
            try:
                return self.diskcache_reader.get_frame(frame_idx)
            except Exception as e:
                print(f"ERROR reading from disk cache: {e}")
                return []

        # Fallback: use cv2.VideoCapture
        all_frames = []
        for i, cap in enumerate(self.video_captures):
            cache = self.onthefly_cache[i]
            frame = None

            # Check cache first
            if frame_idx in cache:
                frame = cache[frame_idx]
                cache.move_to_end(frame_idx)

            # if not, read from video file
            else:
                read_successful = False
                new_frame = None

                # For sequential frames, use the fast way
                if is_sequential:
                    ret, new_frame = cap.read()
                    if ret:
                        read_successful = True

                # For non-sequential frames, use the slow way
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, new_frame = cap.read()
                    if ret:
                        read_successful = True

                if read_successful:
                    frame = new_frame
                    cache[frame_idx] = frame

                    # Enforce the cache size limit
                    if len(cache) > self.onthefly_cache_size:
                        cache.popitem(last=False)  # remove oldest item

            if frame is None:
                return []

            all_frames.append(frame)

        return all_frames

    def _read_loop(self):
        """Main reading loop."""

        prev_frame_idx = -1

        # Get FPS from metadata to limit playback speed
        fps = self.app_state.video_metadata.get('fps', 30.0)
        if fps <= 0: fps = 30.0
        target_frame_duration = 1.0 / fps

        while not self.shutdown_event.is_set():
            loop_start_time = time.perf_counter()

            try:
                command = self.command_queue.get_nowait()
                action = command.get("action")
                if action == "shutdown":
                    self.shutdown_event.set()
                    break
                elif action == "reload_cache":
                    # Hotswap the cache reader
                    new_cache_reader = command.get("cache_reader")
                    if new_cache_reader is not None:
                        self.diskcache_reader = new_cache_reader
                        self.onthefly_cache = None  # on the fly cache is not needed in this case
                        print("VideoReaderWorker: Cache reader reloaded successfully")
            except queue.Empty:
                pass

            with self.app_state.lock:
                current_frame_idx = self.app_state.frame_idx
                is_paused = self.app_state.paused
                is_seeking = self.app_state.is_seeking
                num_frames = self.app_state.video_metadata['num_frames']

            if is_seeking:
                time.sleep(0.01)
                continue

            if current_frame_idx != prev_frame_idx:
                # Sequential read is only possible forward
                is_sequential_forward = (current_frame_idx == prev_frame_idx + 1)
                # But for tracking we just need to know if we moved by 1 frame
                is_adjacent = abs(current_frame_idx - prev_frame_idx) == 1

                frames = self._get_frames(current_frame_idx, is_sequential_forward)

                if not frames:
                    with self.app_state.lock:
                        self.app_state.paused = True
                    continue

                frame_data = {
                    "frame_idx": current_frame_idx,
                    "raw_frames": frames,
                    "was_sequential": is_sequential_forward,
                    "is_adjacent": is_adjacent # signal to tracker that we moved 1 step
                }
                self._distribute_frames(frame_data)
                prev_frame_idx = current_frame_idx

            if not is_paused:
                with self.app_state.lock:
                    if self.app_state.frame_idx < num_frames - 1:
                        self.app_state.frame_idx += 1
                    else:
                        self.app_state.paused = True

                # Calculate how much time weas spent working, sleep the rest
                work_duration = time.perf_counter() - loop_start_time
                sleep_duration = target_frame_duration - work_duration

                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            else:
                # If paused, sleep a standard amount to save CPU
                time.sleep(0.01)

    def _distribute_frames(self, frame_data: dict):
        """Send frame data to all output queues, clearing old data."""

        for q in self.output_queues:
            # Clear any stale frames
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            q.put(frame_data)

    def _cleanup(self):
        """Release all cv2 video captures."""

        if self.diskcache_reader is None:
            for cap in self.video_captures:
                cap.release()
        print("Video reader worker shut down.")
