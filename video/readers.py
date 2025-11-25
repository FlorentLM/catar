"""
This worker is responsible for:
- Managing the video backend (cached or direct)
- Distributing frames to downstream workers (tracking, rendering)
- Handling playback control (play/pause, seeking)
- Hot-swapping cache when it becomes available
"""
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from state import AppState
    from video.backends import VideoBackend


class VideoReaderWorker(threading.Thread):
    """
    Reads video frames using a VideoBackend and distributes them to processing workers.

    The backend can be hot-swapped at runtime (e.g., when cache becomes available).
    """

    def __init__(
            self,
            app_state: 'AppState',
            video_backend: 'VideoBackend',
            command_queue: queue.Queue,
            output_queues: List[queue.Queue],
    ):
        """
        Initialize VideoReaderWorker.

        Args:
            app_state: Shared application state
            video_backend: VideoBackend instance for frame access
            command_queue: Queue for receiving commands
            output_queues: List of queues to send frames to
        """
        super().__init__(daemon=True, name="VideoReaderWorker")
        self.app_state = app_state
        self.backend = video_backend
        self.command_queue = command_queue
        self.output_queues = output_queues
        self.shutdown_event = threading.Event()

    def run(self):
        """Main reading loop."""

        print("VideoReaderWorker started")

        try:
            self._read_loop()
        except Exception as e:
            print(f"VideoReaderWorker error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _read_loop(self):
        """(Inner) main loop for reading and distributing frames."""

        prev_frame_idx = -1

        # Get FPS from metadata to limit playback speed
        default_fps = 60.0
        fps = self.app_state.video_metadata.get('fps', default_fps)
        if fps <= 0:
            fps = default_fps
        target_frame_duration = 1.0 / fps

        while not self.shutdown_event.is_set():
            loop_start_time = time.perf_counter()

            # Process commands
            self._process_commands()

            # Get current state
            with self.app_state.lock:
                current_frame_idx = self.app_state.frame_idx
                is_paused = self.app_state.paused
                is_seeking = self.app_state.is_seeking
                num_frames = self.app_state.video_metadata['num_frames']

            # Don't read during seek operations
            if is_seeking:
                time.sleep(0.01)
                continue

            # Read and distribute frames if index changed
            if current_frame_idx != prev_frame_idx:
                # Determine adjacency for tracking (needs to know if we moved by 1 frame)
                is_adjacent = abs(current_frame_idx - prev_frame_idx) == 1
                was_sequential_forward = (current_frame_idx == prev_frame_idx + 1)

                try:
                    frames = self.backend.get_frame(current_frame_idx)

                    if not frames:
                        print(f"Failed to read frame {current_frame_idx}")
                        with self.app_state.lock:
                            self.app_state.paused = True
                        continue

                    frame_data = {
                        "frame_idx": current_frame_idx,
                        "raw_frames": frames,
                        "was_sequential": was_sequential_forward,
                        "is_adjacent": is_adjacent
                    }
                    self._distribute_frames(frame_data)
                    prev_frame_idx = current_frame_idx

                except Exception as e:
                    print(f"Error reading frame {current_frame_idx}: {e}")
                    with self.app_state.lock:
                        self.app_state.paused = True
                    continue

            # Auto-advance if playing
            if not is_paused:
                with self.app_state.lock:
                    if self.app_state.frame_idx < num_frames - 1:
                        self.app_state.frame_idx += 1
                    else:
                        self.app_state.paused = True

                # Framerate limiting
                work_duration = time.perf_counter() - loop_start_time
                sleep_duration = target_frame_duration - work_duration
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            else:
                # If paused, sleep to save CPU
                time.sleep(0.01)

    def _process_commands(self):
        """Process commands from the command queue."""

        try:
            command = self.command_queue.get_nowait()
            action = command.get("action")

            if action == "shutdown":
                self.shutdown_event.set()

            elif action == "update_backend":
                # Hotswap backend
                new_backend = command.get("backend")
                if new_backend is not None:
                    print("VideoReaderWorker: Updating backend...")

                    # Close old backend
                    self.backend.close()

                    # Switch to new backend
                    self.backend = new_backend
                    print(f"VideoReaderWorker: Backend updated to {type(new_backend).__name__}")

            elif action == "clear_cache":
                self.backend.clear_cache()
                print("VideoReaderWorker: Backend cache cleared")

            elif action == "print_stats":
                stats = self.backend.get_stats()
                print(f"VideoReaderWorker stats: {stats}")

        except queue.Empty:
            pass

    def _distribute_frames(self, frame_data: dict):
        """
        Send frame data to all output queues, clearing old data.

        Args:
            frame_data: Dictionary containing frame info and raw frames
        """
        for q in self.output_queues:
            # Clear any stale frames
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            q.put(frame_data)

    def _cleanup(self):
        print("VideoReaderWorker: Cleaning up...")
        if self.backend:
            self.backend.close()
        print("VideoReaderWorker shut down")


class BatchVideoReader:
    """
    Helper class for batch processing with VideoBackend.

    Provides efficient sequential or random access for batch tracking operations.
    Used by TrackingWorker for batch tracking.
    """

    def __init__(self, backend: 'VideoBackend'):
        """
        Initialize batch reader.

        Args:
            backend: VideoBackend instance to use
        """
        self.backend = backend
        self.cache = {}  # Simple cache for recently read frames
        self.max_cache_size = 10

    def read_frame(self, frame_idx: int) -> Optional[List[np.ndarray]]:
        """
        Read a frame with simple caching.

        Args:
            frame_idx: Frame index to read

        Returns:
            List of frames (one per view) or None on error
        """
        # Check cache first
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        try:
            frames = self.backend.get_frame(frame_idx)

            # Add to cache
            self.cache[frame_idx] = frames

            # Enforce cache size
            if len(self.cache) > self.max_cache_size:
                # Remove oldest
                oldest_key = min(self.cache.keys())
                del self.cache[oldest_key]

            return frames

        except Exception as e:
            print(f"BatchVideoReader: Error reading frame {frame_idx}: {e}")
            return None

    def prefetch_range(self, start_idx: int, end_idx: int):
        """
        Hint that a range of frames will be needed.

        Args:
            start_idx: Start frame index
            end_idx: End frame index (inclusive)
        """
        frame_indices = list(range(start_idx, end_idx + 1))
        self.backend.prefetch(frame_indices)

    def clear_cache(self):
        self.cache.clear()