"""
Worker threads for video reading, tracking, rendering, and calibration.
"""
import collections
import threading
import multiprocessing
import queue
import time
import cv2
import numpy as np
from typing import List, Optional

import config
from core import create_camera_visual, run_genetic_step, process_frame, run_adjustment_perframe
from state import AppState
from viz_3d import Open3DVisualizer, SceneObject
from video_cache import VideoCacheReader

from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.tracking import MultiObjectTracker


class VideoReaderWorker(threading.Thread):
    """Reads video frames and distributes them to processing workers."""

    def __init__(
        self,
        app_state: AppState,
        video_paths: List[str],
        command_queue: queue.Queue,
        output_queues: List[queue.Queue],
        cache_reader: Optional[VideoCacheReader] = None,
    ):
        super().__init__(daemon=True, name="VideoReaderWorker")
        self.app_state = app_state
        self.video_paths = video_paths
        self.command_queue = command_queue
        self.output_queues = output_queues
        self.cache_reader = cache_reader
        self.shutdown_event = threading.Event()
        self.video_captures = []

        # We only need the on-the-fly cache if not using the cached reader
        if self.cache_reader is None:
            self.onthefly_cache = [collections.OrderedDict() for _ in video_paths]
            self.onthefly_cache_size = 300
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
        
        if self.cache_reader is not None:
            print("Video reader using cached frames.")
            return

        self.video_captures = [cv2.VideoCapture(path) for path in self.video_paths]
        if not all(cap.isOpened() for cap in self.video_captures):
            print("ERROR: Video reader failed to open one or more videos.")
            self.shutdown_event.set()

    def _get_frames(self, frame_idx: int, is_sequential: bool) -> List[np.ndarray]:
        """
        Gets frames for a specific index.
        Uses cache reader if available, otherwise falls back to cv2.VideoCapture with on-the-fly caching.
        """
        # Fast path: use cache reader if available
        if self.cache_reader is not None:
            try:
                return self.cache_reader.get_frame(frame_idx)
            except Exception as e:
                print(f"ERROR reading from cache: {e}")
                return []

        # Fallback: use cv2.VideoCapture with on the fly frame caching
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

        while not self.shutdown_event.is_set():
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
                        self.cache_reader = new_cache_reader
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
                is_sequential = (current_frame_idx == prev_frame_idx + 1)

                frames = self._get_frames(current_frame_idx, is_sequential)

                if not frames:
                    with self.app_state.lock:
                        self.app_state.paused = True
                    continue

                frame_data = {
                    "frame_idx": current_frame_idx,
                    "raw_frames": frames,
                }
                self._distribute_frames(frame_data)
                prev_frame_idx = current_frame_idx

            if not is_paused:
                with self.app_state.lock:
                    if self.app_state.frame_idx < num_frames - 1:
                        self.app_state.frame_idx += 1
                    else:
                        self.app_state.paused = True
            else:
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
        """Release all video captures."""
        if self.cache_reader is None:
            for cap in self.video_captures:
                cap.release()
        print("Video reader worker shut down.")


class TrackingWorker(threading.Thread):
    """Point tracking using optic flow."""

    def __init__(
        self,
        app_state: AppState,
        reconstructor: Reconstructor,
        tracker: MultiObjectTracker,
        frames_in_queue: queue.Queue,
        progress_out_queue: queue.Queue,
        video_paths: List[str],
        command_queue: queue.Queue
    ):
        super().__init__(daemon=True, name="TrackingWorker")
        self.app_state = app_state
        self.reconstructor = reconstructor
        self.tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.progress_out_queue = progress_out_queue
        self.video_paths = video_paths
        self.command_queue = command_queue
        self.shutdown_event = threading.Event()
        self.prev_frames = None
        self.prev_frame_idx = -1

    def run(self):
        print("Tracking worker started.")
        while not self.shutdown_event.is_set():
            # Check for special commands (batch tracking)
            try:
                command = self.command_queue.get_nowait()
                if command.get("action") == "batch_track":
                    self._run_batch_tracking(command["start_frame"])
                    continue
            except queue.Empty:
                pass

            # Realtime tracking mode
            try:
                data = self.frames_in_queue.get(timeout=0.1)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                self._process_frame(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in tracking worker: {e}")
                import traceback
                traceback.print_exc()

        print("Tracking worker shut down.")

    def _process_frame(self, data: dict):
        """Process a single frame for tracking."""
        frame_idx = data["frame_idx"]

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        # runs entire LK -> Mokap -> Feedback loop
        if is_tracking_enabled and data["was_sequential"] and self.prev_frames:
            print(f"[Frame {frame_idx}] TrackingWorker: Running full processing pipeline.")
            process_frame(
                frame_idx,
                self.app_state,
                self.reconstructor,
                self.tracker,
                self.prev_frames,
                data["raw_frames"]
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int):
        """Track points forward using the full feedback loop on every frame."""
        print(f"Starting batch track from frame {start_frame}...")

        caps = [cv2.VideoCapture(path) for path in self.video_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Batch track error: Could not open videos.")
            return

        num_frames = self.app_state.video_metadata['num_frames']
        total_to_process = num_frames - start_frame - 1

        # Read initial frame
        prev_frames = self._read_frames_at(caps, start_frame)
        if not prev_frames:
            self._cleanup_captures(caps)
            return

        # Track through remaining frames
        for i, frame_idx in enumerate(range(start_frame + 1, num_frames)):
            if self.app_state.stop_batch_track.is_set():
                print(f"Batch track stopped by user at frame {frame_idx}.")
                break

            # Read current frames
            current_frames = [cap.read()[1] for cap in caps]
            if not all(f is not None for f in current_frames):
                break


            # This single call does LK prediction and mokap correction
            process_frame(
                frame_idx,
                self.app_state,
                self.reconstructor,
                self.tracker,
                prev_frames,
                current_frames
            )

            prev_frames = current_frames

            # Report progress
            if i % 5 == 0:
                self.progress_out_queue.put({
                    "status": "running", "progress": i / total_to_process,
                    "current_frame": frame_idx, "total_frames": num_frames
                })

        self.progress_out_queue.put({"status": "complete", "final_frame": num_frames - 1})
        self._cleanup_captures(caps)
        print("Batch track complete.")

    def _read_frames_at(self, caps: List[cv2.VideoCapture], frame_idx: int) -> List[np.ndarray]:
        """Read frames from all captures at specific index."""
        frames = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return []
            frames.append(frame)
        return frames

    def _cleanup_captures(self, caps: List[cv2.VideoCapture]):
        """Release video captures."""
        for cap in caps:
            cap.release()


class RenderingWorker(threading.Thread):
    """Handles rendering of 2D overlays and 3D visualisation."""

    def __init__(
        self,
        app_state: AppState,
        open3d_viz: Open3DVisualizer,
        reconstructor: Reconstructor,
        tracker: MultiObjectTracker,
        frames_in_queue: queue.Queue,
        results_out_queue: queue.Queue,
    ):
        super().__init__(daemon=True, name="RenderingWorker")
        self.app_state = app_state
        self.open3d_viz = open3d_viz
        self.reconstructor = reconstructor
        self.tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.results_out_queue = results_out_queue
        self.shutdown_event = threading.Event()

    def run(self):
        print("Rendering worker started.")
        while not self.shutdown_event.is_set():
            try:
                data = self.frames_in_queue.get(timeout=1.0)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                self._render_frame(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in rendering worker: {e}")
                import traceback
                traceback.print_exc()
                self.shutdown_event.set()

        print("Rendering worker shut down.")

    def _render_frame(self, data: dict):
        """Renders the current state for visualisation without modifying it."""

        with self.app_state.lock:
            # frame_idx = self.app_state.frame_idx

            # Cache the raw video frames for the loupe tool
            self.app_state.current_video_frames = data["raw_frames"]

        scene = self._build_3d_scene()
        self.open3d_viz.queue_update(scene)

        # Resize all frames to display size
        video_frames = [
            cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            for frame in data["raw_frames"]
        ]

        # Send to GUI for display
        self._send_results({
            'frame_idx': data['frame_idx'],
            'video_frames_bgr': video_frames,
        })

    def _build_3d_scene(self) -> List[SceneObject]:
        """Build list of 3D objects to render."""

        scene = []

        with self.app_state.lock:
            # Add camera visualisations
            if self.app_state.show_cameras_in_3d and self.app_state.best_individual:
                for i, cam_params in enumerate(self.app_state.best_individual):
                    scene.extend(
                        create_camera_visual(cam_params, self.app_state.video_names[i], self.app_state.scene_centre)
                    )

            # Add reconstructed points and skeleton
            points_3d = self.app_state.reconstructed_3d_points[self.app_state.frame_idx]
            point_names = self.app_state.point_names
            point_colors = self.app_state.point_colors
            skeleton = self.app_state.skeleton

        # Debug: count valid points
        valid_points = 0

        # Draw points
        for i, point in enumerate(points_3d):
            if not np.isnan(point).any():
                valid_points += 1
                color = tuple(point_colors[i])
                scene.append(SceneObject(
                    type='point',
                    coords=point,
                    color=color,
                    label=point_names[i]
                ))

                # # Debug: print 1st point
                # if valid_points == 1:
                #     print(f"Adding point '{point_names[i]}' at {point} with color {color}")

                # Draw skeleton connections
                for connected_name in skeleton.get(point_names[i], []):
                    try:
                        j = point_names.index(connected_name)
                        if not np.isnan(points_3d[j]).any():
                            scene.append(SceneObject(
                                type='line',
                                coords=np.array([point, points_3d[j]]),
                                color=(128, 128, 128),
                                label=None
                            ))
                    except ValueError:
                        pass

        # # Debug: Print scene info sometimes
        # if self.app_state.frame_idx % 30 == 0:  # every 30 frames
        #     print(f"3D Scene: {len(scene)} objects, {valid_points} valid keypoints")

        return scene

    def _send_results(self, results: dict):
        """Sends rendered results to GUI and clears old data."""

        while not self.results_out_queue.empty():
            try:
                self.results_out_queue.get_nowait()
            except queue.Empty:
                break
        self.results_out_queue.put(results)


class GAWorker(multiprocessing.Process):
    """Runs genetic algorithm for camera calibration in separate process."""

    def __init__(self, command_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue):
        super().__init__(name="GAWorker")
        self.command_queue = command_queue
        self.progress_queue = progress_queue
        self.ga_state = {"is_running": False}

    def run(self):
        print("GA worker started.")

        while True:
            # Check for commands
            try:
                command = self.command_queue.get_nowait()

                if command.get("action") == "shutdown":
                    break
                elif command.get("action") == "start":
                    print("GA worker received start command.")
                    self.ga_state = command.get("ga_state_snapshot")
                    self.ga_state["is_running"] = True
                    self.ga_state["population"] = None  # Reset population
                elif command.get("action") == "stop":
                    print("GA worker received stop command.")
                    self.ga_state["is_running"] = False

            except queue.Empty:
                pass

            # Run GA step if active
            if self.ga_state.get("is_running"):
                progress = run_genetic_step(self.ga_state)

                # Update state for next iteration
                self.ga_state["best_fitness"] = progress["new_best_fitness"]
                self.ga_state["best_individual"] = progress["new_best_individual"]
                self.ga_state["generation"] = progress["generation"]
                self.ga_state["population"] = progress["next_population"]

                progress_payload = {
                    "status": "running",
                    "generation": progress["generation"],
                    "best_fitness": progress["new_best_fitness"],
                    "mean_fitness": progress["mean_fitness"],
                    "new_best_individual": progress["new_best_individual"]
                }
                self.progress_queue.put(progress_payload)
            else:
                time.sleep(0.01)

        print("GA worker shut down.")


class BAWorker(multiprocessing.Process):
    """Runs bundle adjustment in a separate process."""

    def __init__(self, command_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue):
        super().__init__(name="BAWorker")
        self.command_queue = command_queue
        self.results_queue = results_queue

    def run(self):
        print("BA worker started.")
        while True:
            try:
                command = self.command_queue.get()  # Block until a command arrives

                if command.get("action") == "shutdown":
                    break

                if command.get("action") == "start":
                    print("BA worker received start command.")
                    snapshot = command.get("ba_state_snapshot")
                    try:
                        results = run_adjustment_perframe(snapshot)
                        self.results_queue.put(results)
                    except Exception as e:
                        import traceback
                        print(f"--- BA WORKER EXCEPTION ---")
                        traceback.print_exc()
                        print(f"--- END EXCEPTION ---")
                        self.results_queue.put({"status": "error", "message": str(e)})

            except queue.Empty:
                continue

        print("BA worker shut down.")