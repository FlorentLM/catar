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
from core import create_camera_visual, run_genetic_step, process_frame, run_refinement
from state import AppState
from viz_3d import Open3DVisualizer, SceneObject
from cache_utils import DiskCacheReader

from mokap.utils.geometry import projective
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.tracking import MultiObjectTracker


class VideoReaderWorker(threading.Thread):
    """Reads video frames and distributes them to processing workers."""

    def __init__(
        self,
        app_state: AppState,
        command_queue: queue.Queue,
        output_queues: List[queue.Queue],
        diskcache_reader: Optional[DiskCacheReader] = None,
    ):
        super().__init__(daemon=True, name="VideoReaderWorker")
        self.app_state = app_state
        self.video_paths = app_state.videos.filepaths
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


class TrackingWorker(threading.Thread):
    """
    Handles automated point tracking and on-demand 3D reconstruction.
    """

    def __init__(
        self,
        app_state: AppState,
        reconstructor: Reconstructor,
        tracker: MultiObjectTracker,
        frames_in_queue: queue.Queue,
        progress_out_queue: queue.Queue,
        command_queue: queue.Queue,
        stop_batch_track: threading.Event
    ):
        super().__init__(daemon=True, name="TrackingWorker")

        self.app_state = app_state
        self.reconstructor = reconstructor
        self.tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.progress_out_queue = progress_out_queue
        self.video_paths = app_state.videos.filepaths
        self.command_queue = command_queue
        self.stop_batch_track_event = stop_batch_track
        self.shutdown_event = threading.Event()     # TODO Events should all be defined at the same place probably...
        self.prev_frames = None
        self.prev_frame_idx = -1

    def run(self):
        print("Tracking worker started.")

        while not self.shutdown_event.is_set():
            # Check for special commands (batch tracking)
            try:
                command = self.command_queue.get_nowait()

                if command.get("action") == "batch_track":
                    start_frame = command["start_frame"]
                    direction = command.get("direction", 1) # default to forward (1)
                    self._run_batch_tracking(start_frame, direction)
                    continue

                elif command.get("action") == "update_calibration":
                    print("TrackingWorker: Received calibration update command.")
                    self.reconstructor.update_camera_parameters(command["calibration"])
                    continue

            except queue.Empty:
                pass

            # Check if an on-demand reconstruction is needed from the GUI
            with self.app_state.lock:
                needs_reconstruction = self.app_state.needs_3d_reconstruction
                frame_idx = self.app_state.frame_idx

            if needs_reconstruction:
                self._reconstruct_for_display(frame_idx)

            # Realtime tracking mode
            try:
                data = self.frames_in_queue.get(timeout=0.01)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                self._process_frame_for_tracking(data)

            except queue.Empty:
                continue

            except Exception as e:
                print(f"ERROR in tracking worker: {e}")
                import traceback
                traceback.print_exc()

        print("Tracking worker shut down.")

    def _reconstruct_for_display(self, frame_idx: int):
        """
        Triangulates points from manual annotations for immediate UI feedback.
        This is called when the user clicks or modifies a point.
        """

        with self.app_state.lock:
            # Get only (x, y) coordinates
            annotations = self.app_state.annotations[frame_idx, ..., :2]
            calibration = self.app_state.calibration

        if calibration.best_calibration is not None:
            proj_matrices = calibration.P_mats

            # Triangulate all points for the current frame
            points_3d = projective.triangulate_points_from_projections(
                points2d=annotations,
                P_mats=proj_matrices
            )

            with self.app_state.lock:
                self.app_state.reconstructed_3d_points[frame_idx] = np.asarray(points_3d)

                # Reset the flag now that reconstruction is done
                self.app_state.needs_3d_reconstruction = False

    def _process_frame_for_tracking(self, data: dict):
        """Process a single frame for the automated tracking pipeline (Live Tracking)."""

        frame_idx = data["frame_idx"]

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        # We can track if enabled, and if we have adjacent frames (either forward or backward)
        # ('is_adjacent' is populated by VideoReaderWorker)
        can_track = is_tracking_enabled and self.prev_frames and data.get("is_adjacent", False)

        # Double check adjacency manually to be safe
        if can_track and abs(frame_idx - self.prev_frame_idx) != 1:
            can_track = False

        if can_track:
            print(f"[Frame {frame_idx}] TrackingWorker: Live tracking from {self.prev_frame_idx}.")
            process_frame(
                frame_idx=frame_idx,
                source_frame_idx=self.prev_frame_idx,
                app_state=self.app_state,
                reconstructor=self.reconstructor,
                tracker=self.tracker,
                source_frames=self.prev_frames,
                dest_frames=data["raw_frames"]
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int, direction: int = 1):
        """
        Track points starting from start_frame in the given direction (1 = forward or -1 = backward).
        """
        # TODO: This should use the VideoReaderWorker so we don't have to have separate VideoCaptures, and so we can uptate the UI in real time

        dir_str = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"Starting batch track {dir_str} from frame {start_frame}...")

        # Check cache
        cache_reader = self.app_state.cache_reader
        use_cache = cache_reader is not None

        caps = []
        caps_file_pointer = -1

        if use_cache:
            print("Batch tracking using DiskCacheReader.")
        else:
            print("Batch tracking using VideoCapture.")
            # Only initialize caps if we have to
            caps = [cv2.VideoCapture(path) for path in self.video_paths]
            if not all(cap.isOpened() for cap in caps):
                print("Batch track error: Could not open videos.")
                return

        num_frames = self.app_state.video_metadata['num_frames']

        # Determine the range of frames to process
        if direction == 1:
            # Forward: start + 1 -> end
            frame_range = range(start_frame + 1, num_frames)
            total_to_process = num_frames - start_frame - 1
        else:
            # Backward: start - 1 -> 0
            frame_range = range(start_frame - 1, -1, -1)
            total_to_process = start_frame

        if total_to_process <= 0:
            print("Batch track: No frames to process in this direction.")
            self._cleanup_captures(caps)
            return

        # Read the initial source frames
        source_frames = []

        if use_cache:
            # Priority 1: cache
            try:
                source_frames = cache_reader.get_frame(start_frame)
            except Exception as e:
                print(f"Cache read error: {e}")
        else:
            # Priority 2/3: VideoCapture
            # Since we just opened them we must seek to start_frame
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()
                if ret: source_frames.append(frame)

            # Update pointer (VideoCapture is now at start_frame + 1)
            if len(source_frames) == len(caps):
                caps_file_pointer = start_frame + 1

        if not source_frames:
            print("Could not read start frame.")
            self._cleanup_captures(caps)
            return

        current_source_idx = start_frame

        # Main tracking loop
        for i, dest_frame_idx in enumerate(frame_range):
            if self.stop_batch_track_event.is_set():
                print(f"Batch track stopped by user at frame {dest_frame_idx}.")
                break

            dest_frames = []

            if use_cache:
                # Always use cache if available
                try:
                    dest_frames = cache_reader.get_frame(dest_frame_idx)
                except Exception as e:
                    print(f"Cache read error: {e}")
                    break

            else:
                # VideoCapture sequential read (forward only)
                if (direction == 1) and (dest_frame_idx == caps_file_pointer):
                    for cap in caps:
                        ret, f = cap.read()
                        if ret: dest_frames.append(f)

                    if len(dest_frames) == len(caps):
                        caps_file_pointer += 1
                    else:
                        dest_frames = [] # Signal error

                # VideoCapture seek (backward or jump)
                else:
                    dest_frames = []
                    for cap in caps:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, dest_frame_idx)
                        ret, f = cap.read()
                        if ret: dest_frames.append(f)

                    if len(dest_frames) == len(caps):
                        caps_file_pointer = dest_frame_idx + 1 # Update pointer after seek+read

            # Check if read was successful
            if not dest_frames:
                print(f"Failed to read frames at index {dest_frame_idx}.")
                break

            # Run the core logic source -> dest
            process_frame(
                frame_idx=dest_frame_idx,
                source_frame_idx=current_source_idx,
                app_state=self.app_state,
                reconstructor=self.reconstructor,
                tracker=self.tracker,
                source_frames=source_frames,
                dest_frames=dest_frames
            )

            # Destination becomes source for the next iteration
            source_frames = dest_frames
            current_source_idx = dest_frame_idx

            # Report progress
            if i % 5 == 0:
                self.progress_out_queue.put({
                    "status": "running",
                    "progress": i / total_to_process,
                    "current_frame": dest_frame_idx,
                    "total_frames": num_frames
                })

        # Signal completion with the final frame for the UI to jump to
        final_frame = frame_range[-1] if len(frame_range) > 0 else start_frame
        self.progress_out_queue.put({"status": "complete", "final_frame": final_frame})

        self._cleanup_captures(caps)
        print(f"Batch track {dir_str} complete.")

    def _cleanup_captures(self, caps: List[cv2.VideoCapture]):
        """Release video captures if they were created."""
        if caps:
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
            calibration = self.app_state.calibration

            if self.app_state.show_cameras_in_3d and calibration.best_calibration:
                for cam_name in calibration.camera_names:
                    cam_params = calibration.get(cam_name)
                    scene.extend(
                        create_camera_visual(cam_params, cam_name, self.app_state.scene_centre)
                    )

            # Add reconstructed points and skeleton
            points_3d = self.app_state.reconstructed_3d_points[self.app_state.frame_idx]
            point_names = self.app_state.point_names
            point_colors = self.app_state.point_colors
            skeleton = self.app_state.skeleton

        # Debug: count valid points
        # valid_points = 0

        # Draw points
        for i, point in enumerate(points_3d):
            if not np.isnan(point).any():
                # valid_points += 1

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

    def __init__(self, command_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
        super().__init__(name="BAWorker")
        self.command_queue = command_queue
        self.results_queue = results_queue
        self.stop_event = stop_event

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
                        results = run_refinement(snapshot)

                        if self.stop_event.is_set():
                            print("BA worker finished, but a stop was requested. Discarding results.")
                        else:
                            # if not cancelled send the results
                            self.results_queue.put(results)

                    except Exception as e:
                        import traceback

                        print(f"[ BA WORKER EXCEPTION ]")
                        traceback.print_exc()
                        print(f"[     END EXCEPTION   ]")

                        # check for cancellation before reporting an error
                        if not self.stop_event.is_set():
                            self.results_queue.put({"status": "error", "message": str(e)})

            except queue.Empty:
                continue

        print("BA worker shut down.")

