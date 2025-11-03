"""
Worker threads for video reading, tracking, rendering, and calibration.
"""
import threading
import multiprocessing
import queue
import time
import cv2
import numpy as np
from typing import List

import config
from core import track_points, update_3d_reconstruction, create_camera_visual, run_genetic_step
from state import AppState
from old.viz_3d import SceneVisualizer, SceneObject


class VideoReaderWorker(threading.Thread):
    """Reads video frames and distributes them to processing workers."""

    def __init__(
            self,
            app_state: AppState,
            video_paths: List[str],
            command_queue: queue.Queue,
            output_queues: List[queue.Queue],
    ):
        super().__init__(daemon=True, name="VideoReaderWorker")
        self.app_state = app_state
        self.video_paths = video_paths
        self.command_queue = command_queue
        self.output_queues = output_queues
        self.shutdown_event = threading.Event()
        self.video_captures = []

    def run(self):
        print("Video reader worker started.")
        try:
            self._initialize_captures()
            self._read_loop()
        finally:
            self._cleanup()

    def _initialize_captures(self):
        """Open all video files."""

        self.video_captures = [cv2.VideoCapture(path) for path in self.video_paths]
        if not all(cap.isOpened() for cap in self.video_captures):
            print("ERROR: Video reader failed to open one or more videos.")
            self.shutdown_event.set()

    def _read_loop(self):
        """Main reading loop."""

        prev_frame_idx = -1

        while not self.shutdown_event.is_set():
            # Check for shutdown command
            try:
                command = self.command_queue.get_nowait()
                if command.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break
            except queue.Empty:
                pass

            # Get current state
            with self.app_state.lock:
                current_frame_idx = self.app_state.frame_idx
                is_paused = self.app_state.paused
                num_frames = self.app_state.video_metadata['num_frames']

            # Read frames only if index changed
            if current_frame_idx != prev_frame_idx:
                is_sequential = (current_frame_idx == prev_frame_idx + 1)

                if is_sequential:
                    frames = self._read_next_frames()
                else:
                    frames = self._seek_and_read(current_frame_idx)

                if not frames:
                    with self.app_state.lock:
                        self.app_state.paused = True
                    continue

                # Put frames in all output queues
                frame_data = {
                    "frame_idx": current_frame_idx,
                    "raw_frames": frames,
                    "was_sequential": is_sequential
                }
                self._distribute_frames(frame_data)
                prev_frame_idx = current_frame_idx

            # Advance if not paused
            if not is_paused:
                with self.app_state.lock:
                    if self.app_state.frame_idx < num_frames - 1:
                        self.app_state.frame_idx += 1
                    else:
                        self.app_state.paused = True
            else:
                time.sleep(0.01)

    def _seek_and_read(self, frame_idx: int) -> List[np.ndarray]:
        """Seek to specific frame and read from all cameras."""
        frames = []
        for cap in self.video_captures:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return []
            frames.append(frame)
        return frames

    def _read_next_frames(self) -> List[np.ndarray]:
        """Read next frame from all cameras (sequentially)."""
        frames = []
        for cap in self.video_captures:
            ret, frame = cap.read()
            if not ret:
                return []
            frames.append(frame)
        return frames

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
        for cap in self.video_captures:
            cap.release()
        print("Video reader worker shut down.")


class TrackingWorker(threading.Thread):
    """Point tracking using optic flow."""

    def __init__(
            self,
            app_state: AppState,
            frames_in_queue: queue.Queue,
            progress_out_queue: queue.Queue,
            video_paths: List[str],
            command_queue: queue.Queue
    ):
        super().__init__(daemon=True, name="TrackingWorker")
        self.app_state = app_state
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

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        if is_tracking_enabled and data["was_sequential"] and self.prev_frames:
            track_points(
                self.app_state,
                self.prev_frames,
                data["raw_frames"],
                data["frame_idx"]
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int):
        """Track points forward from start_frame to end of video."""

        print(f"Starting batch track from frame {start_frame}...")

        # Open video captures
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

            # Perform tracking
            track_points(self.app_state, prev_frames, current_frames, frame_idx)
            prev_frames = current_frames

            # Report progress
            if i % 5 == 0:
                self.progress_out_queue.put({
                    "status": "running",
                    "progress": i / total_to_process,
                    "current_frame": frame_idx,
                    "total_frames": num_frames
                })

        # Complete
        self.progress_out_queue.put({
            "status": "complete",
            "final_frame": num_frames - 1
        })
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
        for cap in caps:
            cap.release()


class RenderingWorker(threading.Thread):
    """Handles rendering of 2D overlays and 3D visualisation."""

    def __init__(
            self,
            app_state: AppState,
            scene_visualizer: SceneVisualizer,
            frames_in_queue: queue.Queue,
            results_out_queue: queue.Queue,
    ):
        super().__init__(daemon=True, name="RenderingWorker")
        self.app_state = app_state
        self.scene_visualizer = scene_visualizer
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
        """Render a complete frame with 3D scene."""

        # Update 3D reconstruction if needed
        with self.app_state.lock:
            needs_reconstruction = self.app_state.needs_3d_reconstruction
            best_individual = self.app_state.best_individual

        if needs_reconstruction and best_individual:
            update_3d_reconstruction(self.app_state)
            with self.app_state.lock:
                self.app_state.needs_3d_reconstruction = False

        # Prepare and render 3D scene
        scene = self._build_3d_scene()
        rendered_3d = self.scene_visualizer.draw_scene(scene)

        # Resize all frames to display size
        video_frames = [
            cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            for frame in data["raw_frames"]
        ]
        rendered_3d_resized = cv2.resize(
            rendered_3d,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        )

        # Send to GUI
        self._send_results({
            'frame_idx': data['frame_idx'],
            'video_frames_bgr': video_frames,
            '3d_frame_bgr': rendered_3d_resized,
        })

    def _build_3d_scene(self) -> List[SceneObject]:
        """Build list of 3D objects to render."""
        scene = []

        with self.app_state.lock:
            # Add camera visualisations
            if self.app_state.show_cameras_in_3d and self.app_state.best_individual:
                for i, cam_params in enumerate(self.app_state.best_individual):
                    scene.extend(
                        create_camera_visual(cam_params, self.app_state.video_names[i])
                    )

            # Add reconstructed points and skeleton
            points_3d = self.app_state.reconstructed_3d_points[self.app_state.frame_idx]
            point_names = self.app_state.point_names
            point_colors = self.app_state.point_colors
            skeleton = self.app_state.skeleton

        # Draw points
        for i, point in enumerate(points_3d):
            if not np.isnan(point).any():
                scene.append(SceneObject(
                    type='point',
                    coords=point,
                    color=point_colors[i],
                    label=point_names[i]
                ))

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

                self.progress_queue.put(progress)
            else:
                time.sleep(0.01)

        print("GA worker shut down.")