import threading
import multiprocessing
import queue
import time
import cv2
import numpy as np
from typing import List

from core import (
    track_points,
    update_3d_reconstruction,
    draw_ui,
    create_camera_visual,
    run_genetic_step,
)
from state import AppState
from viz_3d import SceneVisualizer, SceneObject

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

class VideoReaderWorker(threading.Thread):

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
            self.video_captures = [cv2.VideoCapture(path) for path in self.video_paths]
            if not all(cap.isOpened() for cap in self.video_captures):
                print("!!! CRITICAL ERROR: Video reader failed to open one or more videos.")
                return

            prev_frame_idx = -1

            while not self.shutdown_event.is_set():
                # Check for a shutdown command
                try:
                    command = self.command_queue.get_nowait()
                    if command.get("action") == "shutdown":
                        self.shutdown_event.set()
                        break
                except queue.Empty:
                    pass

                with self.app_state.lock:
                    current_frame_idx = self.app_state.frame_idx
                    is_paused = self.app_state.paused

                # If paused and the frame hasn't been changed by the user, do nothing
                if prev_frame_idx == current_frame_idx and is_paused:
                    time.sleep(0.01)
                    continue

                # The core logic: if the requested frame is not the next sequential one,
                # we must perform a slow 'seek'. Otherwise, we do a fast 'read'.
                is_sequential = (current_frame_idx == prev_frame_idx + 1)
                if is_sequential:
                    current_frames = self._read_next_frames()
                else:
                    current_frames = self._read_frames_for_idx(current_frame_idx)

                if not current_frames:
                    with self.app_state.lock:
                        self.app_state.paused = True  # Stop playback at the end of the video
                    continue

                output_data = {
                    "frame_idx": current_frame_idx,
                    "raw_frames": current_frames,
                    "was_sequential": is_sequential,
                }

                for q in self.output_queues:
                    # Clear the queue to prevent consumers from lagging behind
                    while not q.empty():
                        q.get_nowait()
                    q.put(output_data)

                prev_frame_idx = current_frame_idx

                if not is_paused:
                    with self.app_state.lock:
                        num_frames = self.app_state.video_metadata['num_frames']
                        if self.app_state.frame_idx < num_frames - 1:
                            self.app_state.frame_idx += 1
                        else:
                            self.app_state.paused = True
        finally:
            for cap in self.video_captures:
                cap.release()
            print("Video reader worker shut down.")

    def _read_frames_for_idx(self, frame_idx: int) -> List[np.ndarray]:
        frames = []
        for cap in self.video_captures:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: return []
            frames.append(frame)
        return frames

    def _read_next_frames(self) -> List[np.ndarray]:
        frames = []
        for cap in self.video_captures:
            ret, frame = cap.read()
            if not ret: return []
            frames.append(frame)
        return frames


class TrackingWorker(threading.Thread):
    """
    Consumes raw frames and performs optic flow tracking if enabled.
    (does not render or display anything, only updates the annotations in the AppState)
    """
    def __init__(
        self,
        app_state: AppState,
        frames_in_queue: queue.Queue,
    ):
        super().__init__(daemon=True, name="TrackingWorker")
        self.app_state = app_state
        self.frames_in_queue = frames_in_queue
        self.shutdown_event = threading.Event()
        self.prev_frames = None
        self.prev_frame_idx = -1

    def run(self):
        print("Tracking worker started.")
        while not self.shutdown_event.is_set():
            try:
                data = self.frames_in_queue.get(timeout=1.0)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                with self.app_state.lock:
                    is_tracking_enabled = self.app_state.keypoint_tracking_enabled

                # core logic: only track if enabled and frames are sequential
                if is_tracking_enabled and data["was_sequential"] and self.prev_frames:
                    track_points(self.app_state, self.prev_frames, data["raw_frames"])

                # Update state for next iteration
                self.prev_frames = data["raw_frames"]
                self.prev_frame_idx = data["frame_idx"]

            except queue.Empty:
                continue # No new frames, just wait
            except Exception as e:
                print(f"!!! CRITICAL ERROR IN TRACKING WORKER: {e}")
                import traceback
                traceback.print_exc()
                self.shutdown_event.set()
        print("Tracking worker shut down.")


class RenderingWorker(threading.Thread):
    """
    Consumes raw frames, handles rendering (2D overlays and 3D scene),
    and sends the display-ready frames to the GUI thread.
    """
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

                with self.app_state.lock:
                    needs_reconstruction = self.app_state.needs_3d_reconstruction
                    best_individual = self.app_state.best_individual

                if needs_reconstruction and best_individual:
                    update_3d_reconstruction(self.app_state)
                    with self.app_state.lock:
                        self.app_state.needs_3d_reconstruction = False

                scene = self._prepare_3d_scene()
                rendered_3d_frame = self.scene_visualizer.draw_scene(scene)

                rendered_video_frames = [
                    draw_ui(frame.copy(), cam_idx, self.app_state)
                    for cam_idx, frame in enumerate(data["raw_frames"])
                ]

                # Resize all frames to the display dimensions before sending to the GUI
                final_video_frames = [cv2.resize(f, (DISPLAY_WIDTH, DISPLAY_HEIGHT)) for f in rendered_video_frames]
                final_3d_frame = cv2.resize(rendered_3d_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                if self.results_out_queue.empty():
                    self.results_out_queue.put({
                        'frame_idx': data['frame_idx'],
                        'video_frames_bgr': final_video_frames,
                        '3d_frame_bgr': final_3d_frame,
                    })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"!!! CRITICAL ERROR IN RENDERING WORKER: {e}")
                import traceback
                traceback.print_exc()
                self.shutdown_event.set()

        print("Rendering worker shut down.")

    def _prepare_3d_scene(self) -> List[SceneObject]:
        """Prepares the list of 3D objects for rendering from the AppState."""

        scene = []
        with self.app_state.lock:
            if not self.app_state.show_cameras_in_3d:
                pass # Early exit if cameras are hidden

            if self.app_state.best_individual:
                for i, cam_params in enumerate(self.app_state.best_individual):
                    scene.extend(create_camera_visual(cam_params, label=self.app_state.video_names[i]))

            points_3d = self.app_state.reconstructed_3d_points[self.app_state.frame_idx]
            point_names = self.app_state.POINT_NAMES
            point_colors = self.app_state.point_colors
            skeleton_def = self.app_state.SKELETON

            for i, point in enumerate(points_3d):
                if not np.isnan(point).any():
                    scene.append(SceneObject(type='point', coords=point, color=point_colors[i], label=point_names[i]))
                    # Draw skeleton connections
                    for end_point_name in skeleton_def.get(point_names[i], []):
                        try:
                            end_point_idx = point_names.index(end_point_name)
                            if not np.isnan(points_3d[end_point_idx]).any():
                                scene.append(SceneObject(
                                    type='line',
                                    coords=np.array([point, points_3d[end_point_idx]]),
                                    color=(128, 128, 128), label=None
                                ))
                        except ValueError:
                            pass
        return scene


class GAWorker(multiprocessing.Process):
    def __init__(self, command_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue):
        super().__init__(name="GAWorker")
        self.command_queue = command_queue
        self.progress_queue = progress_queue
        self.ga_state = {}

    def run(self):
        print("GA worker started.")
        self.ga_state["is_running"] = False

        while True:
            try:
                command = self.command_queue.get_nowait()
                if command.get("action") == "shutdown":
                    break

                elif command.get("action") == "start":
                    print("GA worker received start command.")
                    self.ga_state = command.get("ga_state_snapshot")
                    self.ga_state["is_running"] = True
                    self.ga_state["population"] = None # reset population

                elif command.get("action") == "stop":
                    print("GA worker received stop command.")
                    self.ga_state["is_running"] = False

            except queue.Empty:
                pass

            if self.ga_state["is_running"]:
                progress_data = run_genetic_step(self.ga_state)
                # Update state for next iteration
                self.ga_state["best_fitness"] = progress_data["new_best_fitness"]
                self.ga_state["best_individual"] = progress_data["new_best_individual"]
                self.ga_state["generation"] = progress_data["generation"]
                self.ga_state["population"] = progress_data["next_population"]

                self.progress_queue.put(progress_data)
            else:
                time.sleep(0.01) # Sleep when not running

        print("GA worker shut down.")