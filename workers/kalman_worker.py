import queue
import threading
from typing import TYPE_CHECKING
from video import BatchVideoReader

from core.trackers.tracker_kalman_simple import SimpleTracker
from core.improved_tracker import create_tracker

if TYPE_CHECKING:
    from state import AppState
    from video import VideoBackend


class KalmanWorker(threading.Thread):
    def __init__(
            self,
            app_state: 'AppState',
            bone_stats: 'BonesStats',
            video_backend: 'VideoBackend',
            reconstructor=None,
            tracker=None,
            frames_in_queue: queue.Queue = None,
            progress_out_queue: queue.Queue = None,
            command_queue: queue.Queue = None,
            stop_batch_track: threading.Event = None
    ):
        super().__init__(daemon=True, name="KalmanWorker")

        self.app_state = app_state
        self.video_backend = video_backend
        self.frames_in_queue = frames_in_queue
        self.progress_out_queue = progress_out_queue
        self.command_queue = command_queue
        self.stop_batch_track_event = stop_batch_track
        self.shutdown_event = threading.Event()

        self.tracker = create_tracker(app_state, True, bone_stats)
        self.prev_frames = None
        self.prev_frame_idx = -1

    def run(self):
        print("SimpleKalmanWorker started (Top-Down Consensus Mode)")

        while not self.shutdown_event.is_set():

            # Process commands
            try:
                try:
                    command = self.command_queue.get_nowait()
                    if command.get("action") == "batch_track":
                        start_frame = command["start_frame"]
                        direction = command.get("direction", 1)
                        self._run_batch_tracking(start_frame, direction)
                        continue
                    elif command.get("action") == "shutdown":
                        self.shutdown_event.set()
                        break
                    elif command.get("action") == "update_calibration":
                        pass

                except queue.Empty:
                    pass

                # Process live frames
                try:
                    data = self.frames_in_queue.get(timeout=0.05)
                    if data.get("action") == "shutdown":
                        self.shutdown_event.set()
                        break
                    self._process_frame_for_tracking(data)
                except queue.Empty:
                    continue

            except Exception as e:
                print(f"ERROR in KalmanWorker: {e}")
                import traceback
                traceback.print_exc()

        print("KalmanWorker shut down")

    def _process_frame_for_tracking(self, data: dict):
        frame_idx = data["frame_idx"]

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        can_track = is_tracking_enabled and self.prev_frames and data.get("is_adjacent", False)
        if can_track and abs(frame_idx - self.prev_frame_idx) != 1:
            can_track = False

        if can_track:
            self.tracker.track_frame(
                frame_idx=frame_idx,
                prev_frame_idx=self.prev_frame_idx,
                source_frames=self.prev_frames,
                dest_frames=data["raw_frames"]
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int, direction: int = 1):
        dir_str = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"Starting batch track {dir_str} from frame {start_frame}...")

        points_to_track = None

        with self.app_state.lock:
            if self.app_state.focus_selected_point:
                idx = self.app_state.selected_point_idx
                if 0 <= idx < self.app_state.num_points:
                    points_to_track = [idx]
                    name = self.app_state.point_names[idx]
                    print(f" >> SELECTIVE TRACKING ACTIVE: Only tracking '{name}'")

        batch_reader = BatchVideoReader(self.video_backend)
        num_frames = self.app_state.video_metadata['num_frames']

        if direction == 1:
            frame_range = range(start_frame + 1, num_frames)
            total_to_process = num_frames - start_frame - 1
        else:
            frame_range = range(start_frame - 1, -1, -1)
            total_to_process = start_frame

        if total_to_process <= 0:
            return

        source_frames = batch_reader.read_frame(start_frame)
        if not source_frames:
            print("Could not read start frame")
            return

        current_source_idx = start_frame

        # Re-init tracker for warm start
        self.tracker = SimpleTracker(self.app_state)
        self.tracker.initialize_tracks(start_frame, direction=direction)

        for i, dest_frame_idx in enumerate(frame_range):
            if self.stop_batch_track_event.is_set():
                print("Batch tracking stopped by user.")
                break

            dest_frames = batch_reader.read_frame(dest_frame_idx)
            if not dest_frames:
                break

            self.tracker.track_frame(
                frame_idx=dest_frame_idx,
                prev_frame_idx=current_source_idx,
                source_frames=source_frames,
                dest_frames=dest_frames,
                active_point_indices=points_to_track
            )

            source_frames = dest_frames
            current_source_idx = dest_frame_idx

            if i % 1 == 0:
                self.progress_out_queue.put({
                    "status": "running",
                    "progress": i / total_to_process,
                    "current_frame": dest_frame_idx,
                    "total_frames": num_frames
                })

        final_frame = frame_range[-1] if len(frame_range) > 0 else start_frame
        self.progress_out_queue.put({"status": "complete", "final_frame": final_frame})
        batch_reader.clear_cache()
        print(f"Batch track {dir_str} complete")