import queue
import threading
from typing import TYPE_CHECKING

from utils import triangulate_and_score
from core import process_frame
from video import BatchVideoReader

if TYPE_CHECKING:
    from state import AppState
    from video import VideoBackend
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


class TrackingWorker(threading.Thread):
    """
    Runs the automated point tracking, and provides on-demand 3D reconstruction.
    """

    def __init__(
            self,
            app_state: 'AppState',
            video_backend: 'VideoBackend',
            reconstructor: 'Reconstructor',
            tracker: 'MultiObjectTracker',
            frames_in_queue: queue.Queue,
            progress_out_queue: queue.Queue,
            command_queue: queue.Queue,
            stop_batch_track: threading.Event
    ):
        """
        Initialize TrackingWorker.

        Args:
            app_state: Shared application state
            video_backend: VideoBackend for batch frame access
            reconstructor: Mokap reconstructor instance
            tracker: Mokap multi-object tracker instance
            frames_in_queue: Queue receiving frames from VideoReaderWorker
            progress_out_queue: Queue for reporting batch tracking progress
            command_queue: Queue for receiving commands
            stop_batch_track: Event to signal batch tracking cancellation
        """
        super().__init__(daemon=True, name="TrackingWorker")

        self.app_state = app_state
        self.video_backend = video_backend
        self.reconstructor = reconstructor
        self.tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.progress_out_queue = progress_out_queue
        self.command_queue = command_queue
        self.stop_batch_track_event = stop_batch_track
        self.shutdown_event = threading.Event()

        # State for live tracking
        self.prev_frames = None
        self.prev_frame_idx = -1

    def run(self):
        """Main worker loop."""

        print("TrackingWorker started")

        while not self.shutdown_event.is_set():
            # Check for special commands (batch tracking)
            try:
                command = self.command_queue.get_nowait()

                if command.get("action") == "batch_track":
                    start_frame = command["start_frame"]
                    direction = command.get("direction", 1)
                    self._run_batch_tracking(start_frame, direction)
                    continue

                elif command.get("action") == "update_calibration":
                    print("TrackingWorker: Received calibration update command")
                    self.reconstructor.update_camera_parameters(command["calibration"])
                    continue

                elif command.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

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

        print("TrackingWorker shut down")

    def _reconstruct_for_display(self, frame_idx: int):
        """
        Perform on-demand 3D reconstruction for a single frame.

        Args:
            frame_idx: Frame to reconstruct
        """
        with self.app_state.lock:
            calibration = self.app_state.calibration

        annotations = self.app_state.data.get_frame_annotations(frame_idx)

        if calibration.best_calibration is not None:
            points_4d = triangulate_and_score(annotations, calibration)
            self.app_state.data.set_frame_points3d(frame_idx, points_4d)

            with self.app_state.lock:
                self.app_state.needs_3d_reconstruction = False

    def _process_frame_for_tracking(self, data: dict):
        """
        Process a single frame for the live tracking.

        Args:
            data: Frame data from VideoReaderWorker
        """
        frame_idx = data["frame_idx"]

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        # We can track if enabled, and if we have adjacent frames
        can_track = is_tracking_enabled and self.prev_frames and data.get("is_adjacent", False)

        # Double check adjacency manually to be safe
        if can_track and abs(frame_idx - self.prev_frame_idx) != 1:
            can_track = False

        if can_track:
            print(f"[Frame {frame_idx}] TrackingWorker: Live tracking from {self.prev_frame_idx}")
            process_frame(
                frame_idx=frame_idx,
                source_frame_idx=self.prev_frame_idx,
                app_state=self.app_state,
                reconstructor=self.reconstructor,
                tracker=self.tracker,
                source_frames=self.prev_frames,
                dest_frames=data["raw_frames"],
                batch_step=1  # always step 1 for realtime tracking
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int, direction: int = 1):
        """
        Track points starting from start_frame in the given direction.

        Args:
            start_frame: Starting frame index
            direction: 1 for forward, -1 for backward
        """

        dir_str = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"Starting batch track {dir_str} from frame {start_frame}...")

        batch_reader = BatchVideoReader(self.video_backend)

        num_frames = self.app_state.video_metadata['num_frames']

        # Determine the range of frames to process
        if direction == 1:
            frame_range = range(start_frame + 1, num_frames)
            total_to_process = num_frames - start_frame - 1
        else:
            frame_range = range(start_frame - 1, -1, -1)
            total_to_process = start_frame

        if total_to_process <= 0:
            print("Batch track: No frames to process in this direction")
            return

        # Read the initial source frames
        source_frames = batch_reader.read_frame(start_frame)

        if not source_frames:
            print("Could not read start frame")
            return

        current_source_idx = start_frame

        # Main tracking loop
        for i, dest_frame_idx in enumerate(frame_range):
            if self.stop_batch_track_event.is_set():
                print(f"Batch track stopped by user at frame {dest_frame_idx}")
                break

            dest_frames = batch_reader.read_frame(dest_frame_idx)

            if not dest_frames:
                print(f"Failed to read frames at index {dest_frame_idx}")
                break

            # Run the core tracking logic: source -> dest
            process_frame(
                frame_idx=dest_frame_idx,
                source_frame_idx=current_source_idx,
                app_state=self.app_state,
                reconstructor=self.reconstructor,
                tracker=self.tracker,
                source_frames=source_frames,
                dest_frames=dest_frames,
                batch_step=i + 1
            )

            # Destination becomes source for next iteration
            source_frames = dest_frames
            current_source_idx = dest_frame_idx

            # Report progress
            # if i % 5 == 0:
            if i % 1 == 0:
                self.progress_out_queue.put({
                    "status": "running",
                    "progress": i / total_to_process,
                    "current_frame": dest_frame_idx,
                    "total_frames": num_frames
                })

        # Signal completion
        final_frame = frame_range[-1] if len(frame_range) > 0 else start_frame
        self.progress_out_queue.put({"status": "complete", "final_frame": final_frame})

        # Clean up
        batch_reader.clear_cache()
        print(f"Batch track {dir_str} complete")