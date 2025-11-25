import queue
import threading
from typing import TYPE_CHECKING, List
import cv2
from utils import triangulate_and_score
from core.tracking import process_frame

if TYPE_CHECKING:
    from state import AppState
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


class TrackingWorker(threading.Thread):
    """
    Handles automated point tracking and on-demand 3D reconstruction.
    """

    def __init__(
        self,
        app_state: 'AppState',
        reconstructor: 'Reconstructor',
        tracker: 'MultiObjectTracker',
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
        self.video_paths = app_state.video_paths
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

        with self.app_state.lock:
            calibration = self.app_state.calibration

        annotations = self.app_state.data.get_frame_annotations(frame_idx)

        if calibration.best_calibration is not None:
            points_4d = triangulate_and_score(annotations, calibration)

            self.app_state.data.set_frame_points3d(frame_idx, points_4d)

            with self.app_state.lock:
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
            # if i % 5 == 0:
            if i % 1 == 0:
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
