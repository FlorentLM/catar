import queue
import threading
from typing import TYPE_CHECKING, Optional

from video import BatchVideoReader
from utils import triangulate_and_score  # Used for on-demand recon
from core.trackers.trackers_commons import create_tracker

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
            stop_batch_track: threading.Event,
            bone_stats: Optional[dict] = None
    ):
        super().__init__(daemon=True, name="TrackingWorker")

        self.app_state = app_state
        self.video_backend = video_backend
        self.reconstructor = reconstructor
        self.mot_tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.progress_out_queue = progress_out_queue
        self.command_queue = command_queue
        self.stop_batch_track_event = stop_batch_track
        self.bone_stats = bone_stats

        self.shutdown_event = threading.Event()

        self.tracker_algorithm = "simple_kalman"
        self.active_tracker = self._build_tracker()

        # State for live tracking
        self.prev_frames = None
        self.prev_frame_idx = -1

    def _build_tracker(self):
        return create_tracker(
            self.tracker_algorithm,
            self.app_state,
            bone_stats=self.bone_stats,
            # These are for the mokap tracker, others will ignore
            reconstructor=self.reconstructor,
            mot_tracker=self.mot_tracker
        )

    def run(self):
        """Main worker loop."""

        print(f"TrackingWorker started (Algo: {self.tracker_algorithm})")

        while not self.shutdown_event.is_set():

            try:
                cmd = self.command_queue.get_nowait()

                if cmd.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                elif cmd.get("action") == "update_calibration":
                    print("TrackingWorker: Updating calibration...")

                    with self.app_state.lock:
                        # Update the reconstructor used by mokap tracker
                        self.reconstructor.rig = self.app_state.rig
                    # trackersSimple/Improved read rig from app_state directly
                    continue

                elif cmd.get("action") == "set_algorithm":
                    new_algo = cmd["algorithm"]

                    if new_algo != self.tracker_algorithm:
                        print(f"Swapping tracker: {self.tracker_algorithm} -> {new_algo}")

                        self.tracker_algorithm = new_algo
                        self.active_tracker = self._build_tracker()
                        self.prev_frames = None  # Reset live tracking state

                elif cmd.get("action") == "batch_track":
                    self._run_batch_tracking(
                        cmd["start_frame"],
                        cmd.get("direction", 1)
                    )
                    continue

            except queue.Empty:
                pass

            # On-demand 3D reconstruction
            with self.app_state.lock:
                needs_reconstruction = self.app_state.needs_3d_reconstruction
                current_frame_idx = self.app_state.frame_idx

            if needs_reconstruction:
                self._reconstruct_for_display(current_frame_idx)

            # Live tracking
            try:
                data = self.frames_in_queue.get(timeout=0.02)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                self._process_live_frame(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in TrackingWorker: {e}")
                import traceback
                traceback.print_exc()

    def _reconstruct_for_display(self, frame_idx: int):
        """Triangulates existing 2D points to 3D for the GUI 3D view."""
        with self.app_state.lock:
            rig = self.app_state.rig

        annotations = self.app_state.data.get_frame_annotations(frame_idx)

        points_4d = triangulate_and_score(annotations, rig)
        self.app_state.data.set_frame_points3d(frame_idx, points_4d)

        with self.app_state.lock:
            self.app_state.needs_3d_reconstruction = False

    def _process_live_frame(self, data: dict):
        """
        Process a single frame for the live tracking.
        """
        frame_idx = data["frame_idx"]

        with self.app_state.lock:
            is_tracking_enabled = self.app_state.keypoint_tracking_enabled

        # We can track if enabled, and if we have adjacent frames
        can_track = is_tracking_enabled and self.prev_frames and data.get("is_adjacent", False)

        if can_track and abs(frame_idx - self.prev_frame_idx) != 1:
            can_track = False

        if can_track:
            self.active_tracker.track_frame(
                frame_idx=frame_idx,
                prev_frame_idx=self.prev_frame_idx,
                source_frames=self.prev_frames,
                dest_frames=data["raw_frames"]
            )

        self.prev_frames = data["raw_frames"]
        self.prev_frame_idx = data["frame_idx"]

    def _run_batch_tracking(self, start_frame: int, direction: int = 1):
        dir_str = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"Starting batch track {dir_str} ({self.tracker_algorithm}) from {start_frame}...")

        self.active_tracker.initialize_tracks(start_frame, direction)

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

        source_frames = batch_reader.read_frame(start_frame)
        if not source_frames:
            print("Could not read start frame")
            return

        current_source_idx = start_frame

        # Main loop
        for i, dest_frame_idx in enumerate(frame_range):
            if self.stop_batch_track_event.is_set():
                print("Batch tracking stopped by user.")
                break

            dest_frames = batch_reader.read_frame(dest_frame_idx)
            if not dest_frames:
                break

            success = self.active_tracker.track_frame(
                frame_idx=dest_frame_idx,
                prev_frame_idx=current_source_idx,
                source_frames=source_frames,
                dest_frames=dest_frames
            )

            # Handle collision / failure (mokap tracker returns False on collision)
            if not success:
                print(f"Tracking interrupted at frame {dest_frame_idx} (Collision or Loss)")
                # Set final frame to previous one so UI knows where it stopped
                final_frame = dest_frame_idx - direction
                self.progress_out_queue.put({"status": "complete", "final_frame": final_frame})
                batch_reader.clear_cache()
                return

            source_frames = dest_frames
            current_source_idx = dest_frame_idx

            # Report progress to UI
            # if i % 5 == 0:
            if i % 1 == 0:
                self.progress_out_queue.put({
                    "status": "running",
                    "progress": i / total_to_process,
                    "current_frame": dest_frame_idx,
                    "total_frames": num_frames,
                    "debug_info": self.active_tracker.get_debug_info()
                })

        # Signal completion
        final_frame = frame_range[-1] if len(frame_range) > 0 else start_frame
        self.progress_out_queue.put({"status": "complete", "final_frame": final_frame})

        # Clean up
        batch_reader.clear_cache()
        print(f"Batch track {dir_str} complete")