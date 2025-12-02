import queue
import time

from dearpygui import dearpygui as dpg


def addremove_calib_frame_callback(sender, app_data, user_data):
    """Add or remove the current frame from the calibration set."""

    app_state = user_data["app_state"]

    with app_state.lock:
        frame_idx = app_state.frame_idx

        if frame_idx in app_state.calibration.calibration_frames:
            app_state.calibration.calibration_frames.remove(frame_idx)
            print(f"Frame {frame_idx} removed from calibration set.")

        else:
            app_state.calibration.calibration_frames.append(frame_idx)
            app_state.calibration.calibration_frames.sort()
            print(f"Frame {frame_idx} added to calibration set.")


def navigate_calib_frame_callback(sender, app_data, user_data):
    """Jump to the next or previous frame in the calibration set."""

    app_state = user_data["app_state"]
    direction = user_data["direction"]  # +1 for next or -1 for previous

    with app_state.lock:
        calib_frames = sorted(app_state.calibration.calibration_frames)
        if not calib_frames:
            print("No calibration frames to navigate.")
            return

        current_frame = app_state.frame_idx

        if direction == 1:  # Next
            next_frames = [f for f in calib_frames if f > current_frame]
            if next_frames:
                app_state.frame_idx = next_frames[0]
            else:
                app_state.frame_idx = calib_frames[0]  # wrap around

        elif direction == -1:  # Previous
            prev_frames = [f for f in calib_frames if f < current_frame]
            if prev_frames:
                app_state.frame_idx = prev_frames[-1]
            else:
                app_state.frame_idx = calib_frames[-1]  # wrap around


def start_ga_callback(sender, app_data, user_data):
    """Start genetic algorithm for calibration."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    with app_state.lock:
        app_state.calibration.best_fitness = float('inf')   # this needs to not be 0.0 on start
        ga_snapshot = app_state.get_ga_snapshot()

    queues.ga_command.put({
        "action": "start",
        "ga_state_snapshot": ga_snapshot
    })

    dpg.show_item("ga_popup")


def stop_ga_callback(sender, app_data, user_data):
    """Stop genetic algorithm."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    queues.ga_command.put({"action": "stop"})
    dpg.hide_item("ga_popup")


def ba_mode_change_callback(sender, app_data, user_data):
    """Updates the help text in the BA config popup when the mode changes."""

    help_text = ""
    if app_data == "Refine Cameras":
        help_text = (
            f"Point labels do not need to be consistent across frames, you can use anything as long as each is consistent across all the cameras.\n\n"
            "Use case: Scaffolding: fixing or improving a suboptimal initial calibration using points on a moving animal or static background features."
        )
    elif app_data == "Refine Cameras and 3D Points":
        help_text = (
            "Keypoints must be labeled consistently for the same body part across all selected frames.\n\n"
            "Use case: Simultaneously improves camera calibration and the 3D reconstruction (Full Bundle Adjustment)."
        )
    elif app_data == "Refine 3D Points only":
        help_text = (
            "Keypoints must be labeled consistently for the same body part across all selected frames.\n\n"
            "Use case: Refining a 3D trajectory (including interpolating gaps) when you are confident your camera calibration is really good."
        )

    dpg.set_value("ba_help_text", help_text)


def start_ba_callback(sender, app_data, user_data):
    """Reads the BA config, starts the BA worker, and manages popups."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    # Clear any stale messages from the results queue
    while not queues.ba_results.empty():
        try:
            queues.ba_results.get_nowait()
        except queue.Empty:
            break

    # Also ensure the stop event is cleared before a new run
    queues.stop_bundle_adjustment.clear()

    # Map UI selection to mode strings for the worker
    selected_mode_label = dpg.get_value("ba_mode_radio")
    mode_map = {
        "Refine Cameras": "refine_cameras_only",
        "Refine Cameras and 3D Points": "full_ba",
        "Refine 3D Points only": "refine_points_only"
    }
    mode = mode_map[selected_mode_label]

    with app_state.lock:
        ba_snapshot = app_state.get_ba_snapshot()

    # Add the selected mode to the snapshot dictionary
    ba_snapshot["mode"] = mode

    queues.ba_command.put({
        "action": "start",
        "ba_state_snapshot": ba_snapshot
    })

    dpg.set_value("ba_status_text", f"Running Bundle Adjustment ({selected_mode_label})...")
    dpg.hide_item("ba_config_popup")
    time.sleep(0.1)
    dpg.show_item("ba_progress_popup")


def stop_ba_callback(sender, app_data, user_data):
    """Signal the BA worker to stop and discard results."""

    queues = user_data["queues"]
    print("Sending stop signal to Bundle Adjustment worker.")
    queues.stop_bundle_adjustment.set()
    dpg.hide_item("ba_progress_popup")


def clear_calib_frames_callback(sender, app_data, user_data):
    """Clears the entire calibration set from the app state."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.calibration.calibration_frames.clear()
    print("Calibration frame set has been cleared.")
