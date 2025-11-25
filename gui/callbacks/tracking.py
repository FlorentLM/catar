from dearpygui import dearpygui as dpg


def toggle_realtime_tracking_callback(sender, app_data, user_data):
    """Toggle realtime keypoint tracking."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.keypoint_tracking_enabled = not app_state.keypoint_tracking_enabled
        is_enabled = app_state.keypoint_tracking_enabled

    if is_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)


def batch_tracking_fwd_callback(sender, app_data, user_data):
    """Start forward batch tracking from current frame."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    with app_state.lock:
        start_frame = app_state.frame_idx
        queues.stop_batch_track.clear()
        queues.tracking_command.put({
            "action": "batch_track",
            "start_frame": start_frame,
            "direction": 1
        })

    dpg.set_value("batch_track_progress", 0.0)
    dpg.show_item("batch_track_popup")


def batch_tracking_bwd_callback(sender, app_data, user_data):
    """Start forward batch tracking from current frame."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    with app_state.lock:
        start_frame = app_state.frame_idx
        queues.stop_batch_track.clear()
        queues.tracking_command.put({
            "action": "batch_track",
            "start_frame": start_frame,
            "direction": -1
        })

    dpg.set_value("batch_track_progress", 0.0)
    dpg.show_item("batch_track_popup")


def stop_batch_tracking_callback(sender, app_data, user_data):
    """Stop batch tracking."""

    queues = user_data["queues"]
    print("Stop command issued to batch tracker.")
    queues.stop_batch_track.set()
    dpg.hide_item("batch_track_popup")