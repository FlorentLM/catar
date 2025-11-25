from dearpygui import dearpygui as dpg


def play_pause_callback(sender, app_data, user_data):
    """Toggle video play/pause."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = not app_state.paused


def next_frame_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        num_frames = app_state.video_metadata['num_frames']
        if app_state.frame_idx < num_frames - 1:
            app_state.frame_idx += 1


def prev_frame_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        if app_state.frame_idx > 0:
            app_state.frame_idx -= 1


def set_frame_callback(sender, app_data, user_data):
    """Set frame from slider (or hist drag line)."""

    app_state = user_data["app_state"]
    new_frame_idx = dpg.get_value(sender)

    if new_frame_idx is None:
        return

    with app_state.lock:
        app_state.is_seeking = True
        app_state.paused = True
        num_frames = app_state.video_metadata['num_frames']
        if 0 <= new_frame_idx < num_frames:
            app_state.frame_idx = int(new_frame_idx)
