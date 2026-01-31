
def set_manual_annotation_callback(sender, app_data, user_data):
    """Mark all previous frames as human-annotated for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_mode:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        keypoint = app_state.selected_keypoint
        app_state.data.is_manual(frame=slice(0, frame_idx), keypoint=keypoint, value=True)

        print(f"Marked previous frames as human-annotated for '{keypoint}'")


def clear_future_annotations_callback(sender, app_data, user_data):
    """Clear all future annotations for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_mode:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        curr_frame_idx = app_state.frame_idx
        keypoint = app_state.selected_keypoint

        app_state.data.set_2d(frame=slice(curr_frame_idx + 1, -1), keypoint=keypoint, data=None)

        print(f"Cleared future annotations for '{keypoint}'")


def set_selected_point_callback(sender, app_data, user_data):
    """Change selected keypoint."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.selected_keypoint = app_data
