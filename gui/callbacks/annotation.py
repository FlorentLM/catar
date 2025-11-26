import numpy as np


def set_human_annotated_callback(sender, app_data, user_data):
    """Mark all previous frames as human-annotated for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        with app_state.data.bulk_lock():
            app_state.data.human_annotated[:frame_idx + 1, :, p_idx] = True
        print(f"Marked previous frames as human-annotated for '{app_state.point_itn[p_idx]}'")


def clear_future_annotations_callback(sender, app_data, user_data):
    """Clear all future annotations for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx

        # Clear x, y, and confidence
        with app_state.data.bulk_lock():
            app_state.data.annotations[frame_idx + 1:, :, p_idx] = np.nan
            app_state.data.human_annotated[frame_idx + 1:, :, p_idx] = False

        print(f"Cleared future annotations for '{app_state.point_itn[p_idx]}'")


def set_selected_point_callback(sender, app_data, user_data):
    """Change selected keypoint."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.selected_point_idx = app_state.point_nti[app_data]
