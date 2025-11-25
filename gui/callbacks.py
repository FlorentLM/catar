from typing import TYPE_CHECKING
import dearpygui.dearpygui as dpg

import config
from gui.annotations_callbacks import set_human_annotated_callback, clear_future_annotations_callback
from gui.calibration_callbacks import addremove_calib_frame_callback
from gui.mouse_handlers import image_mousedrag_callback, leftclick_release_callback

if TYPE_CHECKING:
    from state import AppState, Queues


def register_event_handlers(app_state: 'AppState', queues: 'Queues'):
    """Register event handlers (global)."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press, user_data=user_data)

        dpg.add_mouse_drag_handler(
            button=dpg.mvMouseButton_Left,
            callback=image_mousedrag_callback,
            user_data=user_data
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Left,
            callback=leftclick_release_callback,
            user_data=user_data
        )
        dpg.add_key_down_handler(
            key=dpg.mvKey_LAlt,
            callback=on_alt_down,
            user_data=user_data
        )
        dpg.add_key_release_handler(
            key=dpg.mvKey_LAlt,
            callback=on_alt_up,
            user_data=user_data
        )


# ============================================================================
# Playback
# ============================================================================

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


# ============================================================================
# UI State
# ============================================================================

def toggle_focus_mode_callback(sender, app_data, user_data):
    """Toggle focus mode."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.focus_selected_point = not app_state.focus_selected_point
        status = "Enabled" if app_state.focus_selected_point else "Disabled"
        print(f"Focus mode: {status}")


def toggle_epipolar_lines_callback(sender, app_data, user_data):
    """Toggle visibility of epipolar lines."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_epipolar_lines = dpg.get_value(sender)


def toggle_point_labels_callback(sender, app_data, user_data):
    """Toggle visibility of all keypoint labels."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_all_labels = dpg.get_value(sender)


def toggle_reprojection_error_callback(sender, app_data, user_data):
    """Toggle visibility of the reprojection error lines."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_reprojection_error = dpg.get_value(sender)


def toggle_histogram_callback(sender, app_data, user_data):
    """Toggle histogram visibility."""

    show = dpg.get_value("show_histogram_checkbox")
    dpg.configure_item("annotation_plot", show=show)
    dpg.configure_item("histogram_separator", show=show)
    # Adjust main content window height
    if show:
        dpg.configure_item("main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_FULL)
    else:
        dpg.configure_item("main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_COLLAPSED)


# ============================================================================
# File ops
# ============================================================================

def app_quit_callback(sender, app_data, user_data):
    dpg.stop_dearpygui()


def save_state_callback(sender, app_data, user_data):
    user_data["app_state"].save_to_disk(config.DATA_FOLDER)


def load_state_callback(sender, app_data, user_data):
    user_data["app_state"].load_from_disk(config.DATA_FOLDER)


# ============================================================================
# Keyboard
# ============================================================================

def on_alt_down(sender, app_data, user_data):
    """Temporarily hide overlays when Alt is held down."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.temp_hide_overlays = True


def on_alt_up(sender, app_data, user_data):
    """Show overlays again when Alt is released."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.temp_hide_overlays = False


def on_key_press(sender, app_data, user_data):
    """Handle keyboard shortcuts."""

    app_state = user_data["app_state"]

    match app_data:
        case dpg.mvKey_Spacebar:
            play_pause_callback(sender, app_data, user_data)
        case dpg.mvKey_Right:
            next_frame_callback(sender, app_data, user_data)
        case dpg.mvKey_Left:
            prev_frame_callback(sender, app_data, user_data)
        case dpg.mvKey_Up:
            with app_state.lock:
                new_idx = (app_state.selected_point_idx - 1) % app_state.n_points
                app_state.selected_point_idx = new_idx
        case dpg.mvKey_Down:
            with app_state.lock:
                new_idx = (app_state.selected_point_idx + 1) % app_state.n_points
                app_state.selected_point_idx = new_idx
        case dpg.mvKey_S:
            save_state_callback(sender, app_data, user_data)
        case dpg.mvKey_L:
            load_state_callback(sender, app_data, user_data)
        case dpg.mvKey_Z:
            toggle_focus_mode_callback(sender, app_data, user_data)
        case dpg.mvKey_H:
            set_human_annotated_callback(sender, app_data, user_data)
        case dpg.mvKey_D:
            clear_future_annotations_callback(sender, app_data, user_data)

        case dpg.mvKey_C:
            addremove_calib_frame_callback(sender, app_data, user_data)
        # case dpg.mvKey_Next:
        #     _navigate_calib_frame_callback(sender, None, {"app_state": user_data["app_state"], "direction": 1})
        # case dpg.mvKey_Prior:
        #     _navigate_calib_frame_callback(sender, None, {"app_state": user_data["app_state"], "direction": -1})
