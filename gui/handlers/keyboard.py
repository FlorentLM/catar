from dearpygui import dearpygui as dpg

from gui.callbacks.annotation_callbacks import (
    set_human_annotated_callback,
    clear_future_annotations_callback
)
from gui.callbacks.calibration_callbacks import addremove_calib_frame_callback
from gui.callbacks.general_callbacks import (
    save_state_callback,
    load_state_callback,
    toggle_focus_mode_callback
)
from gui.callbacks.playback_callbacks import (
    play_pause_callback,
    next_frame_callback,
    prev_frame_callback
)

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
                new_idx = (app_state.selected_point_idx - 1) % app_state.num_points
                app_state.selected_point_idx = new_idx
        case dpg.mvKey_Down:
            with app_state.lock:
                new_idx = (app_state.selected_point_idx + 1) % app_state.num_points
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
