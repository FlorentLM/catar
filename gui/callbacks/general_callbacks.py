import dearpygui.dearpygui as dpg
import config


# UI State
# --------

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


# General ops
# -----------

def app_quit_callback(sender, app_data, user_data):
    dpg.stop_dearpygui()


def save_state_callback(sender, app_data, user_data):
    user_data["app_state"].save_to_disk(config.DATA_FOLDER)


def load_state_callback(sender, app_data, user_data):
    user_data["app_state"].load_from_disk(config.DATA_FOLDER)

