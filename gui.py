"""
GUI implementation using DearPyGUI.
"""
import queue
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import config
from state import AppState, Queues
from viz_3d import Open3DVisualizer
from core import snap_annotation
from utils import reproject_points, line_box_intersection
from cache_dialog import CacheManagerDialog

# TODO: Would probably be nice to move all the dialogs in a single separate dialogs file


# ============================================================================
# Main UI
# ============================================================================

def create_ui(app_state: AppState, queues: Queues, open3d_viz: Open3DVisualizer):
    """Create the main DearPyGUI window."""

    dpg.create_context()

    nb_videos = app_state.video_metadata['num_videos']
    if nb_videos > 0:
        # Calculate the most 'square' layout
        n_cols = int(np.ceil(np.sqrt(nb_videos)))
        n_rows = int(np.ceil(nb_videos / n_cols))
    else:
        n_cols = 1
        n_rows = 1

    video_ar = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_width = 480
    item_height = item_width / video_ar

    window_width = int(
        config.CONTROL_PANEL_WIDTH +
        (item_width * n_cols) +
        (config.PADDING * (n_cols + 2))
    )
    window_height = int(
        (item_height * n_rows) +
        config.BOTTOM_PANEL_HEIGHT_FULL +
        100
    )

    # Setup
    _create_textures(app_state.video_metadata)
    _create_themes()
    _register_handlers(app_state, queues)

    # Main window layout
    with dpg.window(label="Main Window", tag="main_window", no_scrollbar=True):
        _create_menu_bar(app_state, queues)

        with dpg.child_window(tag="main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_FULL, no_scrollbar=True):

            with dpg.group(horizontal=True):
                with dpg.child_window(width=config.CONTROL_PANEL_WIDTH, tag="control_panel_window"):
                    _create_control_panel(app_state, queues, open3d_viz)

                with dpg.child_window(width=-1, tag="video_grid_window"):
                    _create_video_grid(app_state, n_cols, n_rows)

        with dpg.child_window(tag="bottom_panel_window", height=config.BOTTOM_PANEL_HEIGHT_FULL, no_scrollbar=True):
            _create_bottom_panel(app_state)

    # Popups
    _create_ga_popup(app_state, queues)
    _create_ba_config_popup(app_state, queues)
    _create_ba_progress_popup(app_state, queues)
    _create_batch_track_popup(app_state, queues)
    _create_loupe_popup()

    # Viewport
    dpg.create_viewport(title="CATAR", width=window_width, height=window_height)
    dpg.set_viewport_resize_callback(resize_video_widgets, user_data={"app_state": app_state})

    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()


# ============================================================================
# UI elements
# ============================================================================

def _create_textures(video_meta: dict):
    """Create GPU textures for video frames and 3D view."""

    with dpg.texture_registry():
        # Video textures
        for i in range(video_meta['num_videos']):
            black = np.zeros(
                (config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 4),
                dtype=np.float32
            )
            dpg.add_raw_texture(
                width=config.DISPLAY_WIDTH,
                height=config.DISPLAY_HEIGHT,
                default_value=black.ravel().tolist(),
                tag=f"video_texture_{i}",
                format=dpg.mvFormat_Float_rgba
            )

        # Add loupe texture
        loupe_size = 128

        black_loupe = np.zeros((loupe_size, loupe_size, 4), dtype=np.float32)
        dpg.add_raw_texture(
            width=loupe_size,
            height=loupe_size,
            default_value=black_loupe.ravel().tolist(),
            tag="loupe_texture",
            format=dpg.mvFormat_Float_rgba
        )


def _create_themes():
    """Create custom UI themes."""

    # Recording button theme (red)
    with dpg.theme(tag="record_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 0, 0, 255))

    # Tracking enabled theme (green)
    with dpg.theme(tag="tracking_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 200, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 0, 255))

    # Frame slider theme (purple)
    with dpg.theme(tag="purple_slider_theme"):
        with dpg.theme_component(dpg.mvSliderInt):
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (200, 135, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (215, 85, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (230, 165, 255, 75))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (230, 165, 255, 75))

    # Sets the loupe window's internal padding to 0 on X and Y
    with dpg.theme(tag="loupe_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)

    # Sets text to a faint gray color
    with dpg.theme(tag="faint_text_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180, 255))

def _create_menu_bar(app_state: AppState, queues: Queues):
    """Create top menu bar."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Save State (S)",
                callback=_save_state_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Load State (L)",
                callback=_load_state_callback,
                user_data=user_data
            )
            dpg.add_separator()
            dpg.add_menu_item(label="Quit", callback=_quit_callback)

        with dpg.menu(label="Display"):
            dpg.add_menu_item(
                label="Show Timeline Histogram",
                tag="show_histogram_checkbox",
                check=True,
                default_value=True,
                callback=_toggle_histogram_visibility_callback
            )
            dpg.add_menu_item(
                label="Show All Labels",
                tag="show_all_labels_checkbox",
                check=True,
                default_value=False,
                callback=_toggle_show_all_labels_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Show Reproj. Errors",
                tag="show_reprojection_error_checkbox",
                check=True,
                default_value=True,
                callback=_toggle_show_reprojection_error_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Show Epipolar Lines",
                tag="show_epipolar_lines_checkbox",
                check=True,
                default_value=True,
                callback=_toggle_show_epipolar_lines_callback,
                user_data=user_data
            )

        with dpg.menu(label="Tools"):
            dpg.add_menu_item(
                label="Manage video cache...",
                callback=_show_cache_manager_callback,
                user_data=user_data
            )


def _create_control_panel(app_state: AppState, queues: Queues, open3d_viz: Open3DVisualizer):
    """Create left control panel."""

    user_data = {"app_state": app_state, "queues": queues}

    dpg.add_text("Info")
    dpg.add_text("Focus Mode: Disabled", tag="focus_text")
    dpg.add_text("Best Fitness: inf", tag="fitness_text")
    dpg.add_separator()

    dpg.add_text("3D View")
    # TODO: this debug section probably needs to be removed now
    dpg.add_button(
        label="Refresh 3D View",
        callback=lambda: open3d_viz.reset_view()
    )
    dpg.add_separator()

    with dpg.collapsing_header(label="Annotate", default_open=True):
        dpg.add_combo(
            label="Keypoint",
            items=app_state.point_names,
            default_value=app_state.point_names[0],
            callback=_set_selected_point_callback,
            user_data=user_data,
            tag="point_combo"
        )
        dpg.add_button(
            label="Toggle live tracking",
            callback=_toggle_tracking_callback,
            user_data=user_data,
            tag="keypoint_tracking_button"
        )
        dpg.add_button(
            label="Track Forward",
            callback=_start_batch_track_callback,
            user_data=user_data,
            tag="batch_track_button"
        )

        dpg.add_button(
            label="Set as Human annotated (H)",
            callback=_set_human_annotated_callback,
            user_data=user_data
        )
        dpg.add_button(
            label="Delete future annots (D)",
            callback=_clear_future_annotations_callback,
            user_data=user_data
        )

    dpg.add_separator()

    with dpg.collapsing_header(label="Calibration", default_open=True):

        dpg.add_text("Calibration Frames: 0", tag="num_calib_frames_text")

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="< Prev",
                callback=_navigate_calib_frame_callback,
                user_data={"app_state": app_state, "direction": -1},
                width=60
            )

            dpg.add_button(
                label="Add (C)",
                tag="toggle_calib_frame_button",
                callback=_toggle_calib_frame_callback,
                user_data=user_data,
                width=75
            )

            dpg.add_button(
                label="Next >",
                callback=_navigate_calib_frame_callback,
                user_data={"app_state": app_state, "direction": 1},
                width=60
            )

        dpg.add_button(
            label="Clear calib set",
            callback=_clear_calib_frames_callback,
            user_data=user_data,
            width=-1
        )

        dpg.add_separator()

        dpg.add_button(
            label="Create calibration (Genetic Algorithm)",
            callback=_start_ga_callback,
            user_data=user_data,
            width=-1
        )
        dpg.add_button(
            label="Refine (Bundle Adjustment)",
            callback=lambda: dpg.show_item("ba_config_popup"),
            user_data={"app_state": app_state, "queues": queues},
            width=-1
        )

        # dpg.add_button(
        #     label="/!\\ DEBUG: Clear Current Calibration /!\\",
        #     callback=_clear_calibration_callback,
        #     user_data=user_data,
        #     width=-1
        # )


def _create_bottom_panel(app_state: AppState):
    """Create bottom panel (player controls and histogram)."""

    user_data = {"app_state": app_state}

    # Player controls
    with dpg.group():
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="<| Prev",
                callback=_prev_frame_callback,
                user_data=user_data
            )
            dpg.add_button(
                label="Play",
                callback=_toggle_pause_callback,
                user_data=user_data,
                tag="play_pause_button"
            )
            dpg.add_button(
                label="Next |>",
                callback=_next_frame_callback,
                user_data=user_data
            )

            slider = dpg.add_slider_int(
                label="Frame",
                min_value=0,
                max_value=app_state.video_metadata['num_frames'] - 1,
                default_value=0,
                callback=_set_frame_callback,
                user_data=user_data,
                tag="frame_slider",
                width=-1
            )
            dpg.bind_item_theme(slider, "purple_slider_theme")

    dpg.add_separator(tag="histogram_separator")

    footer_height = 25

    # Histogram
    with dpg.plot(
        label="Annotation Histogram",
        height=-footer_height,
        width=-1,
        no_menus=True,
        no_box_select=True,
        no_mouse_pos=True,
        tag="annotation_plot"
    ):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="histogram_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Annotations", tag="histogram_y_axis")

        num_frames = app_state.video_metadata['num_frames']
        dpg.add_bar_series(
            list(range(num_frames)),
            [0] * num_frames,
            label="Annotation Count",
            parent="histogram_y_axis",
            tag="annotation_histogram_series"
        )

        dpg.add_drag_line(
            label="Current Frame",
            color=[215, 85, 255],
            vertical=True,
            default_value=0,
            tag="current_frame_line",
            callback=_set_frame_callback,
            user_data=user_data
        )

    # Footer
    dpg.add_separator()
    with dpg.group():
        help_text_content = "Shift (hold): slow cursor | Alt (hold): hide overlays"
        help_text_widget = dpg.add_text(help_text_content, tag="hotkey_help_text")
        dpg.bind_item_theme(help_text_widget, "faint_text_theme")

    with dpg.item_handler_registry(tag="histogram_handler"):
        dpg.add_item_clicked_handler(
            callback=_on_histogram_click,
            user_data=user_data
        )
    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")


def _create_video_grid(app_state: AppState, n_cols: int, n_rows: int):
    """Create grid of videos."""

    nb_videos = app_state.video_metadata['num_videos']

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag="video_table"):
        for _ in range(n_cols):
            dpg.add_table_column()

        for row in range(n_rows):
            with dpg.table_row():
                for col in range(n_cols):
                    idx = row * n_cols + col

                    if idx < nb_videos:
                        _create_video_cell(idx, app_state)


def _create_video_cell(cam_idx: int, app_state: AppState):
    """Create a single video view cell."""

    with dpg.table_cell():
        # Display Camera Name (bold) and Filename (faint)

        with dpg.group(horizontal=True, horizontal_spacing=5):
            camera_name = app_state.camera_names[cam_idx]
            file_name = app_state.video_names[cam_idx]

            dpg.add_text(camera_name)
            faint_text = dpg.add_text(f"({file_name})")
            dpg.bind_item_theme(faint_text, "faint_text_theme")

        with dpg.drawlist(
            width=config.DISPLAY_WIDTH,
            height=config.DISPLAY_HEIGHT,
            tag=f"drawlist_{cam_idx}"
        ):
            color = config.CAMERA_COLORS[cam_idx % len(config.CAMERA_COLORS)]
            frame_thickness = 2

            dpg.draw_image(
                f"video_texture_{cam_idx}",
                pmin=(0, 0),
                pmax=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                tag=f"video_image_{cam_idx}"
            )
            dpg.draw_rectangle(
                pmin=(0, 0),
                pmax=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                tag=f"video_border_{cam_idx}",
                color=color,
                thickness=frame_thickness
            )
            dpg.add_draw_layer(tag=f"annotation_layer_{cam_idx}")

        with dpg.item_handler_registry(tag=f"image_handler_{cam_idx}"):
            dpg.add_item_clicked_handler(
                callback=_image_mouse_down_callback,
                user_data={"cam_idx": cam_idx, "app_state": app_state}
            )
        dpg.bind_item_handler_registry(f"drawlist_{cam_idx}", f"image_handler_{cam_idx}")


def _create_ga_popup(app_state: AppState, queues: Queues):
    """Create genetic algorithm progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
        label="Calibration Progress",
        modal=True,
        show=False,
        tag="ga_popup",
        width=400,
        height=150,
        no_close=True
    ):
        dpg.add_text("Running Genetic Algorithm...", tag="ga_status_text")
        dpg.add_text("Generation: 0", tag="ga_generation_text")
        dpg.add_text("Best Fitness: inf", tag="ga_fitness_text")
        dpg.add_text("Mean Fitness: inf", tag="ga_mean_fitness_text")
        dpg.add_button(
            label="Stop Calibration",
            callback=_stop_ga_callback,
            user_data=user_data,
            width=-1
        )


def _create_ba_config_popup(app_state: AppState, queues: Queues):
    """Creates the popup for configuring the BA run before starting."""

    user_data = {"app_state": app_state, "queues": queues}

    with app_state.lock:
        nb_views = app_state.video_metadata['num_videos']

    with dpg.window(label="Bundle Adjustment Settings", modal=True, show=False, tag="ba_config_popup",
                    width=450, height=270, no_close=False):

        dpg.add_text("Workflow selection:")
        dpg.add_radio_button(
            tag="ba_mode_radio",
            items=[
                "Refine Cameras",
                "Refine Cameras and 3D Points",
                "Refine 3D Points only"
            ],
            default_value="Refine Cameras",
            callback=_ba_mode_changed_callback
        )

        dpg.add_separator()
        with dpg.group():
            dpg.add_text("", tag="ba_help_text", wrap=410)

        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Refinement", callback=_start_ba_callback, user_data=user_data, width=-1)
            dpg.add_button(label="Cancel", callback=lambda: dpg.hide_item("ba_config_popup"), width=80)

    _ba_mode_changed_callback(None, dpg.get_value("ba_mode_radio"), None)


def _create_ba_progress_popup(app_state: AppState, queues: Queues):
    """Create bundle adjustment progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
            label="Refining Calibration", modal=True, show=False, tag="ba_progress_popup",
            width=400, height=100, no_close=True, no_move=False
    ):
        dpg.add_text("Running Bundle Adjustment...", tag="ba_status_text")
        dpg.add_text("This may take a few minutes...")
        dpg.add_separator()
        dpg.add_button(
            label="Cancel Refinement",
            callback=_stop_ba_callback,
            user_data=user_data,
            width=-1
        )

def _create_batch_track_popup(app_state: AppState, queues: Queues):
    """Create batch tracking progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
        label="Tracking Progress",
        modal=True,
        show=False,
        tag="batch_track_popup",
        width=400,
        no_close=True
    ):
        dpg.add_text("Processing frames...", tag="batch_track_status_text")
        dpg.add_progress_bar(tag="batch_track_progress", width=-1)
        dpg.add_button(
            label="Stop",
            callback=_stop_batch_track_callback,
            user_data=user_data,
            width=-1
        )


def _create_loupe_popup():
    """Creates the floating, borderless window for the loupe."""
    loupe_size = 128

    with dpg.window(
            tag="loupe_window",
            show=False,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            width=loupe_size,
            height=loupe_size,
            no_scrollbar=True,
    ):
        with dpg.drawlist(
                width=loupe_size,
                height=loupe_size,
                tag="loupe_drawlist"
        ):
            # Layer for the zoomed video image
            dpg.draw_image(
                "loupe_texture",
                pmin=(0, 0),
                pmax=(loupe_size, loupe_size),
                tag="loupe_image"
            )
            # Layer for drawing annotations and lines on top
            dpg.add_draw_layer(tag="loupe_overlay_layer")

    dpg.bind_item_theme("loupe_window", "loupe_theme")


# ============================================================================
# Event Handlers
# ============================================================================

def _register_handlers(app_state: AppState, queues: Queues):
    """Register event handlers (global)."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key_press, user_data=user_data)

        # 2D annotations
        dpg.add_mouse_drag_handler(
            button=dpg.mvMouseButton_Left,
            callback=_image_drag_callback,
            user_data=user_data
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Left,
            callback=_leftclick_release_callback,
            user_data=user_data
        )
        dpg.add_key_down_handler(
            key=dpg.mvKey_LAlt,
            callback=_on_alt_down,
            user_data=user_data
        )
        dpg.add_key_release_handler(
            key=dpg.mvKey_LAlt,
            callback=_on_alt_up,
            user_data=user_data
        )


# ============================================================================
# Callbacks - Playback
# ============================================================================

def _toggle_pause_callback(sender, app_data, user_data):
    """Toggle play/pause."""
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = not app_state.paused


def _next_frame_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        num_frames = app_state.video_metadata['num_frames']
        if app_state.frame_idx < num_frames - 1:
            app_state.frame_idx += 1


def _prev_frame_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        if app_state.frame_idx > 0:
            app_state.frame_idx -= 1


def _set_frame_callback(sender, app_data, user_data):
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
# Callbacks - Point Selection and Tracking
# ============================================================================

def _set_selected_point_callback(sender, app_data, user_data):
    """Change selected keypoint."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.selected_point_idx = app_state.point_names.index(app_data)


def _toggle_tracking_callback(sender, app_data, user_data):
    """Toggle realtime keypoint tracking."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.keypoint_tracking_enabled = not app_state.keypoint_tracking_enabled
        is_enabled = app_state.keypoint_tracking_enabled

    if is_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)


def _start_batch_track_callback(sender, app_data, user_data):
    """Start forward batch tracking from current frame."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    with app_state.lock:
        start_frame = app_state.frame_idx
        queues.stop_batch_track.clear()
        queues.tracking_command.put({
            "action": "batch_track",
            "start_frame": start_frame
        })

    dpg.set_value("batch_track_progress", 0.0)
    dpg.show_item("batch_track_popup")


def _stop_batch_track_callback(sender, app_data, user_data):
    """Stop batch tracking."""

    queues = user_data["queues"]
    print("Stop command issued to batch tracker.")
    queues.stop_batch_track.set()
    dpg.hide_item("batch_track_popup")


# ============================================================================
# Callbacks - Annotations
# ============================================================================

def _set_human_annotated_callback(sender, app_data, user_data):
    """Mark all previous frames as human-annotated for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        app_state.human_annotated[:frame_idx + 1, :, p_idx] = True
        print(f"Marked previous frames as human-annotated for '{app_state.point_names[p_idx]}'")


def _clear_future_annotations_callback(sender, app_data, user_data):
    """Clear all future annotations for selected point."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx

        # Clear x, y, and confidence
        app_state.annotations[frame_idx + 1:, :, p_idx] = np.nan
        app_state.human_annotated[frame_idx + 1:, :, p_idx] = False

        print(f"Cleared future annotations for '{app_state.point_names[p_idx]}'")


# ============================================================================
# Callbacks - Mouse
# ============================================================================

def _image_mouse_down_callback(sender, app_data, user_data):
    """Handle mouse click on video view (for annotating)."""

    app_state = user_data["app_state"]
    cam_idx = user_data["cam_idx"]
    drawlist_tag = f"drawlist_{cam_idx}"

    with app_state.lock:
        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        annotations = app_state.annotations[frame_idx, cam_idx, :, :]
        video_info = app_state.videos.get(app_state.camera_names[cam_idx])
        video_w = video_info.width
        video_h = video_info.height

    # Get mouse pos (image coordinates)
    container_pos = dpg.get_item_rect_min(drawlist_tag)
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0:
        return

    mouse_pos = dpg.get_mouse_pos(local=False)
    local_pos = (
        mouse_pos[0] - container_pos[0],
        mouse_pos[1] - container_pos[1]
    )
    scaled_pos = np.array(
        (
            local_pos[0] * video_w / container_size[0],
            local_pos[1] * video_h / container_size[1]
        ),
        dtype=np.float32
    )

    # Check if near existing point
    existing_point = annotations[p_idx, :2] # only check x, y
    is_drag_start = False

    if not np.isnan(existing_point).any():
        existing_local = (
            existing_point[0] * container_size[0] / video_w,
            existing_point[1] * container_size[1] / video_h
        )
        dist = np.linalg.norm(np.array(local_pos) - np.array(existing_local))
        if dist < config.ANNOTATION_DRAG_THRESHOLD:
            is_drag_start = True

    with app_state.lock:
        if app_data[0] == 0:  # left click
            annotation_pos_for_drag = None
            if is_drag_start:
                # Use existing point's position for the drag operation
                annotation_pos_for_drag = app_state.annotations[frame_idx, cam_idx, p_idx, :2].copy()

            else:
                # Create a new point
                snapped_pos = snap_annotation(app_state, cam_idx, p_idx, frame_idx, scaled_pos)
                final_pos = snapped_pos if snapped_pos is not None else scaled_pos

                # Assign the new annotation (x, y, confidence=1.0)
                app_state.annotations[frame_idx, cam_idx, p_idx] = [*final_pos, 1.0]

                app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
                app_state.needs_3d_reconstruction = True
                annotation_pos_for_drag = final_pos

            # Offset between the mouse and the point (to prevent 'stickiness')
            drag_offset = scaled_pos - annotation_pos_for_drag

            app_state.drag_state = {
                "cam_idx": cam_idx,
                "p_idx": p_idx,
                "active": True,
                "drag_offset": drag_offset,
                "is_slowing_down": False,
                "slow_down_start_mouse_pos": None,
                "slow_down_start_annotation_pos": None,
            }

            # Show and immediately update the loupe on click
            dpg.show_item("loupe_window")
            _image_drag_callback(sender, app_data, user_data)

        elif app_data[0] == 1:  # right click = delete
            # Delete x, y, and confidence
            app_state.annotations[frame_idx, cam_idx, p_idx] = np.nan
            app_state.human_annotated[frame_idx, cam_idx, p_idx] = False
            app_state.needs_3d_reconstruction = True


def _image_drag_callback(sender, app_data, user_data):
    """Update annotation position while dragging and render the zoom loupe."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return

        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]
        frame_idx = app_state.frame_idx

        video_info = app_state.videos.get(app_state.camera_names[cam_idx])
        video_w = video_info.width
        video_h = video_info.height

        current_frames = app_state.current_video_frames
        all_annotations = app_state.annotations[frame_idx]
        best_calib = app_state.calibration.best_calibration

        show_epipolar_lines = app_state.show_epipolar_lines
        temp_hide_overlays = app_state.temp_hide_overlays

        f_mats = app_state.calibration.F_mats
        point_3d_selected = app_state.reconstructed_3d_points[frame_idx, p_idx]
        camera_colors = app_state.camera_colors
        cam_names = app_state.camera_names

    if current_frames is None:
        return

    video_frame = current_frames[cam_idx]

    # Calculate mouse pos
    drawlist_tag = f"drawlist_{cam_idx}"
    container_pos = dpg.get_item_rect_min(drawlist_tag)
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0:
        return

    mouse_pos = dpg.get_mouse_pos(local=False)
    local_pos = (mouse_pos[0] - container_pos[0], mouse_pos[1] - container_pos[1])
    current_scaled_pos = np.array(
        (local_pos[0] * video_w / container_size[0], local_pos[1] * video_h / container_size[1]),
        dtype=np.float32
    )

    # State for handling normal vs. slow-down (Shift key) movement
    slowdown_factor = 0.25
    shift_is_down = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    drag_state = app_state.drag_state
    is_slowing_down = drag_state["is_slowing_down"]

    if shift_is_down:
        if not is_slowing_down:
            # User just pressed Shift: anchor the current positions for slow movement
            drag_state["is_slowing_down"] = True
            drag_state["slow_down_start_mouse_pos"] = current_scaled_pos.copy()
            with app_state.lock:
                # get (x, y) coordinates
                drag_state["slow_down_start_annotation_pos"] = app_state.annotations[frame_idx, cam_idx, p_idx, :2].copy()

        # Continue slow movement relative to the anchor point
        mouse_delta = current_scaled_pos - drag_state["slow_down_start_mouse_pos"]
        final_scaled_pos = drag_state["slow_down_start_annotation_pos"] + (mouse_delta * slowdown_factor)

    else:  # Shift is not down
        if is_slowing_down:
            # User just released Shift: reset state
            drag_state["is_slowing_down"] = False

        # Normal drag: use the initial offset to make movement direct and non-sticky
        final_scaled_pos = current_scaled_pos - drag_state["drag_offset"]

    # Clamp position to be in video boundaries
    final_scaled_pos[0] = np.clip(final_scaled_pos[0], 0, video_w - 1)
    final_scaled_pos[1] = np.clip(final_scaled_pos[1], 0, video_h - 1)

    with app_state.lock:
        # Update the (x, y) and set confidence to 1.0 (it's a manual annotation)
        app_state.annotations[frame_idx, cam_idx, p_idx] = [*final_scaled_pos, 1.0]
        app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
        app_state.needs_3d_reconstruction = True
        app_state.drag_state = drag_state  # write updated state back

    # Update loupe background texture with subpixel accuracy
    loupe_size = 128
    zoom_factor = 4.0
    src_patch_width = loupe_size / zoom_factor

    src_x = final_scaled_pos[0] - src_patch_width / 2
    src_y = final_scaled_pos[1] - src_patch_width / 2

    center_coords = (final_scaled_pos[0], final_scaled_pos[1])
    patch_size = (int(src_patch_width), int(src_patch_width))
    patch = cv2.getRectSubPix(video_frame, patch_size, center_coords)

    if patch is not None and patch.size > 0:
        # Smoother interpolation for a less blocky look
        zoomed_patch = cv2.resize(patch, (loupe_size, loupe_size), interpolation=cv2.INTER_LINEAR)
        zoomed_patch_rgba = cv2.cvtColor(zoomed_patch, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value("loupe_texture", zoomed_patch_rgba.ravel())

    # Add the overlays to the loupe
    dpg.delete_item("loupe_overlay_layer", children_only=True)
    layer_tag = "loupe_overlay_layer"

    def to_loupe_coords(p):
        return ((p[0] - src_x) * zoom_factor, (p[1] - src_y) * zoom_factor)

    # Draw epipolar lines for selected point
    if show_epipolar_lines and not temp_hide_overlays and best_calib and f_mats:
        for from_cam in range(len(current_frames)):

            point_2d = all_annotations[from_cam, p_idx, :2]
            F = f_mats.get((from_cam, cam_idx))

            if cam_idx == from_cam or np.isnan(point_2d).any() or F is None:
                continue

            p_hom = np.array([point_2d[0], point_2d[1], 1.0])
            line = F @ p_hom
            a, b, c = line
            intersection_points = line_box_intersection(a, b, c, src_x, src_y, src_patch_width, src_patch_width)

            if len(intersection_points) == 2:
                p1_video_coords, p2_video_coords = intersection_points
                p1_loupe, p2_loupe = to_loupe_coords(p1_video_coords), to_loupe_coords(p2_video_coords)
                color = (*camera_colors[from_cam % len(camera_colors)], 120)    # slightly transparent
                dpg.draw_line(p1_loupe, p2_loupe, color=color, thickness=1, parent=layer_tag)

    # Draw reprojection for selected point
    if not temp_hide_overlays and best_calib and not np.isnan(point_3d_selected).any():
        cam_name_target = cam_names[cam_idx]
        reprojected = reproject_points(point_3d_selected, best_calib[cam_name_target])

        if reprojected.size > 0:
            reproj_video_coords = reprojected[0]
            reproj_loupe_coords = to_loupe_coords(reproj_video_coords)

            # only draw if the reprojection is visible within the loupe
            if 0 < reproj_loupe_coords[0] < loupe_size and 0 < reproj_loupe_coords[1] < loupe_size:

                # Draw red X
                color_reproj = (255, 0, 0)
                dpg.draw_line((reproj_loupe_coords[0] - 5, reproj_loupe_coords[1] - 5),
                              (reproj_loupe_coords[0] + 5, reproj_loupe_coords[1] + 5),
                              color=color_reproj, parent=layer_tag, thickness=1)

                dpg.draw_line((reproj_loupe_coords[0] - 5, reproj_loupe_coords[1] + 5),
                              (reproj_loupe_coords[0] + 5, reproj_loupe_coords[1] - 5),
                              color=color_reproj, parent=layer_tag, thickness=1)

                # Dotted error line from the center of the loupe
                color_line = (255, 100, 100)
                p1 = np.array((loupe_size / 2, loupe_size / 2))  # annotation is always at the center
                p2 = np.array(reproj_loupe_coords)
                distance = np.linalg.norm(p1 - p2)

                if distance > 0.1:  # only draw if there's a visible error
                    vec = p2 - p1
                    vec_norm = vec / (distance + 1e-6)
                    dash_length = 4
                    gap_length = 3

                    current_pos = 0
                    while current_pos < distance:
                        start_point = p1 + vec_norm * current_pos
                        end_point = p1 + vec_norm * min(current_pos + dash_length, distance)
                        dpg.draw_line(tuple(start_point), tuple(end_point), color=color_line, thickness=1,
                                      parent=layer_tag)
                        current_pos += dash_length + gap_length

                    # Distance text
                    distance_px = np.linalg.norm(final_scaled_pos - reproj_video_coords)
                    mid_point = p1 + vec * 0.5
                    label_pos = (mid_point[0] + 5, mid_point[1])
                    dpg.draw_text(
                        pos=label_pos,
                        text=f"{distance_px:.1f}px",
                        color=color_line,
                        size=12,
                        parent=layer_tag
                    )

    # Draw a small crosshair
    center = loupe_size / 2
    size = 15  # length of the arms from the center
    gap = 5  # half size of the center gap

    # Horizontal lines
    dpg.draw_line((center - size, center), (center - gap, center), color=(255, 255, 255), thickness=1,
                  parent=layer_tag)
    dpg.draw_line((center + gap, center), (center + size, center), color=(255, 255, 255), thickness=1,
                  parent=layer_tag)
    # Vertical lines
    dpg.draw_line((center, center - size), (center, center - gap), color=(255, 255, 255), thickness=1,
                  parent=layer_tag)
    dpg.draw_line((center, center + size), (center, center + gap), color=(255, 255, 255), thickness=1,
                  parent=layer_tag)

    # Add subpixel coordinates text
    coord_text = f"X: {final_scaled_pos[0]:.2f}\nY: {final_scaled_pos[1]:.2f}"
    dpg.draw_text(pos=(5, 5), text=coord_text, color=(255, 255, 255), size=12, parent=layer_tag)

    dpg.draw_rectangle((0, 0), (loupe_size, loupe_size), color=(255, 255, 255), thickness=1, parent=layer_tag)

    # Position loupe to avoid window edges
    viewport_width, viewport_height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    padding = 20

    loupe_x = mouse_pos[0] + padding
    if loupe_x + loupe_size > viewport_width:
        loupe_x = mouse_pos[0] - padding - loupe_size

    loupe_y = mouse_pos[1] + padding
    if loupe_y + loupe_size > viewport_height:
        loupe_y = mouse_pos[1] - padding - loupe_size

    dpg.configure_item("loupe_window", pos=(loupe_x, loupe_y))


def _leftclick_release_callback(sender, app_data, user_data):
    """Finish annotation drag operation and timeline seeking."""

    app_state = user_data["app_state"]

    with app_state.lock:
        # Check if an annotation drag was active and finalize it
        if app_state.drag_state.get("active"):
            frame_idx = app_state.frame_idx
            cam_idx = app_state.drag_state["cam_idx"]
            p_idx = app_state.drag_state["p_idx"]

            # Note: the annotation (x, y, confidence) was updated in _image_drag_callback
            app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
            app_state.needs_3d_reconstruction = True
            app_state.drag_state = {}

            dpg.hide_item("loupe_window")

        # Check if a seeking operation was active and finish it
        if app_state.is_seeking:
            app_state.is_seeking = False


# ============================================================================
# Callbacks - UI State
# ============================================================================

def _toggle_focus_mode_callback(sender, app_data, user_data):
    """Toggle focus mode."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.focus_selected_point = not app_state.focus_selected_point
        status = "Enabled" if app_state.focus_selected_point else "Disabled"
        print(f"Focus mode: {status}")


def _toggle_show_epipolar_lines_callback(sender, app_data, user_data):
    """Toggle visibility of epipolar lines."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_epipolar_lines = dpg.get_value(sender)


def _toggle_show_all_labels_callback(sender, app_data, user_data):
    """Toggle visibility of all keypoint labels."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_all_labels = dpg.get_value(sender)


def _toggle_show_reprojection_error_callback(sender, app_data, user_data):
    """Toggle visibility of the reprojection error lines."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.show_reprojection_error = dpg.get_value(sender)


def _toggle_histogram_visibility_callback(sender, app_data, user_data):
    """Toggle histogram visibility."""

    show = dpg.get_value("show_histogram_checkbox")
    dpg.configure_item("annotation_plot", show=show)
    dpg.configure_item("histogram_separator", show=show)

    # Adjust main content window height
    if show:
        dpg.configure_item("main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_FULL)
    else:
        dpg.configure_item("main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_COLLAPSED)


def _on_histogram_click(sender, app_data, user_data):
    """Handle click on histogram to jump."""

    app_state = user_data["app_state"]
    mouse_pos = dpg.get_plot_mouse_pos()

    if mouse_pos:
        clicked_frame = int(mouse_pos[0])
        with app_state.lock:
            if 0 <= clicked_frame < app_state.video_metadata['num_frames']:
                app_state.frame_idx = clicked_frame
                app_state.paused = True


# ============================================================================
# Callbacks - File ops
# ============================================================================

def _quit_callback(sender, app_data, user_data):
    dpg.stop_dearpygui()

def _save_state_callback(sender, app_data, user_data):
    user_data["app_state"].save_to_disk(config.DATA_FOLDER)


def _load_state_callback(sender, app_data, user_data):
    user_data["app_state"].load_from_disk(config.DATA_FOLDER)


def _show_cache_manager_callback(sender, app_data, user_data):
    """Show the cache manager dialog."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    def on_cache_complete(metadata, cache_dir):
        """Called when cache build completes."""

        print(f"Cache built successfully: {cache_dir}")
        try:
            from cache_utils import DiskCacheReader
            cache_reader = DiskCacheReader(cache_dir=cache_dir)
            app_state.diskcache_reader = cache_reader

            # Update the video reader worker to use the new cache
            queues.command.put({"action": "reload_cache", "cache_reader": cache_reader})
            print("Cache loaded and VideoReaderWorker notified.")
        except Exception as e:
            print(f"ERROR loading cache after build: {e}")

    # Instantiate and show the dialog
    dialog = CacheManagerDialog(
        app_state=app_state,
        queues=queues,
        data_folder=config.DATA_FOLDER,
        video_format=config.VIDEO_FORMAT,
        on_complete=on_cache_complete
    )
    dialog.show()


# ============================================================================
# Callbacks - Calibration
# ============================================================================


def _toggle_calib_frame_callback(sender, app_data, user_data):
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


def _clear_calib_frames_callback(sender, app_data, user_data):
    """Clears the entire calibration set."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.calibration.calibration_frames.clear()
    print("Calibration frame set has been cleared.")


def _navigate_calib_frame_callback(sender, app_data, user_data):
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


def _start_ga_callback(sender, app_data, user_data):
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


def _stop_ga_callback(sender, app_data, user_data):
    """Stop genetic algorithm."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    queues.ga_command.put({"action": "stop"})
    dpg.hide_item("ga_popup")


def _ba_mode_changed_callback(sender, app_data, user_data):
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


def _start_ba_callback(sender, app_data, user_data):
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


def _stop_ba_callback(sender, app_data, user_data):
    """Signal the BA worker to stop and discard results."""

    queues = user_data["queues"]
    print("Sending stop signal to Bundle Adjustment worker.")
    queues.stop_bundle_adjustment.set()
    dpg.hide_item("ba_progress_popup")


def _clear_calibration_callback(sender, app_data, user_data):
    """Clears the current calibration from the app state."""

    app_state = user_data["app_state"]

    with app_state.lock:
        # Create a new empty calibration dictionary to clear the state
        empty_calib = {name: {} for name in app_state.camera_names}
        app_state.calibration.update_calibration(empty_calib)
        app_state.calibration.best_fitness = float('inf')


# ============================================================================
# Callbacks - Keyboard
# ============================================================================

def _on_alt_down(sender, app_data, user_data):
    """Temporarily hide overlays when Alt is held down."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.temp_hide_overlays = True


def _on_alt_up(sender, app_data, user_data):
    """Show overlays again when Alt is released."""

    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.temp_hide_overlays = False


def _on_key_press(sender, app_data, user_data):
    """Handle keyboard shortcuts."""

    app_state = user_data["app_state"]

    match app_data:
        case dpg.mvKey_Spacebar:
            _toggle_pause_callback(sender, app_data, user_data)
        case dpg.mvKey_Right:
            _next_frame_callback(sender, app_data, user_data)
        case dpg.mvKey_Left:
            _prev_frame_callback(sender, app_data, user_data)
        case dpg.mvKey_Up:
            with app_state.lock:
                new_idx = (app_state.selected_point_idx - 1) % app_state.num_points
                app_state.selected_point_idx = new_idx
        case dpg.mvKey_Down:
            with app_state.lock:
                new_idx = (app_state.selected_point_idx + 1) % app_state.num_points
                app_state.selected_point_idx = new_idx
        case dpg.mvKey_S:
            _save_state_callback(sender, app_data, user_data)
        case dpg.mvKey_L:
            _load_state_callback(sender, app_data, user_data)
        case dpg.mvKey_Z:
            _toggle_focus_mode_callback(sender, app_data, user_data)
        case dpg.mvKey_H:
            _set_human_annotated_callback(sender, app_data, user_data)
        case dpg.mvKey_D:
            _clear_future_annotations_callback(sender, app_data, user_data)

        case dpg.mvKey_C:
            _toggle_calib_frame_callback(sender, app_data, user_data)
        # case dpg.mvKey_Next:
        #     _navigate_calib_frame_callback(sender, None, {"app_state": user_data["app_state"], "direction": 1})
        # case dpg.mvKey_Prior:
        #     _navigate_calib_frame_callback(sender, None, {"app_state": user_data["app_state"], "direction": -1})

# ============================================================================
# UI Update Functions
# ============================================================================

def resize_video_widgets(sender, app_data, user_data):
    """Resize video widgets and maintain aspect ratio."""

    if not user_data or "app_state" not in user_data:
        return

    app_state = user_data["app_state"]

    if not dpg.does_item_exist("video_table"):
        return

    grid_width = dpg.get_item_rect_size("video_grid_window")[0]

    n_cols = len(dpg.get_item_children("video_table", slot=0))
    if n_cols == 0:
        return

    item_width = (grid_width / n_cols) - 20

    nb_videos = app_state.video_metadata['num_videos']

    if item_width <= 0:
        return

    aspect_ratio = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_height = item_width / aspect_ratio
    frame_thickness = 2

    # Resize video views
    for i in range(nb_videos):
        if dpg.does_item_exist(f"drawlist_{i}"):
            dpg.configure_item(f"drawlist_{i}", width=item_width, height=item_height)
            dpg.configure_item(
                f"video_image_{i}",
                pmin=(frame_thickness, frame_thickness),
                pmax=(item_width - frame_thickness, item_height - frame_thickness)
            )
            dpg.configure_item(f"video_border_{i}", pmax=(item_width, item_height))

def update_ui(app_state: AppState):
    """Update all UI elements."""

    update_annotation_overlays(app_state)
    update_histogram(app_state)
    _update_control_panel(app_state)


def _update_control_panel(app_state: AppState):
    """Update control panel texts."""

    with app_state.lock:
        calibration = app_state.calibration

        dpg.set_value("frame_slider", app_state.frame_idx)
        dpg.set_value("current_frame_line", float(app_state.frame_idx))

        dpg.configure_item(
            "play_pause_button",
            label="Play" if app_state.paused else "Pause"
        )
        dpg.set_value("point_combo", app_state.point_names[app_state.selected_point_idx])

        focus_status = "Enabled" if app_state.focus_selected_point else "Disabled"
        dpg.set_value("focus_text", f"Focus Mode: {focus_status}")

        # Update calibration button text
        if app_state.frame_idx in calibration.calibration_frames:
            dpg.configure_item("toggle_calib_frame_button", label="Remove (C)")
        else:
            dpg.configure_item("toggle_calib_frame_button", label="Add (C)")

        dpg.set_value(
            "num_calib_frames_text",
            f"Calibration Frames: {len(calibration.calibration_frames)}"
        )
        dpg.set_value("fitness_text", f"Best Fitness: {calibration.best_fitness:.2f}")


def update_annotation_overlays(app_state: AppState):
    """Draw annotation overlays."""

    with app_state.lock:

        frame_idx = app_state.frame_idx
        num_videos = len(app_state.camera_names)

        focus_mode = app_state.focus_selected_point
        show_all_labels = app_state.show_all_labels
        show_reprojection_error = app_state.show_reprojection_error
        show_epipolar_lines = app_state.show_epipolar_lines
        temp_hide_overlays = app_state.temp_hide_overlays

        calibration = app_state.calibration

        all_annotations = app_state.annotations[frame_idx]
        all_human_annotated = app_state.human_annotated[frame_idx]
        point_names = app_state.point_names
        camera_colors = app_state.camera_colors
        camera_names = app_state.camera_names

    # Selected point data for special overlays
    p_idx = app_state.selected_point_idx
    best_calib = calibration.best_calibration
    f_mats = calibration.F_mats
    selected_annots = app_state.annotations[frame_idx, :, p_idx]
    point_3d = app_state.reconstructed_3d_points[frame_idx, p_idx]

    for cam_idx in range(num_videos):

        layer_tag = f"annotation_layer_{cam_idx}"
        drawlist_tag = f"drawlist_{cam_idx}"
        dpg.delete_item(layer_tag, children_only=True)

        widget_size = dpg.get_item_rect_size(drawlist_tag)

        if widget_size[0] == 0:
            continue

        video_info = app_state.videos.get(camera_names[cam_idx])
        video_w = video_info.width
        video_h = video_info.height

        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h
        cam_name = camera_names[cam_idx]

        # epipolar lines for selected point
        if show_epipolar_lines and not temp_hide_overlays and best_calib and f_mats:
            _draw_epipolar_lines(
                cam_idx, selected_annots, f_mats, num_videos,
                video_w, video_h, scale_x, scale_y,
                camera_colors, camera_names, layer_tag
            )

        # reprojection for selected point
        if not temp_hide_overlays and not np.isnan(point_3d).any() and best_calib:
            _draw_reprojection(
                cam_idx, point_3d, selected_annots,
                best_calib[cam_name],
                scale_x, scale_y, layer_tag,
                show_reprojection_error
            )

        # and all annotated points
        if not temp_hide_overlays:
            _draw_all_points(
                cam_idx, all_annotations, all_human_annotated, point_names,
                p_idx, focus_mode, app_state.num_points, scale_x, scale_y, layer_tag,
                show_all_labels
            )


def _draw_epipolar_lines(
        cam_idx, selected_annots, f_mats, num_videos,
        video_w, video_h, scale_x, scale_y,
        camera_colors, camera_names, layer_tag
):
    """Draws epipolar lines from the other cameras with labels placed inside the view."""

    for from_cam in range(num_videos):
        if cam_idx == from_cam:
            continue

        # Get only x, y
        point_2d = selected_annots[from_cam, :2]
        if np.isnan(point_2d).any():
            continue

        F = f_mats.get((from_cam, cam_idx))
        if F is None:
            continue

        p_hom = np.array([point_2d[0], point_2d[1], 1.0])
        a, b, c = F @ p_hom

        intersection_points = line_box_intersection(a, b, c, 0, 0, video_w, video_h)

        if len(intersection_points) == 2:
            p1_video, p2_video = intersection_points
            color = camera_colors[from_cam % len(camera_colors)]

            p1_scaled = (p1_video[0] * scale_x, p1_video[1] * scale_y)
            p2_scaled = (p2_video[0] * scale_x, p2_video[1] * scale_y)
            dpg.draw_line(p1_scaled, p2_scaled, color=color, thickness=1, parent=layer_tag)

            # Draw the camera name label
            anchor_pos = list(p1_scaled) # use first intersection point as anchor
            widget_size = (video_w * scale_x, video_h * scale_y)
            font_size = 12
            inset = 5

            # Adjust position based on which edge anchor is on
            if anchor_pos[0] < 1:  # Left edge
                anchor_pos[0] = inset
            elif anchor_pos[0] > widget_size[0] - 1:  # Right edge
                text_width_estimate = len(camera_names[from_cam]) * font_size * 0.6
                anchor_pos[0] = widget_size[0] - text_width_estimate - inset

            if anchor_pos[1] < 1:  # Top edge
                anchor_pos[1] = inset
            elif anchor_pos[1] > widget_size[1] - 1:  # Bottom edge
                anchor_pos[1] = widget_size[1] - font_size - inset

            dpg.draw_text(
                pos=anchor_pos,
                text=camera_names[from_cam],
                color=color,
                size=font_size,
                parent=layer_tag
            )


def _draw_reprojection(
        cam_idx, point_3d, selected_annots,
        cam_params, scale_x, scale_y, layer_tag,
        show_reprojection_error
):
    """Draws reproj and error line for reconstructed point."""

    reprojected = reproject_points(point_3d, cam_params)

    if reprojected.size == 0:
        return

    reproj_2d = reprojected[0]
    reproj_scaled = (reproj_2d[0] * scale_x, reproj_2d[1] * scale_y)
    color_reproj = (255, 0, 0)
    color_line = (255, 100, 100)

    # Red X at reprojection
    dpg.draw_line(
        (reproj_scaled[0] - 5, reproj_scaled[1] - 5),
        (reproj_scaled[0] + 5, reproj_scaled[1] + 5),
        color=color_reproj,
        parent=layer_tag
    )
    dpg.draw_line(
        (reproj_scaled[0] - 5, reproj_scaled[1] + 5),
        (reproj_scaled[0] + 5, reproj_scaled[1] - 5),
        color=color_reproj,
        parent=layer_tag
    )

    # Error line and distance label (if annotation exists and is enabled)
    # Check only x, y for existence
    if show_reprojection_error and not np.isnan(selected_annots[cam_idx, :2]).any():
        annot_scaled = (
            selected_annots[cam_idx, 0] * scale_x,
            selected_annots[cam_idx, 1] * scale_y
        )

        # Calculate distance
        p1 = np.array(annot_scaled)
        p2 = np.array(reproj_scaled)
        distance = np.linalg.norm(p1 - p2)

        # Draw dotted line
        # DPG doesn't have dotted lines??
        vec = p2 - p1
        vec_norm = vec / (distance + 1e-6)
        dash_length = 5
        gap_length = 3

        current_pos = 0
        while current_pos < distance:
            start_point = p1 + vec_norm * current_pos
            end_point = p1 + vec_norm * min(current_pos + dash_length, distance)
            dpg.draw_line(tuple(start_point), tuple(end_point), color=color_line, thickness=1, parent=layer_tag)
            current_pos += dash_length + gap_length

        # Draw distance label
        mid_point = p1 + vec * 0.5
        label_pos = (mid_point[0] + 5, mid_point[1] - 5)
        dpg.draw_text(
            pos=label_pos,
            text=f"{distance:.1f}px",
            color=color_line,
            size=12,
            parent=layer_tag
        )


def _draw_all_points(
    cam_idx, annotations, human_annotated, point_names,
    selected_point_idx, focus_mode, num_points, scale_x, scale_y, layer_tag, show_all_labels
):
    """Draws all annotated keypoints and their labels."""

    for i in range(num_points):

        if focus_mode and i != selected_point_idx:
            continue

        # Get only x, y
        point_2d = annotations[cam_idx, i, :2]
        if np.isnan(point_2d).any():
            continue

        center_x = point_2d[0] * scale_x
        center_y = point_2d[1] * scale_y

        non_selected_colour = (255, 255, 255) if human_annotated[cam_idx, i] else (0, 255, 255)
        color = (255, 255, 0) if i == selected_point_idx else non_selected_colour

        # Draw center dot
        dpg.draw_circle(
            center=(center_x, center_y),
            radius=1,
            color=color,
            fill=color,
            parent=layer_tag
        )

        # Label (only for current point)
        if show_all_labels or i == selected_point_idx:
            dpg.draw_text(
                pos=(center_x + 8, center_y - 8),
                text=point_names[i],
                color=color,
                size=14,
                parent=layer_tag
            )


def update_histogram(app_state: AppState):
    """Update annotation count histogram."""

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        point_name = app_state.point_names[selected_idx]
        num_cams = app_state.video_metadata['num_videos']

        # Annotation array is (F, C, P, 3), check for nans in x or y
        if focus_mode:
            is_valid = np.all(~np.isnan(app_state.annotations[:, :, selected_idx, :2]), axis=-1)
        else:
            is_valid = np.all(~np.isnan(app_state.annotations[:, :, :, :2]), axis=-1)

    if focus_mode:
        counts = np.sum(is_valid, axis=1)
        dpg.configure_item("histogram_y_axis", label=f"'{point_name}' Annots")
        dpg.set_axis_limits("histogram_y_axis", 0, num_cams)
    else:
        counts = np.sum(is_valid, axis=(1, 2))
        dpg.configure_item("histogram_y_axis", label="Total Annots")
        if counts.max() > 0:
            dpg.set_axis_limits_auto("histogram_y_axis")

    dpg.set_value(
        "annotation_histogram_series",
        [list(range(len(counts))), counts.tolist()]
    )

