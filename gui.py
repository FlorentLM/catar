"""
GUI implementation using DearPyGUI.
"""
import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import config
from state import AppState, Queues
from viz_3d import SceneVisualizer
from core import reproject_points, refine_annotation


# ============================================================================
# Main UI
# ============================================================================

def create_ui(app_state: AppState, queues: Queues, scene_viz: SceneVisualizer):
    """Create the main DearPyGUI window."""

    dpg.create_context()

    # Calculate window dimensions
    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1  # Videos + 3D view
    num_rows = (num_items + config.GRID_COLS - 1) // config.GRID_COLS

    video_ar = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_width = 480
    item_height = item_width / video_ar

    window_width = int(
        config.CONTROL_PANEL_WIDTH +
        (item_width * config.GRID_COLS) +
        (config.PADDING * (config.GRID_COLS + 2))
    )
    window_height = int(
        (item_height * num_rows) +
        config.BOTTOM_PANEL_HEIGHT_FULL +
        100
    )

    # Setup
    _create_textures(app_state.video_metadata)
    _create_themes()
    _register_handlers(app_state, queues)

    # Main window layout
    with dpg.window(label="Main Window", tag="main_window"):
        _create_menu_bar(app_state, queues)

        with dpg.child_window(
            tag="main_content_window",
            height=-config.BOTTOM_PANEL_HEIGHT_FULL
        ):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=config.CONTROL_PANEL_WIDTH, tag="control_panel_window"):
                    _create_control_panel(app_state, queues)
                with dpg.child_window(width=-1, tag="video_grid_window"):
                    _create_video_grid(app_state, scene_viz, queues)

        with dpg.child_window(tag="bottom_panel_window", height=config.BOTTOM_PANEL_HEIGHT_FULL):
            _create_bottom_panel(app_state)

    # Popups
    _create_ga_popup(app_state, queues)
    _create_batch_track_popup(app_state)
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
    # TODO: Replace the 3D view by a proper 3D plot... maybe need to ditch DPG?

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

        with dpg.menu(label="Calibration"):
            dpg.add_menu_item(
                label="Run Genetic Algorithm",
                callback=_start_ga_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Add Frame to Calibration Set (C)",
                callback=_add_to_calib_frames_callback,
                user_data=user_data
            )


def _create_control_panel(app_state: AppState, queues: Queues):
    """Create left control panel."""

    user_data = {"app_state": app_state, "queues": queues}

    dpg.add_text("--- Info ---")
    dpg.add_text("Focus Mode: Disabled", tag="focus_text")
    dpg.add_text("Calibration Frames: 0", tag="num_calib_frames_text")
    dpg.add_text("Best Fitness: inf", tag="fitness_text")
    dpg.add_separator()

    dpg.add_text("--- Controls ---")
    dpg.add_combo(
        label="Keypoint",
        items=app_state.point_names,
        default_value=app_state.point_names[0],
        callback=_set_selected_point_callback,
        user_data=user_data,
        tag="point_combo"
    )
    dpg.add_button(
        label="Track Keypoints",
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

    dpg.add_separator()

    dpg.add_button(
        label="Set Prev as Annotated (H)",
        callback=_set_human_annotated_callback,
        user_data=user_data
    )
    dpg.add_button(
        label="Delete Future Annots (D)",
        callback=_clear_future_annotations_callback,
        user_data=user_data
    )

    dpg.add_separator()

    dpg.add_checkbox(
        label="Show Histogram",
        default_value=True,
        tag="show_histogram_checkbox",
        callback=_toggle_histogram_visibility_callback
    )


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

    # Histogram
    with dpg.plot(
        label="Annotation Histogram",
        height=-1,
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

    with dpg.item_handler_registry(tag="histogram_handler"):
        dpg.add_item_clicked_handler(
            callback=_on_histogram_click,
            user_data=user_data
        )
    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")


def _create_video_grid(app_state: AppState, scene_viz: SceneVisualizer, queues: Queues):
    """Create grid of videos and 3D view."""

    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
        for _ in range(config.GRID_COLS):
            dpg.add_table_column()

        num_rows = (num_items + config.GRID_COLS - 1) // config.GRID_COLS

        for row in range(num_rows):
            with dpg.table_row():
                for col in range(config.GRID_COLS):
                    idx = row * config.GRID_COLS + col

                    if idx < num_videos:
                        _create_video_cell(idx, app_state)
                    elif idx == num_videos:
                        _create_3d_view_cell(scene_viz)


def _create_video_cell(cam_idx: int, app_state: AppState):
    """Create a single video view cell."""

    with dpg.table_cell():
        dpg.add_text(app_state.video_names[cam_idx])
        with dpg.drawlist(
            width=config.DISPLAY_WIDTH,
            height=config.DISPLAY_HEIGHT,
            tag=f"drawlist_{cam_idx}"
        ):
            dpg.draw_image(
                f"video_texture_{cam_idx}",
                pmin=(0, 0),
                pmax=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                tag=f"video_image_{cam_idx}"
            )
            dpg.add_draw_layer(tag=f"annotation_layer_{cam_idx}")

        with dpg.item_handler_registry(tag=f"image_handler_{cam_idx}"):
            dpg.add_item_clicked_handler(
                callback=_image_mouse_down_callback,
                user_data={"cam_idx": cam_idx, "app_state": app_state}
            )
        dpg.bind_item_handler_registry(f"drawlist_{cam_idx}", f"image_handler_{cam_idx}")


def _create_3d_view_cell(scene_viz: SceneVisualizer):
    """Create 3D visualisation cell."""
    # TODO: Get rid of this cell probably

    with dpg.table_cell():
        dpg.add_text("3D Projection")
        dpg.add_text("")
        dpg.add_text("3D view opens in a separate window", color=(255, 255, 0))
        dpg.add_text("")
        dpg.add_button(
            label="Refresh 3D View",
            callback=lambda: scene_viz.reset_view()
        )


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


def _create_batch_track_popup(app_state: AppState):
    """Create batch tracking progress popup."""

    user_data = {"app_state": app_state}

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
    """Creates the floating, borderless window for the zoom loupe."""

    loupe_size = 128
    with dpg.window(
        tag="loupe_window",
        show=False,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
        width=loupe_size,
        height=loupe_size
    ):
        dpg.add_image("loupe_texture")


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
            callback=_image_release_callback,
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
        app_state.stop_batch_track.clear()
        queues.tracking_command.put({
            "action": "batch_track",
            "start_frame": start_frame
        })

    dpg.set_value("batch_track_progress", 0.0)
    dpg.show_item("batch_track_popup")


def _stop_batch_track_callback(sender, app_data, user_data):
    """Stop batch tracking."""

    app_state = user_data["app_state"]

    print("Stop command issued to batch tracker.")

    app_state.stop_batch_track.set()
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
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

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
    scaled_pos = (
        local_pos[0] * video_w / container_size[0],
        local_pos[1] * video_h / container_size[1]
    )

    # Check if near existing point
    existing_point = annotations[p_idx]
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
            if is_drag_start:
                # Start dragging existing point
                app_state.drag_state = {
                    "cam_idx": cam_idx,
                    "p_idx": p_idx,
                    "active": True
                }
                dpg.show_item("loupe_window")
                print(f"Started dragging point {app_state.point_names[p_idx]} in camera {cam_idx}")
            else:
                # Reprojection-assisted annotation

                # Attempt to refine the click using other views before creating the point
                refined_pos = refine_annotation(
                    app_state,
                    cam_idx,
                    p_idx,
                    frame_idx
                )

                # Use the refined position if available, otherwise use the user's click
                final_pos = refined_pos if refined_pos is not None else scaled_pos

                if refined_pos is not None:
                    print(f"Snapped new annotation for '{app_state.point_names[p_idx]}' using reprojection.")

                # Create new point
                app_state.annotations[frame_idx, cam_idx, p_idx] = final_pos

                app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
                app_state.needs_3d_reconstruction = True
                print(f"Created new annotation: {app_state.point_names[p_idx]} at frame {frame_idx}, camera {cam_idx}, pos {final_pos}")
                # Allow dragging immediately
                app_state.drag_state = {
                    "cam_idx": cam_idx,
                    "p_idx": p_idx,
                    "active": True
                }

        elif app_data[0] == 1:  # right click = delete
            app_state.annotations[frame_idx, cam_idx, p_idx] = np.nan
            app_state.human_annotated[frame_idx, cam_idx, p_idx] = False
            app_state.needs_3d_reconstruction = True
            print(f"Deleted annotation: {app_state.point_names[p_idx]} at frame {frame_idx}, camera {cam_idx}")


def _image_drag_callback(sender, app_data, user_data):
    """Update annotation position while dragging."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return

        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]
        frame_idx = app_state.frame_idx
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

        # Get the cached frames for the loupe
        current_frames = app_state.current_video_frames
        if current_frames is None:
            return
        video_frame = current_frames[cam_idx]

    drawlist_tag = f"drawlist_{cam_idx}"
    container_pos = dpg.get_item_rect_min(drawlist_tag)
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0:
        return

    mouse_pos = dpg.get_mouse_pos(local=False)
    local_pos = (
        mouse_pos[0] - container_pos[0],
        mouse_pos[1] - container_pos[1]
    )
    scaled_pos = (
        local_pos[0] * video_w / container_size[0],
        local_pos[1] * video_h / container_size[1]
    )

    with app_state.lock:
        app_state.annotations[frame_idx, cam_idx, p_idx] = scaled_pos
        app_state.needs_3d_reconstruction = True

    loupe_size = 128
    zoom_factor = 4

    # Calculate the region to extract from the source video frame
    src_half_w = (loupe_size / zoom_factor) / 2
    src_x = int(scaled_pos[0] - src_half_w)
    src_y = int(scaled_pos[1] - src_half_w)

    # Extract the patch
    h, w, _ = video_frame.shape
    x1, y1 = max(0, src_x), max(0, src_y)
    x2, y2 = min(w, src_x + loupe_size // zoom_factor), min(h, src_y + loupe_size // zoom_factor)

    patch = video_frame[y1:y2, x1:x2]

    if patch.size > 0:
        # zoom the patch using nearest-neighbor to see the pixels
        zoomed_patch = cv2.resize(patch, (loupe_size, loupe_size), interpolation=cv2.INTER_NEAREST)
        zoomed_patch_rgba = cv2.cvtColor(zoomed_patch, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        # Update the texture
        dpg.set_value("loupe_texture", zoomed_patch_rgba.ravel())

    # Position the loupe window near the cursor
    dpg.configure_item("loupe_window", pos=(mouse_pos[0] + 15, mouse_pos[1] + 15))


def _image_release_callback(sender, app_data, user_data):
    """Finish drag operation."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return

        frame_idx = app_state.frame_idx
        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]

        app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
        app_state.needs_3d_reconstruction = True
        app_state.drag_state = {}

    dpg.hide_item("loupe_window")  # Hide the loupe


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

def _save_state_callback(sender, app_data, user_data):
    user_data["app_state"].save_to_disk(config.DATA_FOLDER)


def _load_state_callback(sender, app_data, user_data):
    user_data["app_state"].load_from_disk(config.DATA_FOLDER)


# ============================================================================
# Callbacks - Calibration
# ============================================================================

def _add_to_calib_frames_callback(sender, app_data, user_data):
    """Add current frame to calibration set."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if app_state.frame_idx not in app_state.calibration_frames:
            app_state.calibration_frames.append(app_state.frame_idx)
            print(f"Frame {app_state.frame_idx} added to calibration set.")


def _start_ga_callback(sender, app_data, user_data):
    """Start genetic algorithm for calibration."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    with app_state.lock:
        app_state.is_ga_running = True
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

    with app_state.lock:
        app_state.is_ga_running = False

    queues.ga_command.put({"action": "stop"})
    dpg.hide_item("ga_popup")


# ============================================================================
# Callbacks - Keyboard
# ============================================================================

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
        case dpg.mvKey_C:
            _add_to_calib_frames_callback(sender, app_data, user_data)
        case dpg.mvKey_Z:
            _toggle_focus_mode_callback(sender, app_data, user_data)
        case dpg.mvKey_H:
            _set_human_annotated_callback(sender, app_data, user_data)
        case dpg.mvKey_D:
            _clear_future_annotations_callback(sender, app_data, user_data)


# ============================================================================
# UI Update Functions
# ============================================================================

def resize_video_widgets(sender, app_data, user_data):
    """Resize video widgets and maintain aspect ratio."""

    app_state = user_data["app_state"]

    grid_width = dpg.get_item_rect_size("video_grid_window")[0]
    item_width = (grid_width / config.GRID_COLS) - 20

    if item_width <= 0:
        return

    aspect_ratio = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_height = item_width / aspect_ratio

    # Resize video views
    for i in range(app_state.video_metadata['num_videos']):
        dpg.configure_item(f"drawlist_{i}", width=item_width, height=item_height)
        dpg.configure_item(f"video_image_{i}", pmax=(item_width, item_height))


def update_ui(app_state: AppState):
    """Update all UI elements."""

    update_annotation_overlays(app_state)
    update_histogram(app_state)
    _update_control_panel(app_state)


def _update_control_panel(app_state: AppState):
    """Update control panel texts."""

    with app_state.lock:
        dpg.set_value("frame_slider", app_state.frame_idx)
        dpg.set_value("current_frame_line", float(app_state.frame_idx))
        dpg.configure_item(
            "play_pause_button",
            label="Play" if app_state.paused else "Pause"
        )
        dpg.set_value("point_combo", app_state.point_names[app_state.selected_point_idx])

        focus_status = "Enabled" if app_state.focus_selected_point else "Disabled"
        dpg.set_value("focus_text", f"Focus Mode: {focus_status}")

        dpg.set_value(
            "num_calib_frames_text",
            f"Calibration Frames: {len(app_state.calibration_frames)}"
        )
        dpg.set_value("fitness_text", f"Best Fitness: {app_state.best_fitness:.2f}")


def update_annotation_overlays(app_state: AppState):
    """Draw annotation overlays."""

    with app_state.lock:
        frame_idx = app_state.frame_idx
        num_videos = app_state.video_metadata['num_videos']
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

    # Get data for all points
    all_annotations = app_state.annotations[frame_idx]
    all_human_annotated = app_state.human_annotated[frame_idx]
    point_names = app_state.point_names
    point_colors = app_state.point_colors
    camera_colors = app_state.camera_colors

    # Selected point data for special overlays
    p_idx = app_state.selected_point_idx
    best_individual = app_state.best_individual
    f_mats = app_state.fundamental_matrices
    selected_annots = app_state.annotations[frame_idx, :, p_idx]
    point_3d = app_state.reconstructed_3d_points[frame_idx, p_idx]

    # Draw
    for cam_idx in range(num_videos):
        layer_tag = f"annotation_layer_{cam_idx}"
        drawlist_tag = f"drawlist_{cam_idx}"
        dpg.delete_item(layer_tag, children_only=True)

        widget_size = dpg.get_item_rect_size(drawlist_tag)
        if widget_size[0] == 0:
            continue

        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h

        # epipolar lines for selected point
        if best_individual and f_mats:
            _draw_epipolar_lines(
                cam_idx, selected_annots, f_mats, num_videos,
                video_w, video_h, scale_x, scale_y,
                camera_colors, layer_tag
            )

        # reprojection for selected point
        if not np.isnan(point_3d).any() and best_individual:
            _draw_reprojection(
                cam_idx, point_3d, selected_annots,
                best_individual[cam_idx], p_idx,
                point_colors, scale_x, scale_y, layer_tag
            )

        # and all annotated points
        _draw_all_points(
            cam_idx, all_annotations, all_human_annotated,
            point_names, point_colors, app_state.num_points,
            scale_x, scale_y, layer_tag
        )


def _draw_epipolar_lines(
    cam_idx, selected_annots, f_mats, num_videos,
    video_w, video_h, scale_x, scale_y,
    camera_colors, layer_tag
):
    """Draws epipolar lines from the other cameras."""

    for from_cam in range(num_videos):

        if cam_idx == from_cam:
            continue

        point_2d = selected_annots[from_cam]
        if np.isnan(point_2d).any():
            continue

        F = f_mats.get((from_cam, cam_idx))
        if F is None:
            continue

        # epipolar line: l = F @ p
        p_hom = np.array([point_2d[0], point_2d[1], 1.0])
        line = F @ p_hom
        a, b, c = line

        # Find two points on the line
        if abs(b) > 1e-6:
            x0, x1 = 0, video_w
            y0 = (-a * x0 - c) / b
            y1 = (-a * x1 - c) / b
            p1, p2 = (x0, y0), (x1, y1)
        else:
            y0, y1 = 0, video_h
            x0 = (-b * y0 - c) / a
            x1 = (-b * y1 - c) / a
            p1, p2 = (x0, y0), (x1, y1)

        # Scale and draw
        p1_scaled = (p1[0] * scale_x, p1[1] * scale_y)
        p2_scaled = (p2[0] * scale_x, p2[1] * scale_y)
        color = camera_colors[from_cam % len(camera_colors)]
        dpg.draw_line(p1_scaled, p2_scaled, color=color, thickness=1, parent=layer_tag)


def _draw_reprojection(
    cam_idx, point_3d, selected_annots,
    cam_params, p_idx, point_colors,
    scale_x, scale_y, layer_tag
):
    """Draws reproj and error line for reconstructed point."""

    reprojected = reproject_points(point_3d, cam_params)

    if reprojected.size == 0:
        return

    reproj_2d = reprojected[0]
    reproj_scaled = (reproj_2d[0] * scale_x, reproj_2d[1] * scale_y)

    # red X at reprojection
    dpg.draw_line(
        (reproj_scaled[0] - 5, reproj_scaled[1] - 5),
        (reproj_scaled[0] + 5, reproj_scaled[1] + 5),
        color=(255, 0, 0),
        parent=layer_tag
    )
    dpg.draw_line(
        (reproj_scaled[0] - 5, reproj_scaled[1] + 5),
        (reproj_scaled[0] + 5, reproj_scaled[1] - 5),
        color=(255, 0, 0),
        parent=layer_tag
    )

    # error line (if annotation exists)
    if not np.isnan(selected_annots[cam_idx]).any():
        annot_scaled = (
            selected_annots[cam_idx, 0] * scale_x,
            selected_annots[cam_idx, 1] * scale_y
        )
        color = point_colors[p_idx].tolist()
        dpg.draw_line(annot_scaled, reproj_scaled, color=color, thickness=1, parent=layer_tag)


def _draw_all_points(
    cam_idx, annotations, human_annotated,
    point_names, point_colors, num_points,
    scale_x, scale_y, layer_tag
):
    """Draws all annotated keypoints and their labels."""

    for i in range(num_points):
        point_2d = annotations[cam_idx, i]
        if np.isnan(point_2d).any():
            continue

        center_x = point_2d[0] * scale_x
        center_y = point_2d[1] * scale_y
        color = point_colors[i].tolist()

        # White outer ring (human annotations)
        if human_annotated[cam_idx, i]:
            dpg.draw_circle(
                center=(center_x, center_y),
                radius=9,
                color=(255, 255, 255),
                parent=layer_tag
            )

        # Draw colored circle
        dpg.draw_circle(
            center=(center_x, center_y),
            radius=7,
            color=color,
            parent=layer_tag
        )

        # Draw center dot
        dpg.draw_circle(
            center=(center_x, center_y),
            radius=1,
            color=color,
            fill=color,
            parent=layer_tag
        )

        # Label
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
        annotations = app_state.annotations.copy()

    if focus_mode:
        # Count only selected point
        counts = np.sum(~np.isnan(annotations[:, :, selected_idx, 0]), axis=1)
        dpg.configure_item("histogram_y_axis", label=f"'{point_name}' Annots")
        dpg.set_axis_limits("histogram_y_axis", 0, app_state.video_metadata['num_videos'])
    else:
        # all points
        counts = np.sum(~np.isnan(annotations[:, :, :, 0]), axis=(1, 2))
        dpg.configure_item("histogram_y_axis", label="Total Annots")
        if counts.max() > 0:
            dpg.set_axis_limits_auto("histogram_y_axis")

    dpg.set_value(
        "annotation_histogram_series",
        [list(range(len(counts))), counts.tolist()]
    )