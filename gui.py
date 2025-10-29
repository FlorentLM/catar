import dearpygui.dearpygui as dpg
import numpy as np
from queue import Queue
from multiprocessing import Queue as MPQueue

from state import AppState
from viz_3d import SceneVisualizer

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Main UI

def create_dpg_ui(
        app_state: AppState,
        command_queue: Queue,
        ga_command_queue: MPQueue,
        scene_visualizer: SceneVisualizer
):
    dpg.create_context()

    GRID_COLS = 3
    CONTROL_PANEL_WIDTH = 320  # A bit of extra padding
    TARGET_ITEM_HEIGHT = 280  # The desired height for each video feed
    PADDING = 20  # Padding around grid items

    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1
    num_rows = (num_items + GRID_COLS - 1) // GRID_COLS

    video_ar = app_state.video_metadata['width'] / app_state.video_metadata['height']
    target_item_width = TARGET_ITEM_HEIGHT * video_ar

    # Calculate the total dimensions needed
    initial_width = int(CONTROL_PANEL_WIDTH + (target_item_width * GRID_COLS) + (PADDING * GRID_COLS))
    initial_height = int((TARGET_ITEM_HEIGHT * num_rows) + (PADDING * num_rows) + 80)  # +80 for menu bar etc

    # Initial setup for viewport and textures
    _create_textures(app_state.video_metadata)
    _create_themes()

    # DPG event handlers for global key/mouse events
    _register_event_handlers(app_state, command_queue, ga_command_queue, scene_visualizer)

    # Window & widget layout
    with dpg.window(label="Main Window", tag="main_window"):
        _create_menu_bar(app_state, command_queue, ga_command_queue)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=300, tag="control_panel_window"):
                _create_control_panel(app_state, command_queue, ga_command_queue)
            with dpg.child_window(width=-1, tag="video_grid_window"):
                _create_video_grid(app_state, scene_visualizer, command_queue)

    _create_ga_popup(app_state, ga_command_queue)

    dpg.create_viewport(title="CATAR - Refactored", width=initial_width, height=initial_height)
    dpg.set_viewport_resize_callback(resize_video_widgets, user_data={"app_state": app_state})

    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()


# UI helpers

def _create_textures(video_meta):
    """Creates raw textures for video feeds and the 3D projection."""

    with dpg.texture_registry():
        for i in range(video_meta['num_videos']):
            black_screen = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT,
                default_value=black_screen.ravel(), tag=f"video_texture_{i}", format=dpg.mvFormat_Float_rgba
            )

        # 3D view texture
        black_screen_3d = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 4), dtype=np.float32)
        dpg.add_raw_texture(
            width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT,
            default_value=black_screen_3d.ravel(), tag="3d_texture", format=dpg.mvFormat_Float_rgba
        )


def _create_themes():
    """Creates themes for buttons."""

    with dpg.theme(tag="record_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 0, 0, 255))

    with dpg.theme(tag="tracking_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 200, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 0, 255))


def _create_menu_bar(app_state, command_queue, ga_command_queue):
    """Creates the main menu bar."""

    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save State", callback=_save_state_callback, user_data={"app_state": app_state})
        with dpg.menu(label="Calibration"):
            dpg.add_menu_item(label="Run Genetic Algorithm", callback=_start_ga_callback,
                              user_data={"app_state": app_state, "ga_command_queue": ga_command_queue})


def _create_control_panel(app_state, command_queue, ga_command_queue):
    """Creates the control panel."""

    user_data = {"app_state": app_state, "command_queue": command_queue}

    dpg.add_text("--- Info ---")
    dpg.add_text("Frame: 0/0", tag="frame_text")
    dpg.add_text("Status: Paused", tag="status_text")
    dpg.add_text("Tracking: Disabled", tag="tracking_text")
    dpg.add_text(f"Annotating: {app_state.POINT_NAMES[0]}", tag="annotating_point_text")
    dpg.add_separator()
    dpg.add_text("Best Fitness: inf", tag="fitness_text")
    dpg.add_separator()

    dpg.add_text("--- Controls ---")
    with dpg.group(horizontal=True):
        dpg.add_button(label="< Prev", callback=_prev_frame_callback, user_data=user_data)
        dpg.add_button(label="Play", callback=_toggle_pause_callback, user_data=user_data, tag="play_pause_button")
        dpg.add_button(label="Next >", callback=_next_frame_callback, user_data=user_data)

    dpg.add_slider_int(
        label="Frame", min_value=0, max_value=app_state.video_metadata['num_frames'] - 1,
        default_value=0, callback=_set_frame_callback, user_data=user_data, tag="frame_slider"
    )
    dpg.add_combo(
        label="Keypoint", items=app_state.POINT_NAMES, default_value=app_state.POINT_NAMES[0],
        callback=_set_selected_point_callback, user_data=user_data, tag="point_combo"
    )
    dpg.add_button(label="Track Keypoints", callback=_toggle_tracking_callback, user_data=user_data,
                   tag="keypoint_tracking_button")


def _create_video_grid(app_state: AppState, scene_visualizer: SceneVisualizer, command_queue: Queue):
    """Creates a dynamic grid for video feeds and the 3D projection."""

    GRID_COLS = 3
    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1  # (videos + 3D view)

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):

        for _ in range(GRID_COLS):
            dpg.add_table_column()

        num_rows = (num_items + GRID_COLS - 1) // GRID_COLS

        for i in range(num_rows):
            with dpg.table_row():
                for j in range(GRID_COLS):
                    idx = i * GRID_COLS + j

                    if idx < num_videos:
                        with dpg.table_cell():
                            dpg.add_text(app_state.video_names[idx])

                            # Group the image and its drawlist together
                            with dpg.drawlist(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, tag=f"drawlist_{idx}"):

                                dpg.draw_image(f"video_texture_{idx}",
                                               pmin=(0, 0),
                                               pmax=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                               tag=f"video_image_{idx}")

                                dpg.add_draw_layer(tag=f"annotation_layer_{idx}")

                            with dpg.item_handler_registry(tag=f"image_handler_{idx}"):
                                dpg.add_item_clicked_handler(
                                    callback=_image_click_callback,
                                    user_data={
                                        "cam_idx": idx,
                                        "app_state": app_state,
                                        "command_queue": command_queue
                                    }
                                )

                            # We need to bind the handler to the drawlist (not the image inside it)
                            dpg.bind_item_handler_registry(f"drawlist_{idx}", f"image_handler_{idx}")

                    elif idx == num_videos:
                        # This grid cell is the 3D projection

                        with dpg.table_cell():
                            dpg.add_text("3D Projection")
                            dpg.add_image("3d_texture", tag="3d_image", width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
                            with dpg.item_handler_registry(tag="3d_image_handler"):
                                dpg.add_item_clicked_handler(callback=scene_visualizer.dpg_drag_start)
                            dpg.bind_item_handler_registry("3d_image", "3d_image_handler")


def resize_video_widgets(sender, app_data, user_data):
    """Callback to dynamically resize video images to fit the window."""

    app_state = user_data["app_state"]
    GRID_COLS = 3
    grid_width = dpg.get_item_rect_size("video_grid_window")[0]

    item_width = (grid_width / GRID_COLS) - 20
    if item_width <= 0: return

    aspect_ratio = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_height = item_width / aspect_ratio

    for i in range(app_state.video_metadata['num_videos']):
        dpg.configure_item(f"drawlist_{i}", width=item_width, height=item_height)
        dpg.configure_item(f"video_image_{i}", pmax=(item_width, item_height))

    dpg.configure_item("3d_image", width=item_width, height=item_height)


def update_annotation_overlays(app_state: AppState):
    """
    Draws simple 2D annotations directly on the GUI thread for immediate feedback.
    This runs every frame.
    """
    with app_state.lock:
        frame_idx = app_state.frame_idx
        num_videos = app_state.video_metadata['num_videos']
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']
        annotations = app_state.annotations[frame_idx]
        human_annotated = app_state.human_annotated[frame_idx]
        point_names = app_state.POINT_NAMES
        point_colors = app_state.point_colors

    for cam_idx in range(num_videos):
        layer_tag = f"annotation_layer_{cam_idx}"
        drawlist_tag = f"drawlist_{cam_idx}"

        # Clear previous drawings from this layer
        dpg.delete_item(layer_tag, children_only=True)

        # Get the current size of the display widget to scale the points
        widget_size = dpg.get_item_rect_size(drawlist_tag)
        if widget_size[0] == 0 or widget_size[1] == 0:
            continue

        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h

        for p_idx in range(app_state.num_points):
            point_2d = annotations[cam_idx, p_idx]
            if not np.isnan(point_2d).any():
                # Scale from original video coords to displayed widget coords
                center_x = point_2d[0] * scale_x
                center_y = point_2d[1] * scale_y

                color = point_colors[p_idx].tolist()

                # Draw a white outline for points that were manually placed
                if human_annotated[cam_idx, p_idx]:
                    dpg.draw_circle(center=(center_x, center_y), radius=7, color=(255, 255, 255), parent=layer_tag)

                dpg.draw_circle(center=(center_x, center_y), radius=5, color=color, fill=color, parent=layer_tag)
                dpg.draw_text(pos=(center_x + 8, center_y - 8), text=point_names[p_idx], color=color, size=14,
                              parent=layer_tag)

def _create_ga_popup(app_state, ga_command_queue):
    """Creates the popup window for the genetic algorithm."""

    user_data = {"app_state": app_state, "ga_command_queue": ga_command_queue}
    with dpg.window(label="Calibration Progress", modal=True, show=False, tag="ga_popup", width=400, height=150,
                    no_close=True):
        dpg.add_text("Running Genetic Algorithm...", tag="ga_status_text")
        dpg.add_text("Generation: 0", tag="ga_generation_text")
        dpg.add_text("Best Fitness: inf", tag="ga_fitness_text")
        dpg.add_text("Mean Fitness: inf", tag="ga_mean_fitness_text")
        dpg.add_button(label="Stop Calibration", callback=_stop_ga_callback, user_data=user_data, width=-1)


def _register_event_handlers(app_state, command_queue, ga_command_queue, scene_visualizer):
    """Registers global event handlers for keyboard and mouse."""

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key_press, user_data={"app_state": app_state})
        dpg.add_mouse_wheel_handler(callback=scene_visualizer.dpg_on_mouse_wheel, user_data="3d_image")
        dpg.add_mouse_drag_handler(callback=scene_visualizer.dpg_drag_move)
        dpg.add_mouse_release_handler(callback=scene_visualizer.dpg_drag_end)


# Callbacks

def _on_key_press(sender, app_data, user_data):
    """Handles global key presses for shortcuts."""

    if app_data == dpg.mvKey_Spacebar:
        _toggle_pause_callback(sender, app_data, user_data)

    elif app_data == dpg.mvKey_Right:
        _next_frame_callback(sender, app_data, user_data)

    elif app_data == dpg.mvKey_Left:
        _prev_frame_callback(sender, app_data, user_data)


def _toggle_pause_callback(sender, app_data, user_data):
    """Toggles paused state."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = not app_state.paused


def _next_frame_callback(sender, app_data, user_data):
    """Goes to next frame."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        num_frames = app_state.video_metadata['num_frames']
        if app_state.frame_idx < num_frames - 1:
            app_state.frame_idx += 1

def _prev_frame_callback(sender, app_data, user_data):
    """Goes to previous frame."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        if app_state.frame_idx > 0:
            app_state.frame_idx -= 1

def _set_frame_callback(sender, app_data, user_data):
    """Sets frame index from the slider."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.paused = True
        app_state.frame_idx = app_data


def _set_selected_point_callback(sender, app_data, user_data):
    """Sets the currently active keypoint."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.selected_point_idx = app_state.POINT_NAMES.index(app_data)


def _toggle_tracking_callback(sender, app_data, user_data):
    """Toggles the Lucas-Kanade optic flow tracking."""

    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.keypoint_tracking_enabled = not app_state.keypoint_tracking_enabled
        is_enabled = app_state.keypoint_tracking_enabled
    # update the button
    if is_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)


def _image_click_callback(sender, app_data, user_data):
    """Handles clicks on a video feed."""

    app_state = user_data["app_state"]
    cam_idx = user_data["cam_idx"]

    # The container that was clicked is the drawlist
    drawlist_tag = f"drawlist_{cam_idx}"

    # Calculate scaled mouse position relative to the drawlist container
    mouse_pos_abs = dpg.get_mouse_pos(local=False)
    container_pos_abs = dpg.get_item_rect_min(drawlist_tag) # Get rect of the drawlist
    local_pos = (mouse_pos_abs[0] - container_pos_abs[0], mouse_pos_abs[1] - container_pos_abs[1])

    with app_state.lock:
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']
        frame_idx = app_state.frame_idx
        selected_point_idx = app_state.selected_point_idx

    # Get the current size of the drawlist container
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0 or container_size[1] == 0:
        return # Avoid division by zero if the widget is not yet sized

    # Scale the local click position to the original video's resolution
    scaled_pos_x = local_pos[0] * video_w / container_size[0]
    scaled_pos_y = local_pos[1] * video_h / container_size[1]
    original_resolution_pos = (scaled_pos_x, scaled_pos_y)

    # Update state and flag that a 3D reconstruction is needed
    with app_state.lock:
        if app_data[0] == 0:  # Left click to annotate
            app_state.annotations[frame_idx, cam_idx, selected_point_idx] = original_resolution_pos
            app_state.human_annotated[frame_idx, cam_idx, selected_point_idx] = True
            app_state.needs_3d_reconstruction = True
            print(f"Annotated point {selected_point_idx} on cam {cam_idx} at {original_resolution_pos}")


        elif app_data[0] == 1:  # Right click to delete
            app_state.annotations[frame_idx, cam_idx, selected_point_idx] = np.nan
            app_state.human_annotated[frame_idx, cam_idx, selected_point_idx] = False
            app_state.needs_3d_reconstruction = True
            print(f"Deleted point {selected_point_idx} on cam {cam_idx}")


def _save_state_callback(sender, app_data, user_data):
    """Triggers saving of the application state."""

    app_state = user_data["app_state"]

    # For now we hardcode the data folder # TODO: add a file dialog
    from main import DATA_FOLDER

    app_state.save_data_to_files(DATA_FOLDER)


def _start_ga_callback(sender, app_data, user_data):
    """Starts the Genetic Algorithm worker."""

    app_state = user_data["app_state"]
    ga_command_queue = user_data["ga_command_queue"]

    with app_state.lock:
        app_state.is_ga_running = True
        ga_snapshot = app_state.get_ga_state_snapshot()

    ga_command_queue.put({
        "action": "start",
        "ga_state_snapshot": ga_snapshot
    })
    dpg.show_item("ga_popup")


def _stop_ga_callback(sender, app_data, user_data):
    """Stops the Genetic Algorithm worker."""

    app_state = user_data["app_state"]
    ga_command_queue = user_data["ga_command_queue"]

    with app_state.lock:
        app_state.is_ga_running = False

    ga_command_queue.put({"action": "stop"})
    dpg.hide_item("ga_popup")