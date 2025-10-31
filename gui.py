import dearpygui.dearpygui as dpg
import numpy as np
from queue import Queue
from multiprocessing import Queue as MPQueue
from state import AppState
from viz_3d import SceneVisualizer
from pathlib import Path
from core import reproject_points

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DATA_FOLDER = Path.cwd() / 'data'

# Main UI

def create_dpg_ui(
        app_state: AppState,
        command_queue: Queue,
        ga_command_queue: MPQueue,
        scene_visualizer: SceneVisualizer
):
    dpg.create_context()

    # Window size calculation
    GRID_COLS = 3
    CONTROL_PANEL_WIDTH = 320
    BOTTOM_PANEL_HEIGHT = 200  # Initial height for bottom panel
    PADDING = 20

    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1
    num_rows = (num_items + GRID_COLS - 1) // GRID_COLS

    video_ar = app_state.video_metadata['width'] / app_state.video_metadata['height']
    # Start with a reasonable width for each video item
    target_item_width = 480
    target_item_height = target_item_width / video_ar

    initial_width = int(CONTROL_PANEL_WIDTH + (target_item_width * GRID_COLS) + (PADDING * (GRID_COLS + 2)))
    initial_height = int((target_item_height * num_rows) + BOTTOM_PANEL_HEIGHT + 100)

    _create_textures(app_state.video_metadata)
    _create_themes()
    _register_event_handlers(app_state, command_queue, ga_command_queue, scene_visualizer)

    # Window layout
    with dpg.window(label="Main Window", tag="main_window"):
        _create_menu_bar(app_state, command_queue, ga_command_queue)

        # Main content area
        with dpg.child_window(tag="main_content_window", height=-BOTTOM_PANEL_HEIGHT):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=300, tag="control_panel_window"):
                    _create_control_panel(app_state)
                with dpg.child_window(width=-1, tag="video_grid_window"):
                    _create_video_grid(app_state, scene_visualizer, command_queue)

        # Bottom panel for histogram and player controls
        with dpg.child_window(tag="bottom_panel_window", height=BOTTOM_PANEL_HEIGHT):
            _create_bottom_panel(app_state)

    _create_ga_popup(app_state, ga_command_queue)
    _create_batch_track_popup(app_state)

    dpg.create_viewport(title="CATAR", width=initial_width, height=initial_height)
    dpg.set_viewport_resize_callback(resize_video_widgets, user_data={"app_state": app_state})

    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()


# UI helpers

def _create_textures(video_meta):
    with dpg.texture_registry():
        for i in range(video_meta['num_videos']):
            black_screen = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT,
                default_value=black_screen.ravel(), tag=f"video_texture_{i}", format=dpg.mvFormat_Float_rgba
            )
        black_screen_3d = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 4), dtype=np.float32)
        dpg.add_raw_texture(
            width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT,
            default_value=black_screen_3d.ravel(), tag="3d_texture", format=dpg.mvFormat_Float_rgba
        )


def _create_themes():

    with dpg.theme(tag="record_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 0, 0, 255))

    with dpg.theme(tag="tracking_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 200, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 0, 255))

    with dpg.theme(tag="purple_slider_theme"):
        with dpg.theme_component(dpg.mvSliderInt):
            # The main color of the slider grab handle
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (200, 135, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (215, 85, 255, 255))
            # The color of the filled-in part of the bar
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (230, 165, 255, 75))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (230, 165, 255, 75))


def _create_menu_bar(app_state, command_queue, ga_command_queue):

    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save State (S)", callback=_save_state_callback, user_data={"app_state": app_state})
            dpg.add_menu_item(label="Load State (L)", callback=_load_state_callback, user_data={"app_state": app_state})

        with dpg.menu(label="Calibration"):

            dpg.add_menu_item(label="Run Genetic Algorithm", callback=_start_ga_callback,
                              user_data={"app_state": app_state, "ga_command_queue": ga_command_queue})
            dpg.add_menu_item(label="Add Frame to Calibration Set (C)", callback=_add_to_calib_frames_callback,
                              user_data={"app_state": app_state})


def _create_control_panel(app_state: AppState):
    user_data = {"app_state": app_state}

    dpg.add_text("--- Info ---")
    dpg.add_text("Focus Mode: Disabled", tag="focus_text")
    dpg.add_text("Calibration Frames: 0", tag="num_calib_frames_text")
    dpg.add_text("Best Fitness: inf", tag="fitness_text")
    dpg.add_separator()

    dpg.add_text("--- Controls ---")
    dpg.add_combo(
        label="Keypoint", items=app_state.POINT_NAMES, default_value=app_state.POINT_NAMES[0],
        callback=_set_selected_point_callback, user_data=user_data, tag="point_combo"
    )
    dpg.add_button(label="Track Keypoints", callback=_toggle_tracking_callback, user_data=user_data,
                   tag="keypoint_tracking_button")

    dpg.add_button(label="Track Forward", callback=_start_batch_track_callback, user_data=user_data,
                   tag="batch_track_button")

    dpg.add_separator()

    dpg.add_button(label="Set Prev as Annotated (H)", callback=_set_human_annotated_callback, user_data=user_data)
    dpg.add_button(label="Delete Future Annots (D)", callback=_clear_future_annotations_callback, user_data=user_data)

    dpg.add_separator()

    dpg.add_checkbox(label="Show Histogram", default_value=True, tag="show_histogram_checkbox",
                     callback=_toggle_histogram_visibility_callback)


def _create_bottom_panel(app_state: AppState):
    """Creates the bottom panel with player controls and histogram."""

    user_data = {"app_state": app_state}

    # Player controls
    with dpg.group():
        # A single horizontal group for all player controls
        with dpg.group(horizontal=True):
            # Group the buttons on the left
            dpg.add_button(label="<| Prev", callback=_prev_frame_callback, user_data=user_data)
            dpg.add_button(label="Play", callback=_toggle_pause_callback, user_data=user_data, tag="play_pause_button")
            dpg.add_button(label="Next |>", callback=_next_frame_callback, user_data=user_data)

            # The slider will automatically take up the remaining horizontal space
            slider = dpg.add_slider_int(
                label="Frame",
                min_value=0, max_value=app_state.video_metadata['num_frames'] - 1,
                default_value=0, callback=_set_frame_callback,
                user_data=user_data, tag="frame_slider", width=-1  # width=-1 makes it fill space
            )
            dpg.bind_item_theme(slider, "purple_slider_theme")

    dpg.add_separator(tag="histogram_separator")

    # Histogram
    with dpg.plot(label="Annotation Histogram", height=-1, width=-1, no_menus=True, no_box_select=True,
                  no_mouse_pos=True, tag="annotation_plot"):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="histogram_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Annotations", tag="histogram_y_axis")
        dpg.add_bar_series(list(range(app_state.video_metadata['num_frames'])),
                           [0] * app_state.video_metadata['num_frames'], label="Annotation Count",
                           parent="histogram_y_axis", tag="annotation_histogram_series")

        # Callback to link the drag line to the frame index
        dpg.add_drag_line(label="Current Frame", color=[215, 85, 255], vertical=True, default_value=0,
                          tag="current_frame_line",
                          callback=_set_frame_callback, user_data=user_data)

    with dpg.item_handler_registry(tag="histogram_handler"):
        dpg.add_item_clicked_handler(callback=_on_histogram_click, user_data=user_data)
    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")


def _create_video_grid(app_state: AppState, scene_visualizer: SceneVisualizer, command_queue: Queue):

    GRID_COLS = 3
    num_videos = app_state.video_metadata['num_videos']
    num_items = num_videos + 1

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
        for _ in range(GRID_COLS): dpg.add_table_column()
        num_rows = (num_items + GRID_COLS - 1) // GRID_COLS

        for i in range(num_rows):
            with dpg.table_row():
                for j in range(GRID_COLS):
                    idx = i * GRID_COLS + j

                    if idx < num_videos:
                        with dpg.table_cell():
                            dpg.add_text(app_state.video_names[idx])
                            with dpg.drawlist(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, tag=f"drawlist_{idx}"):
                                dpg.draw_image(f"video_texture_{idx}", pmin=(0, 0),
                                               pmax=(DISPLAY_WIDTH, DISPLAY_HEIGHT), tag=f"video_image_{idx}")
                                dpg.add_draw_layer(tag=f"annotation_layer_{idx}")

                            with dpg.item_handler_registry(tag=f"image_handler_{idx}"):
                                dpg.add_item_clicked_handler(callback=_image_mouse_down_callback,
                                                             user_data={"cam_idx": idx, "app_state": app_state})
                            dpg.bind_item_handler_registry(f"drawlist_{idx}", f"image_handler_{idx}")

                    elif idx == num_videos:
                        with dpg.table_cell():
                            dpg.add_text("3D Projection")
                            dpg.add_image("3d_texture", tag="3d_image", width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)

                            with dpg.item_handler_registry(tag="3d_image_handler"):
                                dpg.add_item_clicked_handler(callback=scene_visualizer.dpg_drag_start)
                            dpg.bind_item_handler_registry("3d_image", "3d_image_handler")


def resize_video_widgets(sender, app_data, user_data):

    app_state = user_data["app_state"]
    GRID_COLS = 3
    grid_width = dpg.get_item_rect_size("video_grid_window")[0]
    item_width = (grid_width / GRID_COLS) - 20

    if item_width <= 0:
        return

    aspect_ratio = app_state.video_metadata['width'] / app_state.video_metadata['height']
    item_height = item_width / aspect_ratio

    for i in range(app_state.video_metadata['num_videos']):
        dpg.configure_item(f"drawlist_{i}", width=item_width, height=item_height)
        dpg.configure_item(f"video_image_{i}", pmax=(item_width, item_height))

    dpg.configure_item("3d_image", width=item_width, height=item_height)


def update_annotation_overlays(app_state: AppState):
    with app_state.lock:
        frame_idx = app_state.frame_idx
        num_videos = app_state.video_metadata['num_videos']
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

    # For all points
    all_annotations = app_state.annotations[frame_idx]
    all_human_annotated = app_state.human_annotated[frame_idx]
    point_names = app_state.POINT_NAMES
    point_colors = app_state.point_colors
    camera_colors = app_state.camera_colors

    # Selected point only, for special visualizations
    p_idx = app_state.selected_point_idx
    best_individual = app_state.best_individual
    f_mats = app_state.fundamental_matrices
    selected_annots = app_state.annotations[frame_idx, :, p_idx]
    point_3d = app_state.reconstructed_3d_points[frame_idx, p_idx]

    # Main drawing loop
    for cam_idx in range(num_videos):
        layer_tag = f"annotation_layer_{cam_idx}"
        drawlist_tag = f"drawlist_{cam_idx}"
        dpg.delete_item(layer_tag, children_only=True)

        widget_size = dpg.get_item_rect_size(drawlist_tag)
        if widget_size[0] == 0: continue
        scale_x, scale_y = widget_size[0] / video_w, widget_size[1] / video_h

        # Draw special overlays for the selected point
        if best_individual and f_mats:
            valid_annot_mask = ~np.isnan(selected_annots).any(axis=1)

            # Epipolar Lines
            for from_cam_idx in range(num_videos):
                if cam_idx == from_cam_idx:
                    continue

                point_2d = selected_annots[from_cam_idx]
                if not np.isnan(point_2d).any():
                    F = f_mats.get((from_cam_idx, cam_idx))
                    if F is not None:
                        # Convert point to homogeneous coordinates: p' = [x, y, 1]
                        p_hom = np.array([point_2d[0], point_2d[1], 1.0])

                        # The epipolar line is given by l = F @ p'
                        # l is a 3-element vector [a, b, c] for the line equation ax + by + c = 0
                        line = F @ p_hom
                        a, b, c = line

                        # Calculate two points on the line that are at the edges of the video frame
                        if abs(b) > 1e-6: # Avoid division by zero for horizontal lines
                            x0, x1 = 0, video_w
                            y0 = (-a * x0 - c) / b
                            y1 = (-a * x1 - c) / b
                            p1_orig, p2_orig = (x0, y0), (x1, y1)

                        else: # Handle vertical lines by solving for x: x = (-by - c) / a
                            y0, y1 = 0, video_h
                            x0 = (-b * y0 - c) / a
                            x1 = (-b * y1 - c) / a
                            p1_orig, p2_orig = (x0, y0), (x1, y1)

                        # Scale points to the widget's current size
                        p1_scaled = (p1_orig[0] * scale_x, p1_orig[1] * scale_y)
                        p2_scaled = (p2_orig[0] * scale_x, p2_orig[1] * scale_y)
                        color = camera_colors[from_cam_idx % len(camera_colors)]
                        dpg.draw_line(p1_scaled, p2_scaled, color=color, thickness=1, parent=layer_tag)

            # Reprojection and Error lines
            # if a 3D point has been reconstructed, show where it projects back to
            if not np.isnan(point_3d).any():

                reprojected_points = reproject_points(point_3d, best_individual[cam_idx])

                if reprojected_points.size > 0:
                    reproj_2d = reprojected_points[0]

                    reproj_scaled = (reproj_2d[0] * scale_x, reproj_2d[1] * scale_y)
                    color = point_colors[p_idx].tolist()

                    # Draw a red X to mark the reprojected point
                    dpg.draw_line((reproj_scaled[0] - 5, reproj_scaled[1] - 5),
                                  (reproj_scaled[0] + 5, reproj_scaled[1] + 5), color=(255, 0, 0), parent=layer_tag)
                    dpg.draw_line((reproj_scaled[0] - 5, reproj_scaled[1] + 5),
                                  (reproj_scaled[0] + 5, reproj_scaled[1] - 5), color=(255, 0, 0), parent=layer_tag)

                    # If current view also has an annotation, draw a line showing the reprojection error
                    if valid_annot_mask[cam_idx]:
                        annot_scaled = (selected_annots[cam_idx, 0] * scale_x, selected_annots[cam_idx, 1] * scale_y)
                        dpg.draw_line(annot_scaled, reproj_scaled, color=color, thickness=1, parent=layer_tag)

        # Draw all annotated points (circles and text)
        for i in range(app_state.num_points):
            point_2d = all_annotations[cam_idx, i]
            if not np.isnan(point_2d).any():
                center_x, center_y = point_2d[0] * scale_x, point_2d[1] * scale_y
                color = point_colors[i].tolist()

                if all_human_annotated[cam_idx, i]:
                    dpg.draw_circle(center=(center_x, center_y), radius=9, color=(255, 255, 255), parent=layer_tag)

                # outer circle (not filled)
                dpg.draw_circle(center=(center_x, center_y), radius=7, color=color, parent=layer_tag)
                # tiny dot in the center
                dpg.draw_circle(center=(center_x, center_y), radius=1, color=color, fill=color, parent=layer_tag)

                dpg.draw_text(pos=(center_x + 8, center_y - 8), text=point_names[i], color=color, size=14,
                              parent=layer_tag)


def update_histogram(app_state: AppState):

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        point_name = app_state.POINT_NAMES[selected_idx]
        annotations = app_state.annotations.copy()

    if focus_mode:
        counts = np.sum(~np.isnan(annotations[:, :, selected_idx, 0]), axis=1)
        dpg.configure_item("histogram_y_axis", label=f"'{point_name}' Annots")
        dpg.set_axis_limits("histogram_y_axis", 0, app_state.video_metadata['num_videos'])
    else:
        counts = np.sum(~np.isnan(annotations[:, :, :, 0]), axis=(1, 2))
        dpg.configure_item("histogram_y_axis", label="Total Annots")
        if counts.max() > 0:
            dpg.set_axis_limits_auto("histogram_y_axis")

    dpg.set_value("annotation_histogram_series", [list(range(len(counts))), counts.tolist()])


def _create_ga_popup(app_state, ga_command_queue):
    user_data = {"app_state": app_state, "ga_command_queue": ga_command_queue}

    with dpg.window(label="Calibration Progress", modal=True, show=False, tag="ga_popup", width=400, height=150,
                    no_close=True):
        dpg.add_text("Running Genetic Algorithm...", tag="ga_status_text")
        dpg.add_text("Generation: 0", tag="ga_generation_text")
        dpg.add_text("Best Fitness: inf", tag="ga_fitness_text")
        dpg.add_text("Mean Fitness: inf", tag="ga_mean_fitness_text")
        dpg.add_button(label="Stop Calibration", callback=_stop_ga_callback, user_data=user_data, width=-1)


def _create_batch_track_popup(app_state):
    user_data = {"app_state": app_state}

    with dpg.window(label="Tracking Progress", modal=True, show=False, tag="batch_track_popup", width=400, no_close=True):
        dpg.add_text("Processing frames...", tag="batch_track_status_text")
        dpg.add_progress_bar(tag="batch_track_progress", width=-1)
        dpg.add_button(label="Stop", callback=_stop_batch_track_callback, user_data=user_data, width=-1)


def _start_batch_track_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        start_frame = app_state.frame_idx

        app_state.stop_batch_track.clear()

        # Tell the tracking worker to start its batch job
        app_state.tracking_command_queue.put({
            "action": "batch_track",
            "start_frame": start_frame
        })
    dpg.set_value("batch_track_progress", 0.0)
    dpg.show_item("batch_track_popup")


def _stop_batch_track_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    print("Stop command issued to batch tracker.")
    app_state.stop_batch_track.set() # Set the event to signal the worker to stop
    dpg.hide_item("batch_track_popup")


def _register_event_handlers(app_state, command_queue, ga_command_queue, scene_visualizer):
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key_press, user_data={"app_state": app_state})
        dpg.add_mouse_wheel_handler(callback=scene_visualizer.dpg_on_mouse_wheel, user_data="3d_image")

        # handlers to control the 3D camera
        dpg.add_mouse_drag_handler(callback=scene_visualizer.dpg_drag_move)
        dpg.add_mouse_release_handler(callback=scene_visualizer.dpg_drag_end)

        # handlers for 2D annotation dragging
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=_image_drag_callback,
                                   user_data={"app_state": app_state})
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=_image_release_callback,
                                      user_data={"app_state": app_state})


# Callbacks

def _on_key_press(sender, app_data, user_data):
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


def _on_histogram_click(sender, app_data, user_data):
    app_state = user_data["app_state"]
    mouse_pos = dpg.get_plot_mouse_pos()
    if mouse_pos:
        clicked_frame = int(mouse_pos[0])
        with app_state.lock:
            if 0 <= clicked_frame < app_state.video_metadata['num_frames']:
                app_state.frame_idx = clicked_frame
                app_state.paused = True


def _on_select_calib_file(sender, app_data, user_data):
    """
    This is the secondary callback that is executed after the user
    selects a file from the file dialog or cancels.
    """
    app_state = user_data["app_state"]

    # app_data contains the selection from the file dialog
    # Check if the user selected a file (and didn't cancel)
    if 'file_path_name' in app_data and app_data['file_path_name']:
        file_path = Path(app_data['file_path_name'])

        # Call the method in AppState to perform the actual loading
        app_state.load_calibration(file_path)

    # Clean up by deleting the file dialog from the UI
    dpg.delete_item("external_calib_file_dialog")


def _load_external_calib_callback(sender, app_data, user_data):
    """
    This is the primary callback attached to the menu item.
    It opens the file dialog window.
    """
    # Create and show the file dialog
    with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=_on_select_calib_file,
            tag="external_calib_file_dialog",
            width=700,
            height=400,
            default_path=str(DATA_FOLDER),
            user_data=user_data  # Pass the app_state along to the next callback
    ):
        # Filter for TOML files to guide the user
        dpg.add_file_extension(".toml", color=(255, 255, 0, 255))
        dpg.add_file_extension(".*")

def _toggle_pause_callback(sender, app_data, user_data):
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
    """Sets frame index from the slider or the histogram drag line."""
    app_state = user_data["app_state"]

    new_frame_idx = dpg.get_value(sender)

    if new_frame_idx is None:  # Safety check
        return

    with app_state.lock:
        app_state.paused = True

        # bounds check to be safe
        num_frames = app_state.video_metadata['num_frames']
        if 0 <= new_frame_idx < num_frames:
            app_state.frame_idx = int(new_frame_idx)


def _set_selected_point_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.selected_point_idx = app_state.POINT_NAMES.index(app_data)


def _toggle_tracking_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    with app_state.lock:
        app_state.keypoint_tracking_enabled = not app_state.keypoint_tracking_enabled
        is_enabled = app_state.keypoint_tracking_enabled
    if is_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)


def _toggle_histogram_visibility_callback(sender, app_data, user_data):
    show = dpg.get_value("show_histogram_checkbox")
    dpg.configure_item("annotation_plot", show=show)
    dpg.configure_item("histogram_separator", show=show)

    # Dynamically adjust the height of the main content window to reclaim space
    BOTTOM_PANEL_HEIGHT_FULL = 200
    BOTTOM_PANEL_HEIGHT_COLLAPSED = 75  # Height for just the player controls
    if show:
        dpg.configure_item("main_content_window", height=-BOTTOM_PANEL_HEIGHT_FULL)
    else:
        dpg.configure_item("main_content_window", height=-BOTTOM_PANEL_HEIGHT_COLLAPSED)


def _toggle_focus_mode_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]

    with app_state.lock:
        app_state.focus_selected_point = not app_state.focus_selected_point
        print(f"Focus mode: {'Enabled' if app_state.focus_selected_point else 'Disabled'}")


def _add_to_calib_frames_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]

    with app_state.lock:
        if app_state.frame_idx not in app_state.calibration_frames:
            app_state.calibration_frames.append(app_state.frame_idx)
            print(f"Frame {app_state.frame_idx} added to calibration set.")


def _set_human_annotated_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        app_state.human_annotated[:frame_idx + 1, :, p_idx] = True
        print(f"Marked all previous frames as human-annotated for point {app_state.POINT_NAMES[p_idx]}.")


def _clear_future_annotations_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.focus_selected_point:
            print("Enable Focus Mode (Z) to use this feature.")
            return

        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        app_state.annotations[frame_idx + 1:, :, p_idx] = np.nan
        app_state.human_annotated[frame_idx + 1:, :, p_idx] = False
        print(f"Cleared future annotations for point {app_state.POINT_NAMES[p_idx]}.")


def _image_mouse_down_callback(sender, app_data, user_data):
    """Initiates a drag operation or creates a new point."""

    app_state = user_data["app_state"]
    cam_idx = user_data["cam_idx"]
    drawlist_tag = f"drawlist_{cam_idx}"

    with app_state.lock:
        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        annotations = app_state.annotations[frame_idx, cam_idx, :, :]
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

    container_pos_abs = dpg.get_item_rect_min(drawlist_tag)
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0: return

    # Convert mouse position from screen space to image space
    mouse_pos_abs = dpg.get_mouse_pos(local=False)
    local_pos = (mouse_pos_abs[0] - container_pos_abs[0], mouse_pos_abs[1] - container_pos_abs[1])
    scaled_pos = (local_pos[0] * video_w / container_size[0], local_pos[1] * video_h / container_size[1])

    # Check if the click is near an existing point for the selected keypoint type
    existing_point = annotations[p_idx]
    drag_threshold = 15 # Pixel distance to start a drag
    is_drag_start = False

    if not np.isnan(existing_point).any():
        # Convert existing point from image space to local widget space for distance check
        existing_point_local = (
            existing_point[0] * container_size[0] / video_w,
            existing_point[1] * container_size[1] / video_h
        )
        dist = np.linalg.norm(np.array(local_pos) - np.array(existing_point_local))
        if dist < drag_threshold:
            is_drag_start = True

    with app_state.lock:
        if app_data[0] == 0:  # left mouse down
            if is_drag_start:
                # Start dragging the existing point
                app_state.drag_state = {"cam_idx": cam_idx, "p_idx": p_idx, "active": True}
            else:
                # Create a new point immediately
                app_state.annotations[frame_idx, cam_idx, p_idx] = scaled_pos
                app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
                app_state.needs_3d_reconstruction = True
                # Start a drag so the user can fine-tune the new point's position
                app_state.drag_state = {"cam_idx": cam_idx, "p_idx": p_idx, "active": True}

        elif app_data[0] == 1:  # Right click to delete
            app_state.annotations[frame_idx, cam_idx, p_idx] = np.nan
            app_state.human_annotated[frame_idx, cam_idx, p_idx] = False
            app_state.needs_3d_reconstruction = True

def _image_drag_callback(sender, app_data, user_data):
    """Updates the position of the point being dragged."""

    app_state = user_data["app_state"]
    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return
        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]
        frame_idx = app_state.frame_idx
        video_w = app_state.video_metadata['width']
        video_h = app_state.video_metadata['height']

    drawlist_tag = f"drawlist_{cam_idx}"
    container_pos_abs = dpg.get_item_rect_min(drawlist_tag)
    container_size = dpg.get_item_rect_size(drawlist_tag)
    if container_size[0] == 0: return

    # Convert mouse position to scaled image coordinates
    mouse_pos_abs = dpg.get_mouse_pos(local=False)
    local_pos = (mouse_pos_abs[0] - container_pos_abs[0], mouse_pos_abs[1] - container_pos_abs[1])
    scaled_pos = (local_pos[0] * video_w / container_size[0], local_pos[1] * video_h / container_size[1])

    # Update annotation in real-time
    with app_state.lock:
        app_state.annotations[frame_idx, cam_idx, p_idx] = scaled_pos
        app_state.needs_3d_reconstruction = True

def _image_release_callback(sender, app_data, user_data):
    """Finishes the drag operation and validates the point."""

    app_state = user_data["app_state"]
    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return
        # The point is now considered human-annotated, even if it was originally a tracker point
        frame_idx = app_state.frame_idx
        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]
        app_state.human_annotated[frame_idx, cam_idx, p_idx] = True
        app_state.needs_3d_reconstruction = True
        # Clear the drag state
        app_state.drag_state = {}

def _save_state_callback(sender, app_data, user_data):
    from main import DATA_FOLDER
    user_data["app_state"].save_data_to_files(DATA_FOLDER)


def _load_state_callback(sender, app_data, user_data):
    from main import DATA_FOLDER
    user_data["app_state"].load_data_from_files(DATA_FOLDER)


def _start_ga_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    ga_command_queue = user_data["ga_command_queue"]
    with app_state.lock:
        app_state.is_ga_running = True
        ga_snapshot = app_state.get_ga_state_snapshot()
    ga_command_queue.put({"action": "start", "ga_state_snapshot": ga_snapshot})
    dpg.show_item("ga_popup")


def _stop_ga_callback(sender, app_data, user_data):
    app_state = user_data["app_state"]
    ga_command_queue = user_data["ga_command_queue"]
    with app_state.lock: app_state.is_ga_running = False
    ga_command_queue.put({"action": "stop"})
    dpg.hide_item("ga_popup")