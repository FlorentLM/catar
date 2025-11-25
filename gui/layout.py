from typing import TYPE_CHECKING
import numpy as np
from dearpygui import dearpygui as dpg

import config
from gui import popups, callbacks, rendering, tracking_callbacks, calibration_callbacks, annotations_callbacks, mouse_handlers

if TYPE_CHECKING:
    from state import AppState, Queues
    from gui.viewer_3d import Viewer3D


def create_ui(app_state: 'AppState', queues: 'Queues', viewer_3d: 'Viewer3D'):
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
    create_textures(app_state.video_metadata)
    create_themes()
    callbacks.register_event_handlers(app_state, queues)

    # Main window layout
    with dpg.window(label="Main Window", tag="main_window", no_scrollbar=True):
        create_menu_bar(app_state, queues)

        with dpg.child_window(tag="main_content_window", height=-config.BOTTOM_PANEL_HEIGHT_FULL, no_scrollbar=True):

            with dpg.group(horizontal=True):
                with dpg.child_window(width=config.CONTROL_PANEL_WIDTH, tag="control_panel_window"):
                    create_control_panel(app_state, queues, viewer_3d)

                with dpg.child_window(width=-1, tag="video_grid_window"):
                    create_video_grid(app_state, n_cols, n_rows)

        with dpg.child_window(tag="bottom_panel_window", height=config.BOTTOM_PANEL_HEIGHT_FULL, no_scrollbar=True):
            create_bottom_panel(app_state)

    # Popups
    popups.create_ga_popup(app_state, queues)
    popups.create_ba_config_popup(app_state, queues)
    popups.create_ba_progress_popup(app_state, queues)
    popups.create_batch_tracking_popup(app_state, queues)
    popups.create_loupe()

    # Viewport
    dpg.create_viewport(title="CATAR", width=window_width, height=window_height)
    dpg.set_viewport_resize_callback(rendering.resize_video_widgets, user_data={"app_state": app_state})

    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()


def create_textures(video_meta: dict):
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


def create_themes():
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


def create_menu_bar(app_state: 'AppState', queues: 'Queues'):
    """Create top menu bar."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Save State (S)",
                callback=callbacks.save_state_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Load State (L)",
                callback=callbacks.load_state_callback,
                user_data=user_data
            )
            dpg.add_separator()
            dpg.add_menu_item(label="Quit", callback=callbacks.app_quit_callback)

        with dpg.menu(label="Display"):
            dpg.add_menu_item(
                label="Show Timeline Histogram",
                tag="show_histogram_checkbox",
                check=True,
                default_value=True,
                callback=callbacks.toggle_histogram_callback
            )
            dpg.add_menu_item(
                label="Show All Labels",
                tag="show_all_labels_checkbox",
                check=True,
                default_value=False,
                callback=callbacks.toggle_point_labels_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Show Reproj. Errors",
                tag="show_reprojection_error_checkbox",
                check=True,
                default_value=True,
                callback=callbacks.toggle_reprojection_error_callback,
                user_data=user_data
            )
            dpg.add_menu_item(
                label="Show Epipolar Lines",
                tag="show_epipolar_lines_checkbox",
                check=True,
                default_value=True,
                callback=callbacks.toggle_epipolar_lines_callback,
                user_data=user_data
            )

        with dpg.menu(label="Tools"):
            dpg.add_menu_item(
                label="Manage video cache...",
                callback=popups.cache_manager_callback,
                user_data=user_data
            )


def create_control_panel(app_state: 'AppState', queues: 'Queues', open3d_viz: 'Viewer3D'):
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
            callback=annotations_callbacks.set_selected_point_callback,
            user_data=user_data,
            tag="point_combo"
        )
        dpg.add_button(
            label="Toggle live tracking",
            callback=tracking_callbacks.toggle_realtime_tracking_callback,
            user_data=user_data,
            tag="keypoint_tracking_button"
        )

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="< Track Backward",
                callback=tracking_callbacks.batch_tracking_bwd_callback,
                user_data=user_data,
                width=125
            )
            dpg.add_button(
                label="Track Forward >",
                callback=tracking_callbacks.batch_tracking_fwd_callback,
                user_data=user_data,
                width=125
            )

        dpg.add_button(
            label="Set as Human annotated (H)",
            callback=annotations_callbacks.set_human_annotated_callback,
            user_data=user_data
        )
        dpg.add_button(
            label="Delete future annots (D)",
            callback=annotations_callbacks.clear_future_annotations_callback,
            user_data=user_data
        )

    dpg.add_separator()

    with dpg.collapsing_header(label="Calibration", default_open=True):

        dpg.add_text("Calibration Frames: 0", tag="num_calib_frames_text")

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="< Prev",
                callback=calibration_callbacks.navigate_calib_frame_callback,
                user_data={"app_state": app_state, "direction": -1},
                width=60
            )

            dpg.add_button(
                label="Add (C)",
                tag="toggle_calib_frame_button",
                callback=calibration_callbacks.addremove_calib_frame_callback,
                user_data=user_data,
                width=75
            )

            dpg.add_button(
                label="Next >",
                callback=calibration_callbacks.navigate_calib_frame_callback,
                user_data={"app_state": app_state, "direction": 1},
                width=60
            )

        dpg.add_button(
            label="Clear calib set",
            callback=calibration_callbacks.clear_calib_frames_callback,
            user_data=user_data,
            width=-1
        )

        dpg.add_separator()

        dpg.add_button(
            label="Create calibration (Genetic Algorithm)",
            callback=calibration_callbacks.start_ga_callback,
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


def create_bottom_panel(app_state: 'AppState'):
    """Create bottom panel (player controls and histogram)."""

    user_data = {"app_state": app_state}

    # Player controls
    with dpg.group():
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="<| Prev",
                callback=callbacks.prev_frame_callback,
                user_data=user_data
            )
            dpg.add_button(
                label="Play",
                callback=callbacks.play_pause_callback,
                user_data=user_data,
                tag="play_pause_button"
            )
            dpg.add_button(
                label="Next |>",
                callback=callbacks.next_frame_callback,
                user_data=user_data
            )

            slider = dpg.add_slider_int(
                label="Frame",
                min_value=0,
                max_value=app_state.video_metadata['num_frames'] - 1,
                default_value=0,
                callback=callbacks.set_frame_callback,
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
            callback=callbacks.set_frame_callback,
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
            callback=mouse_handlers.histogram_leftclick,
            user_data=user_data
        )
    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")


def create_video_grid(app_state: 'AppState', n_cols: int, n_rows: int):
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
                        create_video_cell(idx, app_state)


def create_video_cell(cam_idx: int, app_state: 'AppState'):
    """Create a single video view cell."""

    with dpg.table_cell():
        # Display Camera Name (bold) and Filename (faint)

        with dpg.group(horizontal=True, horizontal_spacing=5):
            camera_name = app_state.camera_names[cam_idx]
            file_name = app_state.video_filenames[cam_idx]

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
                callback=mouse_handlers.image_mousedown_callback,
                user_data={"cam_idx": cam_idx, "app_state": app_state}
            )
        dpg.bind_item_handler_registry(f"drawlist_{cam_idx}", f"image_handler_{cam_idx}")
