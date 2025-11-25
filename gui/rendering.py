from typing import TYPE_CHECKING, List
import numpy as np
from dearpygui import dearpygui as dpg

from gui.viewer_3d import Object3D
from utils import line_box_intersection, get_confidence_color

from mokap.utils.geometry import transforms

if TYPE_CHECKING:
    from state import AppState
    from state.calibration_state import CalibrationState


def update_ui(app_state: 'AppState'):
    """Update all UI elements."""

    update_annotation_overlays(app_state)
    update_histogram(app_state)
    update_control_panel(app_state)


def update_annotation_overlays(app_state: 'AppState'):
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

        point_names = app_state.point_names
        camera_colors = app_state.camera_colors
        camera_names = app_state.camera_names

    all_annotations = app_state.data.get_frame_annotations(frame_idx)
    all_human_annotated = app_state.data.get_human_annotated_flags(frame_idx)

    # Selected point data for special overlays
    p_idx = app_state.selected_point_idx
    best_calib = calibration.best_calibration
    f_mats = calibration.F_mats

    selected_annots = app_state.data.get_point_annotations(frame_idx, p_idx)
    point_3d = app_state.data.get_point3d(frame_idx, p_idx)

    for cam_idx in range(num_videos):

        layer_tag = f"annotation_layer_{cam_idx}"
        drawlist_tag = f"drawlist_{cam_idx}"
        dpg.delete_item(layer_tag, children_only=True)

        widget_size = dpg.get_item_rect_size(drawlist_tag)

        if widget_size[0] == 0:
            continue

        camera_name = camera_names[cam_idx]
        video_meta = app_state.get_video_metadata(camera_name)
        video_w = video_meta['width']
        video_h = video_meta['height']

        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h
        cam_name = camera_names[cam_idx]

        # epipolar lines for selected point
        if show_epipolar_lines and not temp_hide_overlays and best_calib and f_mats:
            draw_epipolar_lines(
                cam_idx, selected_annots, f_mats, num_videos,
                video_w, video_h, scale_x, scale_y,
                camera_colors, camera_names, layer_tag
            )

        # reprojection for selected point
        if not temp_hide_overlays and not np.isnan(point_3d).any() and best_calib:
            draw_reprojection_errors(
                cam_idx, point_3d, selected_annots,
                calibration, cam_name,
                scale_x, scale_y, layer_tag,
                show_reprojection_error
            )

        # and all annotated points
        if not temp_hide_overlays:
            draw_all_points(
                cam_idx, all_annotations, all_human_annotated,
                point_names, p_idx, focus_mode, app_state.num_points, scale_x, scale_y, layer_tag,
                show_all_labels
            )


def update_histogram(app_state: 'AppState'):
    """Update annotation count histogram."""

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        point_name = app_state.point_names[selected_idx]
        num_cams = app_state.video_metadata['num_videos']

    if focus_mode:
        with app_state.data.bulk_lock():
            annots = app_state.data.annotations[:, :, selected_idx, :2]
            is_valid = np.all(~np.isnan(annots), axis=-1)
    else:
        with app_state.data.bulk_lock():
            annots = app_state.data.annotations[:, :, :, :2]
            is_valid = np.all(~np.isnan(annots), axis=-1)

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


def update_control_panel(app_state: 'AppState'):
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


def draw_epipolar_lines(
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


def draw_reprojection_errors(
        cam_idx, point_3d, selected_annots,
        calibration, cam_name,
        scale_x, scale_y, layer_tag,
        show_reprojection_error
):
    """Draws reproj and error line for reconstructed point."""

    reprojected = calibration.reproject_to_one(
        point_3d[:3].reshape(1, 3),
        cam_name
    ).flatten()

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


def draw_all_points(
    cam_idx, annotations, human_annotated, point_names,
    selected_point_idx, focus_mode, num_points, scale_x, scale_y, layer_tag, show_all_labels
):
    """Draws all annotated keypoints and their labels."""

    for i in range(num_points):

        if focus_mode and i != selected_point_idx:
            continue

        # Check existence (x, y)
        point_data = annotations[cam_idx, i]
        if np.isnan(point_data[0]) or np.isnan(point_data[1]):
            continue

        center_x = point_data[0] * scale_x
        center_y = point_data[1] * scale_y

        if i == selected_point_idx:
            color = (255, 255, 0)
        elif human_annotated[cam_idx, i]:
            color = (255, 255, 255)
        else:
            color = get_confidence_color(point_data[2])

        # Draw center dot
        dpg.draw_circle(
            center=(center_x, center_y),
            radius=1,
            color=color,
            fill=color,
            parent=layer_tag
        )

        # Label
        if show_all_labels or i == selected_point_idx:
            dpg.draw_text(
                pos=(center_x + 8, center_y - 8),
                text=f"{point_names[i]} ({point_data[2]:.2f})",
                color=color,
                size=12,
                parent=layer_tag
            )


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


def create_camera_visual(
        calibration: 'CalibrationState',
        camera_name: str,
        scene_centre: np.ndarray
) -> List['Object3D']:
    """
    Generate 3D visualisation objects for a camera frustum.

    Args:
        calibration: CalibrationState object
        camera_name: Name of the camera to visualize
        scene_centre: Center of the scene for scaling

    Returns:
        List of SceneObject instances for rendering
    """
    # Get camera-to-world parameters for visualization
    cam_params = calibration.get_camera_params_c2w(camera_name)
    rvec, tvec = cam_params['rvec'], cam_params['tvec']

    R_c2w = transforms.rodrigues(rvec)
    camera_center_world = tvec.flatten()

    distance_to_center = np.linalg.norm(camera_center_world - scene_centre)
    scale = distance_to_center * 0.2  # Frustum size = 20% of distance to center

    w, h, depth = 0.3 * scale, 0.2 * scale, 0.5 * scale

    # Frustum pyramid in local camera coordinates
    # (OpenCV camera convention is +X right, +Y down, +Z forward)
    pyramid_pts_cam = np.array([
        [0, 0, 0],  # Camera center (pyramid apex)
        [-w, -h, depth],  # Bottom left of image plane (in camera space)
        [w, -h, depth],  # Bottom right
        [w, h, depth],  # Top right
        [-w, h, depth]  # Top left
    ])

    # Transform local camera pyramid points into world coordinates
    pyramid_pts_world = (R_c2w @ pyramid_pts_cam.T).T + camera_center_world
    apex, bl, br, tr, tl = pyramid_pts_world

    color = (255, 255, 0)
    objects = [Object3D(type='point', coords=apex, color=color, label=camera_name)]

    # Lines from apex to corners
    for corner in [bl, br, tr, tl]:
        objects.append(Object3D(
            type='line',
            coords=np.array([apex, corner]),
            color=color,
            label=None
        ))

    # Base rectangle
    for p1, p2 in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
        objects.append(Object3D(
            type='line',
            coords=np.array([p1, p2]),
            color=color,
            label=None
        ))

    return objects
