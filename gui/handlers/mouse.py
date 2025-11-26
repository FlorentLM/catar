import cv2
import numpy as np
from dearpygui import dearpygui as dpg

import config
from core.annotation import snap_annotation
from utils import line_box_intersection


def image_mousedown_callback(sender, app_data, user_data):
    """Handle mouse click on video view (for annotating)."""

    app_state = user_data["app_state"]
    cam_idx = user_data["cam_idx"]
    drawlist_tag = f"drawlist_{cam_idx}"

    with app_state.lock:
        frame_idx = app_state.frame_idx
        p_idx = app_state.selected_point_idx
        camera_name = app_state.camera_itn[cam_idx]
        video_meta = app_state.get_video_metadata(camera_name)

    annotations = app_state.data.get_camera_annotations(frame_idx, cam_idx)

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
            local_pos[0] * video_meta['width'] / container_size[0],
            local_pos[1] * video_meta['height'] / container_size[1]
        ),
        dtype=np.float32
    )

    # Check if near existing point
    existing_point = annotations[p_idx, :2] # only check x, y
    is_drag_start = False

    if not np.isnan(existing_point).any():
        existing_local = (
            existing_point[0] * container_size[0] / video_meta['width'],
            existing_point[1] * container_size[1] / video_meta['height']
        )
        dist = np.linalg.norm(np.array(local_pos) - np.array(existing_local))
        if dist < config.ANNOTATION_DRAG_THRESHOLD:
            is_drag_start = True

    with app_state.lock:
        if app_data[0] == 0:  # left click
            annotation_pos_for_drag = None
            if is_drag_start:
                # Use existing point's position for the drag operation
                annotation_pos_for_drag = app_state.data.get_annotation(frame_idx, cam_idx, p_idx)[:2].copy()

            else:
                # Create a new point
                snapped_pos = snap_annotation(app_state, cam_idx, p_idx, frame_idx, scaled_pos)
                final_pos = snapped_pos if snapped_pos is not None else scaled_pos

                # Assign the new annotation (x, y, confidence=1.0)
                app_state.data.set_annotation(
                    frame_idx, cam_idx, p_idx,
                    final_pos, confidence=1.0, is_human=True
                )
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
            image_mousedrag_callback(sender, app_data, user_data)

        elif app_data[0] == 1:  # right click = delete
            # Delete x, y, and confidence
            app_state.data.clear_annotation(frame_idx, cam_idx, p_idx)
            app_state.needs_3d_reconstruction = True


def image_mousedrag_callback(sender, app_data, user_data):
    """Update annotation position while dragging and render the zoom loupe."""

    app_state = user_data["app_state"]

    with app_state.lock:
        if not app_state.drag_state.get("active"):
            return

        cam_idx = app_state.drag_state["cam_idx"]
        p_idx = app_state.drag_state["p_idx"]
        frame_idx = app_state.frame_idx

        camera_name = app_state.camera_itn[cam_idx]
        video_meta = app_state.get_video_metadata(camera_name)
        video_w = video_meta['width']
        video_h = video_meta['height']

        current_frames = app_state.current_video_frames
        best_calib = app_state.calibration.best_calibration

        show_epipolar_lines = app_state.show_epipolar_lines
        temp_hide_overlays = app_state.temp_hide_overlays

        f_mats = app_state.calibration.F_mats
        camera_colors = app_state.camera_colors
        cam_names = app_state.camera_names

    all_annotations = app_state.data.get_frame_annotations(frame_idx)
    point_3d_selected = app_state.data.get_point3d(frame_idx, p_idx)

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
                drag_state["slow_down_start_annotation_pos"] = app_state.data.get_annotation(frame_idx, cam_idx, p_idx)[:2].copy()

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
        app_state.data.set_annotation(
            frame_idx, cam_idx, p_idx,
            final_scaled_pos, confidence=1.0, is_human=True
        )
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
    if not temp_hide_overlays and best_calib and not np.isnan(point_3d_selected[:3]).any():
        cam_name_target = app_state.camera_itn[cam_idx]
        reprojected = app_state.calibration.reproject_to_one(
            point_3d_selected[:3].reshape(1, 3),
            cam_name_target
        ).flatten()

        if reprojected.size > 0:
            reproj_loupe_coords = to_loupe_coords(reprojected)

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
                    distance_px = np.linalg.norm(final_scaled_pos - reprojected)
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


def leftclick_release_callback(sender, app_data, user_data):
    """Finish annotation drag operation and timeline seeking."""

    app_state = user_data["app_state"]

    with app_state.lock:
        # Check if an annotation drag was active and finalize it
        if app_state.drag_state.get("active"):
            app_state.drag_state = {}
            dpg.hide_item("loupe_window")

        # Check if a seeking operation was active and finish it
        if app_state.is_seeking:
            app_state.is_seeking = False


def histogram_leftclick(sender, app_data, user_data):
    """Handle click on histogram to jump."""

    app_state = user_data["app_state"]
    mouse_pos = dpg.get_plot_mouse_pos()

    if mouse_pos:
        clicked_frame = int(mouse_pos[0])
        with app_state.lock:
            if 0 <= clicked_frame < app_state.video_metadata['num_frames']:
                app_state.frame_idx = clicked_frame
                app_state.paused = True
