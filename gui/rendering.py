import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
import numpy as np
from dearpygui import dearpygui as dpg

from utils import line_box_intersection, get_confidence_color

from mokap.utils.geometry import transforms

if TYPE_CHECKING:
    from state import AppState
    from state.calibration_state import CalibrationState


@dataclass
class Object3D:
    """Represents a 3D object to render."""
    type: str                       # 'point' or 'line'
    coords: np.ndarray              # (3,) for point or (2, 3) for line
    color: Tuple[int, int, int]     # RGB 0-255
    label: Optional[str] = None


class Viewer3D:
    """
    Hardware-accelerated 3D visualization with Open3D.
    Runs in separate window with non-blocking updates.
    """

    def __init__(self, window_name: str = "3D Reconstruction"):
        self.window_name = window_name
        self.vis = None
        self.is_initialized = False
        self.lock = threading.Lock()
        self.init_failed = False  # track if initialisation has permanently failed

        # Queue for scene updates from other threads
        self.update_queue = queue.Queue(maxsize=2)

        # Cached geom
        self.point_clouds = {}
        self.line_sets = {}
        self.needs_reset = True

        # View state
        self.view_control = None
        self.camera_params = None

    def initialize(self):
        """Initialise Open3D window."""
        if self.is_initialized or self.init_failed:
            return

        try:
            import open3d as o3d

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.window_name,
                width=800,
                height=600,
                left=50,
                top=50
            )

            # Get render options before setting them
            opt = self.vis.get_render_option()
            if opt is None:
                print("Warning: Could not get render options")
                self.vis.destroy_window()
                self.vis = None
                self.init_failed = True
                return

            # Set up nice rendering options
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.point_size = 5.0
            opt.line_width = 1.0

            # Add coordinate frame at scene center
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=30.0,
                origin=[0, 0, 0]
                # origin=[-5, -140, 70]
            )
            self.vis.add_geometry(coord_frame)

            # Get view control for camera manipulation
            self.view_control = self.vis.get_view_control()

            # Set initial camera view
            self.reset_view()

            self.is_initialized = True
            print("Open3D 3D viewer initialized")

        except Exception as e:
            print(f"Failed to initialize Open3D viewer: {e}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
            self.init_failed = True
            if self.vis is not None:
                try:
                    self.vis.destroy_window()
                except:
                    pass
                self.vis = None

    def queue_update(self, scene_objects: List[Object3D]):
        """Queue a scene update to be processed from main thread."""
        try:
            # clear old updates and add new one
            while not self.update_queue.empty():
                try:
                    self.update_queue.get_nowait()
                except queue.Empty:
                    break
            self.update_queue.put(scene_objects)
        except queue.Full:
            pass  # drop update if queue is full

    def process_updates(self) -> bool:
        """
        Process queued scene updates (needs to be called from main trhead)
        """
        if self.init_failed:
            return False  # so it does not keep trying if init permanently failed

        if not self.is_initialized:
            self.initialize()

        if not self.is_initialized or self.vis is None:
            return False

        try:
            # Check if window is still open
            if not self.vis.poll_events():
                self.is_initialized = False
                return False

            # Process queued updates
            try:
                scene_objects = self.update_queue.get_nowait()
                self._update_scene_internal(scene_objects)
            except queue.Empty:
                pass  # no update available

            # and update renderer
            self.vis.update_renderer()

            return True

        except Exception as e:
            print(f"Error processing Open3D updates: {e}")
            self.is_initialized = False
            return False

    def _update_scene_internal(self, scene_objects: List[Object3D]):
        """Internal method to update the scene (called from main thread)."""

        # Clear old geometries if needed
        if self.needs_reset:
            self.vis.clear_geometries()
            self.point_clouds.clear()
            self.line_sets.clear()
            self.needs_reset = False

        # Separate objects by type
        points_data = []
        lines_data = []

        for obj in scene_objects:
            # (this handles both SceneObject instances and dicts)
            obj_type = obj.type if hasattr(obj, 'type') else obj.get('type')
            obj_coords = obj.coords if hasattr(obj, 'coords') else obj.get('coords')
            obj_color = obj.color if hasattr(obj, 'color') else obj.get('color', (255, 255, 255))
            obj_label = obj.label if hasattr(obj, 'label') else obj.get('label')

            if obj_type == 'point' and obj_coords is not None and not np.isnan(obj_coords).any():
                points_data.append({
                    'pos': obj_coords,
                    'color': np.array(obj_color) / 255.0,
                    'label': obj_label
                })
            elif obj_type == 'line' and obj_coords is not None and not np.isnan(obj_coords).any():
                lines_data.append({
                    'points': obj_coords,
                    'color': np.array(obj_color) / 255.0
                })

        # Update or create point cloud
        if points_data:
            self._update_points(points_data)

        # Update or create line sets
        if lines_data:
            self._update_lines(lines_data)

    def _update_points(self, points_data: List[dict]):
        """Update point cloud visualisation."""

        positions = np.array([p['pos'] for p in points_data])
        colors = np.array([p['color'] for p in points_data])

        # Create or update point cloud
        if 'main_points' not in self.point_clouds:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(pcd)
            self.point_clouds['main_points'] = pcd

        else:
            pcd = self.point_clouds['main_points']
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(pcd)

    def _update_lines(self, lines_data: List[dict]):
        """Update line set visualisation."""

        # Get all line segments
        all_points = []
        all_lines = []
        all_colors = []

        point_offset = 0
        for line_obj in lines_data:
            points = line_obj['points']
            color = line_obj['color']

            # Add points
            all_points.extend(points)

            # Add line indices
            for i in range(len(points) - 1):
                all_lines.append([point_offset + i, point_offset + i + 1])
                all_colors.append(color)

            point_offset += len(points)

        if not all_points:
            return

        # Create or update line set
        if 'main_lines' not in self.line_sets:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
            line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            self.vis.add_geometry(line_set)
            self.line_sets['main_lines'] = line_set

        else:
            line_set = self.line_sets['main_lines']
            line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
            line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            self.vis.update_geometry(line_set)

    def reset_view(self):
        """Reset view to default position."""

        if self.is_initialized and self.view_control:
            self.view_control.set_lookat([0, 0, 0])
            # self.view_control.set_lookat([-5, -140, 70])
            self.view_control.set_front([1, 0.5, -0.5])  # view from angle
            self.view_control.set_up([0, 1, 0])
            self.view_control.set_zoom(0.5)

    def close(self):
        """Close the 3D window."""

        if self.is_initialized and self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.is_initialized = False
            print("Open3D 3D viewer closed")

    def is_open(self) -> bool:
        """Check if the window is still open."""
        return self.is_initialized and self.vis is not None


def update_ui(app_state: 'AppState'):
    """Update all UI elements."""

    update_annotation_overlays(app_state)
    update_histogram(app_state)
    update_control_panel(app_state)


def update_annotation_overlays(app_state: 'AppState'):
    """Draw annotation overlays."""

    with app_state.lock:

        frame_idx = app_state.frame_idx
        num_videos = app_state.num_videos

        focus_mode = app_state.focus_selected_point
        show_all_labels = app_state.show_all_labels
        show_reprojection_error = app_state.show_reprojection_error
        show_epipolar_lines = app_state.show_epipolar_lines
        temp_hide_overlays = app_state.temp_hide_overlays

        calibration = app_state.calibration
        camera_colors = app_state.camera_colors

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

        cam_name = app_state.camera_itn[cam_idx]
        video_meta = app_state.get_video_metadata(cam_name)
        video_w = video_meta['width']
        video_h = video_meta['height']

        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h

        # epipolar lines for selected point
        if show_epipolar_lines and not temp_hide_overlays and best_calib and f_mats:
            draw_epipolar_lines(
                app_state=app_state,
                cam_idx=cam_idx,
                selected_annots=selected_annots,
                f_mats=f_mats,
                num_videos=num_videos,
                video_w=video_w,
                video_h=video_h,
                scale_x=scale_x,
                scale_y=scale_y,
                camera_colors=camera_colors,
                layer_tag=layer_tag
            )

        # reprojection for selected point
        if not temp_hide_overlays and not np.isnan(point_3d).any() and best_calib:
            draw_reprojection_errors(
                cam_idx=cam_idx,
                point_3d=point_3d,
                selected_annots=selected_annots,
                calibration=calibration,
                cam_name=cam_name,
                scale_x=scale_x,
                scale_y=scale_y,
                layer_tag=layer_tag,
                show_reproj_error=show_reprojection_error
            )

        # and all annotated points
        if not temp_hide_overlays:
            draw_all_points(
                app_state=app_state,
                cam_idx=cam_idx,
                selected_point_idx=p_idx,
                annotations=all_annotations,
                human_annotated=all_human_annotated,
                scale_x=scale_x,
                scale_y=scale_y,
                layer_tag=layer_tag,
                focus_mode=focus_mode,
                show_all_labels=show_all_labels
            )



def update_histogram(app_state: 'AppState'):
    """Update annotation count histogram."""

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        point_name = app_state.point_itn[selected_idx]
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
        dpg.set_value("point_combo", app_state.point_itn[app_state.selected_point_idx])

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
        app_state: 'AppState',
        cam_idx: int,
        selected_annots: np.ndarray,
        f_mats: Dict[Tuple[int, int], np.ndarray],
        num_videos: int,
        video_w: int,
        video_h: int,
        scale_x: float,
        scale_y: float,
        camera_colors: List[Tuple[int, int, int]],
        layer_tag: str
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

            from_cam_str = app_state.camera_itn[from_cam]

            # Adjust position based on which edge anchor is on
            if anchor_pos[0] < 1:  # Left edge
                anchor_pos[0] = inset
            elif anchor_pos[0] > widget_size[0] - 1:  # Right edge
                text_width_estimate = len(from_cam_str) * font_size * 0.6
                anchor_pos[0] = widget_size[0] - text_width_estimate - inset

            if anchor_pos[1] < 1:  # Top edge
                anchor_pos[1] = inset
            elif anchor_pos[1] > widget_size[1] - 1:  # Bottom edge
                anchor_pos[1] = widget_size[1] - font_size - inset

            dpg.draw_text(
                pos=anchor_pos,
                text=from_cam_str,
                color=color,
                size=font_size,
                parent=layer_tag
            )


def draw_reprojection_errors(
        cam_idx: int,
        point_3d: np.ndarray,
        selected_annots: np.ndarray,
        calibration: 'CalibrationState',
        cam_name: str,
        scale_x: float,
        scale_y: float,
        layer_tag: str,
        show_reproj_error: bool
):
    """Draws reproj and error line for reconstructed point."""

    reprojected = calibration.reproject_to_one(
        point_3d[:3].reshape(1, 3),
        cam_name
    ).flatten()

    if reprojected.size == 0:
        return

    reproj_scaled = (reprojected[0] * scale_x, reprojected[1] * scale_y)
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
    if show_reproj_error and not np.isnan(selected_annots[cam_idx, :2]).any():
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
        app_state: 'AppState',
        cam_idx: int,
        selected_point_idx: int,
        annotations: np.ndarray,
        human_annotated: np.ndarray,
        scale_x: float,
        scale_y: float,
        layer_tag: str,
        focus_mode: bool,
        show_all_labels: bool
):
    """Draws all annotated keypoints and their labels."""

    for i in range(app_state.num_points):

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
                text=f"{app_state.point_itn[i]} ({point_data[2]:.2f})",
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

