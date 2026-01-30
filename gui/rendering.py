import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np
from dearpygui import dearpygui as dpg

from utils import line_box_intersection, get_confidence_color
from lucida import CameraModel, CameraRig

if TYPE_CHECKING:
    from state import AppState


@dataclass
class Object3D:
    """Represents a 3D object to render."""
    type: str                       # 'point' or 'line'
    coords: np.ndarray              # (3,) for point or (2, 3) for line
    color: Tuple[int, int, int]     # RGB 0-255
    label: Optional[str] = None


class Viewer3D:
    """
    Hardware-accelerated 3D visualisation with Open3D.
    """

    def __init__(self, window_name: str = "3D Reconstruction"):
        self.window_name = window_name
        self.vis = None
        self.is_initialized = False
        self.lock = threading.Lock()
        self.init_failed = False

        # Queue for scene updates from other threads
        self.update_queue = queue.Queue(maxsize=2)

        # Cached geom
        self.point_clouds = {}
        self.line_sets = {}
        self.needs_reset = True

        # View state
        self.view_control = None

    def initialize(self):
        """Initialise Open3D window."""

        if self.is_initialized or self.init_failed:
            return

        try:
            import open3d as o3d

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name, width=800, height=600, left=50, top=50)

            opt = self.vis.get_render_option()
            if opt:
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
                opt.point_size = 5.0
                opt.line_width = 1.0
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0, origin=[0, 0, 0])

            self.vis.add_geometry(coord_frame)
            self.view_control = self.vis.get_view_control()

            self.reset_view()

            self.is_initialized = True

        except Exception as e:
            print(f"Failed to initialize Open3D viewer: {e}")

            self.init_failed = True
            if self.vis:
                self.vis.destroy_window()
            self.vis = None

    def queue_update(self, scene_objects: List[Object3D]):
        """Queue a scene update to be processed from main thread."""

        try:
            while not self.update_queue.empty():
                try:
                    self.update_queue.get_nowait()
                except queue.Empty:
                    break
            self.update_queue.put(scene_objects)
        except queue.Full:
            pass

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
                pass

            self.vis.update_renderer()
            return True

        except Exception:
            self.is_initialized = False
            return False

    def _update_scene_internal(self, scene_objects: List[Object3D]):
        import open3d as o3d

        # Clear old geometries if needed
        if self.needs_reset:
            self.vis.clear_geometries()
            self.point_clouds.clear()
            self.line_sets.clear()
            self.needs_reset = False

        points_data = []
        lines_data = []

        for obj in scene_objects:
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
            positions = np.array([p['pos'] for p in points_data])
            colors = np.array([p['color'] for p in points_data])

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

        if lines_data:
            all_points, all_lines, all_colors = [], [], []

            offset = 0
            for l in lines_data:
                pts, col = l['points'], l['color']
                all_points.extend(pts)
                for i in range(len(pts) - 1):
                    all_lines.append([offset + i, offset + i + 1])
                    all_colors.append(col)
                offset += len(pts)

            if all_points:
                if 'main_lines' not in self.line_sets:
                    ls = o3d.geometry.LineSet()
                    ls.points = o3d.utility.Vector3dVector(np.array(all_points))
                    ls.lines = o3d.utility.Vector2iVector(np.array(all_lines))
                    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
                    self.vis.add_geometry(ls)
                    self.line_sets['main_lines'] = ls
                else:
                    ls = self.line_sets['main_lines']
                    ls.points = o3d.utility.Vector3dVector(np.array(all_points))
                    ls.lines = o3d.utility.Vector2iVector(np.array(all_lines))
                    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
                    self.vis.update_geometry(ls)

    def reset_view(self):
        if self.is_initialized and self.view_control:
            self.view_control.set_lookat([0, 0, 0])
            self.view_control.set_front([1, 0.5, -0.5])
            self.view_control.set_up([0, 1, 0])
            self.view_control.set_zoom(0.5)

    def close(self):
        if self.is_initialized and self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.is_initialized = False

    def is_open(self) -> bool:
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
        rig = app_state.rig

        # UI flags
        show_epipolar = app_state.show_epipolar_lines
        show_reproj = app_state.show_reprojection_error
        temp_hide = app_state.temp_hide_overlays
        camera_colors = app_state.camera_colors
        p_idx = app_state.selected_point_idx

        focus_mode = app_state.focus_selected_point
        show_all_labels = app_state.show_all_labels

    all_annotations = app_state.data.get_2d(frame=frame_idx)
    all_manual_flags = app_state.data.is_manual(frame_idx)

    selected_annots = app_state.data.get_2d(frame=frame_idx, keypoint=p_idx)
    point_3d = app_state.data.get_3d(frame=frame_idx, keypoint=p_idx)

    for cam_idx, cam_name in enumerate(rig.names):
        layer_tag = f"annotation_layer_{cam_idx}"
        dpg.delete_item(layer_tag, children_only=True)

        # Get widget size for scaling
        widget_size = dpg.get_item_rect_size(f"drawlist_{cam_idx}")
        if widget_size[0] == 0:
            continue

        cam = rig[cam_name]

        video_w, video_h = cam.image_size
        scale_x = widget_size[0] / video_w
        scale_y = widget_size[1] / video_h

        if show_epipolar and not temp_hide and len(rig) > 1:
            draw_epipolar_lines(
                target_cam_name=cam_name,
                selected_annots=selected_annots,
                rig=rig,
                video_w=video_w, video_h=video_h,
                scale_x=scale_x, scale_y=scale_y,
                camera_colors=camera_colors,
                layer_tag=layer_tag
            )

        if not temp_hide and not np.isnan(point_3d).any():
            draw_reprojection_errors(
                point_3d=point_3d,
                selected_annot=selected_annots[cam_idx],
                cam=cam,
                scale_x=scale_x, scale_y=scale_y,
                layer_tag=layer_tag,
                show_reproj_error=show_reproj
            )

        if not temp_hide:
            draw_all_points(
                app_state=app_state,
                cam_idx=cam_idx,
                selected_point_idx=p_idx,
                annotations=all_annotations,
                manual_flags=all_manual_flags,
                scale_x=scale_x, scale_y=scale_y,
                layer_tag=layer_tag,
                focus_mode=focus_mode,
                show_all_labels=show_all_labels
            )


def update_histogram(app_state: 'AppState'):

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        point_name = app_state.point_itn[selected_idx]
        num_cams = len(app_state.rig)

    with app_state.data.read_lock():

        # TODO: This hist is broken

        if focus_mode:
            annots = app_state.data.get_2d(frame=selected_idx, keypoint=selected_idx)
            counts = [np.sum(~np.isnan(annots[..., 0]))]

            dpg.configure_item("histogram_y_axis", label=f"'{point_name}' Annots")
            dpg.set_axis_limits("histogram_y_axis", 0, num_cams)
        else:
            annots = app_state.data.get_2d(frame=selected_idx)
            counts = np.sum(~np.isnan(annots[..., 0]), axis=1)

            dpg.configure_item("histogram_y_axis", label="Total Annots")
            if counts.max() > 0:
                dpg.set_axis_limits_auto("histogram_y_axis")

    dpg.set_value("annotation_histogram_series", [list(range(len(counts))), counts])


def update_control_panel(app_state: 'AppState'):
    """Update control panel texts."""

    with app_state.lock:
        dpg.set_value("frame_slider", app_state.frame_idx)
        dpg.set_value("current_frame_line", float(app_state.frame_idx))
        dpg.configure_item("play_pause_button", label="Play" if app_state.paused else "Pause")
        dpg.set_value("point_combo", app_state.point_itn[app_state.selected_point_idx])
        dpg.set_value("focus_text", f"Focus Mode: {'Enabled' if app_state.focus_selected_point else 'Disabled'}")

        is_calib = app_state.frame_idx in app_state.calibration_frames
        dpg.configure_item("toggle_calib_frame_button", label="Remove (C)" if is_calib else "Add (C)")
        dpg.set_value("num_calib_frames_text", f"Calibration Frames: {len(app_state.calibration_frames)}")
        dpg.set_value("fitness_text", f"Best Fitness: {app_state.ga_best_fitness:.2f}")


def draw_epipolar_lines(
        target_cam_name: str,
        selected_annots: np.ndarray,
        rig: CameraRig,
        video_w: int, video_h: int,
        scale_x: float, scale_y: float,
        camera_colors: List,
        layer_tag: str
):
    for from_cam_name in rig.names:
        if from_cam_name == target_cam_name:
            continue
        from_cam_idx = rig.get_index(from_cam_name)

        # Get point from source camera
        point_2d = selected_annots[from_cam_idx, :2]
        if np.isnan(point_2d).any():
            continue

        F = rig.F_between(from_cam_name, target_cam_name)
        p_hom = np.array([point_2d[0], point_2d[1], 1.0])
        a, b, c = F @ p_hom

        intersects = line_box_intersection(a, b, c, 0, 0, video_w, video_h)
        if len(intersects) == 2:
            p1, p2 = intersects
            color = camera_colors[from_cam_idx % len(camera_colors)]

            p1s = (p1[0] * scale_x, p1[1] * scale_y)
            p2s = (p2[0] * scale_x, p2[1] * scale_y)
            dpg.draw_line(p1s, p2s, color=color, thickness=1, parent=layer_tag)

            # Draw label
            anchor = list(p1s)
            if anchor[0] < 10:
                anchor[0] = 5
            elif anchor[0] > (video_w * scale_x - 30):
                anchor[0] = video_w * scale_x - 30
            if anchor[1] < 10:
                anchor[1] = 5
            elif anchor[1] > (video_h * scale_y - 20):
                anchor[1] = video_h * scale_y - 20

            dpg.draw_text(pos=anchor, text=from_cam_name, color=color, size=12, parent=layer_tag)


def draw_reprojection_errors(
        point_3d: np.ndarray,
        selected_annot: np.ndarray,
        cam: CameraModel,
        scale_x: float, scale_y: float,
        layer_tag: str,
        show_reproj_error: bool
):

    p3d = point_3d[:3].reshape(1, 3)

    reproj = cam.project(p3d).flatten()
    if np.isnan(reproj).any():
        return

    p_reproj = (reproj[0] * scale_x, reproj[1] * scale_y)

    # Draw red X
    dpg.draw_line((p_reproj[0] - 5, p_reproj[1] - 5), (p_reproj[0] + 5, p_reproj[1] + 5), color=(255, 0, 0),
                  parent=layer_tag)
    dpg.draw_line((p_reproj[0] - 5, p_reproj[1] + 5), (p_reproj[0] + 5, p_reproj[1] - 5), color=(255, 0, 0),
                  parent=layer_tag)

    if show_reproj_error and not np.isnan(selected_annot[:2]).any():
        p_obs = (selected_annot[0] * scale_x, selected_annot[1] * scale_y)
        dist = np.linalg.norm(np.array(p_obs) - np.array(p_reproj))

        dpg.draw_line(
            p_obs, p_reproj,
            color=(255, 100, 100),
            thickness=1,
            parent=layer_tag
        )

        dpg.draw_text(
            pos=(p_obs[0] + 5, p_obs[1] - 5),
            text=f"{dist:.1f}px",
            color=(255, 100, 100),
            size=12,
            parent=layer_tag
        )


def draw_all_points(
        app_state: 'AppState',
        cam_idx: int,
        selected_point_idx: int,
        annotations: np.ndarray,
        manual_flags: np.ndarray,
        scale_x: float,
        scale_y: float,
        layer_tag: str,
        focus_mode: bool,
        show_all_labels: bool
):
    """Draws all keypoints and their labels."""

    for i in range(app_state.num_points):
        if focus_mode and i != selected_point_idx:
            continue

        pt = annotations[cam_idx, i]
        if np.isnan(pt[:2]).any():
            continue

        cx, cy = pt[0] * scale_x, pt[1] * scale_y

        if i == selected_point_idx:
            color = (255, 255, 0)
        elif manual_flags[cam_idx, i]:
            color = (255, 255, 255)
        else:
            color = get_confidence_color(pt[2])

        dpg.draw_circle(center=(cx, cy), radius=2, color=color, fill=color, parent=layer_tag)

        if show_all_labels or i == selected_point_idx:
            dpg.draw_text(pos=(cx + 8, cy - 8), text=f"{app_state.point_itn[i]} ({pt[2]:.2f})", color=color, size=12,
                          parent=layer_tag)


def resize_video_widgets(sender, app_data, user_data):
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
    if item_width <= 0:
        return

    # Use first camera aspect ratio
    first_cam = app_state.rig.names[0]
    meta = app_state.get_video_metadata(first_cam)
    ar = meta['width'] / meta['height']
    item_height = item_width / ar

    for i in range(len(app_state.rig)):
        if dpg.does_item_exist(f"drawlist_{i}"):
            dpg.configure_item(f"drawlist_{i}", width=item_width, height=item_height)
            dpg.configure_item(f"video_image_{i}", pmin=(2, 2), pmax=(item_width - 2, item_height - 2))
            dpg.configure_item(f"video_border_{i}", pmax=(item_width, item_height))


def create_camera_visual(rig: CameraRig, camera_name: str, scene_centre: np.ndarray) -> List[Object3D]:
    """Generates 3D visual for a camera frustum."""

    cam = rig[camera_name]
    apex = cam.center  # world coords

    # Scale based on distance to scene centre
    dist = np.linalg.norm(apex - scene_centre)
    scale = dist * 0.15

    # Get frustum corners at depth 'scale'
    frustum = cam.frustum_points(depth=scale)
    tl, tr, bl, br, apex = frustum

    color = (255, 255, 0)
    objs = [Object3D('point', apex, color, camera_name)]

    # Edges from apex
    for corner in [tl, tr, bl, br]:
        objs.append(Object3D('line', np.array([apex, corner]), color))

    # Base rectangle
    for p1, p2 in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
        objs.append(Object3D('line', np.array([p1, p2]), color))

    return objs