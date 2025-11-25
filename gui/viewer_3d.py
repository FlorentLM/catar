import open3d as o3d
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import queue


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