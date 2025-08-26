import cv2
import numpy as np
from typing import TypedDict, List, Tuple, Optional
import dearpygui.dearpygui as dpg


class SceneObject(TypedDict):
    """Base type for scene objects."""
    type: str  # 'point' or 'line'
    coords: np.ndarray  # 3D coordinates of the object
    color: Tuple[int, int, int]  # RGB color
    label: Optional[str]  # Optional label for the object

class SceneVisualizer:
    """
    An interactive 3D scene viewer that renders lists of point and line objects.
    """
    def __init__(self, frame_size=(1280, 720), show_axis=True):
        self.frame_size = frame_size
        self.window_name = '3D Scene Viewer'
        self.show_axis = show_axis

        # Viewer state in spherical coordinates
        self.distance = 20.0
        self.azimuth = np.pi / 4
        self.elevation = np.pi / 6

        # Mouse interaction state
        self.is_dragging = False
        self.last_mouse_pos = {'x': 0, 'y': 0}

        # cv2.namedWindow(self.window_name)
        # cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handles mouse events for rotation and zooming."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_pos = {'x': x, 'y': y}
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            dx = x - self.last_mouse_pos['x']
            dy = y - self.last_mouse_pos['y']
            
            if flags & cv2.EVENT_FLAG_SHIFTKEY:  # Check if shift key is held
                self.distance = max(1.0, self.distance - dy * 0.05)  # Adjust zoom
            else:
                self.azimuth -= dx * 0.005
                self.elevation -= dy * 0.005
            
            self.last_mouse_pos = {'x': x, 'y': y}
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 0.9 if flags > 0 else 1.1
            self.distance = max(1.0, self.distance * zoom_factor)
    
    def dpg_drag_start(self, sender, app_data, user_data):
        """Callback for when dragging starts."""
        self.is_dragging = True
    
    def dpg_drag_end(self, sender, app_data, user_data):
        """Callback for when dragging ends."""
        self.is_dragging = False

    def dpg_drag_move(self, sender, app_data, user_data):
        """Callback for when the image is actively being dragged."""
        if self.is_dragging:
            # dx, dy = app_data[0], app_data[1] # dx, dy are in app_data for this handler
            dx, dy = dpg.get_mouse_drag_delta()
            
            self.azimuth -= dx * 0.0005
            self.elevation -= dy * 0.0005


    def dpg_on_mouse_wheel(self, sender, app_data, user_data):
        """Global callback for the mouse wheel."""
        # IMPORTANT: Only zoom if the mouse is over our image
        if dpg.is_item_hovered(user_data):
            # app_data will be +1.0 (up) or -1.0 (down)
            zoom_factor = 0.9 if app_data > 0 else 1.1
            self.distance = max(1.0, self.distance * zoom_factor)
    
    def draw_scene(self, scene_objects: List[SceneObject]) -> np.ndarray:
        """Renders a dynamic list of scene objects onto a 2D canvas."""
        width, height = self.frame_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Calculate the view matrix from the current camera state
        
        # The negative sign on 'z' was incorrect and broke the look-at geometry.
        x = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        y = self.distance * np.sin(self.elevation)
        z = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        viewer_pos = np.array([x, y, z])
        
        # Display camera position for debugging
        pos_str = f"({viewer_pos[0]:.1f}, {viewer_pos[1]:.1f}, {viewer_pos[2]:.1f})"
        cv2.putText(canvas, f"Camera Position: {pos_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # This is a standard "look-at" implementation.
        target = np.array([0., 0., 0.])
        world_up = np.array([0., 1., 0.])
        
        # a. Create camera's coordinate frame in the world
        forward = (target-viewer_pos) / np.linalg.norm(target-viewer_pos)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(forward, right)
        
        # b. Create the world-to-camera rotation matrix (R) and translation vector (tvec)
        # R has the camera's basis vectors (right, cam_up, forward) as its rows.
        R = np.array([right, cam_up, forward])
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ viewer_pos
        
        # Define camera intrinsics (K)
        focal_length = float(width)
        K_view = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ])

        # 2. Batch-project all 3D coordinates for efficiency
        if self.show_axis:
            # Draw world axes
            ax_length = 5.0
            # Create world axes as LineObjects
            scene_objects.append(SceneObject(coords=np.array([[0,0,0], [ax_length,0,0]]), color=(96,96,96), label="X", type="line"))
            scene_objects.append(SceneObject(coords=np.array([[0,0,0], [0,ax_length,0]]), color=(96,96,96), label="Y", type="line"))
            scene_objects.append(SceneObject(coords=np.array([[0,0,0], [0,0,ax_length]]), color=(96,96,96), label="Z", type="line"))

        all_3d_coords = []
        for obj in scene_objects:
            if obj['type'] == 'line':
                all_3d_coords.extend(obj['coords'])
            elif obj['type'] == 'point':
                all_3d_coords.append(obj['coords'])
        all_3d_coords = np.vstack(all_3d_coords)

        projected_2d, _ = cv2.projectPoints(all_3d_coords, rvec, tvec, K_view, None)
        projected_2d = projected_2d.reshape(-1, 2).astype(int)

        # 3. Draw the projected objects
        coord_idx = 0
        for obj in scene_objects:
            color = (255, 255, 255)  # Default color if not specified
            if 'color' in obj:
                color = obj['color']
                if isinstance(color, tuple):
                    color = tuple(int(c) for c in color)
                if isinstance(color, np.ndarray):
                    color = tuple(color.tolist())
            label = obj.get('label')
            
            if obj['type'] == 'line':
                p1 = tuple(projected_2d[coord_idx])
                p2 = tuple(projected_2d[coord_idx + 1])
                cv2.line(canvas, p1, p2, color, 1)
                if label:
                    cv2.putText(canvas, label, (p2[0]+5, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
                coord_idx += 2
            elif obj['type'] == 'point':
                p = tuple(projected_2d[coord_idx])
                cv2.circle(canvas, p, 8, color, -1)
                if label:
                    cv2.putText(canvas, label, (p[0]+5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
                coord_idx += 1

        return canvas

def create_demo_scene() -> List[SceneObject]:
    """Generates a list of objects for demonstration."""
    scene = []
    radius = 5.0

    # Generate points for a sphere
    num_points = 1000
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)

    # Add points to the scene as PointObjects
    for i in range(num_points):
        point_coords = np.array([x[i], y[i], z[i]]) # (3,)
        # Color points by their hemisphere
        point_color = (200, 200, 50) if z[i] > 0 else (200, 50, 200)
        scene.append(SceneObject(coords=point_coords, color=point_color, type="point"))

    return scene

if __name__ == '__main__':
    visualizer = SceneVisualizer()
    scene_data = create_demo_scene()

    # Main application loop
    while True:
        # Pass the current scene data to be rendered
        vis_frame = visualizer.draw_scene(scene_data)
        cv2.imshow(visualizer.window_name, vis_frame)

        key = cv2.waitKey(20) & 0xFF # Added a small delay
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Regenerating scene...")
            scene_data = create_demo_scene()

    cv2.destroyAllWindows()