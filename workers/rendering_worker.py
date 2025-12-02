import queue
import threading
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

import config
from gui.rendering import create_camera_visual, Object3D, Viewer3D

if TYPE_CHECKING:
    from state import AppState
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


class RenderingWorker(threading.Thread):
    """Handles rendering of 2D overlays and 3D visualisation."""

    def __init__(
        self,
        app_state: 'AppState',
        reconstructor: 'Reconstructor',
        tracker: 'MultiObjectTracker',
        frames_in_queue: queue.Queue,
        results_out_queue: queue.Queue,
        viewer_3d: Optional['Viewer3D'],
    ):
        super().__init__(daemon=True, name="RenderingWorker")
        self.app_state = app_state
        self.viewer_3d = viewer_3d
        self.reconstructor = reconstructor
        self.tracker = tracker
        self.frames_in_queue = frames_in_queue
        self.results_out_queue = results_out_queue
        self.shutdown_event = threading.Event()

    def run(self):
        print("Rendering worker started.")
        while not self.shutdown_event.is_set():
            try:
                data = self.frames_in_queue.get(timeout=1.0)
                if data.get("action") == "shutdown":
                    self.shutdown_event.set()
                    break

                self._render_frame(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in rendering worker: {e}")
                import traceback
                traceback.print_exc()
                self.shutdown_event.set()

        print("Rendering worker shut down.")

    def _render_frame(self, data: dict):
        """Renders the current state for visualisation without modifying it."""

        with self.app_state.lock:
            # Cache the raw video frames for the loupe tool
            self.app_state.current_video_frames = data["raw_frames"]

        if not config.DISABLE_3D_VIEW and self.viewer_3d is not None:
            scene = self._build_3d_scene()
            self.viewer_3d.queue_update(scene)

        # Resize all frames to display size
        video_frames = [
            cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            for frame in data["raw_frames"]
        ]

        # Send to GUI for display
        self._send_results({
            'frame_idx': data['frame_idx'],
            'video_frames_bgr': video_frames,
        })

    def _build_3d_scene(self) -> List[Object3D]:
        """Build list of 3D objects to render."""

        scene = []

        with self.app_state.lock:
            calibration = self.app_state.calibration

            if self.app_state.show_cameras_in_3d and calibration.best_calibration:
                for cam_name in calibration.camera_names:
                    scene.extend(
                        create_camera_visual(calibration, cam_name, self.app_state.scene_centre)
                    )

            # Add reconstructed points and skeleton
            frame_idx = self.app_state.frame_idx
            point_names = self.app_state.point_names
            point_colors = self.app_state.point_colors
            skeleton = self.app_state.skeleton

        points_3d = self.app_state.data.get_frame_points3d(frame_idx)

        # Debug: count valid points
        # valid_points = 0

        # Draw points
        for i, point in enumerate(points_3d):
            if not np.isnan(point[:3]).any():
                # valid_points += 1

                color = tuple(point_colors[i])
                scene.append(Object3D(
                    type='point',
                    coords=point[:3],
                    color=color,
                    label=self.app_state.point_itn[i]
                ))

                # # Debug: print 1st point
                # if valid_points == 1:
                #     print(f"Adding point '{point_names[i]}' at {point} with color {color}")

                # Draw skeleton connections
                for connected_name in skeleton.get(point_names[i], []):
                    try:
                        j = self.app_state.point_nti[connected_name]
                        if not np.isnan(points_3d[j][:3]).any():
                            scene.append(Object3D(
                                type='line',
                                coords=np.array([point, points_3d[j, :3]]),
                                color=(128, 128, 128),
                                label=None
                            ))
                    except ValueError:
                        pass

        # # Debug: Print scene info sometimes
        # if self.app_state.frame_idx % 30 == 0:  # every 30 frames
        #     print(f"3D Scene: {len(scene)} objects, {valid_points} valid keypoints")

        return scene

    def _send_results(self, results: dict):
        """Sends rendered results to GUI and clears old data."""

        while not self.results_out_queue.empty():
            try:
                self.results_out_queue.get_nowait()
            except queue.Empty:
                break
        self.results_out_queue.put(results)
