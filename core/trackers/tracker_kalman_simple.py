import numpy as np
import cv2
from typing import Dict, Optional
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from lucida.geometry.backend import xp
from core.trackers.trackers_base import BaseTracker

from state import AppState
from utils import snap_to_ray, robust_triangulate


class KalmanFilter3D:
    """Standard Constant Velocity 3D Kalman Filter."""

    def __init__(self, dt=1.0):
        # State vector: [x, y, z, vx, vy, vz]
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # Transition matrix (Constant Velocity model)
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

        # Measurement matrix (We measure x, y, z)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])

        # Initial Covariance (High uncertainty)
        self.kf.P *= 100.0
        # Measurement Noise
        self.kf.R *= 1.0
        # Process Noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.5, block_size=3)

        self.initialized = False

    def predict(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3)
        self.kf.predict()
        return self.kf.x[:3].flatten()

    def update(self, measurement: np.ndarray, is_forced: bool = False, velocity: Optional[np.ndarray] = None):
        if not self.initialized or is_forced:
            self.kf.x[:3] = measurement.reshape(3, 1)
            if velocity is not None:
                self.kf.x[3:] = velocity.reshape(3, 1)
            else:
                self.kf.x[3:] = 0
            self.kf.P = np.eye(6) * 10.0
            self.initialized = True
        else:
            self.kf.update(measurement)

    def kill(self):
        self.initialized = False
        self.kf.x[:] = 0


class SimpleTracker(BaseTracker):
    def __init__(self, app_state: 'AppState'):
        super().__init__(app_state)

        self.filters: Dict[str, KalmanFilter3D] = {
            pt: KalmanFilter3D() for pt in app_state.point_names
        }
        self.max_reproj_error = 15.0
        self.min_cameras_consensus = 2

    def initialize_tracks(self, frame_idx: int, direction: int = 1):
        """
        Initialises tracker state from a specific frame (keyframe).
        """
        print(f"[Tracker] Initializing constraints from keyframe {frame_idx}...")

        prev_idx = frame_idx - direction
        if prev_idx < 0: prev_idx = 0
        if prev_idx >= self.app_state.num_frames: prev_idx = self.app_state.num_frames - 1

        with self.app_state.data.read_lock():
            annots = self.app_state.data.get_frame_annotations(frame_idx, copy=True)
            current_3d_state = self.app_state.data.get_frame_points3d(frame_idx, copy=True)
            prev_3d_state = self.app_state.data.get_frame_points3d(prev_idx, copy=True)

        rig = self.app_state.rig
        if len(rig) == 0:
            return

        for p_idx, p_name in enumerate(self.app_state.point_names):
            kf = self.filters[p_name]
            valid_cams = np.where(~np.isnan(annots[:, p_idx, 0]))[0]

            if len(valid_cams) == 0:
                kf.kill()
                continue

            pt_3d = None

            # Case A: Multi-view triangulation
            if len(valid_cams) >= 2:
                points_xp = xp.asarray(annots[valid_cams, p_idx, :2])
                pt_3d_xp, _ = robust_triangulate(rig, points_xp, valid_cams)
                if pt_3d_xp is not None:
                    pt_3d = np.asarray(pt_3d_xp)

            # Case B: Single-view snap
            if pt_3d is None and len(valid_cams) == 1:
                cam_idx = valid_cams[0]
                uv = annots[cam_idx, p_idx, :2]

                candidate_depth_source = None
                old_3d = current_3d_state[p_idx, :3]

                # Check 1: old 3D
                if not np.isnan(old_3d).any():
                    reproj = rig[rig.names[cam_idx]].project(old_3d.reshape(1, 3)).flatten()
                    if np.linalg.norm(reproj - uv) < 10.0:
                        candidate_depth_source = old_3d

                # Check 2: prev 3D
                if candidate_depth_source is None:
                    prev_3d = prev_3d_state[p_idx, :3]
                    if not np.isnan(prev_3d).any():
                        candidate_depth_source = prev_3d

                if candidate_depth_source is not None:
                    pt_3d = snap_to_ray(rig, cam_idx, uv, candidate_depth_source)

            # Velocity warm start
            initial_velocity = None
            if pt_3d is not None:
                prev_3d = prev_3d_state[p_idx, :3]
                if not np.isnan(prev_3d).any():
                    initial_velocity = (pt_3d - prev_3d) * direction

            if pt_3d is not None and not np.isnan(pt_3d).any():
                kf.update(pt_3d, is_forced=True, velocity=initial_velocity)
            else:
                kf.kill()

    def track_frame(self, frame_idx, prev_frame_idx, source_frames, dest_frames, active_point_indices=None):

        with self.app_state.data.read_lock():
            prev_annots = self.app_state.data.get_frame_annotations(prev_frame_idx, copy=True)
            dest_existing = self.app_state.data.get_frame_annotations(frame_idx, copy=True)
            dest_human_flags = self.app_state.data.get_human_annotated_flags(frame_idx, copy=True)
            prev_3d_points = self.app_state.data.get_frame_points3d(prev_frame_idx, copy=True)
            existing_3d_full = self.app_state.data.get_frame_points3d(frame_idx, copy=True)

        current_annots = dest_existing.copy()
        current_3d = existing_3d_full.copy()
        rig = self.app_state.rig

        indices_to_track = active_point_indices if active_point_indices else range(self.app_state.num_points)

        for p_idx in indices_to_track:
            p_name = self.app_state.point_names[p_idx]
            kf = self.filters[p_name]

            # Optical flow
            valid_cams_prev = np.where(~np.isnan(prev_annots[:, p_idx, 0]))[0]
            raw_2d_proposals = {}

            for cam_idx in valid_cams_prev:
                if dest_human_flags[cam_idx, p_idx]:
                    raw_2d_proposals[cam_idx] = dest_existing[cam_idx, p_idx, :2]
                    continue

                p0 = prev_annots[cam_idx, p_idx, :2].reshape(-1, 1, 2)
                img_prev = cv2.cvtColor(source_frames[cam_idx], cv2.COLOR_BGR2GRAY)
                img_curr = cv2.cvtColor(dest_frames[cam_idx], cv2.COLOR_BGR2GRAY)

                p1, st, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, p0, None, winSize=(21, 21), maxLevel=3)
                if st[0][0] == 1:
                    raw_2d_proposals[cam_idx] = p1.flatten()

            # Inject human annpotations
            has_human_input = False
            for cam_idx in range(len(rig)):
                if dest_human_flags[cam_idx, p_idx]:
                    raw_2d_proposals[cam_idx] = dest_existing[cam_idx, p_idx, :2]
                    has_human_input = True

            #Solve 3d
            final_3d_pt = None
            inlier_cams = []

            if len(raw_2d_proposals) >= 2:
                cams = list(raw_2d_proposals.keys())
                pts = np.array(list(raw_2d_proposals.values()))
                pts_xp = xp.asarray(pts)
                pt_3d_xp, inliers_xp = robust_triangulate(rig, pts_xp, cams)

                if pt_3d_xp is not None:
                    final_3d_pt = np.asarray(pt_3d_xp)
                    inlier_cams = inliers_xp

            if final_3d_pt is None and has_human_input:
                for c, pt in raw_2d_proposals.items():
                    if dest_human_flags[c, p_idx]:
                        target_depth_pt = kf.predict()
                        if np.allclose(target_depth_pt, 0):
                            target_depth_pt = prev_3d_points[p_idx, :3]

                        if not np.isnan(target_depth_pt).any() and not np.allclose(target_depth_pt, 0):
                            final_3d_pt = snap_to_ray(rig, c, pt, target_depth_pt)
                            inlier_cams = [c]
                            kf.update(final_3d_pt, is_forced=True)
                        break

            # Kalman update and reproject
            if final_3d_pt is not None and not np.isnan(final_3d_pt).any():
                if not has_human_input:
                    kf.update(final_3d_pt)
                valid_3d_pos = final_3d_pt

                current_3d[p_idx, :3] = valid_3d_pos
                current_3d[p_idx, 3] = 1.0

                reproj_np = np.asarray(rig.project(valid_3d_pos.reshape(1, 3))).reshape(len(rig), 2)

                for cam_idx in range(len(rig)):
                    if dest_human_flags[cam_idx, p_idx]:
                        current_annots[cam_idx, p_idx] = dest_existing[cam_idx, p_idx]
                        continue

                    if cam_idx in inlier_cams:
                        current_annots[cam_idx, p_idx, :2] = raw_2d_proposals[cam_idx]
                        current_annots[cam_idx, p_idx, 2] = 1.0
                    else:
                        rep = reproj_np[cam_idx]
                        w, h = self.app_state.frame_width, self.app_state.frame_height
                        if 0 <= rep[0] < w and 0 <= rep[1] < h:
                            current_annots[cam_idx, p_idx, :2] = rep
                            current_annots[cam_idx, p_idx, 2] = 0.6
                        else:
                            current_annots[cam_idx, p_idx] = np.nan
            else:
                kf.kill()
                current_3d[p_idx] = np.nan
                for cam_idx in range(len(rig)):
                    if not dest_human_flags[cam_idx, p_idx]:
                        current_annots[cam_idx, p_idx] = np.nan

        with self.app_state.data.bulk_lock():
            self.app_state.data.set_frame_annotations(frame_idx, current_annots, human_flags=dest_human_flags)
            self.app_state.data.set_frame_points3d(frame_idx, current_3d)

        return True

    def get_debug_info(self):
        # TODO: proper debug info
        return {}