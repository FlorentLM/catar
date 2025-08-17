# -*- coding: utf-8 -*-
"""
Multi-Camera 3D Point Tracking and Calibration using OpenCV and Genetic Algorithms.

This script loads multiple synchronized videos, allows for manual annotation of points,
tracks these points using Lucas-Kanade optical flow, and uses a genetic algorithm
to solve for camera intrinsics, extrinsics, and distortion parameters. The final
output is a 3D reconstruction of the tracked points.

Instructions:
1.  Place your synchronized video files (e.g., cam1.mp4, cam2.mp4) in a 'data' subdirectory.
2.  Run the script: python multi_cam_tracker.py
3.  Use the controls:
    - 'space': Play/Pause
    - 'n': Next frame
    - 'b': Previous frame
    - 'q': Quit
    - '1', '2', ...: Select point P1, P2, ... to annotate.
    - Mouse Click: Annotate the selected point on a video feed.
4.  The script will attempt to solve for camera parameters and show the 3D reconstruction.
"""
import os
import cv2
import numpy as np
import pickle
import json
import glob
import random
import itertools
from pprint import pprint
from tqdm import tqdm
from typing import TypedDict, List, Tuple, Optional
from viz_3d import SceneObject, SceneVisualizer

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# --- Configuration ---
DATA_FOLDER = 'data'
VIDEO_FORMAT = '*.mp4'
SKELETON = {
        "thorax": [ "neck", "leg_f_L0", "leg_f_R0", "leg_m_L0", "leg_m_R0" ],
        "neck": [ "thorax", "a_R0", "a_L0", "eye_L", "eye_R", "m_L0", "m_R0" ],
        "eye_L": [ "neck" ],
        "eye_R": [ "neck" ],
        "a_L0": [ "neck", "a_L1" ],
        "a_L1": [ "a_L2", "a_L0" ],
        "a_L2": [ "a_L1" ],
        "a_R0": [ "neck", "a_R1" ],
        "a_R1": [ "a_R2", "a_R0" ],
        "a_R2": [ "a_R1" ],
        "leg_f_L0": [ "thorax", "leg_f_L1" ],
        "leg_f_L1": [ "leg_f_L2", "leg_f_L0" ],
        "leg_f_L2": [ "leg_f_L1" ],
        "leg_f_R0": [ "thorax", "leg_f_R1" ],
        "leg_f_R1": [ "leg_f_R2", "leg_f_R0" ],
        "leg_f_R2": [ "leg_f_R1" ],
        "leg_m_L0": [ "thorax", "leg_m_L1" ],
        "leg_m_L1": [ "leg_m_L2", "leg_m_L0" ],
        "leg_m_L2": [ "leg_m_L1" ],
        "leg_m_R0": [ "thorax", "leg_m_R1" ],
        "leg_m_R1": [ "leg_m_R2", "leg_m_R0" ],
        "leg_m_R2": [ "leg_m_R1" ],
        "m_L0": [ "neck", "m_L1" ],
        "m_L1": [ "m_L0" ],
        "m_R0": [ "neck", "m_R1" ],
        "m_R1": [ "m_R0" ],
        "s_small": [ "s_large" ],
        "s_large": []
}
GROUND_PLANE_POINTS = [
    "leg_m_R2", "leg_m_L2", "leg_f_R2", "leg_f_L2",
]
GROUND_PLANE_INDICES = np.array([list(SKELETON.keys()).index(p) for p in GROUND_PLANE_POINTS if p in SKELETON.keys()]) # (len(GROUND_PLANE_POINTS),)
POINT_NAMES = list(SKELETON.keys())
NUM_POINTS = len(POINT_NAMES)

# Genetic Algorithm Parameters
POPULATION_SIZE = 400
ELITISM_RATE = 0.1 # Keep the top 10%

# GA State
generation = 0
train_ga = False
population = []
best_fitness_so_far = float('inf')  # Initialize to a very high value
best_individual = None

# --- Global State ---
video_names = []
video_captures = []
video_metadata = {
    'width': 0,
    'height': 0,
    'num_frames': 0,
    'num_videos': 0,
    'fps': 30
}

# Data Structures
# Shape: (num_frames, num_camera, num_points, 2) for (x, y) coordinates
# Using np.nan for un-annotated points
annotations = None
human_annotated = None  # (num_frames, num_camera, num_points) boolean array indicating if a point is annotated
calibration_frames = []  # Frames selected for calibration, empty if not set

# Shape: (num_frames, num_points, 3) for (X, Y, Z) coordinates
reconstructed_3d_points = None
needs_3d_reconstruction = False

# UI and Control State
frame_idx = 300
paused = True
selected_point_idx = 0  # Default to P1
focus_selected_point = False  # Whether to focus on the selected point in the visualization

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(9, 9),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
tracking_enabled = False

# Point colors
point_colors = np.array([
    [255, 0, 0],      # P1 - Red
    [0, 255, 0],      # P2 - Green
    [0, 0, 255],      # P3 - Blue
    [255, 255, 0],    # P4 - Yellow
    [0, 255, 255],    # P5 - Cyan
    [255, 0, 255],    # P6 - Magenta
    [192, 192, 192],  # P7 - Silver
    [255, 128, 0],    # P8 - Orange
    [128, 0, 255],    # P9 - Purple
    [255, 128, 128],  # P10 - Light Red
    [128, 128, 0],    # P11 - Olive
    [0, 128, 128],    # P12 - Teal
    [128, 0, 128],    # P13 - Maroon
    [192, 128, 128],  # P14 - Salmon
    [128, 192, 128],  # P15 - Light Green
    [128, 128, 192],  # P16 - Light Blue
    [192, 192, 128],  # P17 - Khaki
    [192, 128, 192],  # P18 - Plum
    [128, 192, 192],  # P19 - Light Cyan
    [255, 255, 255],  # P20 - White
    [0, 0, 0],        # P21 - Black
    [128, 128, 128],  # P22 - Gray
    [255, 128, 64],   # P23 - Light Orange
    [128, 64, 255],   # P24 - Light Purple
    [210, 105, 30],   # P25 - Chocolate
    [128, 255, 64],   # P26 - Light Yellow
    [128, 64, 0],     # P27 - Brown
    [64, 128, 255]   # P28 - Light Blue
], dtype=np.uint8)

assert NUM_POINTS <= len(point_colors), "Not enough colors defined for the number of points."
    
class CameraParams(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

    def flattened(self) -> np.ndarray:
        """Returns a flattened array of camera parameters for genetic algorithm."""
        return np.concatenate((np.array([self.fx, self.fy, self.cx, self.cy]), self.dist, self.rvec, self.tvec))
    
    def from_flattened(self, flattened: np.ndarray):
        """Creates a CameraParams object from a flattened array."""
        num_dist_coeffs = 5
        self.fx = flattened[0]
        self.fy = flattened[1]
        self.cx = flattened[2]
        self.cy = flattened[3]
        self.dist = flattened[4:4 + num_dist_coeffs]
        self.rvec = flattened[4 + num_dist_coeffs:4 + num_dist_coeffs + 3]
        self.tvec = flattened[4 + num_dist_coeffs + 3:4 + num_dist_coeffs + 6]

def get_camera_matrix(cam_params: CameraParams) -> np.ndarray:
    """Constructs the camera matrix from camera parameters."""
    K = np.array([[cam_params['fx'], 0, cam_params['cx']],
                  [0, cam_params['fy'], cam_params['cy']],
                  [0, 0, 1]], dtype=np.float32)
    return K # (3, 3)

def get_projection_matrix(cam_params: CameraParams) -> np.ndarray:
    """Constructs the projection matrix from camera parameters."""
    K = get_camera_matrix(cam_params)
    R, _ = cv2.Rodrigues(cam_params['rvec'])
    return K @ np.hstack((R, cam_params['tvec'].reshape(-1, 1))) # (3, 4)

def undistort_points(points_2d: np.ndarray, cam_params: CameraParams) -> np.ndarray:
    """Undistorts 2D points using camera parameters."""
    # points_2d: (..., 2)
    valid_points_mask = ~np.isnan(points_2d).any(axis=-1)  # Mask for valid points
    valid_points = points_2d[valid_points_mask]  # Extract valid points (num_valid_points, 2)
    undistorted_full = np.full_like(points_2d, np.nan, dtype=np.float32)  # Prepare output array with NaNs
    # The following function returns normalised coordinates, not pixel coordinates
    undistorted_points = cv2.undistortImagePoints(valid_points.reshape(-1, 1, 2), get_camera_matrix(cam_params), cam_params['dist']) # (num_valid_points, 1, 2)
    undistorted_full[valid_points_mask] = undistorted_points.reshape(-1, 2)  # Fill only valid points
    return undistorted_full  # Shape: (..., 2) with NaNs for invalid points

def combination_triangulate(frame_annotations: np.ndarray, proj_matrices: np.ndarray) -> np.ndarray:
    """Triangulates 3D points from 2D correspondences using multiple camera views."""
    # frame_annotations: (num_frames, num_cams, num_points, 2) and proj_matrices: (num_cams, 3, 4)
    # returns (points_3d: (num_frames, num_points, 3))
    assert frame_annotations.shape[1] == proj_matrices.shape[0], "Number of cameras must match annotations."
    combs = list(itertools.combinations(range(proj_matrices.shape[0]), 2))
    # Every combination makes a prediction, some combinations may not have enough points to triangulate
    points_3d = np.full((frame_annotations.shape[0], len(combs), frame_annotations.shape[2], 3), np.nan, dtype=np.float32)  # (num_frames, num_combs, num_points, 3)
    for idx, (i, j) in enumerate(combs):
        # Get 2D points from both cameras
        p1_2d = frame_annotations[:, i] # (num_frames, num_points, 2)
        p2_2d = frame_annotations[:, j] # (num_frames, num_points, 2)
        common_mask = ~np.isnan(p1_2d).any(axis=2) & ~np.isnan(p2_2d).any(axis=2)  # (num_frames, num_points,)
        if not np.any(common_mask):
            continue
        # Prepare 2D points for triangulation (requires shape [2, N])
        p1_2d = p1_2d[common_mask] # (num_common_points, 2)
        p2_2d = p2_2d[common_mask] # (num_common_points, 2)
        # Expects (3, 4) project matrices and (2, N) points
        points_4d_hom = cv2.triangulatePoints(proj_matrices[i], proj_matrices[j], p1_2d.T, p2_2d.T) # (4, num_common_points) homogenous coordinates
        triangulated_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Convert to 3D coordinates (num_common_points, 3)
        points_3d[:, idx][common_mask] = triangulated_3d
    # Average the triangulated points across all combinations
    average = np.nanmean(points_3d, axis=1)  # (num_frames, num_points, 3)
    return average

def estimate_pose(frame_annotations: np.ndarray, individual: List[CameraParams]):
    """Triangulates 3D points from 2D correspondences using multiple camera views."""
    # frame_annotations: (num_frames, num_cams, num_points, 2) and proj_matrices: (num_cams, 3, 4)
    # returns (points_3d: (num_frames, num_points, 3))
    assert frame_annotations.shape[1] == len(individual), "Number of cameras must match annotations."
    individual[0]['rvec'][:] = 0 # No rotation for the first camera
    individual[0]['tvec'][:] = 0 # No translation for the first camera
    # Construct poses from first 2 cameras
    p1_2d = frame_annotations[:, 0] # (num_frames, num_points, 2)
    p2_2d = frame_annotations[:, 1] # (num_frames, num_points, 2)
    common_mask = ~np.isnan(p1_2d).any(axis=2) & ~np.isnan(p2_2d).any(axis=2)  # (num_frames, num_points,)
    if not np.any(common_mask):
        return
    # Prepare 2D points for triangulation (requires shape [2, N])
    pts1 = p1_2d[common_mask] # (num_common_points, 2)
    pts2 = p2_2d[common_mask] # (num_common_points, 2)
    # Find essential matrix
    _, E, R, t, mask_e = cv2.recoverPose(pts1, pts2, get_camera_matrix(individual[0]), individual[0]['dist'], get_camera_matrix(individual[1]), individual[1]['dist'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # The 'mask' is an output array that specifies which points were considered inliers.
    # We should use only the inliers for further calculations.
    pts1_inliers = pts1[mask_e.ravel() == 1]
    pts2_inliers = pts2[mask_e.ravel() == 1]
    print(f"{np.sum(mask_e)} inliers found out of {len(pts1)} points.\n")
    individual[1]['rvec'] = cv2.Rodrigues(R)[0].flatten()  # Convert rotation matrix to rotation vector
    individual[1]['tvec'] = t.flatten()  # Translation vector
    # Now we can triangulate the points using the first two cameras
    pts1_inliers = undistort_points(pts1_inliers, individual[0])  # (num_common_points, 2)
    pts2_inliers = undistort_points(pts2_inliers, individual[1])  # (num_common_points, 2)
    points_3d = cv2.triangulatePoints(get_projection_matrix(individual[0]), get_projection_matrix(individual[1]), pts1_inliers.T, pts2_inliers.T)  # (4, num_common_points)
    points_3d = (points_3d[:3] / points_3d[3]).T  # Convert to 3D coordinates (num_common_points, 3)
    # Solve for other cameras using PnP
    for i in range(2, len(individual)):
        # # We only use the inliers from the first view that were successfully triangulated.
        pnp_points = frame_annotations[:, i][common_mask][mask_e.ravel() == 1]  # (num_common_points, 2)
        # We do this masking because some points might not be visible in the third camera.
        valid_mask = ~np.isnan(pnp_points).any(axis=1)  # Mask for valid points
        pnp_points = pnp_points[valid_mask]  # (num_valid_points, 2)
        valid_3d_points = points_3d[valid_mask] # (num_valid_points, 3)
        # We use solvePnPRansac to find the pose of the third camera.
        # Distortion coefficients are assumed to be zero as we're using undistorted points.
        success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(valid_3d_points, pnp_points, get_camera_matrix(individual[i]), individual[i]['dist'])
        print(success)
        raise NotImplementedError()

def estimate_independent_pose(frame_annotations: np.ndarray, individual: List[CameraParams]):
    """Triangulates 3D points from 2D correspondences using multiple camera views."""
    # frame_annotations: (num_frames, num_cams, num_points, 2)
    assert frame_annotations.shape[1] == len(individual), "Number of cameras must match annotations."
    individual[0]['rvec'][:] = 0 # No rotation for the first camera
    individual[0]['tvec'][:] = 0 # No translation for the first camera
    # Construct poses from first 2 cameras
    for j in range(1, len(individual)):
        p1_2d = frame_annotations[:, 0] # (num_frames, num_points, 2)
        p2_2d = frame_annotations[:, j] # (num_frames, num_points, 2)
        common_mask = ~np.isnan(p1_2d).any(axis=2) & ~np.isnan(p2_2d).any(axis=2)  # (num_frames, num_points,)
        if not np.any(common_mask):
            return
        # Prepare 2D points for triangulation (requires shape [2, N])
        pts1 = p1_2d[common_mask] # (num_common_points, 2)
        pts2 = p2_2d[common_mask] # (num_common_points, 2)
        # Find essential matrix
        _, E, R, t, mask_e = cv2.recoverPose(pts1, pts2, get_camera_matrix(individual[0]), individual[0]['dist'], get_camera_matrix(individual[j]), individual[j]['dist'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # print(f"{np.sum(mask_e)} inliers found out of {len(pts1)} points.\n")
        individual[j]['rvec'] = cv2.Rodrigues(R)[0].flatten()  # Convert rotation matrix to rotation vector
        individual[j]['tvec'] = t.flatten()  # Translation vector

def reproject_points(points_3d: np.ndarray, cam_params: CameraParams) -> np.ndarray:
    """Reprojects 3D points back to 2D image plane using camera parameters."""
    # points_3d: (N, 3)
    # cam_params: CameraParams
    points_3d = points_3d.reshape(-1, 1, 3)  # Shape: (N, 1, 3)
    reprojected_pts_2d, _ = cv2.projectPoints(
        points_3d, cam_params['rvec'], cam_params['tvec'], get_camera_matrix(cam_params), cam_params['dist']
    )  # (N, 1, 2)
    return reprojected_pts_2d.squeeze(axis=1)  # Shape: (N, 2)


# --- Main Application Logic ---

def load_videos():
    """Loads all videos from the specified data folder."""
    global annotations, reconstructed_3d_points, human_annotated
    video_paths = sorted(glob.glob(os.path.join(DATA_FOLDER, VIDEO_FORMAT)))
    if not video_paths:
        print(f"Error: No videos found in '{DATA_FOLDER}/' with format '{VIDEO_FORMAT}'")
        exit()

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            continue
        video_captures.append(cap)
        video_names.append(os.path.basename(path))

    if not video_captures:
        print("Error: No videos were loaded successfully.")
        exit()

    # Get metadata from the first video (assuming they are synchronized and have same properties)
    video_metadata['num_videos'] = len(video_captures)
    video_metadata['width'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    video_metadata['height'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_metadata['num_frames'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
    video_metadata['fps'] = video_captures[0].get(cv2.CAP_PROP_FPS)

    # Initialize data structures based on metadata
    annotations = np.full((video_metadata['num_frames'], video_metadata['num_videos'], NUM_POINTS, 2), np.nan, dtype=np.float32)
    reconstructed_3d_points = np.full((video_metadata['num_frames'], NUM_POINTS, 3), np.nan, dtype=np.float32)
    human_annotated = np.zeros((video_metadata['num_frames'], video_metadata['num_videos'], NUM_POINTS), dtype=bool)

    print(f"Loaded {video_metadata['num_videos']} videos.")
    print(f"Resolution: {video_metadata['width']}x{video_metadata['height']}, Frames: {video_metadata['num_frames']}")


dragging_kp_idx = None
def mouse_callback(event, x, y, flags, param):
    global selected_point_idx, dragging_kp_idx, needs_3d_reconstruction
    cam_idx = param['cam_idx']
    kp_idx = selected_point_idx
    # print(f"Mouse event: {event}, Position: ({x}, {y}), Cam: {video_names[cam_idx]}, Point: {POINT_NAMES[kp_idx]}")

    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:
        # Check if clicking near an existing point to start dragging
        min_dist = 10 # pixels
        clicked_on_existing = False
        dists = np.linalg.norm(annotations[frame_idx, cam_idx, :] - np.array([x, y]), axis=1) # (num_keypoints,)
        # Find the closest visible keypoint
        dists[np.isnan(dists)] = np.inf # Ignore NaN points
        if np.min(dists) < min_dist:
            clicked_on_existing = True
            # Get the index of the closest keypoint
            closest_kp_idx = np.argmin(dists)
            selected_point_idx = closest_kp_idx
            kp_idx = selected_point_idx
        dragging_kp_idx = kp_idx
        print(f"Dragging {POINT_NAMES[kp_idx]} at ({x}, {y}) in Cam {cam_idx} at frame {frame_idx}")

        if not clicked_on_existing:
            # Add new point
            annotations[frame_idx, cam_idx, kp_idx] = (float(x), float(y))
            human_annotated[frame_idx, cam_idx, kp_idx] = True
            print(f"Annotated {POINT_NAMES[kp_idx]} at ({x}, {y}) in Cam {cam_idx} at frame {frame_idx}")
        needs_3d_reconstruction = True

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_SHIFTKEY:
        if dragging_kp_idx == kp_idx:
            annotations[frame_idx, cam_idx, kp_idx] = (float(x), float(y))
            human_annotated[frame_idx, cam_idx, kp_idx] = True
            needs_3d_reconstruction = True

    elif event == cv2.EVENT_LBUTTONUP:
        print(f"Released {POINT_NAMES[kp_idx]} at ({x}, {y}) in Cam {cam_idx} at frame {frame_idx}")
        if dragging_kp_idx == kp_idx:
            dragging_kp_idx = None
            needs_3d_reconstruction = True

    elif event == cv2.EVENT_RBUTTONDOWN: # Remove point
        annotations[frame_idx, cam_idx, kp_idx] = np.nan
        human_annotated[frame_idx, cam_idx, kp_idx] = False
        needs_3d_reconstruction = True


def track_points(prev_gray, current_gray, cam_idx):
    """Tracks points from previous frame to current frame using Lucas-Kanade."""
    global annotations
    # Get points from the previous frame that are valid
    p0 = annotations[frame_idx - 1, cam_idx, :, :]
    valid_points_indices = ~np.isnan(p0).any(axis=1)
    if focus_selected_point:
        # Only track the selected point
        valid_points_indices[:] = False
        valid_points_indices[selected_point_idx] = True
    
    if not np.any(valid_points_indices):
        return

    p0_valid = p0[valid_points_indices].reshape(-1, 1, 2)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0_valid, None, **lk_params)

    # Update annotations with tracked points
    if p1 is not None and st.any():
        good_new = p1[st == 1]
        
        # Get original indices of good points
        original_indices = np.where(valid_points_indices)[0]
        good_original_indices = original_indices[st.flatten() == 1]

        for i, idx in enumerate(good_original_indices):
             # Only update if the point is not already manually annotated in the current frame
            if np.isnan(annotations[frame_idx, cam_idx, idx]).any() or not human_annotated[frame_idx, cam_idx, idx]:
                annotations[frame_idx, cam_idx, idx] = good_new[i]


# --- Genetic Algorithm for Camera Calibration ---

def create_individual() -> List[CameraParams]:
    """Creates a single individual with a robust lookAt orientation."""
    w, h = video_metadata['width'], video_metadata['height']
    num_cameras = video_metadata['num_videos']
    radius = 5  # Initial guess for camera distance from origin

    individual = []
    for i in range(num_cameras):
        # Intrinsics
        fx = random.uniform(w * 0.8, w * 1.5)
        fy = random.uniform(h * 0.8, h * 1.5)
        cx = w / 2 + random.uniform(-w * 0.05, w * 0.05)
        cy = h / 2 + random.uniform(-h * 0.05, h * 0.05)
        
        # Distortion (keep it small initially)
        dist = np.random.normal(0.0, 0.001, size=5).astype(np.float32)
        
        # 1. Calculate tvec: Position the camera in a circle
        angle = (2 * np.pi / num_cameras) * i
        x = radius * np.cos(angle) + random.uniform(-0.1, 0.1)
        y = 2 + random.uniform(-0.5, 0.5)
        z = radius * np.sin(angle) + random.uniform(-0.1, 0.1)
        cam_in_world = np.array([x, y, z], dtype=np.float32)

        # 2. Calculate rvec: Make the camera "look at" the target
        # Forward vector (from camera to target)

        # The point all cameras are looking at
        target = np.array([0, 0, 0], dtype=np.float32)
        world_up = np.array([0, 1, 0], dtype=np.float32) 
        forward = (target - cam_in_world) / np.linalg.norm(target - cam_in_world)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        cap_up = np.cross(forward, right)

        R = np.array([right, cap_up, forward])
        rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to rotation vector
        tvec = -R @ cam_in_world  # Translation vector to move the camera to the origin

        individual.append(CameraParams(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, rvec=rvec.flatten(), tvec=tvec.flatten()))
        
    return individual

def mutate(individual: List[CameraParams]) -> List[CameraParams]:
    """Mutates an individual by applying small random changes."""
    mutated = []
    alpha = 0.01
    for cam_params in individual:
        # Mutate intrinsics
        fx = cam_params['fx'] + np.random.normal(0, alpha)
        fy = cam_params['fy'] + np.random.normal(0, alpha)
        cx = cam_params['cx'] + np.random.normal(0, alpha)
        cy = cam_params['cy'] + np.random.normal(0, alpha)

        # Mutate distortion
        dist = cam_params['dist'] + np.random.normal(0, alpha, size=cam_params['dist'].shape[0])
        
        # Mutate extrinsics
        rvec = cam_params['rvec'] + np.random.normal(0, np.pi/180, size=3)
        tvec = cam_params['tvec'] + np.random.normal(0, alpha, size=3)
        
        mutated.append(CameraParams(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, rvec=rvec, tvec=tvec))
    
    # Anchor camera 0 to the origin
    # mutated[0]['rvec'] = np.zeros(3, dtype=np.float32)  # No rotation
    # mutated[0]['tvec'] = np.zeros(3, dtype=np.float32)  # No translation
    
    return mutated

def fitness(individual: List[CameraParams], annotations: np.ndarray):
    """
    Refactored fitness function that iterates frame-by-frame and processes points in batches.
    
    This version calculates multiple triangulation points from pairs of cameras for a given valid frame.
    Lower reprojection error results in a higher fitness score.
    """
    reprojection_errors = []

    num_cams = annotations.shape[1]
    
    # --- Pre-computation for Efficiency ---
    # Get projection matrices and camera parameters once to avoid redundant calculations in the loop.
    # **We are ignoring the intrinsics because we undistort the points before triangulation.**
    proj_matrices = np.array([get_projection_matrix(i) for i in individual]) # (num_cams, 3, 4)

    # Find frames with at least one valid annotation to process.
    valid_frames_mask = np.any(~np.isnan(annotations), axis=(1, 2, 3)) & np.any(human_annotated, axis=(1, 2)) # (num_frames,)
    calibration_frames_mask = np.zeros_like(valid_frames_mask, dtype=bool)  # (num_frames,)
    calibration_frames_mask[calibration_frames] = True  # Mark calibration frames as valid
    valid_frames_mask = valid_frames_mask & calibration_frames_mask  # Only consider frames that are both valid and in the calibration set
    valid_annotations = annotations[valid_frames_mask]  # (num_valid_frames, num_cams, num_points, 2)
    undistorted_annotations = np.full_like(valid_annotations, np.nan, dtype=np.float32)  # (num_valid_frames, num_cams, num_points, 2)

    for c in range(num_cams):
        undistorted_annotations[:, c] = undistort_points(valid_annotations[:, c], individual[c])  # (num_valid_frames, num_points, 2)
    
    points_3d = combination_triangulate(undistorted_annotations, proj_matrices) # (num_valid_frames, num_points, 3)
    valid_3d_mask = ~np.isnan(points_3d).any(axis=-1)  # (num_valid_frames, num_points)
    for c in range(num_cams):
        # Reproject 3d points back to 2d for this camera
        valid_2d_mask = ~np.isnan(valid_annotations[:, c]).any(axis=-1)  # (num_valid_frames, num_points)
        common_mask = valid_3d_mask & valid_2d_mask  # Points that are valid in both 3D and 2D
        valid_3d_points = points_3d[common_mask]  # (num_common_points, 3)
        valid_2d_points = valid_annotations[:, c][common_mask]  # (num_common_points, 2)
        reprojected = reproject_points(valid_3d_points, individual[c]) # (num_common_points, 2)
        # Calculate the reprojection error for valid points
        error = np.linalg.norm(reprojected - valid_2d_points, axis=1)  # Euclidean distance, (num_common_points,)
        reprojection_errors.extend(error)  # Append the error for this camera

    if len(reprojection_errors) == 0:
        return float('inf')  # No valid points to evaluate
        
    # average_error = total_reprojection_error / points_evaluated
    average_error = np.sum(reprojection_errors)
    # print("Descriptive statistics of reprojection errors:")
    # print(f"  Min: {np.min(reprojection_errors):.2f}, Max: {np.max(reprojection_errors):.2f}, Mean: {average_error:.2f}, Std Dev: {np.std(reprojection_errors):.2f}")
    
    # Fitness is the error
    return average_error

def calculate_all_reprojection_errors() -> List[dict]:
    """Finds the top_k worst reprojection errors across all frames, cameras, and points."""
    # Check if the camera parameters (best_individual) are available
    if 'best_individual' not in globals() or best_individual is None:
        print("Optimized camera parameters ('best_individual') not found.")
        return []

    # Compute reconstruction
    undistorted_annotations = np.full_like(annotations, np.nan, dtype=np.float32)  # (frames, num_cams, num_points, 2)

    for c in range(video_metadata['num_videos']):
        undistorted_annotations[:, c] = undistort_points(annotations[:, c], best_individual[c])  # (frames, num_points, 2)
    proj_matrices = np.array([get_projection_matrix(i) for i in best_individual])
    points_3d = combination_triangulate(undistorted_annotations, proj_matrices) # (frames, num_points, 3)
    reconstructed_3d_points[:] = points_3d  # Update the global 3D points for this frame

    all_errors = []

    # Pre-calculate masks for valid 3D points and 2D annotations to avoid re-computation
    valid_3d_mask = ~np.isnan(reconstructed_3d_points).any(axis=-1)  # Shape: (num_frames, num_points)
    valid_2d_mask = ~np.isnan(annotations).any(axis=-1)             # Shape: (num_frames, num_cams, num_points)

    # Iterate over each camera to calculate its reprojection errors
    for cam_idx in range(video_metadata['num_videos']):
        # Find points that are valid in both the 3D data and this camera's 2D annotations
        common_mask = valid_3d_mask & valid_2d_mask[:, cam_idx]

        # Get the (frame_idx, point_idx) coordinates for all valid points
        frame_indices, point_indices = np.where(common_mask)

        # If no valid points exist for this camera, skip to the next one
        if frame_indices.size == 0:
            continue

        # Select the corresponding 3D points and 2D ground truth annotations
        points_3d = reconstructed_3d_points[frame_indices, point_indices]
        points_2d = annotations[frame_indices, cam_idx, point_indices]

        # Reproject the 3D points onto the current camera's 2D image plane
        reprojected_points = reproject_points(points_3d, best_individual[cam_idx]) # (num_valid_points, 2)

        valid_reprojection_mask = (reprojected_points[:, 0] >= 0) & (reprojected_points[:, 0] < video_metadata['width']) & \
                                    (reprojected_points[:, 1] >= 0) & (reprojected_points[:, 1] < video_metadata['height'])
        # Filter out points that are outside the image bounds
        reprojected_points = reprojected_points[valid_reprojection_mask]
        points_2d = points_2d[valid_reprojection_mask]
        frame_indices = frame_indices[valid_reprojection_mask]
        point_indices = point_indices[valid_reprojection_mask]

        # Calculate the Euclidean distance (L2 norm) between reprojected and annotated points
        errors = np.square(reprojected_points - points_2d)
        errors = np.sqrt(np.sum(errors, axis=-1))  # Shape: (num_valid_points,)

        # Store each error along with its full context (frame, camera, point index)
        for i in range(len(errors)):
            all_errors.append({
                'error': float(errors[i]),
                'frame': int(frame_indices[i]),
                'camera': video_names[cam_idx],
                'point': POINT_NAMES[point_indices[i]],
                'annotated_point': annotations[frame_indices[i], cam_idx, point_indices[i]].tolist(),
                'reprojected_point': reprojected_points[i].tolist(),
                '3d_point': points_3d[i].tolist()
            })

    # Sort the collected errors in descending order to find the largest ones
    sorted_errors = sorted(all_errors, key=lambda x: x['error'], reverse=True)
    return sorted_errors

def find_worst_frame():
    """Finds the frame with the worst reprojection error."""
    sorted_errors = calculate_all_reprojection_errors()
    frame_errors = {}
    for e in sorted_errors:
        if e['frame'] in calibration_frames:
            continue
        frame_errors.setdefault(e['frame'], 0)
        frame_errors[e['frame']] += e['error']
    sorted_frame_errors = sorted(frame_errors.items(), key=lambda x: x[1], reverse=True) # (frame, total_error)
    print("Worst frames by total reprojection error:")
    pprint(sorted_frame_errors[:10])  # Print top 10 worst frames
    global frame_idx
    frame_idx = sorted_frame_errors[0][0] if sorted_frame_errors else 0

def find_worst_reprojection():
    sorted_errors = calculate_all_reprojection_errors()
    # Compute mean error across all cameras and points
    mean_error = np.mean([e['error'] for e in sorted_errors])
    print(f"Mean reprojection error across all cameras and points: {mean_error:.2f}")
    global frame_idx, selected_point_idx
    if len(sorted_errors) > 0:
        pprint(sorted_errors[0])
        frame_idx = sorted_errors[0]['frame']
        selected_point_idx = POINT_NAMES.index(sorted_errors[0]['point'])

def permutation_optimization(individual: List[CameraParams]):
    """It may at random initialisation the order of cameras are not optimal
    forcing the GA to move a camera across the scene instead of picking one that
    is already in a good position."""
    # Evaluate the fitness of the current individual for every permutation
    perms = list(itertools.permutations(individual))
    fitness_scores = np.array([fitness(list(p), annotations) for p in perms]) # (num_permutations,)
    best_perm = perms[np.argmin(fitness_scores)]
    individual[:] = list(best_perm)  # Update the individual with the best permutation

def run_genetic_step():
    """The main loop for the genetic algorithm to find best camera parameters."""
    global population, best_fitness_so_far, best_individual, generation
    
    # Check if there are enough annotations
    num_annotations = np.sum(~np.isnan(annotations))
    if num_annotations < (2 * NUM_POINTS * 2): # Need at least 2 points in 2 views
        print("Not enough annotations to run calibration. Please annotate more points.")
        return None

    # 1. Initialization
    if len(population) == 0:
        print("Initializing population for Genetic Algorithm...")
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        for i in tqdm(population, desc="Finding optimal initial permutation"):
            permutation_optimization(i)
        best_fitness_so_far = float('inf')  # Initialize to a very high value
        best_individual = None

    # 2. Fitness Evaluation
    fitness_scores = np.array([fitness(ind, annotations) for ind in population]) # (Population,)
    
    # 3. Selection (Elitism + Tournament)
    elite_size = int(POPULATION_SIZE * ELITISM_RATE)
    sorted_population_indices = np.argsort(fitness_scores)  # Sort in ascending order
    
    if fitness_scores[sorted_population_indices[0]] < best_fitness_so_far:
        best_fitness_so_far = fitness_scores[sorted_population_indices[0]]
        best_individual = population[sorted_population_indices[0]]

    print(f"Generation {generation}: Best Fitness (err): {best_fitness_so_far:.2f} Mean Error: {np.nanmean(fitness_scores):.2f} Std Dev: {np.nanstd(fitness_scores):.2f}")

    next_generation = [population[i] for i in sorted_population_indices[:elite_size]]

    # 4. Crossover & Mutation (here simplified to mutation of the best)
    while len(next_generation) < POPULATION_SIZE:
        # Select a parent from the elite group
        parent = random.choice(next_generation[:elite_size])
        # Create a new individual by mutating the parent
        child = mutate(parent)
        next_generation.append(child)
        
    population = next_generation
    generation += 1

def update_3d_reconstruction(best_params: List[CameraParams]):
    """Uses the best camera parameters to reconstruct all 3D points in the current frame."""
    proj_matrices = np.array([get_projection_matrix(i) for i in best_params])
    frame_annotations = annotations[frame_idx]  # (num_cams, num_points, 2)
    undistorted_annotations = np.full_like(frame_annotations, np.nan, dtype=np.float32)  # (num_cams, num_points, 2)
    for c in range(video_metadata['num_videos']):
        undistorted_annotations[c] = undistort_points(frame_annotations[c], best_params[c])  # (num_points, 2)
    points_3d = combination_triangulate(frame_annotations[None], proj_matrices)[0]  # (num_points, 3)
    reconstructed_3d_points[frame_idx] = points_3d  # Update the global 3D points for this frame

# --- Visualization ---

def draw_ui(frame, cam_idx):
    """Draws UI elements on the frame."""
    if best_individual is not None:
        reprojected = reproject_points(reconstructed_3d_points[frame_idx], best_individual[cam_idx])  # (num_points, 2)
    # Draw annotated points for the current frame
    p_idxs = np.arange(NUM_POINTS) if not focus_selected_point else [selected_point_idx]
    for p_idx in p_idxs:
        point = annotations[frame_idx, cam_idx, p_idx]
        if not np.isnan(point).any():
            if human_annotated[frame_idx, cam_idx, p_idx]:
                # Draw a while square around annotated points
                cv2.circle(frame, tuple(point.astype(int)), 5 + 2, (255, 255, 255), -1) # White outline for human
            cv2.circle(frame, tuple(point.astype(int)), 5, point_colors[p_idx].tolist(), -1)
            cv2.putText(frame, POINT_NAMES[p_idx], tuple(point.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 2)
            # Reproject the 3D point back to 2D
            if best_individual is None:
                continue
            point_2d_from_3d = reprojected[p_idx] # (2,)
            if not np.isnan(point_2d_from_3d).any() and (point_2d_from_3d > 0).all():
                # Draw a line from the reprojected point to the annotated point
                cv2.line(frame, tuple(point.astype(int)), tuple(point_2d_from_3d.astype(int)), point_colors[p_idx].tolist(), 1)
                # Euclidean distance text
                distance = np.linalg.norm(point - point_2d_from_3d)
                cv2.putText(frame, f"{distance:.2f}", tuple(point_2d_from_3d.astype(int) + np.array([5, -5])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 1)
    cv2.putText(frame, video_names[cam_idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def create_control_window():
    """Creates the control and info panel."""
    # Display info
    texts = [
        f"Frame: {frame_idx}/{video_metadata['num_frames']}",
        f"Status: {'Paused' if paused else 'Playing'}",
        f"Annotating Point: {POINT_NAMES[selected_point_idx]}",
        f"Focus on Point: {'Enabled' if focus_selected_point else 'Disabled'}",
        f"Best Fitness: {best_fitness_so_far:.2f}",
        f"Num Annotations: {np.sum(~np.isnan(annotations[frame_idx])) // 2} / {NUM_POINTS * len(video_names)}",
        f"Num 3D Points: {np.sum(~np.isnan(reconstructed_3d_points[frame_idx]).any(axis=1))} / {NUM_POINTS}",
        f"Tracking: {'Enabled' if tracking_enabled else 'Disabled'}",
        f"Num Calibration Frames: {len(calibration_frames)}",
        "--- Controls ---",
        "space: play/pause",
        "n/b: next/prev frame",
        "j/k: next/prev keypoint",
        "t: toggle tracking",
        "u: run genetic algorithm",
        "q: quit",
    ]
    w, h = 300, len(texts) * 20 + 20
    control_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, text in enumerate(texts):
        colour = (255, 255, 255)
        if "Annotating Point" in text:
            # Highlight the currently selected point
            colour = point_colors[selected_point_idx].tolist()
        if "Enabled" in text or "Disabled" in text:
            # Highlight tracking status
            colour = (0, 255, 0) if "Enabled" in text else (0, 0, 255)
        cv2.putText(control_img, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        
    cv2.imshow("Controls", control_img)

def reproj_3d(frame: np.ndarray, camera_idx: int):
    """Reprojects 3D points using the given camera parameters and displays the points on a new window."""
    points_3d = reconstructed_3d_points[frame_idx]  # (num_points, 3)
    if np.isnan(points_3d).all():
        print("No 3D points to reproject.")
        return
    points_2d = reproject_points(points_3d, best_individual[camera_idx])  # (num_points, 2)
    for i, point in enumerate(points_2d):
        if not np.isnan(point).any() and (point > 0).all():
            cv2.circle(frame, tuple(point.astype(int)), 5, point_colors[i].tolist(), -1)
            cv2.putText(frame, POINT_NAMES[i], tuple(point.astype(int) + np.array([5, -5])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[i].tolist(), 2)
    cv2.putText(frame, f"{video_names[camera_idx]} Reprojection", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(f"{video_names[camera_idx]} Reprojection", frame)

def create_camera_visual(
    cam_params: CameraParams,
    scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 0),
    label: Optional[str] = None
) -> List[SceneObject]:
    """
    Generates a list of SceneObjects to represent a camera's pose as a pyramid.

    The function computes the camera's world position from its extrinsic parameters
    (which define the world-to-camera transformation) and then transforms a
    canonical pyramid shape from the camera's local coordinates into world coordinates.

    Args:
        cam_params: The camera's parameters, containing rvec and tvec.
        scale: The size of the rendered pyramid.
        color: The color for the camera visualization.
        label: An optional label displayed at the camera's center.

    Returns:
        A list of SceneObjects representing the camera.
    """
    # 1. Get extrinsic parameters (world-to-camera transformation)
    rvec = cam_params['rvec'] # (3,)
    tvec = cam_params['tvec'] # (3,)
    
    # 2. Get the camera-to-world rotation matrix (inverse of world-to-camera)
    R, _ = cv2.Rodrigues(rvec) # (3, 3)

    # 3. Define a canonical camera pyramid in its local coordinate system
    #    (apex at the origin, pointing down the +Z axis)
    #    Note: OpenCV's camera convention is +Y down, +X right.
    w = 0.5 * scale
    h = 0.4 * scale
    depth = 0.8 * scale
    
    # Points in the camera's local coordinate system
    pyramid_points_cam = np.array([
        [0, 0, 0],      # p0: Apex (camera center)
        [-w, -h, depth],# p1: Top-left corner of base
        [w, -h, depth], # p2: Top-right corner
        [w, h, depth],  # p3: Bottom-right corner
        [-w, h, depth], # p4: Bottom-left corner
    ]) # (5, 3) - points in camera local coordinates

    # 4. Transform local camera points to world coordinates.
    # The camera center in the world is C = -R' * t
    # A point in the world is p_w = C + R' * p_c = R' * (p_c - t)
    cam_center_world = -R.T @ tvec # (3,)
    # pyramid_points_world = (R.T @ (pyramid_points_cam - tvec).T).T # (5, 3)
    pyramid_points_world = (pyramid_points_cam - tvec) @ R # (5, 3)

    # 5. Unpack points for clarity
    p0_w, p1_w, p2_w, p3_w, p4_w = pyramid_points_world

    # 6. Create scene objects for the lines and the center point
    scene_objects: List[SceneObject] = []
    
    # Camera center
    # scene_objects.append(SceneObject(type='point', coords=p0_w, color=color, label=label))
    scene_objects.append(SceneObject(type='point', coords=cam_center_world, color=color, label=label))
    
    # Pyramid edges (from apex to base corners)
    scene_objects.append(SceneObject(type='line', coords=np.array([p0_w, p1_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p0_w, p2_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p0_w, p3_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p0_w, p4_w]), color=color, label=None))

    # Pyramid base
    scene_objects.append(SceneObject(type='line', coords=np.array([p1_w, p2_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p2_w, p3_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p3_w, p4_w]), color=color, label=None))
    scene_objects.append(SceneObject(type='line', coords=np.array([p4_w, p1_w]), color=color, label=None))
    
    # Add a line to indicate the 'up' direction (+Y is down in camera coords)
    up_color = (255, 255, 255) # White for the 'up' vector
    scene_objects.append(SceneObject(type='line', coords=np.array([p1_w, p2_w]), color=up_color, label=None))

    return scene_objects

def save_state():
    """Saves the current state of annotations and 3D points."""
    np.save(os.path.join(DATA_FOLDER, 'annotations.npy'), annotations)
    np.save(os.path.join(DATA_FOLDER, 'human_annotated.npy'), human_annotated)
    np.save(os.path.join(DATA_FOLDER, 'reconstructed_3d_points.npy'), reconstructed_3d_points)
    json.dump(calibration_frames, open(os.path.join(DATA_FOLDER, 'calibration_frames.json'), 'w'))
    if best_individual is not None:
        pickle.dump(best_individual, open(os.path.join(DATA_FOLDER, 'best_individual.pkl'), 'wb'))
    print("State saved successfully.")

def load_state():
    """Loads the saved state of annotations and 3D points."""
    global annotations, human_annotated, reconstructed_3d_points, best_individual, best_fitness_so_far, calibration_frames
    try:
        annotations = np.load(os.path.join(DATA_FOLDER, 'annotations.npy'))
        human_annotated = np.load(os.path.join(DATA_FOLDER, 'human_annotated.npy'))
        reconstructed_3d_points = np.load(os.path.join(DATA_FOLDER, 'reconstructed_3d_points.npy'))
        calibration_frames = json.load(open(os.path.join(DATA_FOLDER, 'calibration_frames.json')))
        best_individual = pickle.load(open(os.path.join(DATA_FOLDER, 'best_individual.pkl'), 'rb'))
        best_fitness_so_far = fitness(best_individual, annotations)  # Recalculate fitness
        print("State loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading state: {e}")

# --- Main Loop ---
def main():
    global frame_idx, paused, selected_point_idx, annotations, train_ga, population, best_individual
    global needs_3d_reconstruction, tracking_enabled, focus_selected_point, best_fitness_so_far
    global save_output_video

    load_videos()
    load_state()  # Load previous state if available

    # Create windows and set mouse callbacks
    # Arrange windows in a grid
    grid_cols = int(np.ceil(np.sqrt(video_metadata['num_videos'])))
    # grid_rows = int(np.ceil(video_metadata['num_videos'] / grid_cols))
    win_w = 700
    win_h = 550    
    for i in range(video_metadata['num_videos']):
        win_name = video_names[i]
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback, {'cam_idx': i})
        row = i // grid_cols
        col = i % grid_cols
        x = col * win_w
        y = row * win_h
        cv2.moveWindow(win_name, x, y)

    prev_frames = [None] * video_metadata['num_videos']
    prev_frame_idx = -1
    scene = []
    scene_viz = SceneVisualizer(frame_size=(video_metadata['width'], video_metadata['height']))
    show_3d_viz = True

    video_save_output = None
    last_written_frame = -1
    num_columns = 3
    num_videos = len(video_names) + 1 # +1 for the 3D visualization
    num_rows = num_videos // num_columns + (1 if num_videos % num_columns > 0 else 0)
    video_recording_buffer = np.zeros((video_metadata['height']*num_rows, video_metadata['width']*num_columns, 3), dtype=np.uint8)

    while True:
        # Set all captures to the current frame index
        current_frames = []
        if prev_frame_idx != frame_idx:
            for i, cap in enumerate(video_captures):
                if prev_frame_idx != frame_idx-1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error.")
                    paused = True
                    frame_idx = max(0, frame_idx -1) # Go back one frame if at end
                    continue
                current_frames.append(frame)

            needs_3d_reconstruction = True  # Tracking may change points
            # If not paused and not the first frame, track points
            if tracking_enabled and prev_frame_idx != -1 and prev_frame_idx == frame_idx - 1:
                for i in range(video_metadata['num_videos']):
                    prev_gray = cv2.cvtColor(prev_frames[i].copy(), cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.cvtColor(current_frames[i].copy(), cv2.COLOR_BGR2GRAY)
                    track_points(prev_gray, gray_frame, i)
                if focus_selected_point:
                    # Check if the selected point is annotated in the current frame in at least two cameras
                    count = np.sum(~np.isnan(annotations[frame_idx, :, selected_point_idx, 0]))
                    if count < 2:
                        print(f"Selected point {POINT_NAMES[selected_point_idx]} is not annotated in at least two cameras. Please annotate more points.")
                        paused = True

            prev_frame_idx = frame_idx
            prev_frames = current_frames
        else:
            current_frames = prev_frames  # Use previous frames if not changing index

        if needs_3d_reconstruction and best_individual is not None:
            needs_3d_reconstruction = False
            update_3d_reconstruction(best_individual)
            # Debugging reprojection
            # reproj_3d(current_frames[2].copy(), 2)
            scene = []
            for i, cam in enumerate(best_individual):
                # Create a camera visualization object
                cam_viz = create_camera_visual(
                    cam_params=cam,
                    scale=1.0,
                    color=point_colors[i % len(point_colors)],
                    label=video_names[i]
                )
                scene.extend(cam_viz)
            # Draw 3d points
            points_3d = reconstructed_3d_points[frame_idx]  # (num_points, 3)
            for i, point in enumerate(points_3d):
                if not np.isnan(point).any():
                    scene.append(SceneObject(type='point', coords=point, color=point_colors[i % len(point_colors)], label=POINT_NAMES[i]))
                from_name = POINT_NAMES[i] # e.g. thorax
                for to in SKELETON[from_name]:
                    # e.g. to = neck
                    to_id = POINT_NAMES.index(to)
                    if not np.isnan(point).any() and not np.isnan(points_3d[to_id]).any():
                        scene.append(SceneObject(
                            type='line',
                            coords=np.array([point, points_3d[to_id]]),
                            color=point_colors[i % len(point_colors)],
                            label=None
                        ))

        # Draw UI on frame
        if not scene_viz.is_dragging:
            for i, frame in enumerate(current_frames):
                frame_with_ui = draw_ui(frame.copy(), i)
                # Place the frame in the recording buffer
                row = i // num_columns
                col = i % num_columns
                y_start = row * video_metadata['height']
                x_start = col * video_metadata['width']
                video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = frame_with_ui
                cv2.imshow(video_names[i], frame_with_ui)

        # prev_gray_frames = current_gray_frames
        if train_ga:
            run_genetic_step()
            needs_3d_reconstruction = True
        
        # Update control and 3D plot windows
        create_control_window()
        if show_3d_viz:
            viz_3d_frame = scene_viz.draw_scene(scene)
            # Place the 3D visualization in the recording buffer
            row = (num_videos-1) // num_columns
            col = (num_videos-1) % num_columns
            y_start = row * video_metadata['height']
            x_start = col * video_metadata['width']
            video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = viz_3d_frame
            cv2.imshow(scene_viz.window_name, viz_3d_frame)
        
        if save_output_video and video_save_output is not None and last_written_frame != frame_idx:
            # Save the current frame to the video output
            video_save_output.write(video_recording_buffer)
            last_written_frame = frame_idx

        # --- Keyboard Controls ---
        # key = cv2.waitKey(1 if not paused else 0) & 0xFF
        # wait_time = 25 if train_ga else 500
        # if scene_viz.is_dragging:
        #     wait_time = 1  # Fast updates while dragging
        wait_time = 10
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('v'):
            show_3d_viz = not show_3d_viz
            if show_3d_viz:
                print("Showing 3D visualization.")
            else:
                print("Hiding 3D visualization.")
                cv2.destroyWindow(scene_viz.window_name)
        elif key == ord('s'):
            save_state()
        elif key == ord('l'):
            load_state()
        elif key == ord('h'):
            if focus_selected_point:
                human_annotated[:frame_idx + 1, :, selected_point_idx] = True
                print(f"Marked all previous frames as human-annotated for point {POINT_NAMES[selected_point_idx]}.")
        elif key == ord('f'):
            focus_selected_point = not focus_selected_point
            print(f"Focus on selected point {'enabled' if focus_selected_point else 'disabled'}.")
        elif key == ord('n'):
            paused = True
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
        elif key == ord('b'):
            paused = True
            if frame_idx > 0:
                frame_idx -= 1
        elif key == ord('d'):
            if focus_selected_point:
                annotations[frame_idx:, :, selected_point_idx] = np.nan
                human_annotated[frame_idx:, :, selected_point_idx] = False
                print(f"Deleted annotations for point {POINT_NAMES[selected_point_idx]} from frame {frame_idx} onwards.")
        elif key == ord('g'):
            goto_frame_number = input("Enter frame number to go to: ")
            try:
                goto_frame_number = int(goto_frame_number)
                if 0 <= goto_frame_number < video_metadata['num_frames']:
                    frame_idx = goto_frame_number
                    paused = True
                else:
                    print(f"Frame number must be between 0 and {video_metadata['num_frames'] - 1}.")
            except ValueError:
                print("Invalid frame number. Please enter a valid integer.")
        elif key == ord('u'):
            paused = True
            train_ga = not train_ga
        elif key == ord('r'):
            save_output_video = not save_output_video
            print(f"Output video recording {'enabled' if save_output_video else 'disabled'}.")
            if save_output_video:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'hvc1')
                output_filename = 'recording.mp4'
                video_save_output = cv2.VideoWriter(output_filename, fourcc, 30.0, 
                                                    (video_metadata['width'] * num_columns, 
                                                     video_metadata['height'] * num_rows))
                print(f"Output video will be saved to {output_filename}.")
        elif key == ord('p'):
            population = []
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}.")
        elif ord('1') <= key <= ord('9'):
            point_num = key - ord('1')
            if point_num < NUM_POINTS:
                selected_point_idx = point_num
                print(f"Selected point P{selected_point_idx} for annotation.")
        elif key == ord('w'):
            find_worst_reprojection()
        elif key == ord('e'):
            find_worst_frame()
        # Cycle through points
        elif key == ord('j'):
            selected_point_idx = (selected_point_idx + 1) % NUM_POINTS
            print(f"Selected point P{selected_point_idx} for annotation.")
        elif key == ord('k'):
            selected_point_idx = (selected_point_idx - 1) % NUM_POINTS
            print(f"Selected point P{selected_point_idx} for annotation.")
        elif key == ord('c'):
            # Toggle calibration frame for the current frame
            if frame_idx in calibration_frames:
                calibration_frames.remove(frame_idx)
                print(f"Frame {frame_idx} removed from calibration frames.")
            else:
                calibration_frames.append(frame_idx)
                print(f"Frame {frame_idx} added to calibration frames.")

        # If playing, advance frame
        if not paused:
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
            else:
                paused = True # Pause at the end

    # Cleanup
    for cap in video_captures:
        cap.release()
    if video_save_output is not None:
        video_save_output.release()
    cv2.destroyAllWindows()
    print("Exiting application.")


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos there.")
    main()
