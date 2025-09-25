# -*- coding: utf-8 -*-
"""
Multi-Camera 3D Point Tracking and Calibration using DearPyGui and OpenCV

This script provides a DearPyGui interface for the multi-camera 3D point tracking and calibration project.
It loads multiple synchronized videos, allows for manual annotation of points, tracks these points using
Lucas-Kanade optical flow, and uses a genetic algorithm to solve for camera intrinsics, extrinsics, and
distortion parameters. The final output is a 3D reconstruction of the tracked points.
"""
import cv2
import numpy as np
import pathlib
import pickle
import json
import glob
import random
import itertools
from pprint import pprint
from tqdm import tqdm
from typing import TypedDict, List, Tuple, Optional
from viz_3d import SceneObject, SceneVisualizer
import dearpygui.dearpygui as dpg
import tkinter as tk
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from segment_anything import sam_model_registry, SamPredictor
import torch

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# --- Configuration ---
DATA_FOLDER = pathlib.Path.cwd() / 'data'
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
GRID_COLS = 3  # Fixed number of columns to display the videos etc. for now

# Genetic Algorithm Parameters
POPULATION_SIZE = 200
ELITISM_RATE = 0.1 # Keep the top 10%

# GA State
generation = 0
train_ga = False
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
current_frames = []

# Data Structures
# Shape: (num_frames, num_camera, num_points, 2) for (x, y) coordinates
# Using np.nan for un-annotated points
annotations = None
human_annotated = None  # (num_frames, num_camera, num_points) boolean array indicating if a point is annotated
calibration_frames = []  # Frames selected for calibration, empty if not set

# Shape: (num_frames, num_points, 3) for (X, Y, Z) coordinates
reconstructed_3d_points = None
needs_3d_reconstruction = False

# Seed mesh reconstruction state
seed_points_2d = []
reconstructed_seed_mesh = None
sam_predictor = None
auto_segment_mode = False
SAM_MODEL_PATH = DATA_FOLDER / "sam_vit_b_01ec64.pth"
seed_mesh_poses: np.ndarray = None  # (num_frames, 6) for (tvec, rvec) which is (tx, ty, tz, rx, ry, rz)

# UI and Control State
frame_idx = 300
paused = True
selected_point_idx = 0  # Default to P1
focus_selected_point = False  # Whether to focus on the selected point in the visualisation
save_output_video = False
video_save_output = None

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(9, 9),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
keypoint_tracking_enabled = False
seed_pose_tracking_enabled = False

point_colors = np.array([
    [255, 0, 0],       # P1 - Red
    [0, 255, 0],       # P2 - Green
    [0, 0, 255],       # P3 - Blue
    [255, 255, 0],     # P4 - Yellow
    [0, 255, 255],     # P5 - Cyan
    [255, 0, 255],     # P6 - Magenta
    [192, 192, 192],   # P7 - Silver
    [255, 128, 0],     # P8 - Orange
    [128, 0, 255],     # P9 - Purple
    [255, 128, 128],   # P10 - Light Red
    [128, 128, 0],     # P11 - Olive
    [0, 128, 128],     # P12 - Teal
    [128, 0, 128],     # P13 - Maroon
    [192, 128, 128],   # P14 - Salmon
    [128, 192, 128],   # P15 - Light Green
    [128, 128, 192],   # P16 - Light Blue
    [192, 192, 128],   # P17 - Khaki
    [192, 128, 192],   # P18 - Plum
    [128, 192, 192],   # P19 - Light Cyan
    [255, 255, 255],   # P20 - White
    [0, 0, 0],         # P21 - Black
    [128, 128, 128],   # P22 - Gray
    [255, 128, 64],    # P23 - Light Orange
    [128, 64, 255],    # P24 - Light Purple
    [210, 105, 30],    # P25 - Chocolate
    [128, 255, 64],    # P26 - Light Yellow
    [128, 64, 0],      # P27 - Brown
    [64, 128, 255]     # P28 - Light Blue
], dtype=np.uint8)

assert NUM_POINTS <= len(point_colors), "Not enough colors defined for the number of points."

NUM_DIST_COEFFS = 14  # Number of distortion coefficients
class CameraParams(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

def flat_camera_params(cam: CameraParams) -> np.ndarray:
    """Returns a flattened array of camera parameters for genetic algorithm."""
    """Flattens the CameraParams into a single array."""
    return np.concatenate([
        np.array([cam['fx'], cam['fy'], cam['cx'], cam['cy']], dtype=np.float32),
        cam['dist'][:NUM_DIST_COEFFS].astype(np.float32),
        cam['rvec'].astype(np.float32),
        cam['tvec'].astype(np.float32)
    ])

def unflat_camera_params(cam: CameraParams, flattened: np.ndarray):
    """Creates a CameraParams object from a flattened array."""
    """Unflattens a single array into CameraParams."""
    cam['fx'] = flattened[0]
    cam['fy'] = flattened[1]
    cam['cx'] = flattened[2]
    cam['cy'] = flattened[3]
    cam['dist'] = flattened[4:4 + NUM_DIST_COEFFS].astype(np.float32)
    cam['rvec'] = flattened[4 + NUM_DIST_COEFFS:4 + NUM_DIST_COEFFS + 3].astype(np.float32)
    cam['tvec'] = flattened[4 + NUM_DIST_COEFFS + 3:4 + NUM_DIST_COEFFS + 6].astype(np.float32)

def flat_individual(individual: List[CameraParams]) -> np.ndarray:
    """Flattens a list of CameraParams into a single array."""
    return np.concatenate([flat_camera_params(cam) for cam in individual])

def unflat_individual(flattened: np.ndarray) -> List[CameraParams]:
    """Unflattens a single array into a list of CameraParams."""
    cam_params_list = []
    offset = 0
    num_cameras = video_metadata['num_videos']
    for _ in range(num_cameras):
        cam_params = CameraParams(
            fx=flattened[offset],
            fy=flattened[offset + 1],
            cx=flattened[offset + 2],
            cy=flattened[offset + 3],
            dist=flattened[offset + 4:offset + 4 + NUM_DIST_COEFFS],
            rvec=flattened[offset + 4 + NUM_DIST_COEFFS:offset + 4 + NUM_DIST_COEFFS + 3],
            tvec=flattened[offset + 4 + NUM_DIST_COEFFS + 3:offset + 4 + NUM_DIST_COEFFS + 6]
        )
        cam_params_list.append(cam_params)
        offset += (4 + NUM_DIST_COEFFS + 6)  # Move to the next camera's parameters
    return cam_params_list

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
    if not np.any(valid_points_mask):
        return np.full_like(points_2d, np.nan, dtype=np.float32)
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

def reproject_points(points_3d: np.ndarray, cam_params: CameraParams) -> np.ndarray:
    """Reprojects 3D points back to 2D image plane using camera parameters."""
    # points_3d: (N, 3)
    # cam_params: CameraParams
    points_3d = points_3d.reshape(-1, 1, 3)  # Shape: (N, 1, 3)
    reprojected_pts_2d, _ = cv2.projectPoints(
        points_3d, cam_params['rvec'], cam_params['tvec'], get_camera_matrix(cam_params), cam_params['dist']
    )  # (N, 1, 2)
    if reprojected_pts_2d is None:
        raise ValueError("Reprojection failed, check camera parameters and 3D points.")
    return reprojected_pts_2d.squeeze(axis=1)  # Shape: (N, 2)

# --- Main Application Logic ---

def load_videos():
    """Loads all videos from the specified data folder."""
    global annotations, reconstructed_3d_points, human_annotated, seed_points_2d, seed_mesh_poses
    video_paths = sorted(DATA_FOLDER.glob(VIDEO_FORMAT))
    if not video_paths:
        print(f"Error: No videos found in '{DATA_FOLDER}/' with format '{VIDEO_FORMAT}'")
        exit()

    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            continue
        video_captures.append(cap)
        video_names.append(path.name)

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
    seed_points_2d = [[] for _ in range(video_metadata['num_videos'])]
    seed_mesh_poses = np.zeros((video_metadata['num_frames'], 6), dtype=np.float32)  # (num_frames, 6) for (tx, ty, tz, rx, ry, rz)

    print(f"Loaded {video_metadata['num_videos']} videos.")
    print(f"Resolution: {video_metadata['width']}x{video_metadata['height']}, Frames: {video_metadata['num_frames']}")

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


# --- Genetic algorithm for camera calibration ---

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
        dist = np.random.normal(0.0, 0.001, size=NUM_DIST_COEFFS).astype(np.float32)
        # Calculate tvec: position the camera in a circle
        angle = (2 * np.pi / num_cameras) * i
        x = radius * np.cos(angle) + random.uniform(-0.1, 0.1)
        y = 2 + random.uniform(-0.5, 0.5)
        z = radius * np.sin(angle) + random.uniform(-0.1, 0.1)
        cam_in_world = np.array([x, y, z], dtype=np.float32)
        # Calculate rvec: make the camera "look at" the target
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
    # Get projection matrices and camera parameters once to avoid redundant calculations in the loop.
    # We are ignoring the intrinsics because we undistort the points before triangulation.
    proj_matrices = np.array([get_projection_matrix(i) for i in individual]) # (num_cams, 3, 4)

    if len(calibration_frames) == 0:
        dpg.set_value("status_message", "No calibration frames found / selected.")
        dpg.show_item("status_message")
        return float('inf')
    
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
    return average_error  # Fitness is the error

def calculate_all_reprojection_errors() -> List[dict]:
    """Finds the top_k worst reprojection errors across all frames, cameras, and points."""
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
    valid_2d_mask = ~np.isnan(annotations).any(axis=-1)  # Shape: (num_frames, num_cams, num_points)

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
    """Finds the frame with the worst total reprojection error."""
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
    dpg.set_value("frame_slider", frame_idx)
    message = (f"Worst frames by total reprojection error (frame, error):\n"
               f"{np.round(sorted_frame_errors[:10],2)}"
    )
    dpg.set_value("status_message", message)
    dpg.show_item("status_message")

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
        dpg.set_value("frame_slider", frame_idx)
        dpg.set_value("point_combo", POINT_NAMES[selected_point_idx])
        message = (f"Mean reprojection error across all cameras and points for frame {frame_idx}: {mean_error:.2f}\n"
                   f"Worst keypoint in frame: '{POINT_NAMES[selected_point_idx]}'\n"
                   f"For camera: {sorted_errors[0]['camera']}"
        )
        dpg.set_value("status_message", message)
        dpg.show_item("status_message")

def permutation_optimization(individual: List[CameraParams]):
    """It may at random initialisation the order of cameras are not optimal
    forcing the GA to move a camera across the scene instead of picking one that
    is already in a good position."""
    # Evaluate the fitness of the current individual for every permutation
    perms = list(itertools.permutations(individual))
    fitness_scores = np.array([fitness(list(p), annotations) for p in perms]) # (num_permutations,)
    best_perm = perms[np.argmin(fitness_scores)]
    individual[:] = list(best_perm)  # Update the individual with the best permutation

mean_params = None
def run_genetic_step():
    """The main loop for the genetic algorithm to find best camera parameters."""
    global best_fitness_so_far, best_individual, generation, mean_params

    # Check if there are enough annotations
    num_annotations = np.sum(~np.isnan(annotations))
    if num_annotations < (2 * NUM_POINTS * 2): # Need at least 2 points in 2 views
        print("Not enough annotations to run calibration. Please annotate more points.")
        return None

    # Fitness evaluation
    std_dev = 0.001
    if best_individual is not None:
        mean_params = flat_individual(best_individual)  # Flatten the best individual parameters
    else:
        dpg.set_value("ga_status_text", "Finding optimal initial permutation... Re-fitting will start automatically.")
        dpg.set_value("ga_progress_text", "")
        dpg.render_dearpygui_frame()
        # Initialise the population with random individuals
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        for i in tqdm(population, desc="Finding optimal initial permutation"):
            permutation_optimization(i)
        best_fitness_so_far = float('inf')  # Initialise to a very high value
        best_individual = None
        pop_fitness = np.array([fitness(ind, annotations) for ind in population])  # (Population,)
        mean_params = flat_individual(population[np.argmin(pop_fitness)])  # Get the best parameters from the population
        dpg.set_value("ga_status_text", "Running genetic algorithm...")

    num_params = mean_params.shape[0]  # Number of parameters in an individual
    noise = np.random.normal(0, std_dev, size=(POPULATION_SIZE, num_params))  # (Population, num_params)
    pop_params = noise + mean_params # Add noise to the best parameters for exploration (Population, num_params)
    fitness_scores = np.zeros(POPULATION_SIZE, dtype=np.float32)  # (Population,)
    temp_individual = create_individual()
    for i in range(POPULATION_SIZE):
        temp_individual = unflat_individual(pop_params[i])  # Unflatten the parameters
        fitness_scores[i] = fitness(temp_individual, annotations)  # Calculate fitness for the individual

    # Selection (elitism + tournament)
    sorted_population_indices = np.argsort(fitness_scores)  # Sort in ascending order
    if fitness_scores[sorted_population_indices[0]] < best_fitness_so_far:
        best_fitness_so_far = fitness_scores[sorted_population_indices[0]]
        best_individual = unflat_individual(pop_params[sorted_population_indices[0]])  # Update the best individual
    dpg.set_value("ga_progress_text", f"Generation {generation}: Best fitness (err): {best_fitness_so_far:.2f} Mean error: {np.nanmean(fitness_scores):.2f} SD: {np.nanstd(fitness_scores):.2f}")
    normalised_scores = (fitness_scores - np.nanmean(fitness_scores)) / (np.nanstd(fitness_scores) + 1e-8)  # Normalize scores
    # Update based on evolution strategy
    mean_params = mean_params + 0.1 * np.sum(noise * -normalised_scores[:, None], axis=0) / (POPULATION_SIZE * std_dev)  # Weighted sum of noise

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

# --- Segmentation ---

def translate_rotate_seed_mesh_3d(pose: np.ndarray) -> np.ndarray:
    """Translates and rotates the seed mesh points based on the current frame's seed mesh pose."""
    # pose (6,) tx, ty, tz, rx, ry, rz
    if reconstructed_seed_mesh is None:
        return
    points_3d = reconstructed_seed_mesh['points']  # (points, 3)
    tvec = pose[:3]
    rvec = pose[3:]
    R, _ = cv2.Rodrigues(rvec)
    transformed_points = (R @ points_3d.T).T + tvec  # Apply rotation and translation
    return transformed_points  # (points, 3)

def reproject_seed_segmentation_as_contour(cam_idx: int, pose: np.ndarray) -> np.ndarray:
    """Reproject 2d seed contour (shadow) from 3d reconstructed mesh."""
    if reconstructed_seed_mesh is None or best_individual is None:
        return
    points_3d = translate_rotate_seed_mesh_3d(pose) # (points, 3)
    vertex_points_3d = points_3d[reconstructed_seed_mesh['vertices']] # (vertices, 3)
    reprojected = reproject_points(vertex_points_3d, best_individual[cam_idx]) # (vertices, 2)
    hull = ConvexHull(reprojected)
    hull = reprojected[hull.vertices]  # (hull_points, 2)
    return hull

def resample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """Resamples a 2D contour to have a specific number of points."""
    closed_contour = np.vstack([contour, contour[0]])
    distances = np.cumsum(np.sqrt(np.sum(np.diff(closed_contour, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    total_length = distances[-1]
    if total_length < 1e-6: # Handle case of single or identical points
        return np.repeat(contour[:1], num_points, axis=0)
    resampled_distances = np.linspace(0, total_length, num_points, endpoint=False) # Use endpoint=False for closed loop
    resampled_x = np.interp(resampled_distances, distances, closed_contour[:, 0])
    resampled_y = np.interp(resampled_distances, distances, closed_contour[:, 1])
    return np.vstack((resampled_x, resampled_y)).T


def estimate_seed_pose():
    """Estimates the seed mesh pose for the current frame using PnP."""
    if reconstructed_seed_mesh is None or best_individual is None:
        return
    def calc_error(pose: np.ndarray) -> float:
        # Assume seed_points_2d, reproject_seed_segmentation_as_contour,
        # and resample_contour are defined elsewhere.
        cameras_with_seed_segmentation = [i for i, seeds in enumerate(seed_points_2d) if len(seeds) > 0]
        total_error = 0
        count = 0
        for i in cameras_with_seed_segmentation:
            contour_2d = reproject_seed_segmentation_as_contour(i, pose)
            if contour_2d is None or len(contour_2d) < 5:
                continue
            seed_2d = np.array(seed_points_2d[i], dtype=np.float32)
            if len(seed_2d) < 5:
                continue
            # Resample both contours to a fixed number of points
            resampled_projected = resample_contour(contour_2d, 100)
            resampled_observed = resample_contour(seed_2d, 100)
            # Find the best alignment by shifting the starting point
            best_error_for_cam = float('inf')
            for offset in range(len(resampled_projected)):
                rolled_contour = np.roll(resampled_projected, offset, axis=0)
                pointwise_distances = np.linalg.norm(resampled_observed - rolled_contour, axis=1)
                # Use the 90th percentile instead of the mean to ignore the worst 10% of errors (outliers)
                error = np.percentile(pointwise_distances, 90)                
                if error < best_error_for_cam:
                    best_error_for_cam = error
            total_error += best_error_for_cam
            count += 1
        print("Total robust seed reprojection error:", total_error, "Count:", count, "pose", pose)
        return total_error / count if count > 0 else float('inf') 
    # Initial guess for pose
    initial_pose = seed_mesh_poses[frame_idx]
    print("Optimizing seed mesh pose for frame", frame_idx)
    result = minimize(calc_error, initial_pose, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-2, 'fatol': 1e-2, 'disp': False})
    print("Result", result)
    if result.success:
        seed_mesh_poses[frame_idx] = result.x
    else:
        print("Seed mesh pose optimization failed for frame", frame_idx)

# --- Visualisation ---

def draw_ui(frame, cam_idx):
    """Draws UI elements on the frame."""
    if not auto_segment_mode:
        if best_individual is not None:
            reprojected = reproject_points(reconstructed_3d_points[frame_idx], best_individual[cam_idx])  # (num_points, 2)
        # Draw annotated points for the current frame
        p_idxs = np.arange(NUM_POINTS) if not focus_selected_point else [selected_point_idx]
        for p_idx in p_idxs:
            point = annotations[frame_idx, cam_idx, p_idx]
            if not np.isnan(point).any():
                if human_annotated[frame_idx, cam_idx, p_idx]:
                    cv2.circle(frame, tuple(point.astype(int)), 5 + 2, (255, 255, 255), -1) # White outline
                cv2.circle(frame, tuple(point.astype(int)), 5, point_colors[p_idx].tolist(), -1)
                cv2.putText(frame, POINT_NAMES[p_idx], tuple(point.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 2)
                # Reproject the 3D point back to 2D
                if best_individual is None:
                    continue
                point_2d_from_3d = reprojected[p_idx] # (2,)
                if not np.isnan(point_2d_from_3d).any() and (point_2d_from_3d > 0).all():
                    # Draw a line from the reprojected point to the annotated point
                    cv2.line(frame, tuple(point.astype(int)), tuple(point_2d_from_3d.astype(int)), point_colors[p_idx].tolist(), 1)
                    distance = np.linalg.norm(point - point_2d_from_3d)
                    cv2.putText(frame, f"{distance:.2f}", tuple(point_2d_from_3d.astype(int) + np.array([5, -5])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 1)

    # Draw 3D reconstructed mesh reprojected back down to 2d cameras
    if reconstructed_seed_mesh is not None and best_individual is not None:
        contour_2d = reproject_seed_segmentation_as_contour(cam_idx, seed_mesh_poses[frame_idx])
        if contour_2d is not None and len(contour_2d) > 0:
            cv2.drawContours(frame, [contour_2d.astype(np.int32)], -1, (0, 255, 0), 1)
            # resampled_contour = resample_contour(contour_2d, 100)
            # for point in resampled_contour:
            #     cv2.circle(frame, tuple(point.astype(int)), 2, (255, 255, 0), -1)
        # points_3d = reconstructed_seed_mesh.get("points") # (points, 3)
        # faces = reconstructed_seed_mesh.get("faces") # (faces, 3) the 3 is the index of the vertex that make up the triangle face
        # if points_3d is not None and len(points_3d) > 0:
        #     reprojected_points_2d = reproject_points(points_3d, best_individual[cam_idx])
        #     if faces is not None and len(faces) > 0:
        #         # for face in faces:
        #         #     # Get the 2D coords of the face's vertices
        #         #     p1 = reprojected_points_2d[face[0]].astype(int)
        #         #     p2 = reprojected_points_2d[face[1]].astype(int)
        #         #     p3 = reprojected_points_2d[face[2]].astype(int)
        #         #     # Draw the edges of the face in yellow
        #         #     cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 255), 1)
        #         #     cv2.line(frame, tuple(p2), tuple(p3), (0, 255, 255), 1)
        #         #     cv2.line(frame, tuple(p3), tuple(p1), (0, 255, 255), 1)
        #         projected_vertices = reprojected_points_2d[reconstructed_seed_mesh['vertices']]
        #         hull = cv2.convexHull(projected_vertices.astype(np.int32))
        #         cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
        #     else:  # If no faces, draw the convex hull of the points
        #         valid_points_mask = ~np.isnan(reprojected_points_2d).any(axis=1)
        #         if np.any(valid_points_mask):
        #             contour = reprojected_points_2d[valid_points_mask].astype(np.int32)
        #             if len(contour) > 2:
        #                 hull = cv2.convexHull(contour)
        #                 cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 255), thickness=1)

    # Draw 2D segmentations
    if auto_segment_mode and cam_idx < len(seed_points_2d):
        contour = np.array(seed_points_2d[cam_idx], dtype=np.int32)
        if len(contour) > 5:
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)

    return frame

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
    # Extrinsic parameters (world-to-camera transformation)
    rvec = cam_params['rvec'] # (3,)
    tvec = cam_params['tvec'] # (3,)
    # Get camera-to-world rotation matrix (inverse of world-to-camera)
    R, _ = cv2.Rodrigues(rvec) # (3, 3)
    # Define canonical camera pyramid in its local coordinate system
    #  (apex at the origin, pointing down the +Z axis)
    #  Note: OpenCV's camera convention is +Y down, +X right.
    w = 0.5 * scale
    h = 0.4 * scale
    depth = 0.8 * scale

    # Points in the camera's local coordinate system
    pyramid_points_cam = np.array([
        [0, 0, 0],       # p0: Apex (camera center)
        [-w, -h, depth], # p1: Top-left corner of base
        [w, -h, depth],  # p2: Top-right corner
        [w, h, depth],   # p3: Bottom-right corner
        [-w, h, depth],  # p4: Bottom-left corner
    ]) # (5, 3) - points in camera local coordinates

    # 4. Transform local camera points to world coordinates.
    # The camera center in the world is C = -R' * t
    # A point in the world is p_w = C + R' * p_c = R' * (p_c - t)
    cam_center_world = -R.T @ tvec # (3,)
    pyramid_points_world = (pyramid_points_cam - tvec) @ R # (5, 3)
    p0_w, p1_w, p2_w, p3_w, p4_w = pyramid_points_world
    scene_objects: List[SceneObject] = []

    # Camera center
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
    np.save(DATA_FOLDER / 'annotations.npy', annotations)
    np.save(DATA_FOLDER / 'human_annotated.npy', human_annotated)
    np.save(DATA_FOLDER / 'reconstructed_3d_points.npy', reconstructed_3d_points)
    json.dump(calibration_frames, open(DATA_FOLDER / 'calibration_frames.json', 'w'))
    if best_individual is not None:
        with open(DATA_FOLDER / 'best_individual.pkl', 'wb') as f:
            pickle.dump(best_individual, f)
    print("State saved successfully.")

def load_state():
    """Loads the saved state of annotations and 3D points."""
    global annotations, human_annotated, reconstructed_3d_points, best_individual, best_fitness_so_far, calibration_frames
    try:
        annotations = np.load(DATA_FOLDER / 'annotations.npy')
        human_annotated = np.load(DATA_FOLDER / 'human_annotated.npy')
        reconstructed_3d_points = np.load(DATA_FOLDER / 'reconstructed_3d_points.npy')
        with open(DATA_FOLDER / 'calibration_frames.json', 'r') as f:
            calibration_frames = json.load(f)
        with open(DATA_FOLDER / 'best_individual.pkl', 'rb') as f:
            best_individual = pickle.load(f)
        best_fitness_so_far = fitness(best_individual, annotations)  # Recalculate fitness
        print("State loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading state: {e}")

# --- DearPyGui UI ---

def get_screen_dimensions():
    """Gets the screen dimensions using tkinter."""
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height

def resize_callback(sender, app_data, user_data):
    """Callback for when the viewport is resized."""
    viewport_width = app_data[0]
    video_grid_width = viewport_width - 320  # Leave space for the control panel
    aspect_ratio = video_metadata['width'] / video_metadata['height']
    item_width = (video_grid_width / GRID_COLS) - 20 # Subtract some padding
    item_height = item_width / aspect_ratio
    for i in range(video_metadata['num_videos']):
        dpg.configure_item(f"video_image_{i}", width=item_width, height=item_height)
    dpg.configure_item("3d_image", width=item_width, height=item_height)

def on_key_press(sender, app_data):
    """Key press handler."""
    allowed_keys = {dpg.mvKey_Q, dpg.mvKey_S, dpg.mvKey_P} # In autosegment mode
    if auto_segment_mode and app_data not in allowed_keys:    
        return
        
    match app_data:
        case dpg.mvKey_Spacebar:
            toggle_pause(sender, app_data, None)
        case dpg.mvKey_Left:
            prev_frame(sender, app_data, None)
        case dpg.mvKey_Right:
            next_frame(sender, app_data, None)
        case dpg.mvKey_Up:
            set_selected_point(None, POINT_NAMES[(selected_point_idx - 1) % NUM_POINTS], None)
        case dpg.mvKey_Down:
            set_selected_point(None, POINT_NAMES[(selected_point_idx + 1) % NUM_POINTS], None)
        case dpg.mvKey_T:
            toggle_keypoint_tracking()
        case dpg.mvKey_P:
            toggle_seed_pose_tracking()            
        case dpg.mvKey_G:
            toggle_ga(sender, app_data, None)
        case dpg.mvKey_H:
            set_human_annotated(sender, app_data, None)
        case dpg.mvKey_D:
            clear_future_annotations(sender, app_data, None)
        case dpg.mvKey_S:
            save_state()
        case dpg.mvKey_L:
            load_state()
        case dpg.mvKey_F:
            find_worst_reprojection()
        case dpg.mvKey_W:
            find_worst_frame()
        case dpg.mvKey_C:
            add_to_calib_frames(sender, app_data, None)
        case dpg.mvKey_Q:
            dpg.stop_dearpygui()
        case dpg.mvKey_R:
            toggle_record()
        case dpg.mvKey_Z:
            global focus_selected_point
            focus_selected_point = not focus_selected_point
            
def create_ga_popup():
    """Creates the popup window for the genetic algorithm."""
    with dpg.window(label="Calibration", modal=False, show=False, tag="ga_popup", width=540, height=120, on_close=toggle_ga_pause):
        dpg.add_text("Running genetic algorithm...", tag="ga_status_text")
        dpg.add_text("", tag="ga_progress_text")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset", callback=reset_ga)
            dpg.add_button(label="Pause", callback=toggle_ga_pause, tag="ga_pause_button")

def create_dpg_ui(textures: np.ndarray, scene_viz: SceneVisualizer):
    """Creates the DearPyGui UI."""
    dpg.create_context()
    screen_width, screen_height = get_screen_dimensions()
    viewport_width = int(screen_width * 0.9)
    viewport_height = int(screen_height * 0.9)

    # Calculate the position to center the viewport
    x_pos = int((screen_width - viewport_width) * 0.5)
    y_pos = int((screen_height - viewport_height) * 0.5)

    dpg.create_viewport(title="CATAR", width=viewport_width, height=viewport_height, x_pos=x_pos, y_pos=y_pos)
    
    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save state", callback=save_state)
            dpg.add_menu_item(label="Load state", callback=load_state)
        with dpg.menu(label="Calibration"):
            dpg.add_menu_item(label="Run genetic algorithm", callback=toggle_ga)
            dpg.add_menu_item(label="Find worst calibration", callback=find_worst_frame)
            dpg.add_menu_item(label="Find worst reprojection", callback=find_worst_reprojection)
            dpg.add_menu_item(label="Add calibration frame", callback=add_to_calib_frames)
        with dpg.menu(label="Segmentation"):
            dpg.add_menu_item(label="Enable 2D auto-segmentation", callback=lambda: toggle_auto_segment_mode(), check=True, tag="auto_segment_checkbox")
            dpg.add_menu_item(label="Reconstruct 3D mesh", callback=reconstruct_seed_mesh)
            dpg.add_menu_item(label="Clear segmentation points", callback=clear_seed_points)
            dpg.add_menu_item(label="Estimate seed pose in current frame", callback=estimate_seed_pose)

    create_ga_popup()
    dpg.setup_dearpygui()
    dpg.set_viewport_resize_callback(resize_callback)

    # Create textures for each video feeds and one for the 3D reprojection
    with dpg.texture_registry():
        for i in range(video_metadata['num_videos']):
            dpg.add_raw_texture(
                width=video_metadata['width'], 
                height=video_metadata['height'], 
                default_value=textures[i].ravel(),
                tag=f"video_texture_{i}",
                format=dpg.mvFormat_Float_rgba
            )
        dpg.add_raw_texture(
            width=video_metadata['width'], 
            height=video_metadata['height'], 
            default_value=textures[-1].ravel(), 
            tag="3d_texture",
            format=dpg.mvFormat_Float_rgba
        )

    with dpg.theme(tag="record_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 50, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 100, 100, 255))
    with dpg.theme(tag="tracking_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 255, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 255, 100, 255))

    with dpg.window(label="Main Window", tag="main_window"):
        with dpg.group(horizontal=True):
            # Left side control panel
            with dpg.child_window(width=300):
                create_control_panel()
            # Right side video and 3D projection
            with dpg.child_window(width=-1):
                with dpg.child_window(height=-150, tag="video_grid_window"):
                    create_video_grid(scene_viz)
                # Annotation histogram
                with dpg.child_window(height=150, tag="histogram_window"):
                    with dpg.plot(label="Annotation Histogram", height=-1, width=-1, no_menus=True, no_box_select=True, no_mouse_pos=True, tag="annotation_plot"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="histogram_x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Annotations", tag="histogram_y_axis")
                        dpg.add_bar_series(list(range(video_metadata['num_frames'])), [0]*video_metadata['num_frames'], label="Annotation Count", parent="histogram_y_axis", tag="annotation_histogram_series")
                        dpg.add_drag_line(label="Current Frame", color=[255, 0, 0, 255], vertical=True, default_value=frame_idx, tag="current_frame_line")
                    with dpg.item_handler_registry(tag="histogram_handler"):
                        dpg.add_item_clicked_handler(callback=on_histogram_click)
                    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press)
        dpg.add_mouse_wheel_handler(callback=scene_viz.dpg_on_mouse_wheel, user_data="3d_image")
        dpg.add_mouse_move_handler(callback=scene_viz.dpg_drag_move)
        dpg.add_mouse_release_handler(callback=scene_viz.dpg_drag_end)

    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()

def create_control_panel():
    """Creates the control panel on the left side of the UI."""
    dpg.add_text("--- Info ---")
    dpg.add_text(f"Frame: {frame_idx}/{video_metadata['num_frames']}", tag="frame_text")
    dpg.add_text(f"Status: {'Paused' if paused else 'Playing'}", tag="status_text")
    dpg.add_text(f"Save output video: {'Enabled' if save_output_video else 'Disabled'}", tag="save_video_text")
    dpg.add_text(f"Autosegment mode: {'Enabled' if auto_segment_mode else 'Disabled'}", tag="autosegment_text")
    dpg.add_text(f"Tracking: {'Enabled' if (keypoint_tracking_enabled or seed_pose_tracking_enabled) else 'Disabled'}", tag="tracking_text")
    dpg.add_text(f"Focus mode: {'Enabled' if focus_selected_point else 'Disabled'}", tag="focus_text")
    dpg.add_text(f"Annotating keypoint: {POINT_NAMES[selected_point_idx]}", tag="annotating_point_text")
    dpg.add_spacing(count=5)
    dpg.add_text(f"Best fitness: {best_fitness_so_far:.2f}", tag="fitness_text")
    dpg.add_text(f"Num annotation: {np.sum(~np.isnan(annotations[frame_idx])) // 2} / {NUM_POINTS * len(video_names)}", tag="num_annotations_text")
    dpg.add_text(f"Num 3D points: {np.sum(~np.isnan(reconstructed_3d_points[frame_idx]).any(axis=1))} / {NUM_POINTS}", tag="num_3d_points_text")
    dpg.add_text(f"Num calibration frames: {len(calibration_frames)}", tag="num_calib_frames_text")
    dpg.add_separator()
    dpg.add_text("--- Controls ---")
    with dpg.group(horizontal=True):
        dpg.add_button(label="< Prev", callback=prev_frame)
        play_label = "Play" if paused else "Pause"
        dpg.add_button(label=play_label, callback=toggle_pause, tag="play_pause_button")
        dpg.add_button(label="Next >", callback=next_frame)
    dpg.add_slider_int(label="Frame", min_value=0, max_value=video_metadata['num_frames'] - 1, default_value=frame_idx, callback=set_frame, tag="frame_slider")
    dpg.add_combo(label="Keypoint", items=POINT_NAMES, default_value=POINT_NAMES[selected_point_idx], callback=set_selected_point, tag="point_combo")
    dpg.add_button(label="Set all previous to 'human annotated'", callback=set_human_annotated)
    dpg.add_button(label="Delete future annotations", callback=clear_future_annotations)
    dpg.add_button(label="Track keypoints", callback=toggle_keypoint_tracking, tag="keypoint_tracking_button")
    # dpg.add_button(label="Track seed pose", callback=toggle_seed_pose_tracking, tag="seed_pose_tracking_button")
    dpg.add_button(label="Record", callback=toggle_record, tag="record_button")
    dpg.add_checkbox(label="Show histogram", default_value=True, callback=toggle_histogram)
    dpg.add_separator()
    dpg.add_text("--- Messages ---")
    dpg.add_text("", tag="status_message", color=(255, 100, 100), wrap=280, show=False)
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=125)
        dpg.add_loading_indicator(tag="loading_indicator", show=False, style=1, speed=0.5)

def create_video_grid(scene_viz: SceneVisualizer):
    """Creates the grid for video feeds and 3D projection."""
    num_items = video_metadata['num_videos'] + 1
    with dpg.table(header_row=False):
        for _ in range(GRID_COLS):
            dpg.add_table_column()
        num_rows = (num_items + GRID_COLS - 1) // GRID_COLS
        for i in range(num_rows):
            with dpg.table_row():
                for j in range(GRID_COLS):
                    idx = i * GRID_COLS + j
                    if idx < video_metadata['num_videos']:
                        with dpg.table_cell():
                            dpg.add_text(video_names[idx])
                            dpg.add_image(f"video_texture_{idx}", tag=f"video_image_{idx}")
                            with dpg.item_handler_registry(tag=f"image_handler_{idx}"):
                                dpg.add_item_clicked_handler(callback=image_click_callback, user_data=idx)
                            dpg.bind_item_handler_registry(f"video_image_{idx}", f"image_handler_{idx}")
                    elif idx == video_metadata['num_videos']:
                        with dpg.table_cell():
                            dpg.add_text("3D Projection")
                            dpg.add_image("3d_texture", tag="3d_image")
                            # Bind scene visualizer mouse events to the 3D image
                            with dpg.item_handler_registry(tag="3d_image_handler"):
                                dpg.add_item_clicked_handler(callback=scene_viz.dpg_drag_start)
                            dpg.bind_item_handler_registry("3d_image", "3d_image_handler")

# --- DearPyGui callbacks ---

def toggle_record():
    global save_output_video, video_save_output
    save_output_video = not save_output_video
    if save_output_video:
        dpg.bind_item_theme("record_button", "record_button_theme")
    else:
        dpg.bind_item_theme("record_button", 0) # 0 resets to the default theme
    if save_output_video:  # Initialise video writer
        num_rows = video_metadata["num_videos"] // GRID_COLS + (1 if video_metadata["num_videos"] % GRID_COLS > 0 else 0)
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')
        video_save_output = cv2.VideoWriter("recording.mp4", fourcc, 30.0, 
                                            (video_metadata['width'] * GRID_COLS, 
                                            video_metadata['height'] * num_rows))
    print(f"Recording toggled: {save_output_video}")

def toggle_pause(sender, app_data, user_data):
    global paused
    paused = not paused
    if paused:
        dpg.configure_item("play_pause_button", label="Play")
    else:
        dpg.configure_item("play_pause_button", label="Pause")

def next_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if frame_idx < video_metadata['num_frames'] - 1:
        frame_idx += 1
    dpg.set_value("frame_slider", frame_idx)

def prev_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if frame_idx > 0:
        frame_idx -= 1
    dpg.set_value("frame_slider", frame_idx)

def set_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    frame_idx = app_data

def set_selected_point(sender, app_data, user_data):
    global selected_point_idx
    selected_point_idx = POINT_NAMES.index(app_data)
    dpg.set_value("point_combo", POINT_NAMES[selected_point_idx])

def set_human_annotated(sender, app_data, user_data):
    if focus_selected_point:
        human_annotated[:frame_idx + 1, :, selected_point_idx] = True
        print(f"Marked all previous frames as human-annotated for point {POINT_NAMES[selected_point_idx]}.")

def clear_future_annotations(sender, app_data, user_data):
    if focus_selected_point:
        annotations[frame_idx:, :, selected_point_idx] = np.nan
        human_annotated[frame_idx:, :, selected_point_idx] = False
        print(f"Cleared all annotations for {POINT_NAMES[selected_point_idx]} from frame {frame_idx}")

def toggle_keypoint_tracking():
    global keypoint_tracking_enabled, seed_pose_tracking_enabled
    keypoint_tracking_enabled = not keypoint_tracking_enabled
    if keypoint_tracking_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
        seed_pose_tracking_enabled = False
        dpg.bind_item_theme("seed_pose_tracking_button", 0)
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)

def toggle_seed_pose_tracking():
    global seed_pose_tracking_enabled, keypoint_tracking_enabled
    seed_pose_tracking_enabled = not seed_pose_tracking_enabled
    if seed_pose_tracking_enabled:
        dpg.bind_item_theme("seed_pose_tracking_button", "tracking_button_theme")
        keypoint_tracking_enabled = False
        dpg.bind_item_theme("keypoint_tracking_button", 0)
    else:
        dpg.bind_item_theme("seed_pose_tracking_button", 0)

def toggle_ga(sender, app_data, user_data):
    global train_ga, best_fitness_so_far, paused
    train_ga = not train_ga
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if train_ga:
        dpg.configure_item("ga_popup", show=True)
    else:
        dpg.configure_item("ga_popup", show=False)

def toggle_ga_pause():
    global train_ga
    train_ga = not train_ga

def reset_ga():
    global generation, best_fitness_so_far, best_individual
    generation = 0
    best_fitness_so_far = float('inf')
    best_individual = None
    
def add_to_calib_frames(sender, app_data, user_data):
    global calibration_frames
    if frame_idx not in calibration_frames:
        calibration_frames.append(frame_idx)
        print(f"Frame {frame_idx} added to calibration frames.")

def image_click_callback(sender, app_data, user_data):
    """Callback function for handling mouse clicks on video images."""
    global needs_3d_reconstruction, current_frames
    cam_idx = user_data
    image_tag = f"video_image_{cam_idx}"
    # Mouse pos in absolute window coords
    mouse_pos_abs = dpg.get_mouse_pos(local=False)
    # Get the absolute position of the top-left corner of the frame
    image_pos_abs = dpg.get_item_rect_min(image_tag)
    local_pos_x = mouse_pos_abs[0] - image_pos_abs[0]
    local_pos_y = mouse_pos_abs[1] - image_pos_abs[1]
    # Current size of frame
    image_size = dpg.get_item_rect_size(image_tag)
    # Scale local mouse position to original video's resolution
    scaled_x = local_pos_x * video_metadata['width'] / image_size[0]
    scaled_y = local_pos_y * video_metadata['height'] / image_size[1]
    # Clip coordinates within frame bounds
    mouse_pos = (max(0, min(video_metadata['width'] - 1, scaled_x)),
                 max(0, min(video_metadata['height'] - 1, scaled_y)))
    
    if auto_segment_mode:
        if app_data[0] == 0: # Left-click to auto-segment
            dpg.show_item("loading_indicator")
            try:
                print(f"Running auto-segmentation for camera {cam_idx} with prompt at {mouse_pos}...")
                frame_to_segment = current_frames[cam_idx] # Get current frame for segmentation
                contour_points = get_sam_segmentation(frame_to_segment, mouse_pos)
                if contour_points is not None:
                    seed_points_2d[cam_idx] = contour_points.tolist() # Replace the old points with new contour
                    # print(f"Successfully segmented seed with {len(contour_points)} points.")
                else:
                    print("Auto-segmentation failed.")
            finally:
                dpg.hide_item("loading_indicator")
        return # Don't allow other functionality whilst in autosegment mode

    if app_data[0] == 0:  # Left click to annotate
        annotations[frame_idx, cam_idx, selected_point_idx] = (float(mouse_pos[0]), float(mouse_pos[1]))
        human_annotated[frame_idx, cam_idx, selected_point_idx] = True
        print(f"Annotated {POINT_NAMES[selected_point_idx]} at ({mouse_pos[0]:.2f}, {mouse_pos[1]:.2f}) in Cam {cam_idx} at frame {frame_idx}")
        needs_3d_reconstruction = True
    elif app_data[0] == 1: # Right click to remove annotation
        annotations[frame_idx, cam_idx, selected_point_idx] = np.nan
        human_annotated[frame_idx, cam_idx, selected_point_idx] = False
        print(f"Removed annotation for {POINT_NAMES[selected_point_idx]} in Cam {cam_idx} at frame {frame_idx}")
        needs_3d_reconstruction = True

def clear_seed_points():
    """Clears all selected sam-segmented seed points."""
    global seed_points_2d, reconstructed_seed_mesh, needs_3d_reconstruction
    for i in range(len(seed_points_2d)):
        seed_points_2d[i].clear()
    # reconstructed_seed_mesh = None
    needs_3d_reconstruction = True

def reconstruct_seed_mesh():
    """Reconstructs a 3D mesh from the 2D seed contours using triangulation
    and find the best correspondence between contour points."""
    global reconstructed_seed_mesh, needs_3d_reconstruction
    if best_individual is None:
        dpg.set_value("status_message", "Error: Provide calibration before reconstructing mesh.")
        dpg.show_item("status_message")
        return

    cams_with_points = [i for i, pts in enumerate(seed_points_2d) if len(pts) > 2]
    if len(cams_with_points) < 2:
        dpg.set_value("status_message", "Error: Segment object in at least two views.")
        dpg.show_item("status_message")
        return

    proj_matrices = np.array([get_projection_matrix(cam) for cam in best_individual])
    camera_pairs = list(itertools.combinations(cams_with_points, 2))
    all_3d_points = []
    num_resampled_points = 50
    dpg.show_item("loading_indicator")
    for cam_idx_1, cam_idx_2 in camera_pairs:
        print(f"Processing pair: Camera {cam_idx_1} and Camera {cam_idx_2}")
        contour1_orig = np.array(seed_points_2d[cam_idx_1], dtype=np.float32)
        contour2_orig = np.array(seed_points_2d[cam_idx_2], dtype=np.float32)
        contour1 = resample_contour(contour1_orig, num_resampled_points)
        contour2 = resample_contour(contour2_orig, num_resampled_points)

        # Find the best rotational alignment by minimising reprojection error
        min_error = float('inf')
        best_offset = 0
        for offset in range(num_resampled_points):
            contour2_rolled = np.roll(contour2, offset, axis=0)
            contour1_undistorted = undistort_points(contour1, best_individual[cam_idx_1])
            contour2_rolled_undistorted = undistort_points(contour2_rolled, best_individual[cam_idx_2])
            points_4d_hom = cv2.triangulatePoints(proj_matrices[cam_idx_1], proj_matrices[cam_idx_2], contour1_undistorted.T, contour2_rolled_undistorted.T)
            points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
            if np.any(np.isinf(points_3d)) or np.any(np.isnan(points_3d)):
                continue # Skip if any points are at infinity

            reproj1 = reproject_points(points_3d, best_individual[cam_idx_1])
            reproj2 = reproject_points(points_3d, best_individual[cam_idx_2])
            error1 = np.linalg.norm(contour1 - reproj1, axis=1).sum()
            error2 = np.linalg.norm(contour2_rolled - reproj2, axis=1).sum()
            total_error = error1 + error2
            if total_error < min_error:
                min_error = total_error
                best_offset = offset
        print(f"  - Found best alignment with offset {best_offset} and error {min_error:.2f}")

        # Triangulate with the best alignment using undistorted points for final accuracy
        contour2_aligned = np.roll(contour2, best_offset, axis=0)
        contour1_final_undistorted = undistort_points(contour1, best_individual[cam_idx_1])
        contour2_aligned_final_undistorted = undistort_points(contour2_aligned, best_individual[cam_idx_2])

        points_4d_hom = cv2.triangulatePoints(proj_matrices[cam_idx_1], proj_matrices[cam_idx_2], contour1_final_undistorted.T, contour2_aligned_final_undistorted.T)
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
        all_3d_points.append(points_3d)
    if not all_3d_points:
        dpg.set_value("status_message", "Error: Triangulation failed.")
        dpg.show_item("status_message")
        return

    final_points_3d = np.vstack(all_3d_points)
    try:
        hull = ConvexHull(final_points_3d)
        faces = hull.simplices
        reconstructed_seed_mesh = {"points": final_points_3d, "faces": faces, "vertices": hull.vertices}
        needs_3d_reconstruction = True # Trigger redraw
        print(f"Successfully reconstructed seed into a mesh with {len(final_points_3d)} vertices and {len(faces)} faces.")
        dpg.set_value("status_message", "Seed mesh reconstructed.")
        dpg.show_item("status_message")
    except Exception as e:
        print(f"Error creating Convex Hull: {e}. Storing as point cloud.")
        reconstructed_seed_mesh = {"points": final_points_3d, "faces": []} # Store as point cloud if hull fails
        needs_3d_reconstruction = True
        dpg.set_value("status_message", "Could not form mesh, showing point cloud.")
        dpg.show_item("status_message")
    dpg.hide_item("loading_indicator")

# --- Main Loop (DPG) ---

def calculate_annotation_counts():
    """Calculates the number of annotations for each frame."""
    return np.sum(~np.isnan(annotations[:, :, :, 0]), axis=(1, 2))

def calculate_keypoint_annotation_counts(point_idx):
    """Calculates the number of annotations for a specific keypoint for each frame."""
    return np.sum(~np.isnan(annotations[:, :, point_idx, 0]), axis=1)

def on_histogram_click(sender, app_data):
    """Callback for when the annotation histogram is clicked."""
    global frame_idx
    mouse_pos = dpg.get_plot_mouse_pos()
    if mouse_pos:
        clicked_frame = int(mouse_pos[0])
        if 0 <= clicked_frame < video_metadata['num_frames']:
            frame_idx = clicked_frame
            dpg.set_value("frame_slider", frame_idx)

def toggle_histogram(sender, app_data, user_data):
    """Toggles the visibility of the annotation histogram."""
    show = dpg.is_item_shown("histogram_window")
    if show:
        dpg.configure_item("video_grid_window", height=-1)
        dpg.hide_item("histogram_window")
    else:
        dpg.configure_item("video_grid_window", height=-150)
        dpg.show_item("histogram_window")

def initialise_sam_model():
    """Loads the SAM model into memory and prepares it for inference."""
    global sam_predictor
    if not SAM_MODEL_PATH.exists():
        print(f"SAM model checkpoint not found at '{SAM_MODEL_PATH}'.")
        print("Please download it and place it in the 'data' folder.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_MODEL_PATH))
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        sam_predictor = None

def get_sam_segmentation(frame, point_prompt):
    """Uses the SAM model to segment an object based on a single point prompt."""
    if sam_predictor is None:
        print("SAM model is not initialized.")
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb_frame)
    input_point = np.array([point_prompt])
    input_label = np.array([1]) # 1 indicates a foreground point
    masks, scores, logits = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False, # We want the single best mask
    )
    if masks is None or len(masks) == 0:
        return None
    mask = masks[0].astype(np.uint8) # Convert the boolean mask to contour points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.squeeze(axis=1) # Shape: (N, 2)

def toggle_auto_segment_mode():
    """Toggles the automatic seed segmentation mode."""
    global auto_segment_mode
    auto_segment_mode = not auto_segment_mode
    if auto_segment_mode and sam_predictor is None:
        initialise_sam_model() # Load the model when mode is first enabled
        if sam_predictor is None:
            auto_segment_mode = False
            dpg.set_value("status_message", "Loading of SAM2 model failed. Please check its .pth file is in 'data'.")
    dpg.set_value("auto_segment_checkbox", auto_segment_mode)
    if auto_segment_mode:
        dpg.set_value("status_message", "Left-click in a video frame to segment an object.")
    else:
        dpg.set_value("status_message", "")
    dpg.show_item("status_message")
    print(f"Auto-segment mode: {'On' if auto_segment_mode else 'Off'}")

def main_dpg():
    """Main loop for the DearPyGui application."""
    global frame_idx, paused, needs_3d_reconstruction, best_individual, keypoint_tracking_enabled, seed_pose_tracking_enabled, focus_selected_point, save_output_video, current_frames
    load_videos()
    load_state()

    scene = []
    scene_viz = SceneVisualizer(frame_size=(video_metadata['width'], video_metadata['height']))
    textures = np.zeros((video_metadata['num_videos'] + 1, video_metadata['height'], video_metadata['width'], 4), dtype=np.float32) # RGBA
    create_dpg_ui(textures, scene_viz)
    
    # Call resize_callback once to set initial sizes
    resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()], None)

    prev_frames = [None] * video_metadata['num_videos']
    prev_frame_idx = -1

    last_written_frame = -1
    num_videos = video_metadata["num_videos"] + 1 # +1 for the 3D visualization
    num_rows = num_videos // GRID_COLS + (1 if num_videos % GRID_COLS > 0 else 0)
    video_recording_buffer = np.zeros((video_metadata["height"]*num_rows, video_metadata["width"]*GRID_COLS, 3), dtype=np.uint8)

    while dpg.is_dearpygui_running():
        # Update UI text
        dpg.set_value("frame_text", f"Frame: {frame_idx}/{video_metadata['num_frames']}")
        dpg.set_value("status_text", f"Status: {'Paused' if paused else 'Playing'}")
        dpg.set_value("annotating_point_text", f"Annotating keypoint: {POINT_NAMES[selected_point_idx] if not auto_segment_mode else 'Disabled'}") # TODO: disable other functions
        dpg.set_value("focus_text", f"Focus mode: {'Enabled' if focus_selected_point else 'Disabled'}")
        dpg.set_value("fitness_text", f"Best fitness: {best_fitness_so_far:.2f}")
        dpg.set_value("num_annotations_text", f"Num annotations: {np.sum(~np.isnan(annotations[frame_idx])) // 2} / {NUM_POINTS * len(video_names)}")
        dpg.set_value("num_3d_points_text", f"Num 3D points: {np.sum(~np.isnan(reconstructed_3d_points[frame_idx]).any(axis=1))} / {NUM_POINTS}")
        dpg.set_value("tracking_text", f"Tracking: {'Enabled' if (keypoint_tracking_enabled or seed_pose_tracking_enabled) else 'Disabled'}")
        dpg.set_value("autosegment_text", f"Autosegment mode: {'Enabled' if auto_segment_mode else 'Disabled'}")
        dpg.set_value("num_calib_frames_text", f"Num calibration frames: {len(calibration_frames)}")
        dpg.set_value("save_video_text", f"Save output video: {'Enabled' if save_output_video else 'Disabled'}")
        dpg.set_value("current_frame_line", float(frame_idx))

        # Update annotation histogram based on focus mode
        if focus_selected_point:
            keypoint_annotation_counts = calculate_keypoint_annotation_counts(selected_point_idx)
            dpg.configure_item("histogram_y_axis", label=f"'{POINT_NAMES[selected_point_idx]}'")
            dpg.set_value("annotation_histogram_series", [list(range(video_metadata['num_frames'])), keypoint_annotation_counts.tolist()])
            dpg.set_axis_limits("histogram_y_axis", 0, video_metadata['num_videos'])
        else:
            total_annotation_counts = calculate_annotation_counts()
            dpg.configure_item("histogram_y_axis", label="Total annotations")
            dpg.set_value("annotation_histogram_series", [list(range(video_metadata['num_frames'])), total_annotation_counts.tolist()])
            dpg.set_axis_limits("histogram_y_axis", 0, max(total_annotation_counts.tolist()))

        # Frame update logic
        current_frames = []
        if prev_frame_idx != frame_idx:
            for i, cap in enumerate(video_captures):
                if prev_frame_idx != frame_idx - 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    paused = True
                    dpg.configure_item("play_pause_button", label="Play")
                    frame_idx = max(0, frame_idx - 1)
                    continue
                current_frames.append(frame)
            if keypoint_tracking_enabled and prev_frame_idx != -1 and prev_frame_idx == frame_idx - 1 and prev_frames and prev_frames[0] is not None:
                for i in range(video_metadata['num_videos']):
                    prev_gray = cv2.cvtColor(prev_frames[i], cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.cvtColor(current_frames[i], cv2.COLOR_BGR2GRAY)
                    track_points(prev_gray, gray_frame, i)
            if focus_selected_point and keypoint_tracking_enabled:
                # Check if the selected point is annotated in the current frame in at least two cameras
                count = np.sum(~np.isnan(annotations[frame_idx, :, selected_point_idx, 0]))
                if count < 2:
                    message = f"Selected keypoint '{POINT_NAMES[selected_point_idx]}' is not annotated in at least two cameras. Please annotate more points and continue."
                    dpg.set_value("status_message", message)
                    dpg.show_item("status_message")
                    paused = True
                else:
                    dpg.hide_item("status_message")
            prev_frame_idx = frame_idx
            prev_frames = current_frames
            needs_3d_reconstruction = True
        else:
            current_frames = prev_frames

        if needs_3d_reconstruction and best_individual is not None:
            needs_3d_reconstruction = False
            update_3d_reconstruction(best_individual)
            scene.clear()
            for i, cam in enumerate(best_individual):
                cam_viz = create_camera_visual(cam_params=cam, scale=1.0, color=point_colors[i % len(point_colors)], label=video_names[i])
                scene.extend(cam_viz)
            points_3d = reconstructed_3d_points[frame_idx]
            for i, point in enumerate(points_3d):
                if not np.isnan(point).any():
                    scene.append(SceneObject(type='point', coords=point, color=point_colors[i % len(point_colors)], label=POINT_NAMES[i]))
                from_name = POINT_NAMES[i]
                for to in SKELETON[from_name]:
                    to_id = POINT_NAMES.index(to)
                    if not np.isnan(point).any() and not np.isnan(points_3d[to_id]).any():
                        scene.append(SceneObject(type='line', coords=np.array([point, points_3d[to_id]]), color=point_colors[i % len(point_colors)], label=None))
            
            if reconstructed_seed_mesh is not None:
                points_3d = translate_rotate_seed_mesh_3d(seed_mesh_poses[frame_idx]) # (N, 3)
                faces = reconstructed_seed_mesh["faces"]
                for face in faces:
                    p1 = points_3d[face[0]]
                    p2 = points_3d[face[1]]
                    p3 = points_3d[face[2]]
                    scene.append(SceneObject(type='line', coords=np.array([p1, p2]), color=(0, 255, 255), label=None))
                    scene.append(SceneObject(type='line', coords=np.array([p2, p3]), color=(0, 255, 255), label=None))
                    scene.append(SceneObject(type='line', coords=np.array([p3, p1]), color=(0, 255, 255), label=None))

        # Update video textures
        if current_frames:
            for i, frame in enumerate(current_frames):
                frame_with_ui = draw_ui(frame.copy(), i) # (H, W, 3)
                # Convert BGR to RGBA, then to float32 and normalize
                rgba_frame = cv2.cvtColor(frame_with_ui, cv2.COLOR_BGR2RGBA) # (H, W, 4)
                textures[i, :, :, :] = rgba_frame.astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", textures[i].ravel())
                # Place the frame in the recording buffer
                row = i // GRID_COLS
                col = i % GRID_COLS
                y_start = row * video_metadata['height']
                x_start = col * video_metadata['width']
                video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = frame_with_ui

        # Update 3D projection texture
        viz_3d_frame = scene_viz.draw_scene(scene)
        rgba_3d_frame = cv2.cvtColor(viz_3d_frame, cv2.COLOR_BGR2RGBA)
        textures[-1, :, :, :] = rgba_3d_frame.astype(np.float32) / 255.0
        dpg.set_value("3d_texture", textures[-1].ravel())
        
        # Place the 3D visualization in the recording buffer
        row = (num_videos-1) // GRID_COLS
        col = (num_videos-1) % GRID_COLS
        y_start = row * video_metadata['height']
        x_start = col * video_metadata['width']
        video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = viz_3d_frame
        
        if save_output_video and video_save_output is not None and last_written_frame != frame_idx:
            video_save_output.write(video_recording_buffer)
            last_written_frame = frame_idx

        if train_ga:
            run_genetic_step()
            needs_3d_reconstruction = True

        if not paused:
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
            else:
                paused = True
                dpg.configure_item("play_pause_button", label="Play")
            dpg.set_value("frame_slider", frame_idx)
        
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    for cap in video_captures:
        cap.release()
    if video_save_output is not None:
        video_save_output.release()

if __name__ == '__main__':
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos there.")
    main_dpg()