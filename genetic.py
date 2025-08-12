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
import glob
import random
import itertools
from tqdm import tqdm
from typing import TypedDict, List, Tuple, Optional
from viz_3d import SceneObject, SceneVisualizer

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
        "m_R1": [ "m_R0" ]
}
POINT_NAMES = list(SKELETON.keys())
NUM_POINTS = len(POINT_NAMES)

# Genetic Algorithm Parameters
POPULATION_SIZE = 200 
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

# Shape: (num_frames, num_points, 3) for (X, Y, Z) coordinates
reconstructed_3d_points = None
needs_3d_reconstruction = False


# UI and Control State
frame_idx = 600
paused = True
selected_point_idx = 0  # Default to P1

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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
    [128, 255, 64]    # P26 - Light Yellow
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

def combination_triangulate(frame_annotations: np.ndarray, proj_matrices: np.ndarray) -> np.ndarray:
    """Triangulates 3D points from 2D correspondences using multiple camera views."""
    # frame_annotations: (num_videos, num_points, 2) and proj_matrices: (num_videos, 3, 4)
    # returns (points_3d: (num_points, 3))
    assert frame_annotations.shape[0] == proj_matrices.shape[0], "Number of cameras must match annotations."
    combs = list(itertools.combinations(range(frame_annotations.shape[0]), 2))
    # Every combination makes a prediction, some combinations may not have enough points to triangulate
    points_3d = np.full((len(combs), frame_annotations.shape[1], 3), np.nan, dtype=np.float32)  # (num_combs, num_points, 3)
    for idx, (i, j) in enumerate(combs):
        # Get 2D points from both cameras
        p1_2d = frame_annotations[i] # (num_points, 2)
        p2_2d = frame_annotations[j] # (num_points, 2)
        common_mask = ~np.isnan(p1_2d).any(axis=1) & ~np.isnan(p2_2d).any(axis=1)  # (num_points,)
        if not np.any(common_mask):
            continue
        # Prepare 2D points for triangulation (requires shape [2, N])
        p1_2d = p1_2d[common_mask] # (num_common_points, 2)
        p2_2d = p2_2d[common_mask] # (num_common_points, 2)
        # Expects (3, 4) project matrices and (2, N) points
        points_4d_hom = cv2.triangulatePoints(proj_matrices[i], proj_matrices[j], p1_2d.T, p2_2d.T) # (4, N) homogenous coordinates
        triangulated_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Convert to 3D coordinates (N, 3)
        points_3d[idx, common_mask] = triangulated_3d
    # Average the triangulated points across all combinations
    average = np.nanmean(points_3d, axis=0)  # (num_points, 3)
    return average

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


# def mouse_callback(event, x, y, flags, param):
#     """Handles mouse clicks for point annotation."""
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cam_idx = param['cam_idx']
#         annotations[frame_idx, cam_idx, selected_point_idx] = [x, y]
#         human_annotated[frame_idx, cam_idx, selected_point_idx] = True
#         print(f"Annotated P{selected_point_idx} on Cam {cam_idx} at frame {frame_idx}: ({x}, {y})")

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
            if np.isnan(annotations[frame_idx, cam_idx, idx, 0]):
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
        dist = np.random.uniform(-0.001, 0.001, 5).astype(np.float32)
        
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
    for cam_params in individual:
        # Mutate intrinsics
        fx = cam_params['fx'] + random.uniform(-0.01, 0.01)
        fy = cam_params['fy'] + random.uniform(-0.01, 0.01)
        cx = cam_params['cx'] + random.uniform(-0.01, 0.01)
        cy = cam_params['cy'] + random.uniform(-0.01, 0.01)

        # Mutate distortion
        dist = cam_params['dist'] + np.random.uniform(-0.001, 0.001, cam_params['dist'].shape[0])
        
        # Mutate extrinsics
        rvec = cam_params['rvec'] + np.random.uniform(-np.pi/180, np.pi/180, 3)
        tvec = cam_params['tvec'] + np.random.uniform(-0.01, 0.01, 3)
        
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
    total_reprojection_error = 0.0
    points_evaluated = 0

    num_cams = annotations.shape[1]
    
    # --- Pre-computation for Efficiency ---
    # Get projection matrices and camera parameters once to avoid redundant calculations in the loop.
    # **We are ignoring the intrinsics because we undistort the points before triangulation.**
    proj_matrices = []
    for i in individual:
        R, _ = cv2.Rodrigues(i['rvec'])  # Convert rotation vector to rotation matrix
        P = np.hstack((R, i['tvec'].reshape(-1, 1)))
        proj_matrices.append(P)
    proj_matrices = np.array(proj_matrices)  # (num_cams, 3, 4)

    # Find frames with at least one valid annotation to process.
    valid_frames_mask = np.any(~np.isnan(annotations), axis=(1, 2, 3))
    valid_frame_indices = np.where(valid_frames_mask)[0]

    # --- Main Logic: Iterate by Frame, then Camera Pair ---
    for f_idx in valid_frame_indices:
        frame_annotations = annotations[f_idx]  # (num_cams, num_points, 2)
        undistorted_annotations = np.full_like(frame_annotations, np.nan, dtype=np.float32)
        for c in range(num_cams):
            camera_annotations = frame_annotations[c] # (num_points, 2)
            valid_2d_mask = ~np.isnan(camera_annotations).any(axis=1) # (num_points,)
            valid_2d_points = camera_annotations[valid_2d_mask]  # (num_valid_points, 2)
            undistorted_camera_anns = undistorted_annotations[c] # (num_points, 2)
            undistorted_camera_anns[valid_2d_mask] = cv2.undistortPoints(valid_2d_points.reshape(-1, 1, 2), get_camera_matrix(individual[c]), individual[c]['dist']).reshape(-1, 2)
        points_3d = combination_triangulate(undistorted_annotations, proj_matrices)  # (num_points, 3)
        valid_3d_mask = ~np.isnan(points_3d).any(axis=1)  # (num_points,)
        for c in range(num_cams):
            camera_annotations = frame_annotations[c]  # (num_points, 2)
            valid_2d_mask = ~np.isnan(camera_annotations).any(axis=1) # (num_points,)
            common_mask = valid_3d_mask & valid_2d_mask  # Points that are valid in both 3D and 2D
            valid_3d_points = points_3d[common_mask]  # (num_valid_points, 3)
            valid_2d_points = camera_annotations[common_mask]  # (num_valid_points, 2)
            reprojected = reproject_points(valid_3d_points, individual[c]) # (num_valid_points, 2)
            # Calculate the reprojection error for valid points
            error = np.square(reprojected - valid_2d_points)
            error = np.sqrt(np.sum(error, axis=1))  # Euclidean distance
            total_reprojection_error += np.sum(error)
            points_evaluated += np.sum(common_mask)  # Count how many points were evaluated

    if points_evaluated == 0:
        return 0.0  # Return a very low fitness if no points could be evaluated.
        
    average_error = total_reprojection_error / points_evaluated
    # average_error = total_reprojection_error
    
    # Fitness is the error
    return average_error

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
    points_3d = combination_triangulate(frame_annotations, proj_matrices)  # (num_points, 3)
    reconstructed_3d_points[frame_idx] = points_3d  # Update the global 3D points for this frame

# --- Visualization ---

def draw_ui(frame, cam_idx):
    """Draws UI elements on the frame."""
    # Draw annotated points for the current frame
    for p_idx in range(NUM_POINTS):
        point = annotations[frame_idx, cam_idx, p_idx]
        if not np.isnan(point).any():
            if human_annotated[frame_idx, cam_idx, p_idx]:
                # Draw a while square around annotated points
                cv2.circle(frame, tuple(point.astype(int)), 5 + 2, (255, 255, 255), -1) # White outline for human
            cv2.circle(frame, tuple(point.astype(int)), 5, point_colors[p_idx].tolist(), -1)
            cv2.putText(frame, POINT_NAMES[p_idx], tuple(point.astype(int) + np.array([5, -5])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 2)
    cv2.putText(frame, video_names[cam_idx], (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def create_control_window():
    """Creates the control and info panel."""
    w, h = 400, 200
    control_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Display info
    texts = [
        f"Frame: {frame_idx}/{video_metadata['num_frames']}",
        f"Status: {'Paused' if paused else 'Playing'}",
        f"Annotating Point: {POINT_NAMES[selected_point_idx]}",
        "--- Controls ---",
        "space: play/pause",
        "n/b: next/prev frame",
        "1,2..: select point",
        "q: quit",
        "t: run genetic algorithm",
    ]
    for i, text in enumerate(texts):
        cv2.putText(control_img, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    if best_individual is not None:
        pickle.dump(best_individual, open(os.path.join(DATA_FOLDER, 'best_individual.pkl'), 'wb'))
    print("State saved successfully.")

def load_state():
    """Loads the saved state of annotations and 3D points."""
    global annotations, human_annotated, reconstructed_3d_points, best_individual
    try:
        annotations = np.load(os.path.join(DATA_FOLDER, 'annotations.npy'))
        human_annotated = np.load(os.path.join(DATA_FOLDER, 'human_annotated.npy'))
        reconstructed_3d_points = np.load(os.path.join(DATA_FOLDER, 'reconstructed_3d_points.npy'))
        best_individual = pickle.load(open(os.path.join(DATA_FOLDER, 'best_individual.pkl'), 'rb'))
        print("State loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading state: {e}")

# --- Main Loop ---
def main():
    global frame_idx, paused, selected_point_idx, annotations, train_ga, population, best_individual, needs_3d_reconstruction

    load_videos()
    load_state()  # Load previous state if available

    # Create windows and set mouse callbacks
    # Arrange windows in a grid
    grid_cols = int(np.ceil(np.sqrt(video_metadata['num_videos'])))
    # grid_rows = int(np.ceil(video_metadata['num_videos'] / grid_cols))
    win_w = 600
    win_h = 500
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

    while True:
        # Set all captures to the current frame index
        current_frames = []
        if prev_frame_idx != frame_idx:
            for i, cap in enumerate(video_captures):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error.")
                    paused = True
                    frame_idx = max(0, frame_idx -1) # Go back one frame if at end
                    continue
                current_frames.append(frame)
            prev_frame_idx = frame_idx
            prev_frames = current_frames
        else:
            current_frames = prev_frames  # Use previous frames if not changing index

        # If not paused and not the first frame, track points
        # if not paused and frame_idx > 0 and prev_gray_frames[i] is not None:
        #     track_points(prev_gray_frames[i], gray_frame, i)

        # Draw UI on frame
        if not train_ga and not scene_viz.is_dragging:
            for i, frame in enumerate(current_frames):
                frame_with_ui = draw_ui(frame.copy(), i)
                cv2.imshow(video_names[i], frame_with_ui)

        if needs_3d_reconstruction and best_individual is not None:
            needs_3d_reconstruction = False
            update_3d_reconstruction(best_individual)
            reproj_3d(current_frames[4].copy(), 4)
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

        # prev_gray_frames = current_gray_frames
        if train_ga:
            run_genetic_step()
            needs_3d_reconstruction = True

        # Update control and 3D plot windows
        create_control_window()
        cv2.imshow(scene_viz.window_name, scene_viz.draw_scene(scene))


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
        elif key == ord('s'):
            save_state()
        elif key == ord('l'):
            load_state()
        elif key == ord('n'):
            paused = True
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
        elif key == ord('b'):
            paused = True
            if frame_idx > 0:
                frame_idx -= 1
        elif key == ord('t'):
            paused = True
            train_ga = not train_ga
            if train_ga:
                print("Training Genetic Algorithm...")
                # for _ in range(100):
                #     run_genetic_step()
        elif key == ord('r'):
            population = []
        elif ord('1') <= key <= ord('9'):
            point_num = key - ord('1')
            if point_num < NUM_POINTS:
                selected_point_idx = point_num
                print(f"Selected point P{selected_point_idx} for annotation.")
        # Cycle through points
        elif key == ord('j'):
            selected_point_idx = (selected_point_idx + 1) % NUM_POINTS
            print(f"Selected point P{selected_point_idx} for annotation.")
        elif key == ord('k'):
            selected_point_idx = (selected_point_idx - 1) % NUM_POINTS
            print(f"Selected point P{selected_point_idx} for annotation.")

        # If playing, advance frame
        if not paused:
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
            else:
                paused = True # Pause at the end

    # Cleanup
    for cap in video_captures:
        cap.release()
    cv2.destroyAllWindows()
    print("Exiting application.")


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos there.")
    main()
