import random
import cv2
import numpy as np
import itertools
from typing import List, Dict, Any
from state import AppState
from viz_3d import SceneObject

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Configuration # TODO: load fron config file
LK_PARAMS = dict(winSize=(9, 9),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
NUM_DIST_COEFFS = 14



def get_camera_matrix(cam_params: Dict[str, Any]) -> np.ndarray:
    """Constructs the 3x3 camera intrinsic matrix."""
    # TODO: Use jax version from mokap

    return np.array([[cam_params['fx'], 0, cam_params['cx']],
                     [0, cam_params['fy'], cam_params['cy']],
                     [0, 0, 1]], dtype=np.float32)

def get_projection_matrix(cam_params: Dict[str, Any]) -> np.ndarray:
    """Constructs the 3x4 projection matrix."""
    # TODO: Use jax version from mokap

    K = get_camera_matrix(cam_params)
    R, _ = cv2.Rodrigues(cam_params['rvec'])
    return K @ np.hstack((R, cam_params['tvec'].reshape(-1, 1)))

def reproject_points(points_3d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """Reprojects 3D points to a 2D image plane for a single camera."""
    # TODO: Use jax version from mokap

    if points_3d.size == 0:
        return np.array([])

    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 1, 3)

    elif points_3d.ndim == 2:
        points_3d = points_3d[:, np.newaxis, :]

    reprojected, _ = cv2.projectPoints(
        points_3d, cam_params['rvec'], cam_params['tvec'],
        get_camera_matrix(cam_params), cam_params['dist']
    )
    return reprojected.squeeze(axis=1) if reprojected is not None else np.array([])

def undistort_points(points_2d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """Undistorts 2D points using camera parameters."""
    # TODO: Use jax version from mokap

    valid_mask = ~np.isnan(points_2d).any(axis=-1)
    if not np.any(valid_mask):
        return np.full_like(points_2d, np.nan)

    valid_points = points_2d[valid_mask]
    undistorted_norm = cv2.undistortImagePoints(
        valid_points.reshape(-1, 1, 2), get_camera_matrix(cam_params), cam_params['dist']
    )

    undistorted_full = np.full_like(points_2d, np.nan)
    undistorted_full[valid_mask] = undistorted_norm.reshape(-1, 2)
    return undistorted_full


# Point Tracking (Optic flow)

def track_points(app_state: AppState, prev_frames: List[np.ndarray], current_frames: List[np.ndarray]):
    """Tracks keypoints from previous to current frames using Lucas-Kanade algo."""

    with app_state.lock:
        frame_idx = app_state.frame_idx
        focus_selected_point = app_state.focus_selected_point
        selected_point_idx = app_state.selected_point_idx
        annotations = app_state.annotations
        human_annotated = app_state.human_annotated

    num_videos = len(current_frames)
    new_annotations = annotations.copy()

    for cam_idx in range(num_videos):
        prev_gray = cv2.cvtColor(prev_frames[cam_idx], cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frames[cam_idx], cv2.COLOR_BGR2GRAY)

        p0 = annotations[frame_idx - 1, cam_idx, :, :]
        valid_indices = ~np.isnan(p0).any(axis=1)

        if focus_selected_point:
            is_valid = valid_indices[selected_point_idx]
            valid_indices[:] = False
            valid_indices[selected_point_idx] = is_valid

        if not np.any(valid_indices):
            continue

        p0_valid = p0[valid_indices].reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0_valid, None, **LK_PARAMS)

        if p1 is not None and st.any():
            good_new = p1[st == 1]
            original_indices = np.where(valid_indices)[0][st.flatten() == 1]

            for i, original_idx in enumerate(original_indices):
                if not human_annotated[frame_idx, cam_idx, original_idx]:
                    new_annotations[frame_idx, cam_idx, original_idx] = good_new[i]

    with app_state.lock:
        app_state.annotations[:] = new_annotations

# 3D reconstruction

def combination_triangulate(frame_annotations: np.ndarray, proj_matrices: np.ndarray) -> np.ndarray:
    """Triangulates 3D points from 2D correspondences using all camera pair combinations."""

    num_cams, num_points, _ = frame_annotations.shape # (num_cams, num_points, 2)

    combs = list(itertools.combinations(range(num_cams), 2))
    points_3d_votes = np.full((len(combs), num_points, 3), np.nan, dtype=np.float32)

    for i, (cam1_idx, cam2_idx) in enumerate(combs):
        p1 = frame_annotations[cam1_idx]
        p2 = frame_annotations[cam2_idx]

        mask = ~np.isnan(p1).any(axis=1) & ~np.isnan(p2).any(axis=1)
        if not np.any(mask):
            continue

        points_4d_hom = cv2.triangulatePoints(proj_matrices[cam1_idx], proj_matrices[cam2_idx], p1[mask].T, p2[mask].T)
        triangulated_3d = (points_4d_hom[:3] / (points_4d_hom[3] + 1e-6)).T
        points_3d_votes[i, mask] = triangulated_3d

    return np.nanmean(points_3d_votes, axis=0)

def update_3d_reconstruction(app_state: AppState):
    """Uses the best camera parameters to reconstruct 3D points for the current frame."""

    with app_state.lock:
        frame_idx = app_state.frame_idx
        annotations = app_state.annotations[frame_idx]   # (num_cams, num_points, 2)
        best_individual = app_state.best_individual
        if best_individual is None:
            return

    proj_matrices = np.array([get_projection_matrix(cam) for cam in best_individual])
    points_3d = combination_triangulate(annotations, proj_matrices)

    with app_state.lock:
        app_state.reconstructed_3d_points[app_state.frame_idx] = points_3d


# Visualization helpers

def draw_ui(frame: np.ndarray, cam_idx: int, app_state: AppState) -> np.ndarray:
    """Draws complex UI elements like reprojection lines on a single video frame."""

    with app_state.lock:
        frame_idx = app_state.frame_idx
        best_individual = app_state.best_individual
        annotations_for_frame = app_state.annotations[frame_idx, cam_idx]
        points_3d_for_frame = app_state.reconstructed_3d_points[frame_idx]
        point_colors = app_state.point_colors

    # Only draw if we have a calibrated camera
    if not best_individual:
        return frame

    cam_params = best_individual[cam_idx]
    valid_3d_points_mask = ~np.isnan(points_3d_for_frame).any(axis=1)

    if np.any(valid_3d_points_mask):
        points_to_reproject = points_3d_for_frame[valid_3d_points_mask]
        reprojected_pts = reproject_points(points_to_reproject, cam_params)
        original_indices = np.where(valid_3d_points_mask)[0]

        for i, p_idx in enumerate(original_indices):
            # Only draw line if the original 2D point exists
            if np.isnan(annotations_for_frame[p_idx]).any():
                continue

            annotated_pt = tuple(annotations_for_frame[p_idx].astype(int))
            reprojected_pt = tuple(reprojected_pts[i].astype(int))
            color = point_colors[p_idx].tolist()

            cv2.line(frame, annotated_pt, reprojected_pt, color, 1)
            cv2.circle(frame, reprojected_pt, 3, color, -1) # Draw a small circle at the reprojected end

    return frame

def create_camera_visual(cam_params: Dict[str, Any], label: str) -> List[SceneObject]:
    """Generates a list of SceneObjects to represent a camera's pose as a pyramid."""

    rvec, tvec = cam_params['rvec'], cam_params['tvec']
    R_w2c, _ = cv2.Rodrigues(rvec)
    R_c2w, t_c2w = R_w2c.T, -R_w2c.T @ tvec.flatten()

    scale = 1.0

    w, h, depth = 0.5 * scale, 0.4 * scale, 0.8 * scale
    pyramid_pts_cam = np.array([
        [0, 0, 0], [-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]
    ])

    pyramid_pts_world = (R_c2w @ pyramid_pts_cam.T).T + t_c2w
    p0, p1, p2, p3, p4 = pyramid_pts_world

    color = (255, 255, 0)

    return [SceneObject(type='point', coords=p0, color=color, label=label)] + [
        SceneObject(type='line', coords=np.array([p0, p_corner]), color=color, label=None) for p_corner in [p1, p2, p3, p4]
    ] + [
        SceneObject(type='line', coords=np.array([p1, p2]), color=color, label=None),
        SceneObject(type='line', coords=np.array([p2, p3]), color=color, label=None),
        SceneObject(type='line', coords=np.array([p3, p4]), color=color, label=None),
        SceneObject(type='line', coords=np.array([p4, p1]), color=color, label=None),
    ]


# Genetic Algorithm core

def _create_individual(video_metadata: Dict) -> List[Dict]:
    """Creates a single random individual (set of camera parameters)."""

    w, h = video_metadata['width'], video_metadata['height']
    num_cameras = video_metadata['num_videos']
    radius = 5.0
    individual = []

    for i in range(num_cameras):
        angle = (2 * np.pi / num_cameras) * i
        cam_in_world = np.array([radius * np.cos(angle), 2.0, radius * np.sin(angle)])

        target = np.array([0., 0., 0.])
        up = np.array([0., 1., 0.])

        forward = (target - cam_in_world) / np.linalg.norm(target - cam_in_world)
        right = np.cross(forward, up)
        cam_up = np.cross(right, forward)
        R = np.array([-right, cam_up, -forward])
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ cam_in_world

        individual.append({
            'fx': random.uniform(w * 0.8, w * 1.5), 'fy': random.uniform(h * 0.8, h * 1.5),
            'cx': w / 2 + random.uniform(-w * 0.05, w * 0.05), 'cy': h / 2 + random.uniform(-h * 0.05, h * 0.05),
            'dist': np.random.normal(0.0, 0.001, size=NUM_DIST_COEFFS),
            'rvec': rvec.flatten(), 'tvec': tvec.flatten()
        })

    return individual

def _fitness(individual: List[Dict], annotations: np.ndarray, calibration_frames: List[int], video_metadata: Dict) -> float:
    """The core fitness function."""

    if not calibration_frames:
        return float('inf')

    calib_mask = np.zeros(annotations.shape[0], dtype=bool)
    calib_mask[calibration_frames] = True

    valid_frames_mask = np.any(~np.isnan(annotations), axis=(1, 2, 3))
    combined_mask = calib_mask & valid_frames_mask

    if not np.any(combined_mask):
        return float('inf')

    valid_annotations = annotations[combined_mask]
    num_cams = video_metadata['num_videos']
    proj_matrices = np.array([get_projection_matrix(cam) for cam in individual])

    undistorted_annotations = np.full_like(valid_annotations, np.nan)

    for c in range(num_cams):
        undistorted_annotations[:, c] = undistort_points(valid_annotations[:, c], individual[c])

    all_points_3d = []
    for frame_idx in range(valid_annotations.shape[0]):
        points_3d_for_frame = combination_triangulate(valid_annotations[frame_idx], proj_matrices)
        all_points_3d.append(points_3d_for_frame)

    points_3d = np.array(all_points_3d)  # (num_valid_frames, num_points, 3)

    total_error = 0
    points_evaluated = 0
    for c in range(num_cams):
        valid_3d_mask = ~np.isnan(points_3d).any(axis=-1)
        valid_2d_mask = ~np.isnan(valid_annotations[:, c]).any(axis=-1)
        common_mask = valid_3d_mask & valid_2d_mask

        if not np.any(common_mask):
            continue

        reprojected = reproject_points(points_3d[common_mask], individual[c])
        error = np.linalg.norm(reprojected - valid_annotations[:, c][common_mask], axis=1)

        total_error += np.sum(error)
        points_evaluated += len(error)

    return total_error / points_evaluated if points_evaluated > 0 else float('inf')

def run_genetic_step(ga_state: Dict[str, Any]) -> Dict[str, Any]:
    """Performs a single generation of the Genetic Algorithm."""

    POPULATION_SIZE = 200
    ELITISM_RATE = 0.1
    MUTATION_RATE = 0.8
    MUTATION_STRENGTH = 0.1

    population = ga_state.get("population")
    best_fitness = ga_state.get("best_fitness", float('inf'))
    best_individual = ga_state.get("best_individual")
    generation = ga_state.get("generation", 0)

    if population is None:
        population = [_create_individual(ga_state['video_metadata']) for _ in range(POPULATION_SIZE)]

    fitness_scores = np.array([
        _fitness(ind, ga_state['annotations'], ga_state['calibration_frames'], ga_state['video_metadata'])
        for ind in population
    ])

    sorted_indices = np.argsort(fitness_scores)

    if fitness_scores[sorted_indices[0]] < best_fitness:
        best_fitness = fitness_scores[sorted_indices[0]]
        best_individual = population[sorted_indices[0]]

    # Create next generation
    num_elites = int(POPULATION_SIZE * ELITISM_RATE)
    next_population = [population[i] for i in sorted_indices[:num_elites]]

    while len(next_population) < POPULATION_SIZE:
        # Tournament selection
        p1_idx, p2_idx = np.random.choice(sorted_indices, 2)
        parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p2_idx]
        p3_idx, p4_idx = np.random.choice(sorted_indices, 2)
        parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p4_idx]

        # Crossover (simple: average parameters)
        child = []
        for i in range(len(parent1)):
            child_cam = {}
            for key in parent1[i]:
                if isinstance(parent1[i][key], np.ndarray):
                    child_cam[key] = (parent1[i][key] + parent2[i][key]) / 2
                else:
                    child_cam[key] = (parent1[i][key] + parent2[i][key]) / 2

            # Mutation
            if np.random.rand() < MUTATION_RATE:
                for key in ['fx', 'fy', 'cx', 'cy']:
                    child_cam[key] += np.random.normal(0, MUTATION_STRENGTH * child_cam[key])
                for key in ['rvec', 'tvec']:
                    child_cam[key] += np.random.normal(0, MUTATION_STRENGTH, size=child_cam[key].shape)

            child.append(child_cam)
        next_population.append(child)

    generation += 1

    return {
        "status": "running",
        "new_best_fitness": best_fitness,
        "new_best_individual": best_individual,
        "generation": generation,
        "mean_fitness": np.mean(fitness_scores),
        "std_fitness": np.std(fitness_scores),
        "next_population": next_population # pass new population to next step
    }