"""
Core algos for tracking, triangulation and calibration.
"""
import random
import cv2
import numpy as np
import itertools
from typing import List, Dict, Any, Tuple, Optional

from viz_3d import SceneObject
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state import AppState


# ============================================================================
# Camera geometry
# ============================================================================

# TODO: These will be replaced by the mokap implementations

def get_camera_matrix(cam_params: Dict[str, Any]) -> np.ndarray:
    """3x3 intrinsic matrix from parameters."""
    return np.array([
        [cam_params['fx'], 0.0, cam_params['cx']],
        [0.0, cam_params['fy'], cam_params['cy']],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def get_projection_matrix(cam_params: Dict[str, Any]) -> np.ndarray:
    """3x4 projection matrix from parameters."""
    K = get_camera_matrix(cam_params)

    # rvec and tvec are camera-to-world but we need world-to-camera for projection
    R_c2w, _ = cv2.Rodrigues(cam_params['rvec'])
    t_c2w = cam_params['tvec'].flatten()

    # Invert to get world-to-camera
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w

    return K @ np.hstack((R_w2c, t_w2c.reshape(-1, 1)))


def reproject_points(points_3d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """
    Reproject 3D points to 2D.
    (Assumes rvec and tvec in cam_params are camera-to-world)
    """
    if points_3d.size == 0:
        return np.array([])

    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 1, 3)
    elif points_3d.ndim == 2:
        points_3d = points_3d[:, np.newaxis, :]

    # rvec and tvec are camera-to-world but cv2.projectPoints needs world-to-camera
    R_c2w, _ = cv2.Rodrigues(cam_params['rvec'])
    t_c2w = cam_params['tvec'].flatten()

    # Invert to get world-to-camera
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    rvec_w2c, _ = cv2.Rodrigues(R_w2c)

    reprojected, _ = cv2.projectPoints(
        points_3d,
        rvec_w2c,
        t_w2c,
        get_camera_matrix(cam_params),
        cam_params['dist']
    )
    return reprojected.squeeze(axis=1) if reprojected is not None else np.array([])


def undistort_points(points_2d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """Undistort 2D points."""

    valid_mask = ~np.isnan(points_2d).any(axis=-1)
    if not np.any(valid_mask):
        return np.full_like(points_2d, np.nan)

    valid_points = points_2d[valid_mask]
    undistorted = cv2.undistortImagePoints(
        valid_points.reshape(-1, 1, 2),
        get_camera_matrix(cam_params),
        cam_params['dist']
    )

    result = np.full_like(points_2d, np.nan)
    result[valid_mask] = undistorted.reshape(-1, 2)
    return result


def calculate_fundamental_matrices(calibration: List[Dict]) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Calculate F mat for each camera pair
    (rvec and tvec are camera-to-world)
    """
    num_cams = len(calibration)
    f_mats = {}

    Ks = [get_camera_matrix(cam) for cam in calibration]
    Rs_c2w = []
    ts_c2w = []

    for cam in calibration:
        R, _ = cv2.Rodrigues(cam['rvec'])
        Rs_c2w.append(R)
        ts_c2w.append(cam['tvec'].flatten())

    for i, j in itertools.product(range(num_cams), repeat=2):

        if i == j:
            continue

        K_i, K_j = Ks[i], Ks[j]
        R_i, t_i = Rs_c2w[i], ts_c2w[i]
        R_j, t_j = Rs_c2w[j], ts_c2w[j]

        # Relative transformation camera i to camera j
        R_rel = R_j.T @ R_i
        t_rel = R_j.T @ (t_i - t_j)

        # Essential mat E = [t]_x @ R
        t_skew = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0]
        ])
        E = t_skew @ R_rel

        # Fundamental mat F = K_j^-T @ E @ K_i^-1
        F = np.linalg.inv(K_j).T @ E @ np.linalg.inv(K_i)

        # rank-2 constraint
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0.0
        F_corrected = U @ np.diag(S) @ Vt

        f_mats[(i, j)] = F_corrected

    return f_mats


# ============================================================================
# Optic flow tracking
# ============================================================================

def track_points(
    app_state: 'AppState',
    prev_frames: List[np.ndarray],
    current_frames: List[np.ndarray],
    frame_idx: int
):
    """Track keypoints using Lucas-Kanade algo."""

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        annotations = app_state.annotations
        human_annotated = app_state.human_annotated

    num_videos = len(current_frames)

    for cam_idx in range(num_videos):
        prev_gray = cv2.cvtColor(prev_frames[cam_idx], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frames[cam_idx], cv2.COLOR_BGR2GRAY)

        p0 = annotations[frame_idx - 1, cam_idx, :, :]
        valid_mask = ~np.isnan(p0).any(axis=1)

        # In focus mode: only track selected point
        if focus_mode:
            is_valid = valid_mask[selected_idx]
            valid_mask[:] = False
            valid_mask[selected_idx] = is_valid

        if not np.any(valid_mask):
            continue

        p0_valid = p0[valid_mask].reshape(-1, 1, 2)
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0_valid, None, **config.LK_PARAMS
        )

        if p1 is not None:
            good_tracks = (status.flatten() == 1) & (error.flatten() < config.LK_ERROR_THRESHOLD)

            # Check if there are any valid tracks after filtering
            if np.any(good_tracks):
                tracked_points = p1[good_tracks]
                original_indices = np.where(valid_mask)[0][good_tracks]

                with app_state.lock:
                    for i, orig_idx in enumerate(original_indices):
                        if not human_annotated[frame_idx, cam_idx, orig_idx]:
                            # Keep human annotations intact
                            annotations[frame_idx, cam_idx, orig_idx] = tracked_points[i]
                            human_annotated[frame_idx, cam_idx, orig_idx] = False


# ============================================================================
# 3D Reconstruction
# ============================================================================

# TODO: Also replaced by mokap implementation

def triangulate_points(
    frame_annotations: np.ndarray,
    proj_matrices: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from 2D correspondences using all camera pairs."""

    num_cams, num_points, _ = frame_annotations.shape
    camera_pairs = list(itertools.combinations(range(num_cams), 2))

    votes = np.full((len(camera_pairs), num_points, 3), np.nan, dtype=np.float32)

    for i, (cam1, cam2) in enumerate(camera_pairs):
        p1 = frame_annotations[cam1]
        p2 = frame_annotations[cam2]

        # Only triangulate where both views have valid points
        valid = ~np.isnan(p1).any(axis=1) & ~np.isnan(p2).any(axis=1)
        if not np.any(valid):
            continue

        points_4d = cv2.triangulatePoints(
            proj_matrices[cam1],
            proj_matrices[cam2],
            p1[valid].T,
            p2[valid].T
        )
        points_3d = (points_4d[:3] / (points_4d[3] + 1e-6)).T
        votes[i, valid] = points_3d

    return np.nanmean(votes, axis=0)


def refine_annotation(
    app_state: 'AppState',
    target_cam_idx: int,
    point_idx: int,
    frame_idx: int
) -> Optional[np.ndarray]:
    """
    Refines a 2D annotation by triangulating from other views and reprojecting.
    This snaps a new user click to a more accurate position based on other existing annotations.

    Returns:
        A (2,) array with the refined 2D coordinates, or None if refinement is not possible.
    """

    with app_state.lock:
        annotations_for_point = app_state.annotations[frame_idx, :, point_idx, :]
        best_individual = app_state.best_individual

    if best_individual is None:
        return None

    # Which cameras have a valid annotation for this point (excluding target camera ofc)?
    num_cams = annotations_for_point.shape[0]
    valid_cam_indices = []
    valid_annotations_2d = []

    for i in range(num_cams):
        if i == target_cam_idx:
            continue
        if not np.isnan(annotations_for_point[i]).any():
            valid_cam_indices.append(i)
            valid_annotations_2d.append(annotations_for_point[i])

    # We need at least two other views to triangulate
    if len(valid_cam_indices) < 2:
        return None

    valid_annotations_2d = np.array(valid_annotations_2d)
    proj_matrices = np.array([get_projection_matrix(best_individual[i]) for i in valid_cam_indices])

    # triangulate_points expects (num_cams, num_points, 2)
    point_to_triangulate = valid_annotations_2d.reshape(len(valid_cam_indices), 1, 2)
    # and returns (num_points, 3), so we get (1, 3)
    point_3d_single = triangulate_points(point_to_triangulate, proj_matrices)

    if np.isnan(point_3d_single).any():
        return None

    point_3d = point_3d_single.flatten()  # (3,)

    # Reproject the point back into the target camera's view
    target_cam_params = best_individual[target_cam_idx]
    reprojected_point_2d = reproject_points(point_3d, target_cam_params)

    if reprojected_point_2d.size == 0:
        return None

    return reprojected_point_2d.flatten()  # (2,)


def update_3d_reconstruction(app_state: 'AppState'):
    """Reconstruct 3D points for the current frame."""

    with app_state.lock:
        frame_idx = app_state.frame_idx
        annotations = app_state.annotations[frame_idx]
        best_individual = app_state.best_individual

        if best_individual is None:
            return

    proj_matrices = np.array([get_projection_matrix(cam) for cam in best_individual])
    points_3d = triangulate_points(annotations, proj_matrices)

    with app_state.lock:
        app_state.reconstructed_3d_points[frame_idx] = points_3d


# ============================================================================
# Visualisation
# ============================================================================

def create_camera_visual(cam_params: Dict[str, Any], label: str) -> List[SceneObject]:
    """Generate 3D visualisation objects for a camera frustum."""

    rvec, tvec = cam_params['rvec'], cam_params['tvec']
    R_w2c, _ = cv2.Rodrigues(rvec)

    # Camera center in world coordinates
    t_c2w = -R_w2c.T @ tvec.flatten()
    R_c2w = R_w2c.T


    # cameras are about 100 units away so make frustums about 20 units in size
    # TODO: This should be automatic

    scale = 20.0
    w, h, depth = 0.3 * scale, 0.2 * scale, 0.5 * scale

    # Frustum pyramid in camera coordinates (camera looks down +Z axis)
    pyramid_pts_cam = np.array([
        [0, 0, 0],           # Camera center (pyramid apex)
        [-w, -h, depth],     # Bottom left of image plane
        [w, -h, depth],      # Bottom right
        [w, h, depth],       # Top right
        [-w, h, depth]       # Top left
    ])

    # Transform to world coords
    pyramid_pts_world = (R_c2w @ pyramid_pts_cam.T).T + t_c2w
    apex, bl, br, tr, tl = pyramid_pts_world

    color = (255, 255, 0)
    objects = [SceneObject(type='point', coords=apex, color=color, label=label)]

    # Lines from apex to corners
    for corner in [bl, br, tr, tl]:
        objects.append(SceneObject(
            type='line',
            coords=np.array([apex, corner]),
            color=color,
            label=None
        ))

    # Base rectangle
    for p1, p2 in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
        objects.append(SceneObject(
            type='line',
            coords=np.array([p1, p2]),
            color=color,
            label=None
        ))

    return objects


# ============================================================================
# Genetic Algorithm
# ============================================================================

def create_individual(video_metadata: Dict) -> List[Dict]:
    """Create random camera calibration individual."""

    w, h = video_metadata['width'], video_metadata['height']
    num_cameras = video_metadata['num_videos']
    radius = 5.0

    individual = []
    for i in range(num_cameras):
        # Position cameras in a circle around origin
        angle = (2 * np.pi / num_cameras) * i
        cam_pos = np.array([
            radius * np.cos(angle),
            2.0,
            radius * np.sin(angle)
        ])

        # Orient camera toward origin
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        forward = (target - cam_pos) / np.linalg.norm(target - cam_pos)
        right = np.cross(forward, up)
        cam_up = np.cross(right, forward)

        R = np.array([-right, cam_up, -forward])
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ cam_pos

        individual.append({
            'fx': random.uniform(w * 0.8, w * 1.5),
            'fy': random.uniform(h * 0.8, h * 1.5),
            'cx': w / 2 + random.uniform(-w * 0.05, w * 0.05),
            'cy': h / 2 + random.uniform(-h * 0.05, h * 0.05),
            'dist': np.random.normal(0.0, 0.001, size=config.NUM_DIST_COEFFS),
            'rvec': rvec.flatten(),
            'tvec': tvec.flatten()
        })

    return individual


def compute_fitness(
    individual: List[Dict],
    annotations: np.ndarray,
    calibration_frames: List[int],
    video_metadata: Dict
) -> float:
    """Compute reprojection error fitness for a calibration."""

    if not calibration_frames:
        return float('inf')

    # Filter to calibration frames with valid data
    calib_mask = np.zeros(annotations.shape[0], dtype=bool)
    calib_mask[calibration_frames] = True
    valid_mask = np.any(~np.isnan(annotations), axis=(1, 2, 3))
    combined_mask = calib_mask & valid_mask

    if not np.any(combined_mask):
        return float('inf')

    valid_annots = annotations[combined_mask]
    num_cams = video_metadata['num_videos']

    # Compute projection matrices
    proj_matrices = np.array([get_projection_matrix(cam) for cam in individual])

    # Undistort all annotations
    undistorted = np.full_like(valid_annots, np.nan)
    for c in range(num_cams):
        undistorted[:, c] = undistort_points(valid_annots[:, c], individual[c])

    # Triangulate for each frame
    points_3d = np.array([
        triangulate_points(frame_annots, proj_matrices)
        for frame_annots in valid_annots
    ])

    # Compute reprojection error
    total_error = 0.0
    total_points = 0

    for c in range(num_cams):
        valid_3d = ~np.isnan(points_3d).any(axis=-1)
        valid_2d = ~np.isnan(valid_annots[:, c]).any(axis=-1)
        valid = valid_3d & valid_2d

        if not np.any(valid):
            continue

        reprojected = reproject_points(points_3d[valid], individual[c])
        errors = np.linalg.norm(reprojected - valid_annots[:, c][valid], axis=1)

        total_error += np.sum(errors)
        total_points += len(errors)

    return total_error / total_points if total_points > 0 else float('inf')


def run_genetic_step(ga_state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one generation of the genetic algorithm."""

    population = ga_state.get("population")
    best_fitness = ga_state.get("best_fitness", float('inf'))
    best_individual = ga_state.get("best_individual")
    generation = ga_state.get("generation", 0)

    # Initialise population if needed
    if population is None:
        population = [
            create_individual(ga_state['video_metadata'])
            for _ in range(config.GA_POPULATION_SIZE)
        ]

    # Evaluate fitness
    fitness_scores = np.array([
        compute_fitness(
            ind,
            ga_state['annotations'],
            ga_state['calibration_frames'],
            ga_state['video_metadata']
        )
        for ind in population
    ])

    sorted_indices = np.argsort(fitness_scores)

    # Update best individual
    if fitness_scores[sorted_indices[0]] < best_fitness:
        best_fitness = fitness_scores[sorted_indices[0]]
        best_individual = population[sorted_indices[0]]

    # Create next generation
    num_elites = int(config.GA_POPULATION_SIZE * config.GA_ELITISM_RATE)
    next_population = [population[i] for i in sorted_indices[:num_elites]]

    while len(next_population) < config.GA_POPULATION_SIZE:
        # Tournament selection
        p1, p2 = np.random.choice(sorted_indices, 2)
        parent1 = population[p1] if fitness_scores[p1] < fitness_scores[p2] else population[p2]
        p3, p4 = np.random.choice(sorted_indices, 2)
        parent2 = population[p3] if fitness_scores[p3] < fitness_scores[p4] else population[p4]

        # Crossover (average parameters)
        child = []
        for i in range(len(parent1)):
            child_cam = {}
            for key in parent1[i]:
                child_cam[key] = (parent1[i][key] + parent2[i][key]) / 2

            # Mutation
            if np.random.rand() < config.GA_MUTATION_RATE:
                for key in ['fx', 'fy', 'cx', 'cy']:
                    child_cam[key] += np.random.normal(
                        0, config.GA_MUTATION_STRENGTH * child_cam[key]
                    )
                for key in ['rvec', 'tvec']:
                    child_cam[key] += np.random.normal(
                        0, config.GA_MUTATION_STRENGTH, size=child_cam[key].shape
                    )

            child.append(child_cam)
        next_population.append(child)

    return {
        "status": "running",
        "new_best_fitness": best_fitness,
        "new_best_individual": best_individual,
        "generation": generation + 1,
        "mean_fitness": np.mean(fitness_scores),
        "std_fitness": np.std(fitness_scores),
        "next_population": next_population
    }