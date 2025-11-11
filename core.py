"""
Core algos for tracking and calibration (and some stuff for visualisation)
"""
import random
import cv2
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Optional

from utils import annotations_to_polars, get_projection_matrix, reproject_points, undistort_points, triangulate_points, \
    calculate_reproj_confidence
from viz_3d import SceneObject
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


# ============================================================================
# Camera geometry
# ============================================================================

def snap_annotation(
    app_state: 'AppState',
    target_cam_idx: int,
    point_idx: int,
    frame_idx: int
) -> Optional[np.ndarray]:
    """
    Refines a 2D annotation by triangulating from other views and reprojecting

    This "snaps" a new user click to a (maybe) more accurate position based
    on the consensus of other existing annotations for the same point.

    Returns:
        A (2,) numpy array with the refined 2D coordinates, or None if
        refinement is not possible (fewer than 2 other views)
    """

    with app_state.lock:
        # all annotations for the specific point in the specific frame
        annotations_for_point = app_state.annotations[frame_idx, :, point_idx, :]
        best_individual = app_state.best_individual

    if best_individual is None:
        return None

    # Which cameras have a valid annotation for this point (excluding target camera)
    num_cams = annotations_for_point.shape[0]
    valid_cam_indices = []
    valid_annotations_2d = []

    for i in range(num_cams):
        if i == target_cam_idx:
            continue
        if not np.isnan(annotations_for_point[i]).any():
            valid_cam_indices.append(i)
            valid_annotations_2d.append(annotations_for_point[i])

    # We need at least 2 other views to triangulate
    if len(valid_cam_indices) < 2:
        return None

    valid_annotations_2d = np.array(valid_annotations_2d)

    proj_matrices = np.array([get_projection_matrix(best_individual[i]) for i in valid_cam_indices])

    # triangulate_points expects shape (num_cams, num_points, 2)
    # Only one point here so reshape
    point_to_triangulate = valid_annotations_2d.reshape(len(valid_cam_indices), 1, 2)

    # The function returns shape (num_points, 3), so we get (1, 3)
    point_3d_single = triangulate_points(point_to_triangulate, proj_matrices)

    if np.isnan(point_3d_single).any():
        return None

    point_3d = point_3d_single.flatten()

    # Reproject the 3D point back onto the target camera's view
    target_cam_params = best_individual[target_cam_idx]
    reprojected_point_2d = reproject_points(point_3d, target_cam_params)

    if reprojected_point_2d.size == 0:
        return None

    return reprojected_point_2d.flatten()


def update_annotations_from_3d(
    app_state: 'AppState',
    frame_idx: int,
    final_3d_skeleton_kps: Dict[str, np.ndarray]
):
    """
    Reprojects a final 3D skeleton back to all 2D views to create a corrected
    set of annotations for the next frame's tracking.
    """
    # TODO: This array building not the most efficient, shuld use the functions in mokap

    if not final_3d_skeleton_kps:
        return

    print(f"[Frame {frame_idx}] Feedback Loop: Correcting 2D annotations based on the final 3D pose.")

    with app_state.lock:
        if app_state.best_individual is None:
            return

        calibration = app_state.best_individual
        point_names = app_state.point_names
        num_cams = len(calibration)

        points_to_reproject_3d = []
        point_indices_in_app_state = []
        for p_name, pos_3d in final_3d_skeleton_kps.items():
            if p_name in point_names:
                points_to_reproject_3d.append(pos_3d)
                point_indices_in_app_state.append(point_names.index(p_name))

        if not points_to_reproject_3d:
            return

        points_to_reproject_3d = np.array(points_to_reproject_3d)

        for cam_idx in range(num_cams):
            # Reproject all points for this camera
            reprojected_points_2d = reproject_points(points_to_reproject_3d, calibration[cam_idx])

            if reprojected_points_2d.size == 0:
                continue

            # Update the annotations array
            for i, p_idx in enumerate(point_indices_in_app_state):
                # only update if the point was not manually annotated
                if not app_state.human_annotated[frame_idx, cam_idx, p_idx]:
                    app_state.annotations[frame_idx, cam_idx, p_idx] = reprojected_points_2d[i]
                    app_state.human_annotated[frame_idx, cam_idx, p_idx] = False


def process_frame(
        frame_idx: int,
        app_state: 'AppState',
        reconstructor: 'Reconstructor',
        tracker: 'MultiObjectTracker',
        prev_frames: List[np.ndarray],
        current_frames: List[np.ndarray]
):
    """
    Runs one full cycle: LK prediction -> mokap correction -> feedback
    """
    with app_state.lock:
        camera_names = app_state.video_names
        point_names = app_state.point_names

    # Get geometric prediction and confidence score from LK tracking
    annotations_from_lk = track_points(
        app_state, prev_frames, current_frames, frame_idx
    )
    num_lk_points = np.sum(~np.isnan(annotations_from_lk[..., 0]))
    print(f"\n--- [Frame {frame_idx}] ---")
    print(f"LK PREDICT:    Started with {num_lk_points} raw 2D keypoints from Optical Flow.")

    # Run Mokap reconstruction using the LK results and their confidence score
    df_frame = annotations_to_polars(annotations_from_lk, frame_idx, camera_names, point_names)
    points_soup = reconstructor.reconstruct_frame(
        df_frame=df_frame,
        keypoint_names=point_names
    )
    print(f"MOKAP RECON:   Input {len(df_frame)} 2D points -> Produced a soup of {len(points_soup)} 3D candidates.")

    active_tracklets = tracker.update(points_soup, frame_idx)

    final_3d_skeleton_kps = None
    if active_tracklets:
        best_tracklet = max(active_tracklets, key=lambda t: len(t.skeleton.keypoints))
        final_3d_skeleton_kps = best_tracklet.skeleton.keypoints
        num_kps = len(final_3d_skeleton_kps)
        score = best_tracklet.skeleton.score
        print(f"MOKAP ASSEMBLE: SUCCESS -> Assembled 1 skeleton with {num_kps} keypoints (Score: {score:.2f}).")
    else:
        print(f"MOKAP ASSEMBLE: Could not assemble any skeletons from the soup.")

    # feedback loop
    with app_state.lock:
        if app_state.annotations.shape[3] != 3:
            old_annotations = app_state.annotations
            app_state.annotations = np.full(
                (old_annotations.shape[0], old_annotations.shape[1], old_annotations.shape[2], 3), np.nan,
                dtype=np.float32)

            app_state.annotations[..., :2] = old_annotations
            app_state.annotations[..., 2] = 1.0  # default confidence for old data

        if final_3d_skeleton_kps:
            calibration = app_state.best_individual
            num_cams = len(calibration)
            points_to_reproject_3d, p_indices = [], []

            for p_name, pos_3d in final_3d_skeleton_kps.items():
                if p_name in point_names:
                    points_to_reproject_3d.append(pos_3d)
                    p_indices.append(point_names.index(p_name))

            points_to_reproject_3d = np.array(points_to_reproject_3d)
            annotations_from_model = np.full_like(app_state.annotations[frame_idx], np.nan)

            if points_to_reproject_3d.size > 0:
                for cam_idx in range(num_cams):
                    reprojected_2d = reproject_points(points_to_reproject_3d, calibration[cam_idx])
                    if reprojected_2d.size > 0:
                        for i, p_idx in enumerate(p_indices):
                            # Rescued points get a fixed (medium high) confidence score
                            annotations_from_model[cam_idx, p_idx] = [*reprojected_2d[i], 0.75]

            final_annotations = np.where(
                ~np.isnan(annotations_from_lk[..., 0, np.newaxis]),
                annotations_from_lk,
                annotations_from_model
            )

            app_state.annotations[frame_idx] = final_annotations
            app_state.human_annotated[frame_idx] = False

        else:
            # Fallback path: Mokap failed. Keep only the successful LK tracks.
            # Create a mask for where to copy
            mask = ~np.isnan(annotations_from_lk[..., 0])
            app_state.annotations[frame_idx][mask] = annotations_from_lk[mask]
            # Set confidence to NaN where LK failed
            app_state.annotations[frame_idx][~mask, 2] = np.nan
            app_state.human_annotated[frame_idx] = False


# ============================================================================
# Optic flow tracking
# ============================================================================

def track_points(
        app_state: 'AppState',
        prev_frames: List[np.ndarray],
        current_frames: List[np.ndarray],
        frame_idx: int
) -> np.ndarray:
    """
    Tracks points and returns new 2D annotations.
    (onfidence score based on (sort of) RANSAC-like multi-view consistency)
    """
    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        annotations_prev = app_state.annotations[frame_idx - 1][..., :2].copy()
        calibration = app_state.best_individual

    output_annotations = np.full((annotations_prev.shape[0], annotations_prev.shape[1], 3), np.nan, dtype=np.float32)

    if calibration is None:
        return output_annotations

    num_videos = len(current_frames)
    proj_matrices = [get_projection_matrix(cam) for cam in calibration]
    raw_lk_coords = np.full_like(annotations_prev, np.nan)

    # Run LK Optic flow
    for cam_idx in range(num_videos):
        prev_gray = cv2.cvtColor(prev_frames[cam_idx], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frames[cam_idx], cv2.COLOR_BGR2GRAY)
        p0_2d_prev = annotations_prev[cam_idx]
        track_mask = ~np.isnan(p0_2d_prev).any(axis=1)

        if focus_mode:
            is_valid = track_mask[selected_idx]
            track_mask[:] = False
            track_mask[selected_idx] = is_valid

        if not np.any(track_mask):
            continue
        start_points_for_lk = p0_2d_prev[track_mask].reshape(-1, 1, 2)
        point_indices_to_track = np.where(track_mask)[0]

        if start_points_for_lk.size == 0:
            continue

        # Froward and backward test
        p1_forward, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, start_points_for_lk, None,
                                                             **config.LK_PARAMS)

        p0_backward, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1_forward, None,
                                                              **config.LK_PARAMS)

        for i, p_idx in enumerate(point_indices_to_track):
            fb_error = np.linalg.norm(start_points_for_lk[i] - p0_backward[i])

            is_lk_successful = status_fwd[i] == 1 and status_bwd[
                i] == 1 and fb_error < config.FORWARD_BACKWARD_THRESHOLD

            if is_lk_successful:
                raw_lk_coords[cam_idx, p_idx] = p1_forward[i].flatten()

    # Confidence scoring
    for p_idx in range(app_state.num_points):

        for cam_idx_to_check in range(num_videos):
            point_to_check = raw_lk_coords[cam_idx_to_check, p_idx]

            if np.isnan(point_to_check).any():
                continue

            peer_cam_indices = []
            peer_annotations_2d = []

            for i in range(num_videos):
                if i == cam_idx_to_check:
                    continue

                peer_point = raw_lk_coords[i, p_idx]

                if not np.isnan(peer_point).any():
                    peer_cam_indices.append(i)
                    peer_annotations_2d.append(peer_point)

            confidence = 1.0
            if len(peer_cam_indices) >= 2:

                # Test every possible pair of peers to find the best possible consensus
                # TODO: This works pretty well but is slow

                min_drift_error = float('inf')

                for peer_pair_indices in combinations(range(len(peer_cam_indices)), 2):
                    idx1, idx2 = peer_pair_indices

                    cam1_global_idx, cam2_global_idx = peer_cam_indices[idx1], peer_cam_indices[idx2]
                    annots_pair = np.array([peer_annotations_2d[idx1], peer_annotations_2d[idx2]])
                    proj_mats_pair = [proj_matrices[cam1_global_idx], proj_matrices[cam2_global_idx]]

                    # Triangulate a hypothesis from this pair
                    point_3d_hypothesis = triangulate_points(
                        annots_pair.reshape(2, 1, 2),
                        proj_mats_pair
                    ).flatten()

                    if np.isnan(point_3d_hypothesis).any():
                        continue

                    # Reproject hypothesis back to the camera we are checking
                    reprojected = reproject_points(point_3d_hypothesis, calibration[cam_idx_to_check])
                    if reprojected.size == 0:
                        continue

                    # Calculate error for this specific hypothesis
                    error = np.linalg.norm(point_to_check - reprojected.flatten())

                    # Keep track of the minimum error found so far
                    if error < min_drift_error:
                        min_drift_error = error

                # at least one valid consensus is found: use it to calculate confidence
                if min_drift_error != float('inf'):
                    confidence = max(0.0, 1.0 - (min_drift_error / config.LK_CONFIDENCE_MAX_ERROR))

            output_annotations[cam_idx_to_check, p_idx] = [*point_to_check, confidence]

    return output_annotations


# ============================================================================
# Visualisation
# ============================================================================

def create_camera_visual(
    cam_params: Dict[str, Any],
    label: str,
    scene_center: np.ndarray
) -> List[SceneObject]:
    """Generate 3D visualisation objects for a camera frustum."""

    rvec, tvec = cam_params['rvec'], cam_params['tvec']

    R_c2w, _ = cv2.Rodrigues(rvec)
    camera_center_world = tvec.flatten()

    distance_to_center = np.linalg.norm(camera_center_world - scene_center)
    scale = distance_to_center * 0.2  # Frustum size = 20% of distance to center

    w, h, depth = 0.3 * scale, 0.2 * scale, 0.5 * scale

    # Frustum pyramid in local camera coordinates
    # (OpenCV camera convention is +X right, +Y down, +Z forward)
    pyramid_pts_cam = np.array([
        [0, 0, 0],           # Camera center (pyramid apex)
        [-w, -h, depth],     # Bottom left of image plane (in camera space)
        [w, -h, depth],      # Bottom right
        [w, h, depth],       # Top right
        [-w, h, depth]       # Top left
    ])

    # Transform local camera pyramid points into world coordinates
    pyramid_pts_world = (R_c2w @ pyramid_pts_cam.T).T + camera_center_world
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