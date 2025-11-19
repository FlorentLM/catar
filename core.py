"""
Core algos for tracking and calibration (and some stuff for visualisation)
"""
import random
from itertools import combinations
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import jax.numpy as jnp

from mokap.calibration import bundle_adjustment
from mokap.utils.geometry import projective
from mokap.utils.geometry import transforms
from mokap.utils.geometry.fitting import quaternion_average

from utils import annotations_to_polars, get_projection_matrix, reproject_points, undistort_points
from viz_3d import SceneObject
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker
    from state import AppState


# ============================================================================
# Camera geometry
# ============================================================================

def snap_annotation(
    app_state: 'AppState',
    target_cam_idx: int,
    point_idx: int,
    frame_idx: int,
    user_click_pos: np.ndarray
) -> Optional[np.ndarray]:
    """
    Snaps a new user click to a (maybe) more accurate position based
    on the consensus of other existing annotations (weighted by confidence).
    """

    with app_state.lock:
        # all annotations for the specific point in the specific frame
        annotations_for_point = app_state.annotations[frame_idx, :, point_idx, :]
        calibration = app_state.calibration_state.best_individual

    if calibration is None:
        return None

    # Which cameras have a valid annotation for this point (excluding target camera)
    num_cams = annotations_for_point.shape[0]
    valid_cam_indices = []
    valid_annotations_2d = []
    valid_scores = []

    cam_names = list(calibration.keys())


    # TODO: This loop is stupid, it can just be array manipulation

    for i in range(num_cams):

        if i == target_cam_idx:
            continue

        # Check if x, y, and score are all valid
        if not np.isnan(annotations_for_point[i]).any():
            valid_cam_indices.append(i)

            # Append only the (x, y) coordinates for triangulation
            valid_annotations_2d.append(annotations_for_point[i, :2])

            # Also grab the confidence score for weighting
            valid_scores.append(annotations_for_point[i, 2])

    # We need at least 2 other views to triangulate
    if len(valid_cam_indices) < 2:
        return None

    valid_annotations_2d = np.array(valid_annotations_2d)

    # Get calibration parameters for valid cameras
    proj_matrices = np.array([get_projection_matrix(calibration[cam_names[i]]) for i in valid_cam_indices])

    # triangulate_points expects shape (num_cams, num_points, 2)
    # Only one point here so reshape
    point_to_triangulate = valid_annotations_2d.reshape(len(valid_cam_indices), 1, 2)

    # The weights need to be shape (num_cams, num_points)
    weights_for_triangulation = np.array(valid_scores).reshape(len(valid_cam_indices), 1)

    # The function returns shape (num_points, 3), so we get (1, 3)
    point_3d_single = projective.triangulate_points_from_projections(
        points2d=jnp.asarray(point_to_triangulate),
        P_mats=jnp.asarray(proj_matrices),
        weights=jnp.asarray(weights_for_triangulation)
    )
    point_3d_single = np.asarray(point_3d_single)

    if np.isnan(point_3d_single).any():
        return None

    point_3d = point_3d_single.flatten()

    # Reproject the 3D point back onto the target camera's view
    target_cam_params = calibration[cam_names[target_cam_idx]]
    reprojected_point_2d = reproject_points(point_3d, target_cam_params)

    if reprojected_point_2d.size == 0:
        return None

    reprojected_point_2d = reprojected_point_2d.flatten()

    # Only snap if the click is close enough
    distance = np.linalg.norm(reprojected_point_2d - user_click_pos)
    if distance > 15:
        return None

    return reprojected_point_2d


def update_annotations_from_3d(
    app_state: 'AppState',
    frame_idx: int,
    final_3d_skeleton_kps: Dict[str, np.ndarray]
):
    """
    Reprojects a final 3D skeleton back to all 2D views to create a corrected
    set of annotations for the next frame's tracking.
    """
    # TODO: This array building not the most efficient

    if not final_3d_skeleton_kps:
        return

    print(f"[Frame {frame_idx}] Feedback Loop: Correcting 2D annotations based on the final 3D pose.")

    with app_state.lock:
        calibration = app_state.calibration_state.best_individual
        point_names = app_state.point_names
        num_cams = len(app_state.camera_names)
        cam_names = app_state.camera_names

    if calibration is None:
        return

    points_to_reproject_3d = []
    point_indices_in_app_state = []
    for p_name, pos_3d in final_3d_skeleton_kps.items():
        if p_name in point_names:
            points_to_reproject_3d.append(pos_3d)
            point_indices_in_app_state.append(point_names.index(p_name))

    if not points_to_reproject_3d:
        return

    points_to_reproject_3d = np.array(points_to_reproject_3d)

    # TODO: Use batched function, no need to loop
    for cam_idx in range(num_cams):
        cam_name = cam_names[cam_idx]
        # Reproject all points for this camera
        reprojected_points_2d = reproject_points(points_to_reproject_3d, calibration[cam_name])

        if reprojected_points_2d.size == 0:
            continue

        # Update the annotations array
        for i, p_idx in enumerate(point_indices_in_app_state):
            with app_state.lock:
                # only update if the point was not manually annotated
                if not app_state.human_annotated[frame_idx, cam_idx, p_idx]:

                    # TODO: Urgent: confidence for reprojections SHOULD NOT BE 1.0 here
                    app_state.annotations[frame_idx, cam_idx, p_idx] = [*reprojected_points_2d[i], 1.0]
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
        camera_names = app_state.camera_names
        point_names = app_state.point_names

    # Get geometric prediction and confidence score from LK tracking
    annotations_from_lk = track_points(
        app_state, prev_frames, current_frames, frame_idx
    )
    num_lk_points = np.sum(~np.isnan(annotations_from_lk[..., 0]))
    print(f"\n[Frame {frame_idx}]")
    print(f"LK PREDICT:    Started with {num_lk_points} raw 2D keypoints from Optical Flow.")

    # Run Mokap reconstruction using the LK results and their confidence score
    # annotations_from_lk is (C, P, 3) where 3 is (x, y, confidence)
    df_frame = annotations_to_polars(annotations_from_lk, frame_idx, camera_names, point_names)
    points_soup = reconstructor.reconstruct_frame(df_frame=df_frame, keypoint_names=point_names)
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
        calibration = app_state.calibration_state.best_individual
        cam_names = app_state.camera_names
        num_cams = len(cam_names)

        # Create base annotation set from the model, rescuing failed multi-view tracks
        # This will be (C, P, 3)
        annotations_from_model = np.full_like(annotations_from_lk, np.nan)
        if final_3d_skeleton_kps:

            points_to_reproject_3d, p_indices = [], []

            for p_name, pos_3d in final_3d_skeleton_kps.items():
                if p_name in point_names:
                    points_to_reproject_3d.append(pos_3d)
                    p_indices.append(point_names.index(p_name))

            points_to_reproject_3d = np.array(points_to_reproject_3d)
            if points_to_reproject_3d.size > 0 and calibration is not None:
                for cam_idx in range(num_cams):
                    cam_params = calibration[cam_names[cam_idx]]
                    reprojected_2d = reproject_points(points_to_reproject_3d, cam_params)
                    if reprojected_2d.size > 0:
                        for i, p_idx in enumerate(p_indices):
                            # Rescued points get a fixed (medium high) confidence score
                            annotations_from_model[cam_idx, p_idx] = [*reprojected_2d[i], 0.75]

        # Start with the model-based rescue for failed tracks

        # If LK failed (nan) but model has a projection (not nan), use model
        # Otherwise use LK

        final_annotations = np.where(
            ~np.isnan(annotations_from_lk[..., 0, np.newaxis]),
            annotations_from_lk,
            annotations_from_model
        )

        # Re-inject single-view tracks

        # Find points that were successfully tracked by LK but only in a single view
        # (these were lost by the 3D reconstructor but are still useful)

        is_valid_lk_track = ~np.isnan(annotations_from_lk[..., 0])
        num_views_per_point = np.sum(is_valid_lk_track, axis=0)

        for p_idx in range(app_state.num_points):
            if num_views_per_point[p_idx] == 1:
                cam_idx = np.where(is_valid_lk_track[:, p_idx])[0][0]

                # Ensure this point is preserved in the final annotation set for next frame
                # (it's possible the model-based rescue filled it with nan so explicitly overwrite it)
                final_annotations[cam_idx, p_idx] = annotations_from_lk[cam_idx, p_idx]

        app_state.annotations[frame_idx] = final_annotations
        app_state.human_annotated[frame_idx] = False

        # Update the 3D points
        app_state.reconstructed_3d_points[frame_idx].fill(np.nan)
        if final_3d_skeleton_kps:
            for p_name, pos_3d in final_3d_skeleton_kps.items():
                if p_name in point_names:
                    p_idx = point_names.index(p_name)
                    app_state.reconstructed_3d_points[frame_idx, p_idx] = pos_3d


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
    Tracks points and returns new 2D annotations (x, y, confidence)
    (Confidence score based on (sort of) RANSAC-like multi-view consistency)
    """

    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx

        # Only take x, y from previous frame annotations for LK input
        annotations_prev = app_state.annotations[frame_idx - 1][..., :2].copy()
        calibration = app_state.calibration_state.best_individual
        cam_names = app_state.camera_names

    # Output array is (C, P, 3) where 3 is (x, y, confidence)
    output_annotations = np.full((annotations_prev.shape[0], annotations_prev.shape[1], 3), np.nan, dtype=np.float32)

    if calibration is None:
        return output_annotations

    num_videos = len(current_frames)
    proj_matrices = {name: get_projection_matrix(calibration[name]) for name in cam_names}
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

            peer_cam_indices = [i for i, p in enumerate(raw_lk_coords[:, p_idx]) if i != cam_idx_to_check and not np.isnan(p).any()]
            if len(peer_cam_indices) < 2:
                # TODO: uhhh actually I am not sure about this one
                confidence = 1.0 if len(peer_cam_indices) < 2 else 0.0
            else:
                min_drift_error = float('inf')
                
                cam_name_to_check = cam_names[cam_idx_to_check]

                for peer_pair_indices in combinations(peer_cam_indices, 2):
                    idx1, idx2 = peer_pair_indices
                    
                    cam_name1, cam_name2 = cam_names[idx1], cam_names[idx2]
                    
                    annots_pair = np.array([raw_lk_coords[idx1, p_idx], raw_lk_coords[idx2, p_idx]])
                    
                    proj_mats_pair = [proj_matrices[cam_name1], proj_matrices[cam_name2]]

                    # Triangulate a hypothesis from this pair
                    point_3d_hypothesis = projective.triangulate_points_from_projections(
                        points2d=jnp.asarray(annots_pair.reshape(2, 1, 2)),
                        P_mats=jnp.asarray(proj_mats_pair)
                    )
                    point_3d_hypothesis = np.asarray(point_3d_hypothesis).flatten()

                    if np.isnan(point_3d_hypothesis).any():
                        continue

                    # Reproject hypothesis back to the camera we are checking
                    reprojected = reproject_points(point_3d_hypothesis, calibration[cam_name_to_check])
                    if reprojected.size == 0:
                        continue

                    # Calculate error for this specific hypothesis
                    error = np.linalg.norm(point_to_check - reprojected.flatten())

                    # Keep track of the minimum error found so far
                    min_drift_error = min(min_drift_error, error)

                # at least one valid consensus is found: use it to calculate confidence
                if min_drift_error != float('inf'):
                    confidence = max(0.0, 1.0 - (min_drift_error / config.LK_CONFIDENCE_MAX_ERROR))
                else:
                    # triangulation failed for all pairs so default to low confidence
                    confidence = 0.1 

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

    R_c2w = transforms.rodrigues(rvec)
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


def create_individual(video_metadata: Dict, cam_names: List[str], scene_centre: np.ndarray) -> Dict[str, Dict]:
    """Create a random camera calibration individual."""

    w, h = video_metadata['width'], video_metadata['height']
    num_cameras = video_metadata['num_videos']

    radius = np.linalg.norm(scene_centre) if np.linalg.norm(scene_centre) > 1 else 100.0

    individual: Dict[str, Dict] = {}
    for i, cam_name in enumerate(cam_names):
        # Position cameras in a circle around the scene centre
        angle = (2 * np.pi / num_cameras) * i

        # Offset camera positions by the scene centre
        cam_pos_world = scene_centre + np.array([radius * np.cos(angle), 2.0, radius * np.sin(angle)])

        up_vector = np.array([0.0, 1.0, 0.0])
        forward = (scene_centre - cam_pos_world) / np.linalg.norm(scene_centre - cam_pos_world)

        right = np.cross(forward, up_vector)
        cam_up = np.cross(right, forward)

        R_w2c = np.array([-right, cam_up, -forward])
        R_c2w = R_w2c.T
        tvec_c2w = cam_pos_world
        rvec_c2w = transforms.inverse_rodrigues(R_c2w)

        # Randomize K components
        # TODO: These factors are a bit large...
        fx = random.uniform(w * 0.8, w * 1.5)
        fy = random.uniform(h * 0.8, h * 1.5)
        cx = w / 2 + random.uniform(-w * 0.05, w * 0.05)
        cy = h / 2 + random.uniform(-h * 0.05, h * 0.05)

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        individual[cam_name] = {
            'camera_matrix': K,
            'dist_coeffs': np.random.normal(0.0, 0.001, size=config.NUM_DIST_COEFFS),
            'rvec': rvec_c2w.flatten(),
            'tvec': tvec_c2w.flatten()
        }

    return individual


def compute_fitness(
    individual: Dict[str, Dict],
    annotations: np.ndarray,
    calibration_frames: List[int],
    video_metadata: Dict,
    cam_names: List[str]
) -> float:
    """Compute reprojection error fitness for a calibration."""

    if not calibration_frames:
        return float('inf')

    # Filter to calibration frames with valid data
    calib_mask = np.zeros(annotations.shape[0], dtype=bool)
    calib_mask[calibration_frames] = True
    valid_mask = np.any(~np.isnan(annotations[..., 0]), axis=(1, 2))
    combined_mask = calib_mask & valid_mask

    if not np.any(combined_mask):
        return float('inf')

    # Get annotations (x, y) for valid frames
    valid_annots = annotations[combined_mask][..., :2]
    num_cams = video_metadata['num_videos']

    # Compute projection matrices
    proj_matrices = np.array([get_projection_matrix(individual[name]) for name in cam_names])

    undistorted_annots = np.full_like(valid_annots, np.nan)
    for c in range(num_cams):
        undistorted_annots[:, c] = undistort_points(valid_annots[:, c], individual[cam_names[c]])

    # Triangulate for each frame
    points_3d = np.array([
        np.asarray(projective.triangulate_points_from_projections(
            points2d=jnp.asarray(frame_annots),
            P_mats=jnp.asarray(proj_matrices)
        ))
        for frame_annots in undistorted_annots
    ])

    # Compute reprojection error
    total_error = 0.0
    total_points = 0

    for c in range(num_cams):
        cam_name = cam_names[c]
        valid_3d = ~np.isnan(points_3d).any(axis=-1)
        valid_2d = ~np.isnan(valid_annots[:, c]).any(axis=-1)
        valid = valid_3d & valid_2d

        if not np.any(valid):
            continue

        reprojected = reproject_points(points_3d[valid], individual[cam_name])
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
    scene_centre = ga_state.get("scene_centre", np.zeros(3))
    stagnation_counter = ga_state.get("stagnation_counter", 0)
    cam_names = ga_state["camera_names"]

    if population is None:
        if best_individual:
            # Seed initial population from best individual
            population = [best_individual]
            for _ in range(config.GA_POPULATION_SIZE - 1):
                mutated_ind = {}

                for cam_name, cam_params in best_individual.items():
                    # TODO: Are these copies really necessary?
                    mutated_cam = cam_params.copy()

                    # Mutate K matrix elements (fx, fy, cx, cy)
                    K = cam_params['camera_matrix'].copy()
                    K[0, 0] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[0, 0]))  # fx
                    K[1, 1] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[1, 1]))  # fy
                    K[0, 2] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[0, 2]))  # cx
                    K[1, 2] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[1, 2]))  # cy
                    mutated_cam['camera_matrix'] = K

                    # Mutate other parameters
                    for key in ['tvec', 'dist_coeffs']:
                        mutated_cam[key] = np.asarray(mutated_cam[key]) + np.random.normal(0,
                                                                                           config.GA_MUTATION_STRENGTH_INIT,
                                                                                           size=mutated_cam[key].shape)

                    mutated_cam['rvec'] = np.asarray(mutated_cam['rvec']) + np.random.normal(0,
                                                                                             config.GA_MUTATION_STRENGTH_INIT * 0.001,
                                                                                             # Massively reduce mutation on rvec because radians
                                                                                             size=mutated_cam['rvec'].shape)
                    mutated_ind[cam_name] = mutated_cam
                population.append(mutated_ind)
        else:
            population = [create_individual(ga_state['video_metadata'], cam_names, scene_centre) for _ in
                          range(config.GA_POPULATION_SIZE)]

    # Evaluate fitness
    fitness_scores = np.array([
        compute_fitness(
            ind,
            ga_state['annotations'],
            ga_state['calibration_frames'],
            ga_state['video_metadata'],
            cam_names
        )
        for ind in population
    ])
    sorted_indices = np.argsort(fitness_scores)

    if fitness_scores[sorted_indices[0]] < best_fitness:
        best_fitness = fitness_scores[sorted_indices[0]]
        best_individual = population[sorted_indices[0]]
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    current_mutation_strength = config.GA_MUTATION_STRENGTH
    if stagnation_counter > 20:
        print("GA is stagnating, temporarily increasing mutation strength.")
        current_mutation_strength *= 2.5

    # Create next generation
    num_elites = int(config.GA_POPULATION_SIZE * config.GA_ELITISM_RATE)
    next_population = [population[i] for i in sorted_indices[:num_elites]]

    while len(next_population) < config.GA_POPULATION_SIZE:

        # Tournament selection
        # TODO: This is a bit simple, could be improved

        p1_idx, p2_idx = np.random.choice(len(population), 2, replace=False)
        parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p2_idx]
        p3_idx, p4_idx = np.random.choice(len(population), 2, replace=False)
        parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p4_idx]

        child = {}
        for cam_name in cam_names:
            p1_cam = parent1[cam_name]
            p2_cam = parent2[cam_name]

            child_cam = {}

            # rvec averaging (quaternion)
            q_batch = transforms.axisangle_to_quaternion_batched(jnp.stack([p1_cam['rvec'], p2_cam['rvec']]))
            q_avg = quaternion_average(q_batch)

            rvec_avg = transforms.quaternion_to_axisangle(q_avg)
            child_cam['rvec'] = np.asarray(rvec_avg)

            # Linear Averaging for other parameters
            for key in p1_cam:
                if key != 'rvec':
                    p1_val = np.asarray(p1_cam[key])
                    p2_val = np.asarray(p2_cam[key])
                    child_cam[key] = (p1_val + p2_val) / 2.0

            # Mutation
            if np.random.rand() < config.GA_MUTATION_RATE:

                # Mutate K elements
                K_mut = child_cam['camera_matrix'].copy()
                K_mut[0, 0] += np.random.normal(0, current_mutation_strength * abs(K_mut[0, 0]))
                K_mut[1, 1] += np.random.normal(0, current_mutation_strength * abs(K_mut[1, 1]))
                K_mut[0, 2] += np.random.normal(0, current_mutation_strength * abs(K_mut[0, 2]))
                K_mut[1, 2] += np.random.normal(0, current_mutation_strength * abs(K_mut[1, 2]))
                child_cam['camera_matrix'] = K_mut

                # Mutate tvec and dist_coeffs
                for key in ['tvec', 'dist_coeffs']:
                    child_cam[key] = child_cam[key] + np.random.normal(0, current_mutation_strength,
                                                                       size=child_cam[key].shape)

                # Mutate rvec
                child_cam['rvec'] = child_cam['rvec'] + np.random.normal(0,
                                                                         current_mutation_strength * 0.001, # Massively reduce mutation on rvec because radians
                                                                         size=child_cam['rvec'].shape)

            child[cam_name] = child_cam
        next_population.append(child)

    mean_fitness = np.nanmean(fitness_scores)
    std_fitness = np.nanstd(fitness_scores)

    return {
        "status": "running",
        "new_best_fitness": best_fitness,
        "new_best_individual": best_individual,
        "generation": generation + 1,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "next_population": next_population,
        "stagnation_counter": stagnation_counter,
    }


def _prepare_ba_data(snapshot: Dict[str, Any], is_independent: bool) -> Dict[str, Any]:
    """
    Prepares data for BA.
    """

    all_annots = snapshot["annotations"]
    initial_calib: Dict[str, Dict] = snapshot["best_individual"]
    calib_frames = snapshot["calibration_frames"]
    cam_names = snapshot["camera_names"]

    # Annotations (x, y) for calibration frames
    annots_in_calib_frames = all_annots[calib_frames, ..., :2]
    P, C, N_potential, _ = annots_in_calib_frames.shape

    proj_matrices = np.array([get_projection_matrix(initial_calib[name]) for name in cam_names])

    if not is_independent:
        initial_structure = np.full((P, N_potential, 3), np.nan, dtype=np.float32)
        for i in range(P):
            triangulated = projective.triangulate_points_from_projections(
                points2d=jnp.asarray(annots_in_calib_frames[i]),
                P_mats=jnp.asarray(proj_matrices)
            )
            initial_structure[i] = np.asarray(triangulated)
        if np.all(np.isnan(initial_structure)):
            raise ValueError("Temporally-consistent triangulation failed for all points.")
        return {
            "image_points": np.nan_to_num(np.transpose(annots_in_calib_frames, (1, 0, 2, 3))),
            "visibility_mask": ~np.isnan(np.transpose(annots_in_calib_frames, (1, 0, 2, 3))[..., 0]),
            "object_points_initial": np.nan_to_num(initial_structure.reshape(-1, 3)),
            "initial_structure_for_masking": initial_structure
        }

    else:
        # Scaffolding mode
        per_frame_3d = []
        per_frame_2d = []
        num_points_per_frame = []

        for p in range(P):
            frame_annots = annots_in_calib_frames[p]
            points_3d = projective.triangulate_points_from_projections(
                points2d=jnp.asarray(frame_annots),
                P_mats=jnp.asarray(proj_matrices)
            )
            points_3d = np.asarray(points_3d)
            valid_mask = ~np.isnan(points_3d).any(axis=-1)

            valid_3d = points_3d[valid_mask]
            valid_2d = frame_annots[:, valid_mask, :]

            per_frame_3d.append(valid_3d)
            per_frame_2d.append(valid_2d)
            num_points_per_frame.append(len(valid_3d))

        if sum(num_points_per_frame) == 0:
            raise ValueError("Scaffolding triangulation failed. No valid 3D points could be generated.")

        N_max = max(num_points_per_frame) if num_points_per_frame else 0

        padded_3d = np.full((P, N_max, 3), np.nan, dtype=np.float32)
        padded_2d = np.full((C, P, N_max, 2), np.nan, dtype=np.float32)

        for p in range(P):
            n_pts = num_points_per_frame[p]
            if n_pts > 0:
                padded_3d[p, :n_pts] = per_frame_3d[p]
                padded_2d[:, p, :n_pts] = per_frame_2d[p]

        return {
            "image_points": np.nan_to_num(padded_2d),
            "visibility_mask": ~np.isnan(padded_2d[:, :, :, 0]),
            "object_points_initial": np.nan_to_num(padded_3d.reshape(-1, 3), nan=1e6),
            "initial_structure_for_masking": None
        }

def run_refinement(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares CATAR data and runs bundle adjustment based on the selected mode.
    """

    print("[BA] Preparing data...")

    mode = snapshot.get("mode", "full_ba")
    initial_calib: Dict[str, Dict] = snapshot.get("best_individual")
    video_meta = snapshot.get("video_metadata")
    calib_frames = snapshot.get("calibration_frames")
    cam_names = snapshot.get("camera_names")

    if not calib_frames or initial_calib is None:
        return {"status": "error", "message": "Missing calibration frames or initial calibration."}

    # Order the calibration list according to cam_names
    ordered_initial_calib = [initial_calib[name] for name in cam_names]

    # Set flags based on mode
    if mode == "refine_cameras_only":
        is_independent_mode = True
        fix_cameras = False
        fix_points = False
    elif mode == "refine_points_only":
        is_independent_mode = False
        fix_cameras = True
        fix_points = False
    else:  # "full_ba"
        is_independent_mode = False
        fix_cameras = False
        fix_points = False

    try:
        ba_data = _prepare_ba_data(snapshot, is_independent_mode)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    print("[BA] Starting optimization engine...")
    image_sizes_wh = np.array([[video_meta['width'], video_meta['height']]] * video_meta['num_videos'])

    # Extract initial parameters as ordered lists of arrays for mokap/JAX
    K_init = jnp.asarray(np.array([c['camera_matrix'] for c in ordered_initial_calib]))
    D_init = jnp.asarray(np.array([c['dist_coeffs'] for c in ordered_initial_calib]))
    r_init = jnp.asarray(np.array([c['rvec'] for c in ordered_initial_calib]))
    t_init = jnp.asarray(np.array([c['tvec'] for c in ordered_initial_calib]))

    success, results = bundle_adjustment.run_bundle_adjustment(
        camera_matrices_initial=K_init,
        distortion_coeffs_initial=D_init,
        cam_rvecs_initial=r_init,
        cam_tvecs_initial=t_init,
        images_sizes_wh=image_sizes_wh,

        image_points=jnp.asarray(ba_data["image_points"]),
        visibility_mask=jnp.asarray(ba_data["visibility_mask"]),
        object_points_initial=jnp.asarray(ba_data["object_points_initial"]),

        # Set flags based on mode
        fix_cameras_intrinsics=fix_cameras,
        fix_cameras_extrinsics=fix_cameras,
        fix_object_points=fix_points,
        fix_poses=True,  # we never optimize board poses in CATAR (well, at least for now)
        time_independent_points=is_independent_mode,

        origin_idx=0, max_nfev=200
    )

    if not success:
        return {"status": "error", "message": "Bundle Adjustment failed to converge."}

    print("[BA] Optimization successful!")
    K_opt, D_opt, r_opt, t_opt = results['K_opt'], results['D_opt'], results['cam_r_opt'], results['cam_t_opt']

    # Convert refined results back to Dict[str, Dict]
    refined_calib: Dict[str, Dict] = {}
    for i, cam_name in enumerate(cam_names):
        refined_calib[cam_name] = {
            'camera_matrix': np.asarray(K_opt[i]),
            'dist_coeffs': np.asarray(D_opt[i]),
            'rvec': np.asarray(r_opt[i]),
            'tvec': np.asarray(t_opt[i]),
        }

    refined_points_to_return = None
    if not is_independent_mode and not fix_points:
        P, _, N, _ = ba_data["image_points"].shape
        refined_points_flat = np.asarray(results['object_points_opt']).copy()
        refined_points_to_return = refined_points_flat.reshape(P, N, 3)
        initial_invalid_mask = np.isnan(ba_data["initial_structure_for_masking"])
        refined_points_to_return[initial_invalid_mask] = np.nan

    return {
        "status": "success",
        "refined_calibration": refined_calib,
        "refined_3d_points": refined_points_to_return,
        "calibration_frame_indices": calib_frames
    }
