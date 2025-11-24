"""
Core algos for tracking and calibration (and some stuff for visualisation)
"""
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import jax.numpy as jnp

from mokap.calibration import bundle_adjustment
from mokap.utils.geometry import projective
from mokap.utils.geometry import transforms
from mokap.utils.geometry.fitting import quaternion_average

from utils import reproject_points
from viz_3d import SceneObject
import config
from state import CalibrationState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state import AppState
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


# TODO: Move these to config

# NCC Threshold: 0.8 is a safe conservative bet
# It means "The new patch is at least 80% similar to the old patch"
NCC_THRESHOLD_WARNING = 0.6
NCC_THRESHOLD_KILL = 0.3
NCC_PATCH_SIZE = 11

# Rescue threshold can be lower here because reprojected points
# might be slightly misaligned compared to perfect LK tracking
NCC_THRESHOLD_RESCUE = 0.5


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
        annotations_for_point = app_state.annotations[frame_idx, :, point_idx, :]
        calibration = app_state.calibration

    if calibration.best_calibration is None:
        return None

    num_cams = annotations_for_point.shape[0]
    cam_indices = np.arange(num_cams)

    # Masks for valid annotations excluding the target camera
    is_not_target_cam = (cam_indices != target_cam_idx)
    is_valid_annotation = ~np.isnan(annotations_for_point).any(axis=1)
    final_mask = is_not_target_cam & is_valid_annotation

    valid_cam_indices = cam_indices[final_mask]

    # We need at least 2 other views to triangulate
    if len(valid_cam_indices) < 2:
        return None

    valid_annotations = annotations_for_point[final_mask]
    valid_annotations_2d = valid_annotations[:, :2]
    valid_scores = valid_annotations[:, 2]

    # Get calibration parameters for valid cameras
    proj_matrices = calibration.P_mats[valid_cam_indices]

    # Reshape for triangulation function
    point_to_triangulate = valid_annotations_2d.reshape(len(valid_cam_indices), 1, 2)
    weights_for_triangulation = valid_scores.reshape(len(valid_cam_indices), 1)

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
    target_cam_params = calibration.get(calibration.camera_names[target_cam_idx])
    reprojected_point_2d = reproject_points(point_3d, target_cam_params)

    if reprojected_point_2d.size == 0:
        return None

    reprojected_point_2d = reprojected_point_2d.flatten()

    # Only snap if the click is close enough
    distance = np.linalg.norm(reprojected_point_2d - user_click_pos)
    if distance > 15:
        return None

    return reprojected_point_2d


def _fuse_annotations(
        human_annots: np.ndarray,
        human_flags: np.ndarray,
        lk_annots: np.ndarray,
        model_annots: np.ndarray
) -> np.ndarray:
    """
    Fuses annotations from multiple sources. It finds the best source of evidence
    and boosts its confidence if other sources agree with its position.

    Args:
        human_annots: Manual annotations
        lk_annots: Annotations from the LK tracker
        model_annots: Annotations reprojected from the 3D skeleton model

    Returns:
        final_annotations (np.ndarray): The fused (x, y, confidence) annotations
    """
    num_cams, num_points, _ = human_annots.shape
    final_annotations = np.full((num_cams, num_points, 3), np.nan, dtype=np.float32)

    # Note: A vectorised implementation of this would be kinda complex (because of the conditional logic)
    # A loop is better for now
    for c in range(num_cams):
        for p in range(num_points):
            sources = []
            # Source 1: Human Annotation
            if human_flags[c, p] and not np.isnan(human_annots[c, p, 0]):
                sources.append({
                    'pos': human_annots[c, p, :2],
                    'conf': human_annots[c, p, 2],
                    'type': 'human'
                })

            # Source 2: LK tracker
            if not np.isnan(lk_annots[c, p, 0]):
                sources.append({
                    'pos': lk_annots[c, p, :2],
                    'conf': lk_annots[c, p, 2],
                    'type': 'lk'
                })

            # Source 3: 3D Model reprojection
            if not np.isnan(model_annots[c, p, 0]):
                sources.append({
                    'pos': model_annots[c, p, :2],
                    'conf': model_annots[c, p, 2],
                    'type': 'model'
                })

            if not sources:
                continue

            # If only one source, just use it
            if len(sources) == 1:
                final_annotations[c, p] = [*sources[0]['pos'], sources[0]['conf']]
                continue

            # Find the best source (highest confidence)
            sources.sort(key=lambda s: s['conf'], reverse=True)
            best_source = sources[0]
            other_sources = sources[1:]

            # Start with the best source's data
            final_conf = best_source['conf']

            # These will be used for the weighted average of *agreeing* sources
            sum_of_weights = best_source['conf']
            weighted_sum_pos = best_source['pos'] * best_source['conf']

            # Check for agreement from other sources
            for other in other_sources:
                distance = np.linalg.norm(best_source['pos'] - other['pos'])

                if distance < config.FUSION_AGREEMENT_RADIUS:
                    # Agreement found: add to the average
                    weighted_sum_pos += other['pos'] * other['conf']
                    sum_of_weights += other['conf']

                    # Add confidence bonus based on how good the agreement is
                    bonus = (1.0 - (distance / config.FUSION_AGREEMENT_RADIUS)) * config.FUSION_AGREEMENT_BONUS
                    final_conf += bonus

            # Calculate final position from agreeing sources
            final_pos = np.divide(
                weighted_sum_pos,
                sum_of_weights,
                out=np.full(2, np.nan),  # default to nan if sum_of_weights is 0 (but that probably will never happenm)
                where=sum_of_weights > 1e-6
            )

            # Confidence ceiling
            # If the highest-confidence source was a human annotation in this frame, ceiling is 1.0
            # Otherwise it's capped at the maximum for automated points
            if best_source['type'] == 'human':
                max_conf = config.FUSION_HUMAN_CONFIDENCE
            else:
                max_conf = config.FUSION_MAX_AUTO_CONFIDENCE
            final_conf = min(final_conf, max_conf)

            final_annotations[c, p] = [*final_pos, final_conf]

    return final_annotations


def process_frame(
        frame_idx: int,
        source_frame_idx: int,
        app_state: 'AppState',
        reconstructor: 'Reconstructor',
        tracker: 'MultiObjectTracker',
        source_frames: List[np.ndarray],
        dest_frames: List[np.ndarray]
):
    """
    Runs the full pipeline (get the state of 'frame_idx' based on 'source_frame_idx').
    """

    with app_state.lock:
        calibration = app_state.calibration
        point_names = app_state.point_names
        cam_names = app_state.camera_names

        # Load source annotations for validation later
        prev_annotations_for_validation = app_state.annotations[source_frame_idx].copy()
        human_flags_for_frame = app_state.human_annotated[frame_idx].copy()
        human_annots_for_frame = app_state.annotations[frame_idx].copy()

    src_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in source_frames]
    dst_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in dest_frames]

    # Get geometric prediction and confidence score from LK tracking
    annotations_from_lk = track_points(
        app_state, src_gray, dst_gray, source_frame_idx, frame_idx
    )

    # Prepare input for Mokap Reconstructor
    cam_indices, point_indices = np.where(~np.isnan(annotations_from_lk[..., 0]))

    num_lk_points = len(cam_indices)

    # Debug print
    # TODO: Add toggle for these prints
    print(f"\n[Frame: {source_frame_idx} -> {frame_idx}]")
    print(f"LK PREDICT:    Started with {num_lk_points} raw 2D keypoints from Optical Flow.")

    if num_lk_points > 0:
        coords = annotations_from_lk[cam_indices, point_indices, :2]
        scores = annotations_from_lk[cam_indices, point_indices, 2]

        # Construct inputs dictionary matching mokap.reconstruction.datatypes requirements
        reconstruction_input = {
            "frame_indices": np.full(num_lk_points, frame_idx, dtype=np.int32),
            "kp_type_ids": point_indices.astype(np.int16),
            "cam_ids": cam_indices.astype(np.int8),
            "coords": coords.astype(np.float32),
            "scores": scores.astype(np.float32)
        }

        # Run Mokap reconstruction
        soup = reconstructor.reconstruct_batch(
            inputs=reconstruction_input,
            keypoint_names=point_names
        )

        print(f"MOKAP RECON:   Input {num_lk_points} 2D points -> Produced a soup of {soup.num_points} 3D points and {len(soup.ray_origins)} orphan rays.")

        # Run multi-object tracking (skeleton assembly + time association)
        active_tracklets = tracker.update(soup, frame_idx)
    else:
        print("MOKAP RECON:   No input points from LK.")
        active_tracklets = []

    # Extract best skeleton for feedback loop
    final_3d_skeleton_kps = None
    skeleton_score = 0.0

    if active_tracklets:
        # Heuristic: Pick tracklet with most keypoints or highest score
        best_tracklet = max(active_tracklets, key=lambda t: (len(t.skeleton.keypoints), t.skeleton.score))
        final_3d_skeleton_kps = best_tracklet.skeleton.keypoints
        num_kps = len(final_3d_skeleton_kps)
        skeleton_score = best_tracklet.skeleton.score
        print(f"MOKAP ASSEMBLE: SUCCESS -> Assembled 1 skeleton with {num_kps} keypoints (Score: {skeleton_score:.2f}).")
    else:
        print(f"MOKAP ASSEMBLE: Could not assemble any skeletons from the soup.")

    # Fusion & cleanup

    # Get model-reprojected annotations as a source of evidence
    annotations_from_model = np.full_like(annotations_from_lk, np.nan)

    if final_3d_skeleton_kps and calibration.best_calibration:
        points_to_reproject_3d = []
        p_indices = []

        for p_name, pos_3d in final_3d_skeleton_kps.items():
            if p_name in point_names:
                points_to_reproject_3d.append(pos_3d)
                p_indices.append(point_names.index(p_name))

        points_to_reproject_3d = np.array(points_to_reproject_3d)

        if points_to_reproject_3d.size > 0:

            # Project into all cameras
            reprojected_all_cams, _ = projective.project_to_multiple_cameras(
                object_points=jnp.asarray(points_to_reproject_3d),
                rvec=jnp.asarray(calibration.rvecs_w2c),
                tvec=jnp.asarray(calibration.tvecs_w2c),
                camera_matrix=jnp.asarray(calibration.K_mats),
                dist_coeffs=jnp.asarray(calibration.dist_coeffs)
            )
            reprojected_all_cams = np.asarray(reprojected_all_cams)  # shape (C, P_reproj, 2)

            # Fill annotation array
            model_confidence = skeleton_score * 0.75
            for i, p_idx in enumerate(p_indices):
                annotations_from_model[:, p_idx, :2] = reprojected_all_cams[:, i, :]
                annotations_from_model[:, p_idx, 2] = model_confidence

    # Step 1: Get best possible 2D annotations by fusing all evidence
    fused_2d_annotations = _fuse_annotations(
        human_annots=human_annots_for_frame,
        human_flags=human_flags_for_frame,
        lk_annots=annotations_from_lk,
        model_annots=annotations_from_model
    )  # shape (C, P, 3)

    # Rescue single-view LK tracks that were lost during fusion because they had no other supporting evidence
    is_valid_lk = ~np.isnan(annotations_from_lk[..., 0])
    num_views_per_lk_point = np.sum(is_valid_lk, axis=0)  # shape (P,)

    is_valid_fused = ~np.isnan(fused_2d_annotations[..., 0])
    is_fused_point_lost = ~np.any(is_valid_fused, axis=0)  # shape (P,)

    for p_idx in range(app_state.num_points):
        # if LK had exactly one view AND the point was lost in fusion...
        if num_views_per_lk_point[p_idx] == 1 and is_fused_point_lost[p_idx]:
            # ...find the camera that had the single track and re-insert it
            cam_idx = np.where(is_valid_lk[:, p_idx])[0][0]
            fused_2d_annotations[cam_idx, p_idx] = annotations_from_lk[cam_idx, p_idx]

    # Validate rescued points
    # Reject SVR/model jumps if appearance changes drastically
    fused_2d_annotations = validate_rescued_points(
        fused_annots=fused_2d_annotations,
        prev_annots=prev_annotations_for_validation,
        lk_annots=annotations_from_lk,
        src_frames_gray=src_gray,
        dst_frames_gray=dst_gray,
        cam_names=cam_names,
        point_names=point_names
    )

    # Step 2: Generate (the hopefully high fidelity) 3D data by triangulating these fused 2D points
    triangulated_3d_points = np.full((app_state.num_points, 3), np.nan, dtype=np.float32)
    if calibration.best_calibration is not None:

        points2d_for_triangulation = fused_2d_annotations[..., :2] # (C, P, 2)
        weights_for_triangulation = fused_2d_annotations[..., 2] # (C, P)

        triangulated_3d_points = projective.triangulate_points_from_projections(
            points2d=jnp.asarray(points2d_for_triangulation),
            P_mats=jnp.asarray(calibration.P_mats),
            weights=jnp.asarray(weights_for_triangulation)
        )
        triangulated_3d_points = np.asarray(triangulated_3d_points)

    # Step 3: Create the final 3D pose for this frame (priority on triangulation)
    final_3d_pose = np.full((app_state.num_points, 3), np.nan, dtype=np.float32)

    for p_idx in range(app_state.num_points):
        p_name = app_state.point_names[p_idx]

        # Priority 1: Use the triangulation result if it's valid
        if not np.isnan(triangulated_3d_points[p_idx, 0]):
            final_3d_pose[p_idx] = triangulated_3d_points[p_idx]

        # Priority 2 (fallback): If triangulation failed, use mokap skeleton's position
        # This helps fill gaps where direct triangulation failed but the skeleton assembler
        # succeeded (via single-view rescue for instance)
        elif final_3d_skeleton_kps and p_name in final_3d_skeleton_kps:
            final_3d_pose[p_idx] = final_3d_skeleton_kps[p_name]

    # Structural feedback
    # If the skeleton assembler rejected a point we penalize the 2D track for this frame.
    # (otherwise the next frame could continue tracking this zombie points)
    if final_3d_skeleton_kps:
        assembled_names = set(final_3d_skeleton_kps.keys())

        for p_idx in range(app_state.num_points):
            p_name = app_state.point_names[p_idx]

            if p_name not in assembled_names:
                # don't necessarily delete it (might be a valid point that was just occluded once or whatever),
                # but drop confidence significantly so it dies quickly if it keeps failing
                valid_mask = ~np.isnan(fused_2d_annotations[:, p_idx, 0])
                if np.any(valid_mask):
                    print(f"ZOMBIE POINT: '{point_names[p_idx]}' was rejected by Mokap")

                    fused_2d_annotations[valid_mask, p_idx, 2] *= 0.5   # slash conf by half

                    # # if confidence drops too low, kill it entirely
                    # # TODO: test if that's useful
                    # kill_mask = fused_2d_annotations[:, p_idx, 2] < 0.1
                    # fused_2d_annotations[kill_mask, p_idx, :] = np.nan

    # Step 4: Update app state with the final data for this frame
    with app_state.lock:
        app_state.annotations[frame_idx] = fused_2d_annotations
        app_state.reconstructed_3d_points[frame_idx] = final_3d_pose

# ============================================================================
# Optic flow tracking
# ============================================================================

def _compute_patch_ncc(
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        p_prev: np.ndarray,
        p_curr: np.ndarray,
        patch_size: int = NCC_PATCH_SIZE
) -> float:
    """
    Computes Normalized Cross Correlation between two patches.
    Returns 1.0 if identical, -1.0 if inverted, 0.0 if uncorrelated.
    Returns -1.0 if patches are out of bounds or invalid.
    """
    h, w = img_prev.shape
    pad = patch_size // 2

    u_p, v_p = p_prev
    u_c, v_c = p_curr

    if (u_p < pad or u_p >= w - pad or v_p < pad or v_p >= h - pad or
            u_c < pad or u_c >= w - pad or v_c < pad or v_c >= h - pad):
        return -1.0

    try:
        patch_prev = cv2.getRectSubPix(img_prev, (patch_size, patch_size), (float(u_p), float(v_p)))
        patch_curr = cv2.getRectSubPix(img_curr, (patch_size, patch_size), (float(u_c), float(v_c)))

        if np.std(patch_prev) < 1e-5 or np.std(patch_curr) < 1e-5:
            return 0.0

        res = cv2.matchTemplate(patch_curr.astype(np.float32), patch_prev.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        return res[0][0]

    except Exception:
        return -1.0


def validate_rescued_points(
        fused_annots: np.ndarray,
        prev_annots: np.ndarray,
        lk_annots: np.ndarray,
        src_frames_gray: List[np.ndarray],
        dst_frames_gray: List[np.ndarray],
        cam_names: List[str],
        point_names: List[str],
) -> np.ndarray:
    """
    Checks points that were lost by LK but rescued by fusion.
    If the rescued point looks too different from the previous frame, reject it.
    """
    num_cams, num_points, _ = fused_annots.shape
    validated_annots = fused_annots.copy()

    for c in range(num_cams):
        for p in range(num_points):
            # Condition: LK failed, but fusion returned a value, and we have history
            if (np.isnan(lk_annots[c, p, 0]) and
                    not np.isnan(fused_annots[c, p, 0]) and
                    not np.isnan(prev_annots[c, p, 0])):

                ncc = _compute_patch_ncc(
                    src_frames_gray[c], dst_frames_gray[c],
                    prev_annots[c, p, :2],
                    fused_annots[c, p, :2]
                )

                if ncc < NCC_THRESHOLD_RESCUE:
                    # Reject rescue
                    validated_annots[c, p] = np.nan
                    print(f"APPEARENCE REJECT: Camera '{cam_names[c]}': '{point_names[p]}' looks too different from previous frame.")

    return validated_annots


def track_points(
        app_state: 'AppState',
        source_frames_gray: List[np.ndarray],
        dest_frames_gray: List[np.ndarray],
        source_frame_idx: int,
        dest_frame_idx: int
) -> np.ndarray:
    """
    Tracks points using LK + Forward-Backward Error + NCC Appearance Check.
    """
    with app_state.lock:
        focus_mode = app_state.focus_selected_point
        selected_idx = app_state.selected_point_idx
        annotations_source_full = app_state.annotations[source_frame_idx].copy()
        calibration = app_state.calibration
        cam_names = app_state.camera_names

    p0_2d_all = annotations_source_full[..., :2]
    num_cams, num_points, _ = annotations_source_full.shape
    output_annotations = np.full((num_cams, num_points, 3), np.nan, dtype=np.float32)

    if calibration.best_calibration is None:
        return output_annotations

    for cam_idx in range(num_cams):
        src_gray = source_frames_gray[cam_idx]
        dst_gray = dest_frames_gray[cam_idx]

        p0_2d_src = p0_2d_all[cam_idx]

        # Filter valid points
        track_mask = ~np.isnan(p0_2d_src).any(axis=1)

        if focus_mode:
            is_valid = track_mask[selected_idx]
            track_mask[:] = False
            track_mask[selected_idx] = is_valid

        if not np.any(track_mask):
            continue

        start_points = p0_2d_src[track_mask].reshape(-1, 1, 2)
        point_indices = np.where(track_mask)[0]

        # Forward flow: source -> dest
        p1_forward, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            src_gray, dst_gray, start_points, None, **config.LK_PARAMS
        )

        # Backward flow: dest -> source
        p0_backward, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            dst_gray, src_gray, p1_forward, None, **config.LK_PARAMS
        )

        for i, p_idx in enumerate(point_indices):
            # Geometric check (Forward-Backward Error)
            fb_error = np.linalg.norm(start_points[i] - p0_backward[i])
            is_geom_valid = (status_fwd[i] == 1 and status_bwd[i] == 1 and
                             fb_error < config.FORWARD_BACKWARD_THRESHOLD)

            if is_geom_valid:
                # Appearance check (NCC)
                ncc_score = _compute_patch_ncc(
                    src_gray, dst_gray,
                    start_points[i].flatten(),
                    p1_forward[i].flatten()
                )

                # If it's terrible, kill it
                if ncc_score < NCC_THRESHOLD_KILL:
                    continue

                # Confidence
                prev_conf = annotations_source_full[cam_idx, p_idx, 2]
                if np.isnan(prev_conf): prev_conf = config.MAX_SINGLE_VIEW_CONFIDENCE

                geom_quality = (1.0 - (fb_error / config.FORWARD_BACKWARD_THRESHOLD))

                # Apply soft penalty for lower NCC scores
                ncc_factor = 1.0
                if ncc_score < NCC_THRESHOLD_WARNING:
                    # Linearly scale down: 0.6 -> 1.0, 0.3 -> 0.0
                    ncc_factor = max(0.0, (ncc_score - NCC_THRESHOLD_KILL) / (
                                NCC_THRESHOLD_WARNING - NCC_THRESHOLD_KILL))

                new_conf = prev_conf * geom_quality * ncc_factor * config.CONFIDENCE_TIME_DECAY

                output_annotations[cam_idx, p_idx] = [*p1_forward[i].flatten(), new_conf]

    # Confidence upgrade via multi-view consensus on the destination frame
    for p_idx in range(app_state.num_points):

        # Find all cameras that have a valid track for this point
        peer_cam_indices = [i for i, p in enumerate(output_annotations[:, p_idx]) if not np.isnan(p).any()]

        # if we have enough views for triangulation we can try to upgrade the confidence
        if len(peer_cam_indices) >= 2:

            for cam_idx_to_check in peer_cam_indices:
                point_to_check = output_annotations[cam_idx_to_check, p_idx, :2]

                # The consensus group is all the other cameras (that have a valid track)
                consensus_peer_indices = [i for i in peer_cam_indices if i != cam_idx_to_check]

                # We need at least 2 other views to form a triangulation hypothesis
                if len(consensus_peer_indices) < 2:
                    continue

                consensus_annots = output_annotations[consensus_peer_indices, p_idx, :2]
                consensus_proj_mats = calibration.P_mats[consensus_peer_indices]

                points_for_triangulation = consensus_annots.reshape(len(consensus_peer_indices), 1, 2)

                point_3d_hypothesis = projective.triangulate_points_from_projections(
                    points2d=jnp.asarray(points_for_triangulation),
                    P_mats=jnp.asarray(consensus_proj_mats)
                )
                point_3d_hypothesis = np.asarray(point_3d_hypothesis).flatten()

                if np.isnan(point_3d_hypothesis).any():
                    continue

                # Reproject the 3D hypothesis back to the camera we are checking
                cam_name_to_check = cam_names[cam_idx_to_check]
                reprojected = reproject_points(point_3d_hypothesis, calibration.get(cam_name_to_check))
                if reprojected.size == 0:
                    continue

                # Geometric error (drift) for this LK track
                error = np.linalg.norm(point_to_check - reprojected.flatten())

                # Convert error into a confidence score
                multi_view_confidence = max(0.0, 1.0 - (error / config.LK_CONFIDENCE_MAX_ERROR))

                # We only update the confidence if the multi-view check provides a better score
                # (This prevents a stable single-view track from being punished by noisy triangulation)
                output_annotations[cam_idx_to_check, p_idx, 2] = max(
                    output_annotations[cam_idx_to_check, p_idx, 2],
                    multi_view_confidence
                )

    # Ensure that no automated track's confidence exceeds the maximum allowed value
    is_valid = ~np.isnan(output_annotations[..., 2])
    output_annotations[is_valid, 2] = np.fmin(
        output_annotations[is_valid, 2],
        config.FUSION_MAX_AUTO_CONFIDENCE
    )

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
    individual: Dict[str, Dict],  # This is the candidate from the GA
    annotations: np.ndarray,
    calibration_frames: List[int],
    video_metadata: Dict,
    cam_names: List[str]
) -> float:
    """Compute reprojection error fitness for a calibration."""

    if not calibration_frames:
        return float('inf')

    # Temporary CalibrationState for candidate individual
    temp_calib_state = CalibrationState(individual, cam_names)

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

    annots_for_undistort = np.transpose(valid_annots, (1, 2, 0, 3)).reshape(num_cams, -1, 2)
    undistorted_flat = projective.undistort_multiple(
        jnp.asarray(annots_for_undistort),
        jnp.asarray(temp_calib_state.K_mats),
        jnp.asarray(temp_calib_state.dist_coeffs)
    )
    undistorted_annots = np.asarray(undistorted_flat).reshape(num_cams, -1, valid_annots.shape[2], 2)
    undistorted_annots = np.transpose(undistorted_annots, (1, 0, 2, 3))

    # Triangulate for each frame
    points_3d_per_frame = [
        np.asarray(projective.triangulate_points_from_projections(
            points2d=jnp.asarray(frame_annots),
            P_mats=jnp.asarray(temp_calib_state.P_mats)
        ))
        for frame_annots in undistorted_annots
    ]
    points_3d = np.array(points_3d_per_frame)  # shape (F, P, 3)

    # Temporary state for candidate calibration
    temp_calib_state = CalibrationState(individual, cam_names)

    num_frames, num_points, _ = points_3d.shape
    points_3d_flat = points_3d.reshape(-1, 3)  # shape (F*P, 3)

    # Project all points into all cameras
    reprojected_flat, _ = projective.project_to_multiple_cameras(
        jnp.asarray(points_3d_flat),
        rvec=jnp.asarray(temp_calib_state.rvecs_c2w),
        tvec=jnp.asarray(temp_calib_state.tvecs_c2w),
        camera_matrix=jnp.asarray(temp_calib_state.K_mats),
        dist_coeffs=jnp.asarray(temp_calib_state.dist_coeffs)
    ) # shape (C, F*P, 2)

    # Reshape back to match annotation structure
    reprojected_unflat = np.asarray(reprojected_flat).reshape(num_cams, num_frames, num_points, 2)
    reprojected_final = np.transpose(reprojected_unflat, (1, 0, 2, 3))  # shape (F, C, P, 2)

    # Calculate error across all points and cameras
    errors = np.linalg.norm(reprojected_final - undistorted_annots, axis=-1)

    # Mask out invalid points and calculate final fitness score
    valid_mask = ~np.isnan(undistorted_annots[..., 0])
    total_error = np.sum(errors[valid_mask])
    total_points = np.sum(valid_mask)

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
        parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p1_idx]
        p3_idx, p4_idx = np.random.choice(len(population), 2, replace=False)
        parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p3_idx]

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
    initial_calib_dict: Dict[str, Dict] = snapshot["best_individual"]
    calib_frames = snapshot["calibration_frames"]
    cam_names = snapshot["camera_names"]

    # Temporary state object to access cached properties
    temp_calib_state = CalibrationState(initial_calib_dict, cam_names)

    # Annotations (x, y) for calibration frames
    annots_in_calib_frames = all_annots[calib_frames, ..., :2]
    P, C, N_potential, _ = annots_in_calib_frames.shape
    proj_matrices = temp_calib_state.P_mats

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

        # Safe padding value (centroid) to prevent numerical instability when projecting invalid points
        all_valid_3d = np.concatenate(per_frame_3d)
        safe_point = np.mean(all_valid_3d, axis=0)

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
            "object_points_initial": np.nan_to_num(padded_3d.reshape(-1, 3), nan=safe_point),
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

    temp_calib_state = CalibrationState(initial_calib, cam_names)

    # Extract initial parameters as ordered lists of arrays for mokap/JAX
    K_init = jnp.asarray(temp_calib_state.K_mats)
    D_init = jnp.asarray(temp_calib_state.dist_coeffs)
    r_init = jnp.asarray(temp_calib_state.rvecs_c2w)
    t_init = jnp.asarray(temp_calib_state.tvecs_c2w)

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
