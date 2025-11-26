from typing import TYPE_CHECKING, List
import cv2
import numpy as np
import config
from core import fuse_annotations
from utils import triangulate_and_score

if TYPE_CHECKING:
    from state import AppState
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


def compute_patch_ncc(
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        p_prev: np.ndarray,
        p_curr: np.ndarray,
        patch_size: int = config.NCC_PATCH_SIZE
) -> float:
    """
    Computes Normalised Cross Correlation between two patches.
    Returns 1.0 if identical, -1.0 if inverted, 0.0 if uncorrelated.
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


def detect_track_collision(
        existing_annots: np.ndarray,
        new_predictions: np.ndarray,
        distance_threshold: float = 30.0,
        ratio_threshold: float = 0.25
) -> bool:
    """
    Checks if the new predictions diverge significantly from existing annotations.

    Args:
        existing_annots: Array (C, P, 3) of existing data.
        new_predictions: Array (C, P, 3) of new LK tracker data.
        distance_threshold: Pixel distance to consider a single point in conflict.
        ratio_threshold: Percentage (0.0-1.0) of total visible points that must
                         be in conflict to trigger a stop.
    """

    valid_existing = ~np.isnan(existing_annots[..., 0])
    valid_new = ~np.isnan(new_predictions[..., 0])

    # Union (total scope of data available in this frame)
    union_mask = valid_existing | valid_new
    total_points_count = np.sum(union_mask)

    if total_points_count == 0:
        return False

    # Intersection (waht can actually be compared)
    overlap_mask = valid_existing & valid_new

    # if no overlap, no conflict
    if not np.any(overlap_mask):
        return False

    # Calculate vector diffs
    diffs = existing_annots[..., :2] - new_predictions[..., :2]
    dists_overlap = np.linalg.norm(diffs[overlap_mask], axis=1)

    # How much of the total skeleton is currently contradicting?
    n_conflicts = np.sum(dists_overlap > distance_threshold)
    conflict_ratio = n_conflicts / total_points_count

    if conflict_ratio > ratio_threshold:
        print(f"!!! COLLISION DETECTED !!!")
        print(f"    {n_conflicts} points conflict out of {total_points_count} visible points.")
        print(f"    Ratio: {conflict_ratio:.2f} (Threshold: {ratio_threshold})")
        print(f"    Stopping batch track.")
        return True

    return False


def process_frame(
        frame_idx: int,
        source_frame_idx: int,
        app_state: 'AppState',
        reconstructor: 'Reconstructor',
        tracker: 'MultiObjectTracker',
        source_frames: List[np.ndarray],
        dest_frames: List[np.ndarray],
        batch_step: int = 0
) -> bool:
    """
    Runs the full processing pipeline for a frame.
    """

    print(f"\n[Frame: {source_frame_idx} -> {frame_idx}] (step {batch_step})")

    with app_state.lock:
        calibration = app_state.calibration
        point_names = app_state.point_names
        collision_stop = app_state.tracker_collision_stop

    existing_annots = app_state.data.get_frame_annotations(frame_idx, copy=True)
    is_human_flags = app_state.data.get_human_annotated_flags(frame_idx, copy=True)

    src_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in source_frames]
    dst_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in dest_frames]

    # Get geometric prediction from LK tracking
    predictions_LK = track_points(
        app_state, src_gray, dst_gray, source_frame_idx, frame_idx
    )

    if collision_stop:
        if detect_track_collision(existing_annots, predictions_LK, distance_threshold=30.0, ratio_threshold=0.25):  # TODO: config for these
            return False  # signal to stop

    # Prepare input for Mokap Reconstructor
    cam_indices, point_indices = np.where(~np.isnan(predictions_LK[..., 0]))
    n_points_LK = len(point_indices)

    if config.VERBOSE:
        print(f"LK PREDICT:    Started with {n_points_LK} raw 2D keypoints from Optical Flow.")

    if n_points_LK > 0:
        coords = predictions_LK[cam_indices, point_indices, :2]
        scores = predictions_LK[cam_indices, point_indices, 2]

        reconstruction_input = {
            "frame_indices": np.full(n_points_LK, frame_idx, dtype=np.int32),
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

        if config.VERBOSE:
            print(f"MOKAP RECON:   Input {n_points_LK} 2D points -> Produced soup "
                  f"of {soup.num_points} 3D points and {len(soup.ray_origins)} orphan rays.")

        # Run multi-object tracking (skeleton assembly + time association)
        active_tracklets = tracker.update(soup, frame_idx)
    else:
        if config.VERBOSE:
            print("MOKAP RECON:   No input points from LK.")
        active_tracklets = []

    # Extract best skeleton for feedback loop
    final_skeleton_kps = None
    norm_skeleton_score = 0.0

    if active_tracklets:
        # Heuristic: Pick tracklet with most keypoints or highest score
        # TODO: maybe this should be smarter
        best_tracklet = max(active_tracklets, key=lambda t: (len(t.skeleton.keypoints), t.skeleton.score))
        final_skeleton_kps = best_tracklet.skeleton.keypoints
        num_kps = len(final_skeleton_kps)

        max_score = reconstructor.max_point_score
        avg_score = best_tracklet.skeleton.score / max(1, num_kps)
        norm_skeleton_score = np.clip(avg_score / max_score, 0.0, 1.0)

        if config.VERBOSE:
            print(f"MOKAP ASSEMBLE: SUCCESS -> Assembled 1 skeleton with {num_kps} keypoints "
                  f"(score: {best_tracklet.skeleton.score:.2f}, or {norm_skeleton_score:.2f} normalised).")
    else:
        if config.VERBOSE:
            print(f"MOKAP ASSEMBLE: Could not assemble any skeletons from the soup.")

    # Get model-reprojected annotations
    predictions_model = np.full_like(predictions_LK, np.nan)

    if final_skeleton_kps and calibration.best_calibration:
        points3d = []
        p_indices = []

        for p_name, pos_3d in final_skeleton_kps.items():
            if p_name in point_names:
                points3d.append(pos_3d)
                p_indices.append(app_state.point_nti[p_name])

        points3d = np.array(points3d)

        if points3d.size > 0:
            reprojected_all_cams = calibration.reproject_to_all(points3d)  # (C, P, 2)

            # Fill annotation array
            for i, p_idx in enumerate(p_indices):
                predictions_model[:, p_idx, :2] = reprojected_all_cams[:, i, :]
                predictions_model[:, p_idx, 2] = norm_skeleton_score

    # Step 1: Fuse all evidence to get best 2D annotations
    annotations_fused = fuse_annotations(
        existing_annots=existing_annots,
        human_flags=is_human_flags,
        lk_annots=predictions_LK,
        model_annots=predictions_model,
    )

    # Rescue single-view LK tracks that were lost during fusion
    valid_mask_LK = ~np.isnan(predictions_LK[..., 0])
    n_views_per_LK_point = np.sum(valid_mask_LK, axis=0)

    valid_mask_fused = ~np.isnan(annotations_fused[..., 0])
    is_fused_point_lost = ~np.any(valid_mask_fused, axis=0)

    for p_idx in range(app_state.num_points):
        #   LK had exactly one view         AND  the point was lost in fusion
        if n_views_per_LK_point[p_idx] == 1 and is_fused_point_lost[p_idx]:

            # find the camera that had the single track and re-insert it
            cam_idx = np.where(valid_mask_LK[:, p_idx])[0][0]
            annotations_fused[cam_idx, p_idx] = predictions_LK[cam_idx, p_idx]

    # Step 2: Triangulate fused 2D points
    if calibration.best_calibration is not None:
        final_points3d = triangulate_and_score(annotations_fused, calibration)
    else:
        final_points3d = np.full((app_state.num_points, 4), np.nan, dtype=np.float32)

    # Step 3: Create final 3D pose (with priority on triangulation)
    if final_skeleton_kps:

        # Calculate mask of points where triangulation failed
        missing_mask = np.isnan(final_points3d[:, 0])

        # Fill where triangulation failed with mokap skeleton
        for name, coords in final_skeleton_kps.items():
            idx = app_state.point_nti.get(name)
            if idx is not None and missing_mask[idx]:
                final_points3d[idx, :3] = coords
                final_points3d[idx, 3] = norm_skeleton_score * 0.8

    # Step 4: Structural feedback (penalize points rejected by skeleton assembler)
    if final_skeleton_kps:
        assembled_kps = set(final_skeleton_kps.keys())

        # Rejected points: all - (assembled + non skeleton)
        rejected_kps = app_state.all_points_set - (assembled_kps | app_state.non_skeleton_points_set)

        for p_name in rejected_kps:
            p_idx = app_state.point_nti[p_name]

            valid_mask = ~np.isnan(annotations_fused[:, p_idx, 0])

            if np.any(valid_mask):
                if config.VERBOSE:
                    print(f"ZOMBIE POINT: '{p_name}' was rejected by Mokap")

                # Slash confidence by half
                annotations_fused[valid_mask, p_idx, 2] *= 0.5

                # Kill if confidence drops too low
                kill_mask = annotations_fused[:, p_idx, 2] < 0.1
                annotations_fused[kill_mask, p_idx, :] = np.nan

    # Step 4: Update app state with final data
    app_state.data.set_frame_annotations(frame_idx, annotations_fused)
    app_state.data.set_frame_points3d(frame_idx, final_points3d)

    return True

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
        calibration = app_state.calibration
        cam_names = app_state.camera_names
        point_names = app_state.point_names
        decay_rate = app_state.tracker_decay_rate

    annotations_source_full = app_state.data.get_frame_annotations(source_frame_idx, copy=True)

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
            # Geometric check (forward-backward error)
            fb_error = np.linalg.norm(start_points[i] - p0_backward[i])
            is_geom_valid = (status_fwd[i] == 1 and status_bwd[i] == 1 and
                             fb_error < config.FORWARD_BACKWARD_THRESHOLD)

            if is_geom_valid:
                # Appearance check (NCC)
                ncc_score = compute_patch_ncc(
                    src_gray, dst_gray,
                    start_points[i].flatten(),
                    p1_forward[i].flatten()
                )

                # Kill if terrible
                if ncc_score < config.NCC_THRESHOLD_KILL:
                    if config.VERBOSE:
                        print(f"NCC CHECK: Killed point '{app_state.point_itn[p_idx]}' in "
                              f"camera '{app_state.camera_itn[cam_idx]}' (NCC score = {ncc_score:.2f})")
                    continue

                # Confidence calculation
                prev_conf = annotations_source_full[cam_idx, p_idx, 2]
                if np.isnan(prev_conf):
                    prev_conf = config.MAX_SINGLE_VIEW_CONFIDENCE

                geom_quality = (1.0 - (fb_error / config.FORWARD_BACKWARD_THRESHOLD))

                # Soft penalty for lower NCC scores
                ncc_factor = 1.0
                if ncc_score < config.NCC_THRESHOLD_WARNING:

                    if config.VERBOSE:
                        print(f"NCC CHECK: Penalty applied to '{app_state.point_itn[p_idx]}' in "
                              f"camera '{app_state.camera_itn[cam_idx]}' (NCC score = {ncc_score:.2f})")

                    ncc_factor = max(0.0, (ncc_score - config.NCC_THRESHOLD_KILL) / (
                                config.NCC_THRESHOLD_WARNING - config.NCC_THRESHOLD_KILL))

                new_conf = prev_conf * geom_quality * ncc_factor * decay_rate

                output_annotations[cam_idx, p_idx] = [*p1_forward[i].flatten(), new_conf]

    # Multi-view consensus upgrade
    for p_idx in range(app_state.num_points):

        # Find all cameras that have a valid track for this point
        peer_cam_indices = [i for i, p in enumerate(output_annotations[:, p_idx]) if not np.isnan(p).any()]

        # if we have enough views for triangulation we can try to upgrade the confidence
        if len(peer_cam_indices) >= 2:

            for cam_idx_to_check in peer_cam_indices:
                point_to_check = output_annotations[cam_idx_to_check, p_idx, :2]

                # The consensus group is all the other cameras (that have a valid track)
                consensus_peer_indices = [i for i in peer_cam_indices if i != cam_idx_to_check]

                if len(consensus_peer_indices) < 2:
                    continue

                consensus_annots = output_annotations[consensus_peer_indices, p_idx, :2]

                points_for_triangulation = consensus_annots.reshape(len(consensus_peer_indices), 1, 2)

                point_3d_hypothesis = calibration.triangulate_subset(
                    points_for_triangulation,
                    np.array(consensus_peer_indices),
                    weights=None
                ).flatten()

                if np.isnan(point_3d_hypothesis).any():
                    continue

                cam_name_to_check = app_state.camera_itn[cam_idx_to_check]
                reprojected = calibration.reproject_to_one(
                    point_3d_hypothesis.reshape(1, 3),
                    cam_name_to_check
                )

                if reprojected.size == 0:
                    continue

                # Geometric error for this LK track
                error = np.linalg.norm(point_to_check - reprojected.flatten())

                # Convert error into confidence score
                multi_view_confidence = max(0.0, 1.0 - (error / config.LK_CONFIDENCE_MAX_ERROR))

                if multi_view_confidence < output_annotations[cam_idx_to_check, p_idx, 2]:
                    output_annotations[cam_idx_to_check, p_idx, 2] = multi_view_confidence

    # Ensure no automated track exceeds maximum
    is_valid = ~np.isnan(output_annotations[..., 2])
    output_annotations[is_valid, 2] = np.fmin(
        output_annotations[is_valid, 2],
        config.FUSION_MAX_AUTO_CONFIDENCE
    )

    return output_annotations
