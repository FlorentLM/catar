from typing import TYPE_CHECKING, List, NamedTuple
import cv2
import numpy as np
import config
from utils import triangulate_and_score # TODO: Can maybe now remove this function

if TYPE_CHECKING:
    from state import AppState
    from lucida import CameraRig
    from mokap.reconstruction.reconstruction import Reconstructor
    from mokap.reconstruction.tracking import MultiObjectTracker


class OverlapStats(NamedTuple):
    """Result of comparing two sets of tracking points."""
    total_overlap: int
    n_conflicts: int
    n_safe: int
    mean_dist: float
    max_dist: float
    conflict_ratio: float
    safe_ratio: float


def compute_patch_ncc(img_prev, img_curr, p_prev, p_curr, patch_size=config.NCC_PATCH_SIZE):
    """Computes NCC between patches."""
    h, w = img_prev.shape
    pad = patch_size // 2
    u_p, v_p = p_prev
    u_c, v_c = p_curr

    if (u_p < pad or u_p >= w - pad or v_p < pad or v_p >= h - pad or
            u_c < pad or u_c >= w - pad or v_c < pad or v_c >= h - pad):
        return -1.0

    try:
        p1 = cv2.getRectSubPix(img_prev, (patch_size, patch_size), (float(u_p), float(v_p)))
        p2 = cv2.getRectSubPix(img_curr, (patch_size, patch_size), (float(u_c), float(v_c)))
        if np.std(p1) < 1e-5 or np.std(p2) < 1e-5: return 0.0
        return float(cv2.matchTemplate(p2.astype(np.float32), p1.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0])
    except:
        return -1.0


def compute_comparison_stats(points_a, points_b, conflict_threshold, safe_threshold):
    valid_a = ~np.isnan(points_a[..., 0])
    valid_b = ~np.isnan(points_b[..., 0])
    overlap = valid_a & valid_b
    total = np.sum(overlap)

    if total == 0: return None

    dists = np.linalg.norm(points_a[overlap, :2] - points_b[overlap, :2], axis=1)

    return OverlapStats(
        total_overlap=total,
        n_conflicts=np.sum(dists > conflict_threshold),
        n_safe=np.sum(dists < safe_threshold),
        mean_dist=np.mean(dists),
        max_dist=np.max(dists),
        conflict_ratio=np.sum(dists > conflict_threshold) / total,
        safe_ratio=np.sum(dists < safe_threshold) / total
    )


def detect_track_collision(existing_annots, new_predictions, rig: 'CameraRig', distance_threshold=30.0,
                           ratio_threshold=0.25):
    """Checks if new predictions conflict with existing annotations."""

    # 2D check
    stats = compute_comparison_stats(existing_annots, new_predictions, distance_threshold, 5.0)

    if stats and stats.safe_ratio > 0.90:
        return False  # safe match

    # 3D check
    existing_3d = triangulate_and_score(existing_annots, rig)
    valid_3d_mask = ~np.isnan(existing_3d[:, 0])

    check_type = "2D"
    if np.any(valid_3d_mask):
        # Reproject expectations
        p3d = existing_3d[:, :3]
        if p3d.ndim == 1: p3d = p3d[None, :]

        reproj = rig.project(p3d)  # (C, P, 2)

        # Handle shape mismatch if rig subset (unlikely with full rig)
        if reproj.shape[:-1] != new_predictions[..., :2].shape[:-1]:
            return False  # cant compare mismatched shapes

        stats_3d = compute_comparison_stats(reproj, new_predictions, distance_threshold, 5.0)
        if stats_3d:
            stats = stats_3d
            check_type = "3D"

    if stats is None:
        return False

    if stats.conflict_ratio > ratio_threshold:
        print(f"[ {check_type} COLLISION ] {stats.conflict_ratio:.1%} conflict. Mean: {stats.mean_dist:.1f}px")
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
    """Runs processing pipeline."""
    print(f"\n[Frame: {source_frame_idx} -> {frame_idx}] (step {batch_step})")

    with app_state.lock:
        rig = app_state.rig
        point_names = app_state.point_names
        collision_stop = app_state.tracker_collision_stop

    existing = app_state.data.get_frame_annotations(frame_idx, copy=True)
    is_human = app_state.data.get_human_annotated_flags(frame_idx, copy=True)

    src_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in source_frames]
    dst_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in dest_frames]

    # LK tracking
    pred_LK = track_points(app_state, src_gray, dst_gray, source_frame_idx, frame_idx)

    if collision_stop and detect_track_collision(existing, pred_LK, rig):
        return False

    # Mokap reconstruction
    cam_indices, point_indices = np.where(~np.isnan(pred_LK[..., 0]))
    n_pts = len(point_indices)

    active_tracklets = []
    if n_pts > 0:
        inputs = {
            "frame_indices": np.full(n_pts, frame_idx, dtype=np.int32),
            "kp_type_ids": point_indices.astype(np.int16),
            "cam_ids": cam_indices.astype(np.int8),
            "coords": pred_LK[cam_indices, point_indices, :2].astype(np.float32),
            "scores": pred_LK[cam_indices, point_indices, 2].astype(np.float32)
        }
        soup = reconstructor.reconstruct_batch(inputs=inputs, keypoint_names=point_names)
        active_tracklets = tracker.update(soup, frame_idx)

    # Skeleton feedback
    final_skel_kps = None
    norm_score = 0.0

    if active_tracklets:
        best = max(active_tracklets, key=lambda t: (len(t.skeleton.keypoints), t.skeleton.score))
        final_skel_kps = best.skeleton.keypoints
        norm_score = np.clip(best.skeleton.score / (max(1, len(final_skel_kps)) * reconstructor.max_point_score), 0, 1)

    # Model reproj
    pred_model = np.full_like(pred_LK, np.nan)
    if final_skel_kps and len(rig) > 0:
        p3d_list = []
        p_ind_list = []
        for name, pos in final_skel_kps.items():
            if name in point_names:
                p3d_list.append(pos)
                p_ind_list.append(app_state.point_nti[name])

        if p3d_list:
            reproj = rig.project(np.array(p3d_list))  # (C, P_subset, 2)
            for i, p_idx in enumerate(p_ind_list):
                pred_model[:, p_idx, :2] = reproj[:, i, :]
                pred_model[:, p_idx, 2] = norm_score

    # Fusion
    annots_fused = fuse_annotations(existing, is_human, pred_LK, pred_model)

    # Rescue single-view points lost in fusion
    valid_LK = ~np.isnan(pred_LK[..., 0])
    valid_Fused = ~np.isnan(annots_fused[..., 0])
    n_views_LK = np.sum(valid_LK, axis=0)
    is_fused_lost = ~np.any(valid_Fused, axis=0)

    for p_idx in range(app_state.num_points):
        if n_views_LK[p_idx] == 1 and is_fused_lost[p_idx]:
            cam_idx = np.where(valid_LK[:, p_idx])[0][0]
            annots_fused[cam_idx, p_idx] = pred_LK[cam_idx, p_idx]

    # Final triangulation
    final_3d = triangulate_and_score(annots_fused, rig)

    # Fill missing triangulation with Skeleton
    if final_skel_kps:
        missing = np.isnan(final_3d[:, 0])
        for name, coords in final_skel_kps.items():
            idx = app_state.point_nti.get(name)
            if idx is not None and missing[idx]:
                final_3d[idx, :3] = coords
                final_3d[idx, 3] = norm_score * 0.8

    # Zombie point filter
    if final_skel_kps:
        assembled_kps = set(final_skel_kps.keys())
        rejected_kps = app_state.all_points_set - (assembled_kps | app_state.non_skeleton_points_set)
        for p_name in rejected_kps:
            p_idx = app_state.point_nti[p_name]
            valid_mask = ~np.isnan(annots_fused[:, p_idx, 0])
            if np.any(valid_mask):
                annots_fused[valid_mask, p_idx, 2] *= 0.5
                kill_mask = annots_fused[:, p_idx, 2] < 0.1
                annots_fused[kill_mask, p_idx, :] = np.nan

    app_state.data.set_frame_annotations(frame_idx, annots_fused)
    app_state.data.set_frame_points3d(frame_idx, final_3d)

    return True


def track_points(app_state, src_gray, dst_gray, src_idx, dst_idx):
    """LK Tracker logic."""
    with app_state.lock:
        rig = app_state.rig
        decay = app_state.tracker_decay_rate

    annots_src = app_state.data.get_frame_annotations(src_idx, copy=True)
    out_annots = np.full_like(annots_src, np.nan)

    if len(rig) == 0: return out_annots

    num_cams = annots_src.shape[0]

    for c in range(num_cams):
        p0 = annots_src[c, :, :2]
        valid = ~np.isnan(p0).any(axis=1)
        if not np.any(valid): continue

        p0_active = p0[valid].reshape(-1, 1, 2)
        idx_active = np.where(valid)[0]

        p1, st, _ = cv2.calcOpticalFlowPyrLK(src_gray[c], dst_gray[c], p0_active, None, **config.LK_PARAMS)
        p0r, st_r, _ = cv2.calcOpticalFlowPyrLK(dst_gray[c], src_gray[c], p1, None, **config.LK_PARAMS)

        for i, idx in enumerate(idx_active):
            fb_err = np.linalg.norm(p0_active[i] - p0r[i])
            if st[i] == 1 and st_r[i] == 1 and fb_err < config.FORWARD_BACKWARD_THRESHOLD:
                ncc = compute_patch_ncc(src_gray[c], dst_gray[c], p0_active[i].flatten(), p1[i].flatten())
                if ncc < config.NCC_THRESHOLD_KILL: continue

                old_conf = annots_src[c, idx, 2]
                geom_q = (1.0 - fb_err / config.FORWARD_BACKWARD_THRESHOLD)

                # Soft penalty
                ncc_factor = 1.0
                if ncc < config.NCC_THRESHOLD_WARNING:
                    ncc_factor = max(0.0, (ncc - config.NCC_THRESHOLD_KILL) / (
                                config.NCC_THRESHOLD_WARNING - config.NCC_THRESHOLD_KILL))

                new_conf = old_conf * geom_q * ncc_factor * decay
                out_annots[c, idx] = [*p1[i].flatten(), new_conf]

    # Multi-view consensus upgrade
    for p_idx in range(app_state.num_points):
        valid_cams = np.where(~np.isnan(out_annots[:, p_idx, 0]))[0]
        if len(valid_cams) < 2: continue

        # Triangulate hypothesis
        obs = out_annots[valid_cams, p_idx, :2]  # (N_valid, 2)
        obs_reshaped = obs[:, None, :]  # (C, 1, 2)

        cam_names = [rig.names[i] for i in valid_cams]
        p3d = rig.triangulate(obs_reshaped, cameras=cam_names).flatten()  # (3,)

        if np.isnan(p3d).any(): continue

        reproj = rig.project(p3d[None, :], cameras=cam_names).reshape(len(valid_cams), 2)
        errs = np.linalg.norm(obs - reproj, axis=1)

        for k, c_idx in enumerate(valid_cams):
            mv_conf = max(0.0, 1.0 - (errs[k] / config.LK_CONFIDENCE_MAX_ERROR))
            if mv_conf < out_annots[c_idx, p_idx, 2]:
                out_annots[c_idx, p_idx, 2] = mv_conf

    # Max confidence cap
    mask = ~np.isnan(out_annots[..., 2])
    out_annots[mask, 2] = np.fmin(out_annots[mask, 2], config.FUSION_MAX_AUTO_CONFIDENCE)

    return out_annots


def snap_annotation(app_state, target_cam_idx, point_idx, frame_idx, click_pos):
    """Snap click to epipolar line intersection from other views."""
    annots = app_state.data.get_point_annotations(frame_idx, point_idx)
    rig = app_state.rig

    valid_mask = ~np.isnan(annots[:, 0])
    valid_mask[target_cam_idx] = False  # exclude self

    valid_cams = np.where(valid_mask)[0]
    if len(valid_cams) < 2: return None

    cam_names = [rig.names[i] for i in valid_cams]
    obs = annots[valid_cams, :2][:, None, :]
    weights = annots[valid_cams, 2][:, None]

    p3d = rig.triangulate(obs, weights=weights, cameras=cam_names).flatten()

    if np.isnan(p3d).any(): return None

    reproj = rig[rig.names[target_cam_idx]].project(p3d).flatten()

    if np.linalg.norm(reproj - click_pos) > 20: return None

    return reproj


def fuse_annotations(
        existing_annots: np.ndarray,
        human_flags: np.ndarray,
        lk_annots: np.ndarray,
        model_annots: np.ndarray,
) -> np.ndarray:
    """
    Fuses annotations from multiple sources.
    """
    num_cams, num_points, _ = existing_annots.shape
    final_annotations = np.full((num_cams, num_points, 3), np.nan, dtype=np.float32)

    for c in range(num_cams):
        for p in range(num_points):
            sources = []

            # Check existing data (Source 1: Human or Source 2: prior auto track)
            if not np.isnan(existing_annots[c, p, 0]):
                type_str = 'human' if human_flags[c, p] else 'prior'
                sources.append({
                    'pos': existing_annots[c, p, :2],
                    'conf': existing_annots[c, p, 2],
                    'type': type_str
                })

            # Source 3: LK tracker
            if not np.isnan(lk_annots[c, p, 0]):
                sources.append({
                    'pos': lk_annots[c, p, :2],
                    'conf': lk_annots[c, p, 2],
                    'type': 'lk'
                })

            # Source 4: 3D Model reprojection
            if not np.isnan(model_annots[c, p, 0]):
                sources.append({
                    'pos': model_annots[c, p, :2],
                    'conf': model_annots[c, p, 2],
                    'type': 'model'
                })

            if not sources:
                continue

            # If only one source, use it directly
            if len(sources) == 1:
                final_annotations[c, p] = [*sources[0]['pos'], sources[0]['conf']]
                continue

            # Find best source (highest confidence)
            sources.sort(key=lambda s: s['conf'], reverse=True)
            best_source = sources[0]
            other_sources = sources[1:]

            # Start with best source's confidence
            final_conf = best_source['conf']

            # Weighted average of agreeing sources
            sum_weights = best_source['conf']
            weighted_pos = best_source['pos'] * best_source['conf']

            # Check for agreement from other sources
            for other in other_sources:

                # Fusion precedence ratio: Ratio by which a better source must exceed a worse source
                # to completely take precedence on it
                if other['type'] != 'human':
                    if best_source['conf'] > (other['conf'] * config.FUSION_PRECEDENCE_RATIO):
                        continue

                distance = np.linalg.norm(best_source['pos'] - other['pos'])

                # If sources agree spatially, we average them
                # (this allows a strong 3D model to slightly shift a Human / Prior annotation)
                if distance < config.FUSION_AGREEMENT_RADIUS:
                    weighted_pos += other['pos'] * other['conf']
                    sum_weights += other['conf']

                    # Confidence bonus based on agreement quality
                    bonus = (1.0 - (distance / config.FUSION_AGREEMENT_RADIUS)) * config.FUSION_AGREEMENT_BONUS
                    final_conf += bonus

            final_pos = weighted_pos / (sum_weights + 1e-6)

            max_conf = config.FUSION_HUMAN_CONFIDENCE if best_source[
                                                             'type'] == 'human' else config.FUSION_MAX_AUTO_CONFIDENCE
            final_conf = min(final_conf, max_conf)

            final_annotations[c, p] = [*final_pos, final_conf]

    return final_annotations