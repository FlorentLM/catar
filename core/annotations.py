from typing import TYPE_CHECKING, NamedTuple
import numpy as np
import config
from utils import triangulate_and_score

if TYPE_CHECKING:
    from lucida import CameraRig


class OverlapStats(NamedTuple):
    """Result of comparing two sets of tracking points."""
    total_overlap: int
    n_conflicts: int
    n_safe: int
    mean_dist: float
    max_dist: float
    conflict_ratio: float
    safe_ratio: float


def compute_comparison_stats(points_a, points_b, conflict_threshold, safe_threshold):
    valid_a = ~np.isnan(points_a[..., 0])
    valid_b = ~np.isnan(points_b[..., 0])

    overlap = valid_a & valid_b
    total = np.sum(overlap)

    if total == 0:
        return None

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
        if p3d.ndim == 1:
            p3d = p3d[None, :]

        reproj = rig.project(p3d)  # (C, P, 2)
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


def snap_annotation(app_state, target_cam_name, keypoint_name, frame_idx, click_pos):
    """Snap click to epipolar line intersection from other views."""

    annots = app_state.data.get_2d(frame=frame_idx, keypoint=keypoint_name)

    rig = app_state.rig
    target_cam_idx = rig.get_index(target_cam_name)

    valid_mask = ~np.isnan(annots[:, 0])
    valid_mask[target_cam_idx] = False  # exclude self

    valid_cams = np.where(valid_mask)[0]
    if len(valid_cams) < 2:
        return None

    cam_names = [rig.names[i] for i in valid_cams]  # This *needs* to be in order # TODO: Fix that in Lucida

    p3d = rig.triangulate(annots[:, :2], weights=annots[:, 2], cameras=cam_names).flatten()
    if np.isnan(p3d).any():
        return None

    reproj = rig[target_cam_name].project(p3d).flatten()

    if np.linalg.norm(reproj - click_pos) > 20:
        return None

    return reproj


def fuse_annotations(
        existing_annots: np.ndarray,
        manual_flags: np.ndarray,
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
                type_str = 'human' if manual_flags[c, p] else 'prior'
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