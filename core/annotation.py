from typing import TYPE_CHECKING, Optional
import numpy as np
import config

if TYPE_CHECKING:
    from state import AppState


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

    annotations_for_point = app_state.data.get_point_annotations(frame_idx, point_idx)
    calibration = app_state.calibration

    if calibration.best_calibration is None:
        return None

    num_cams = annotations_for_point.shape[0]
    cam_indices = np.arange(num_cams)

    is_not_target_cam = (cam_indices != target_cam_idx)
    is_valid_annotation = ~np.isnan(annotations_for_point).any(axis=1)
    final_mask = is_not_target_cam & is_valid_annotation

    valid_cam_indices = cam_indices[final_mask]

    if len(valid_cam_indices) < 2:
        return None

    valid_annotations = annotations_for_point[final_mask]
    valid_annotations_2d = valid_annotations[:, :2]
    valid_scores = valid_annotations[:, 2]

    point_to_triangulate = valid_annotations_2d.reshape(len(valid_cam_indices), 1, 2)
    weights_for_triangulation = valid_scores.reshape(len(valid_cam_indices), 1)

    point_3d_single = calibration.triangulate_subset(
        point_to_triangulate,
        valid_cam_indices,
        weights_for_triangulation
    )

    if np.isnan(point_3d_single).any():
        return None

    point_3d = point_3d_single.flatten()

    cam_name = calibration.camera_itn[target_cam_idx]
    reprojected_point_2d = calibration.reproject_to_one(
        point_3d.reshape(1, 3),
        cam_name
    )

    if reprojected_point_2d.size == 0:
        return None

    reprojected_point_2d = reprojected_point_2d.flatten()

    # Only snap if the click is close enough
    distance = np.linalg.norm(reprojected_point_2d - user_click_pos)
    if distance > 20:
        return None

    return reprojected_point_2d


def fuse_annotations(
        existing_annots: np.ndarray,
        human_flags: np.ndarray,
        lk_annots: np.ndarray,
        model_annots: np.ndarray,
) -> np.ndarray:
    """
    Fuses annotations from multiple sources with confidence-based weighting.

    Finds the best source and boosts confidence if other sources agree.

    Args:
        existing_annots: Current state of annotations (C, P, 3). Used as 'Human' source if flag is set, or 'Prior' source otherwise.
        human_flags: Boolean flags for human annotations (C, P)
        lk_annots: Annotations from LK tracker (C, P, 3)
        model_annots: Annotations from 3D skeleton reprojection (C, P, 3)

    Returns:
        Fused annotations (C, P, 3) with (x, y, confidence)
    """
    num_cams, num_points, _ = existing_annots.shape
    final_annotations = np.full((num_cams, num_points, 3), np.nan, dtype=np.float32)

    for c in range(num_cams):
        for p in range(num_points):
            sources = []

            # Check existing data (Source 1: Human or Source 2: prior auto track)
            if not np.isnan(existing_annots[c, p, 0]):
                if human_flags[c, p]:
                    # Human Annotation (highest priority)
                    sources.append({
                        'pos': existing_annots[c, p, :2],
                        'conf': existing_annots[c, p, 2],
                        'type': 'human'
                    })
                else:
                    # Prior auto track
                    sources.append({
                        'pos': existing_annots[c, p, :2],
                        'conf': existing_annots[c, p, 2],
                        'type': 'prior'
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
            sum_of_weights = best_source['conf']
            weighted_sum_pos = best_source['pos'] * best_source['conf']

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
                    # Agreement found: add to average
                    weighted_sum_pos += other['pos'] * other['conf']
                    sum_of_weights += other['conf']

                    # Confidence bonus based on agreement quality
                    bonus = (1.0 - (distance / config.FUSION_AGREEMENT_RADIUS)) * config.FUSION_AGREEMENT_BONUS
                    final_conf += bonus

            # Calculate final position from agreeing sources
            final_pos = np.divide(
                weighted_sum_pos,
                sum_of_weights,
                out=np.full(2, np.nan),
                where=sum_of_weights > 1e-6
            )

            # Apply confidence ceiling
            # Humans can go up to 1.0 (or whatever config says), automated tracks capped lower
            if best_source['type'] == 'human':
                max_conf = config.FUSION_HUMAN_CONFIDENCE
            else:
                max_conf = config.FUSION_MAX_AUTO_CONFIDENCE
            final_conf = min(final_conf, max_conf)

            final_annotations[c, p] = [*final_pos, final_conf]

    return final_annotations