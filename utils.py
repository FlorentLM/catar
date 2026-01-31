import sys
from pathlib import Path
from typing import List, Tuple, Union

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

from lucida import CameraRig
from lucida.geometry.backend import xp


def get_confidence_color(confidence: float) -> tuple:
    """
    Returns a color tuple (r, g, b) based on confidence (0.0 - 1.0).
    Gradient: Red (0.0) -> Orange (0.5) -> Green (1.0).
    """
    if np.isnan(confidence):
        return (0, 255, 255)

    conf = max(0.0, min(1.0, confidence))

    if conf < 0.5:
        # red (255, 0, 0) to orange (255, 165, 0) interp
        t = conf * 2.0  # 0.0 to 1.0
        r = 255
        g = int(165 * t)
        b = 0
    else:
        # orange (255, 165, 0) to green (0, 255, 0) interp
        t = (conf - 0.5) * 2.0  # 0.0 to 1.0
        r = int(255 * (1.0 - t))
        g = int(165 + (255 - 165) * t)
        b = 0

    return r, g, b


def load_and_match_videos(
        rig_path: Union[str, Path],
        videos_folder: Union[str, Path],
        video_format: str
    ) -> Tuple[CameraRig, List[Path]]:
    """
    Load videos and calibration with smart names matching.
    """

    rig_path = Path(rig_path)
    videos_folder = Path(videos_folder)

    if rig_path.is_file():
        rig = CameraRig.load(rig_path)
        print(f"Loaded camera rig from '{rig_path}'")
    else:
        print(f"ERROR: Calibration file '{rig_path}' not found.")
        sys.exit(1)

    video_paths = sorted(videos_folder.glob(f'*.{video_format.strip("*.")}'))
    if not video_paths:
        print(f"ERROR: No videos matching '{video_format}' found in '{videos_folder}'")
        sys.exit(1)

    nb_cameras = len(rig)
    nb_videos = len(video_paths)

    if nb_cameras != nb_videos:
        print(f"ERROR: Number of cameras in calibration ({nb_cameras}) doesn't match number of videos ({nb_videos})")
        sys.exit(1)
    print(f"Found {nb_videos} videos in {videos_folder}")

    # Match with Levenshtein distance
    cost_matrix = np.zeros((nb_cameras, nb_cameras))
    for i in range(nb_cameras):
        for j in range(nb_cameras):
            cost_matrix[i, j] = Levenshtein.distance(rig.names[i], video_paths[j].name)
    rig_indices, video_indices = linear_sum_assignment(cost_matrix)

    rtv_map = dict(zip(rig_indices, video_indices))
    ordered_paths = [video_paths[rtv_map[i]] for i in range(nb_cameras)]

    return rig, ordered_paths


def line_box_intersection(a: float, b: float, c: float, box_x: float, box_y: float, box_w: float, box_h: float) -> list:
    """
    Calculates the two intersection points of a line (ax + by + c = 0) with a rectangle.
    """
    intersections = []

    # Top edge (y = box_y)
    if abs(a) > 1e-9:
        x = (-c - b * box_y) / a
        if box_x <= x <= box_x + box_w:
            intersections.append((x, box_y))

    # Bottom edge (y = box_y + box_h)
    if abs(a) > 1e-9:
        x = (-c - b * (box_y + box_h)) / a
        if box_x <= x <= box_x + box_w:
            intersections.append((x, box_y + box_h))

    # Left edge (x = box_x)
    if abs(b) > 1e-9:
        y = (-c - a * box_x) / b
        if box_y <= y <= box_y + box_h:
            intersections.append((box_x, y))

    # Right edge (x = box_x + box_w)
    if abs(b) > 1e-9:
        y = (-c - a * (box_x + box_w)) / b
        if box_y <= y <= box_y + box_h:
            if box_y <= y <= box_y + box_h:
                intersections.append((box_x + box_w, y))

    # Remove duplicate points (can happen at corners)
    unique_points = sorted(list(set(intersections)))

    return unique_points


# TODO: Maybe these two functions could be removed / replaced by Lucida's own

def compute_3d_scores(
        points_3d: np.ndarray,  # shape (P, 3)
        annotations: np.ndarray,  # shape (C, P, 3) or (C, P, 2)
        rig: CameraRig
) -> np.ndarray:
    """
    Computes confidence scores (0.0 to 1.0) for 3D points based on reprojection error.
    """
    if len(rig) == 0:
        return np.zeros(points_3d.shape[0], dtype=np.float32)

    valid_3d = ~np.isnan(points_3d).any(axis=1)
    if not np.any(valid_3d):
        return np.zeros(points_3d.shape[0], dtype=np.float32)

    if points_3d.ndim == 1:
        points_3d = points_3d[None, :]

    reprojected = rig.project(points_3d)  # (C, P, 2)

    # Calculate errors against observations
    obs_2d = annotations[..., :2]
    diffs = reprojected - obs_2d
    errors_sq = np.sum(diffs ** 2, axis=-1)  # (C, P)
    errors = np.sqrt(errors_sq)

    # We only care about errors where the 2D annotation actually exists
    if annotations.shape[-1] >= 3:
        valid_obs = (annotations[..., 2] > 0) & (~np.isnan(obs_2d[..., 0]))
    else:
        valid_obs = ~np.isnan(obs_2d[..., 0])

    # Compute mean error per point
    sum_errors = np.sum(np.where(valid_obs, errors, 0.0), axis=0)
    count_obs = np.sum(valid_obs, axis=0)

    mean_errors = np.divide(
        sum_errors,
        count_obs,
        out=np.full_like(sum_errors, np.inf),
        where=count_obs > 0
    )

    # Convert to score: 1.0 / (1.0 + error)
    scores = 1.0 / (1.0 + mean_errors)
    scores[count_obs == 0] = 0.0

    return scores.astype(np.float32)


def triangulate_and_score(
        annotations: np.ndarray,  # shape (C, P, 3)
        rig: CameraRig
) -> np.ndarray:
    """
    Triangulates points and computes their scores.
    """
    if len(rig) < 2:
        return np.full((annotations.shape[1], 4), np.nan, dtype=np.float32)

    points2d = annotations[..., :2]
    weights = annotations[..., 2]

    points_3d = rig.triangulate(points2d, weights=weights)

    # Compute scores
    scores = compute_3d_scores(points_3d, annotations, rig)

    # Combine into (P, 4) format
    points_4d = np.full((points_3d.shape[0], 4), np.nan, dtype=np.float32)
    points_4d[:, :3] = points_3d
    points_4d[:, 3] = scores

    return points_4d

# _______________________________________________________________

# TODO: These might make their way into Lucida directly

def robust_triangulate(rig, points_xp, cams_indices, max_reproj_error=15.0):
    """
    RANSAC-style triangulation.
    """
    current_cams = list(cams_indices)
    current_pts_xp = list(points_xp)

    while len(current_cams) >= 2:
        pts_stack = xp.stack(current_pts_xp)
        pts_in = pts_stack[:, None, :]  # (C, 1, 2)
        cam_names = [rig.names[i] for i in current_cams]

        pt_3d_xp = rig.triangulate(pts_in, cameras=cam_names).flatten()

        if xp.isnan(pt_3d_xp).any():
            break

        reproj_xp = rig.project(pt_3d_xp.reshape(1, 3), cameras=cam_names).flatten().reshape(len(current_cams), 2)

        errors = []
        for i in range(len(current_cams)):
            dist = xp.linalg.norm(reproj_xp[i] - current_pts_xp[i])
            errors.append(float(dist))

        if max(errors) < max_reproj_error:
            return pt_3d_xp, current_cams

        worst = np.argmax(errors)
        current_cams.pop(worst)
        current_pts_xp.pop(worst)

    return None, []


def snap_to_ray(rig, cam_idx, uv, target_point_3d):
    """
    Find point on ray closest to a 3D target.
    """
    cam = rig[rig.names[cam_idx]]
    uv_xp = xp.asarray(uv)
    origin_xp, dir_xp = cam.raycast(uv_xp)
    dir_xp = dir_xp.flatten()
    target_xp = xp.asarray(target_point_3d)

    t = xp.dot(target_xp - origin_xp, dir_xp)
    t = xp.maximum(t, 0.1)
    return np.asarray(origin_xp + t * dir_xp)