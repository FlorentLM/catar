import sys
import tomllib
from pathlib import Path
from typing import List, Tuple

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

from lucida import Intrinsics, Extrinsics, CameraModel, CameraRig


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


def load_and_match_videos(data_folder: Path, video_format: str) -> Tuple[
    List[str], List[str], List[str], CameraRig]:
    """
    Load videos and calibration with smart names matching.

    Returns:
        Tuple of (video_paths, video_filenames, camera_names, camera_rig)
    """

    # Fallback to Lucida specific file if parameters.toml doesn't exist
    rig_file = data_folder / 'rig.toml'

    if rig_file.exists():
        print(f"Loading Lucida rig from '{rig_file}'...")
        rig = CameraRig.load(rig_file)
        toml_names = sorted(rig.names)
    else:
        print(f"ERROR: No calibration file ('rig.toml') found in '{data_folder}'")
        sys.exit(1)

    # Find video files
    video_paths = sorted(data_folder.glob(video_format))
    if not video_paths:
        print(f"ERROR: No videos matching '{video_format}' found in '{data_folder}'")
        sys.exit(1)

    video_filenames = [p.name for p in video_paths]

    if len(toml_names) != len(video_paths):
        print(f"ERROR: Number of cameras in calibration ({len(toml_names)}) "
              f"doesn't match number of videos ({len(video_paths)})")
        sys.exit(1)

    # Match with Levenshtein distance
    n = len(toml_names)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = Levenshtein.distance(toml_names[i], video_filenames[j])

    toml_indices, video_indices = linear_sum_assignment(cost_matrix)

    ordered_paths = ["" for _ in range(n)]
    ordered_filenames = ["" for _ in range(n)]
    ordered_toml_names = toml_names

    # Create a map from the toml_index (from the assignment) back to the video_index
    toml_to_video_map = {ti: vi for ti, vi in zip(toml_indices, video_indices)}

    for i in range(n):  # i is the index in the sorted toml_names list
        matched_video_idx = toml_to_video_map[i]
        ordered_paths[i] = str(video_paths[matched_video_idx])
        ordered_filenames[i] = video_filenames[matched_video_idx]

    return ordered_paths, ordered_filenames, ordered_toml_names, rig


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