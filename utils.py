import itertools
import sys
import tomllib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import Levenshtein
import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment


def annotations_to_polars(annotations_slice, frame_idx, camera_names, point_names):
    """Converts a numpy slice of annotations into a Polars df (for mokap)"""
    rows = []
    num_cams, num_points, _ = annotations_slice.shape

    for c in range(num_cams):
        for p in range(num_points):
            if not np.isnan(annotations_slice[c, p, 0]):
                rows.append({
                    "frame": frame_idx,
                    "camera": camera_names[c],
                    "keypoint": point_names[p],
                    "x": annotations_slice[c, p, 0],
                    "y": annotations_slice[c, p, 1],
                    "score": annotations_slice[c, p, 2],
                })
    if not rows:
        return pl.DataFrame(schema={'frame': pl.Int64, 'camera': pl.Utf8, 'keypoint': pl.Utf8, 'x': pl.Float64, 'y': pl.Float64, 'score': pl.Float64})
    return pl.DataFrame(rows)


# ============================================================================
# 3D reconstruction & confidence
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


def triangulate_points(frame_annotations: np.ndarray, proj_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Triangulates 3D points from 2D correspondences using SVD.
    """
    # TODO: Also use mokap's implementation

    num_cams, num_points, _ = frame_annotations.shape
    points_3d = np.full((num_points, 3), np.nan, dtype=np.float32)

    for p_idx in range(num_points):
        # Gather valid observations for this specific point
        valid_annots = []
        valid_proj_mats = []
        for cam_idx in range(num_cams):
            if not np.isnan(frame_annotations[cam_idx, p_idx]).any():
                valid_annots.append(frame_annotations[cam_idx, p_idx])
                valid_proj_mats.append(proj_matrices[cam_idx])

        if len(valid_annots) < 2:
            continue

        A = []
        for p2d, P in zip(valid_annots, valid_proj_mats):
            A.append(p2d[0] * P[2, :] - P[0, :])
            A.append(p2d[1] * P[2, :] - P[1, :])
        A = np.array(A)

        # Solve using SVD
        u, s, vh = np.linalg.svd(A)
        point_4d = vh[-1, :]

        # Dehomogenize to get 3D coordinates
        if point_4d[3] != 0:
            points_3d[p_idx] = point_4d[:3] / point_4d[3]

    return points_3d


def calculate_reproj_confidence(app_state: 'AppState', frame_idx: int, point_idx: int) -> np.ndarray:
    """
    Calculates reprojection error for a point in a frame using leave-one-out approach.
    """

    with app_state.lock:
        annotations_for_point = app_state.annotations[frame_idx, :, point_idx, :]
        calibration = app_state.best_individual

    num_cams = len(calibration)
    errors = np.full(num_cams, np.nan, dtype=np.float32)

    if calibration is None:
        return errors

    valid_cam_indices = np.where(~np.isnan(annotations_for_point).any(axis=1))[0]

    if len(valid_cam_indices) < 3:  # Need at least 2 other views to triangulate
        return errors

    for cam_to_test in valid_cam_indices:
        peer_indices = [i for i in valid_cam_indices if i != cam_to_test]

        peer_annots = annotations_for_point[peer_indices].reshape(len(peer_indices), 1, 2)
        peer_proj_mats = [get_projection_matrix(calibration[i]) for i in peer_indices]

        point_3d = triangulate_points(peer_annots, peer_proj_mats).flatten()

        if not np.isnan(point_3d).any():
            reprojected = reproject_points(point_3d, calibration[cam_to_test])
            if reprojected.size > 0:
                original_annot = annotations_for_point[cam_to_test]
                errors[cam_to_test] = np.linalg.norm(reprojected.flatten() - original_annot)

    return errors


# ============================================================================
# Videos IO
# ============================================================================
# TODO: These will be merged with or replaced by mokap implementations

def load_and_match_videos(data_folder: Path, video_format: str):
    """
    Load videos and calibration with smart names matching
    """
    # TODO: the smart matching logic should be ported to mokap

    # Load calibration TOML
    calib_file = data_folder / 'parameters.toml'
    if not calib_file.exists():
        print(f"ERROR: 'parameters.toml' not found in '{data_folder}'")
        sys.exit(1)

    with calib_file.open("rb") as f:
        calib_data = tomllib.load(f)

    toml_names = sorted(calib_data.keys())

    # Find video files
    video_paths = sorted(data_folder.glob(video_format))
    if not video_paths:
        print(f"ERROR: No videos matching '{video_format}' found in '{data_folder}'")
        sys.exit(1)

    video_filenames = [p.name for p in video_paths]

    if len(toml_names) != len(video_paths):
        print("ERROR: Number of cameras in TOML doesn't match number of videos")
        sys.exit(1)

    # Match with Levenshtein distance
    n = len(toml_names)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = Levenshtein.distance(toml_names[i], video_filenames[j])

    toml_indices, video_indices = linear_sum_assignment(cost_matrix)

    # Build ordered lists
    ordered_paths = []
    ordered_names = []
    calibration = []

    for i, toml_name in enumerate(toml_names):
        matched_video_idx = video_indices[np.where(toml_indices == i)][0]

        ordered_paths.append(str(video_paths[matched_video_idx]))
        ordered_names.append(video_filenames[matched_video_idx])

        # Convert TOML calibration to dict
        cam = calib_data[toml_name]
        calibration.append({
            'fx': cam['camera_matrix'][0][0],
            'fy': cam['camera_matrix'][1][1],
            'cx': cam['camera_matrix'][0][2],
            'cy': cam['camera_matrix'][1][2],
            'dist': np.array(cam['dist_coeffs'], dtype=np.float32),
            'rvec': np.array(cam['rvec'], dtype=np.float32),
            'tvec': np.array(cam['tvec'], dtype=np.float32),
        })

    return ordered_paths, ordered_names, calibration


def get_video_metadata(video_path: str) -> dict:
    """Extract metadata from a video."""
    # TODO: This is also already implemented in mokap

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    metadata = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return metadata


# ============================================================================
# Other stuff
# ============================================================================

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
