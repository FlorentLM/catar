import itertools
import sys
import tomllib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import Levenshtein

import numpy as np
import polars as pl
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

from mokap.utils.fileio import probe_video
from mokap.utils.geometry import projective
from mokap.utils.geometry import transforms


# TODO: These wrappers for mokap should be removed eventually


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
# 3D reconstruction & confidence (wrappers for mokap)
# ============================================================================

def get_projection_matrix(cam_params: Dict[str, Any]) -> np.ndarray:
    """Computes the 3x4 projection matrix."""

    K = np.array([
        [cam_params['fx'], 0.0, cam_params['cx']],
        [0.0, cam_params['fy'], cam_params['cy']],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Mokap expects world-to-camera poses for projection
    rvec_c2w = cam_params['rvec']
    tvec_c2w = cam_params['tvec']
    rvec_w2c, tvec_w2c = transforms.invert_rtvecs(jnp.asarray(rvec_c2w), jnp.asarray(tvec_c2w))

    E_w2c = transforms.extrinsics_matrix(rvec_w2c, tvec_w2c)
    P = transforms.projection_matrix(K, E_w2c)
    return np.asarray(P)


def reproject_points(points_3d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """Reprojects 3D points to 2D."""

    if points_3d.size == 0:
        return np.array([])

    K = np.array([
        [cam_params['fx'], 0.0, cam_params['cx']],
        [0.0, cam_params['fy'], cam_params['cy']],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # get world-to-camera for projection
    rvec_c2w = cam_params['rvec']
    tvec_c2w = cam_params['tvec']
    rvec_w2c, tvec_w2c = transforms.invert_rtvecs(jnp.asarray(rvec_c2w), jnp.asarray(tvec_c2w))

    reprojected, _ = projective.project_points(
        object_points=jnp.asarray(points_3d),
        rvec=rvec_w2c,
        tvec=tvec_w2c,
        camera_matrix=jnp.asarray(K),
        dist_coeffs=jnp.asarray(cam_params['dist'])
    )

    return np.asarray(reprojected).reshape(-1, 2)


def undistort_points(points_2d: np.ndarray, cam_params: Dict[str, Any]) -> np.ndarray:
    """Undistorts 2D points."""

    valid_mask = ~np.isnan(points_2d).any(axis=-1)
    if not np.any(valid_mask):
        return np.full_like(points_2d, np.nan)

    K = np.array([
        [cam_params['fx'], 0.0, cam_params['cx']],
        [0.0, cam_params['fy'], cam_params['cy']],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    valid_points = points_2d[valid_mask]

    undistorted = projective.undistort_points(
        points2d=jnp.asarray(valid_points),
        camera_matrix=jnp.asarray(K),
        dist_coeffs=jnp.asarray(cam_params['dist'])
    )

    result = np.full_like(points_2d, np.nan)
    result[valid_mask] = np.asarray(undistorted).reshape(-1, 2)
    return result


def calculate_fundamental_matrices(calibration: List[Dict]) -> Dict[Tuple[int, int], np.ndarray]:
    """Calculates F matrix for each camera pair."""

    num_cams = len(calibration)
    if num_cams < 2:
        return {}

    f_mats = {}
    Ks = jnp.asarray([np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]]) for c in calibration])
    rvecs_c2w = jnp.asarray([c['rvec'] for c in calibration])
    tvecs_c2w = jnp.asarray([c['tvec'] for c in calibration])

    # invert poses from camera-to-world to world-to-camera
    rvecs_w2c, tvecs_w2c = transforms.invert_rtvecs(rvecs_c2w, tvecs_c2w)

    # Create (from_cam, to_cam) pairs
    cam_indices = list(range(num_cams))
    pairs = [p for p in itertools.product(cam_indices, repeat=2) if p[0] != p[1]]

    # Prepare for batched mokap function (num_pairs, 2, ...)
    K_pairs = jnp.asarray([(Ks[i], Ks[j]) for i, j in pairs])
    rvecs_w2c_pairs = jnp.asarray([(rvecs_w2c[i], rvecs_w2c[j]) for i, j in pairs])
    tvecs_w2c_pairs = jnp.asarray([(tvecs_w2c[i], tvecs_w2c[j]) for i, j in pairs])

    F_matrices_batched = transforms.batched_fundamental_matrices(K_pairs, rvecs_w2c_pairs, tvecs_w2c_pairs)

    for idx, (i, j) in enumerate(pairs):
        f_mats[(i, j)] = np.asarray(F_matrices_batched[idx])

    return f_mats


def triangulate_points(frame_annotations: np.ndarray, proj_matrices: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Triangulates 3D points using mokap."""

    num_cams, num_points, _ = frame_annotations.shape
    if num_points == 0:
        return np.full((0, 3), np.nan, dtype=np.float32)

    # Expected input shape for points2d is (C, N, 2)
    points_2d_all = jnp.asarray(frame_annotations)
    P_mats_all = jnp.asarray(proj_matrices)

    triangulated = projective.triangulate_points_from_projections(
        points2d=points_2d_all,
        P_mats=P_mats_all
    )

    return np.asarray(triangulated)


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

    # Build ordered lists based on the sorted toml_names order
    ordered_paths = ["" for _ in range(n)]
    ordered_filenames = ["" for _ in range(n)]
    ordered_toml_names = toml_names  # Use the alphabetically sorted TOML names as the canonical order

    # Create a map from the toml_index (from the assignment) back to the video_index
    toml_to_video_map = {ti: vi for ti, vi in zip(toml_indices, video_indices)}

    for i in range(n):  # i is the index in the sorted toml_names list
        matched_video_idx = toml_to_video_map[i]
        ordered_paths[i] = str(video_paths[matched_video_idx])
        ordered_filenames[i] = video_filenames[matched_video_idx]

    # Build calibration list in the same sorted order
    calibration = []
    for toml_name in ordered_toml_names:
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

    # Return the camera names from the TOML as a new list
    return ordered_paths, ordered_filenames, ordered_toml_names, calibration


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
