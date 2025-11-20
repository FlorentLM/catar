import itertools
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Tuple, Union

import Levenshtein

import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

from mokap.utils.fileio import probe_video
from mokap.utils.geometry import projective
from mokap.utils.geometry import transforms

CameraParameters = Dict[str, Union[float, np.ndarray]]
CalibrationDict = Dict[str, CameraParameters]


# ============================================================================
# 3D reconstruction & confidence (wrappers for mokap)
# ============================================================================
# TODO: These wrappers will be removed


def get_projection_matrix(cam_params: CameraParameters) -> np.ndarray:
    """
    Computes the 3x4 projection matrix.
    """
    K = cam_params['camera_matrix']

    # Mokap expects world-to-camera poses for projection
    rvec_c2w = cam_params['rvec']
    tvec_c2w = cam_params['tvec']
    rvec_w2c, tvec_w2c = transforms.invert_rtvecs(jnp.asarray(rvec_c2w), jnp.asarray(tvec_c2w))

    E_w2c = transforms.extrinsics_matrix(rvec_w2c, tvec_w2c)
    P = transforms.projection_matrix(K, E_w2c)
    return np.asarray(P)


def reproject_points(points_3d: np.ndarray, cam_params: CameraParameters) -> np.ndarray:
    """
    Reprojects 3D points to 2D.
    """

    if points_3d.size == 0:
        return np.array([])

    K = cam_params['camera_matrix']

    # get world-to-camera for projection
    rvec_c2w = cam_params['rvec']
    tvec_c2w = cam_params['tvec']
    rvec_w2c, tvec_w2c = transforms.invert_rtvecs(jnp.asarray(rvec_c2w), jnp.asarray(tvec_c2w))

    reprojected, _ = projective.project_points(
        object_points=jnp.asarray(points_3d),
        rvec=rvec_w2c,
        tvec=tvec_w2c,
        camera_matrix=jnp.asarray(K),
        dist_coeffs=jnp.asarray(cam_params['dist_coeffs'])
    )

    return np.asarray(reprojected).reshape(-1, 2)


def undistort_points(points_2d: np.ndarray, cam_params: CameraParameters) -> np.ndarray:
    """
    Undistorts 2D points.
    """

    valid_mask = ~np.isnan(points_2d).any(axis=-1)
    if not np.any(valid_mask):
        return np.full_like(points_2d, np.nan)

    K = cam_params['camera_matrix']

    valid_points = points_2d[valid_mask]

    undistorted = projective.undistort_points(
        points2d=jnp.asarray(valid_points),
        camera_matrix=jnp.asarray(K),
        dist_coeffs=jnp.asarray(cam_params['dist_coeffs'])
    )

    result = np.full_like(points_2d, np.nan)
    result[valid_mask] = np.asarray(undistorted).reshape(-1, 2)
    return result


def calculate_fundamental_matrices(calibration: CalibrationDict, cam_names: List[str]) -> Dict[
    Tuple[int, int], np.ndarray]:
    """
    Calculates F matrix for each camera pair.

    Args:
        calibration: calibration Dict[cam_name, cam_params].
        cam_names: Ordered list of camera names (must match video index order).  # TODO: Videos should not be accessed by index, this is error-prone

    Returns:
        Dict mapping (from_cam_index, to_cam_index) to the F matrix.
    """

    num_cams = len(cam_names)
    if num_cams < 2:
        return {}

    f_mats = {}

    # Get ordered parameters from the dictionary
    ordered_params = [calibration[name] for name in cam_names]

    Ks = jnp.asarray([c['camera_matrix'] for c in ordered_params])
    rvecs_c2w = jnp.asarray([c['rvec'] for c in ordered_params])
    tvecs_c2w = jnp.asarray([c['tvec'] for c in ordered_params])

    # invert poses from camera-to-world to world-to-camera
    rvecs_w2c, tvecs_w2c = transforms.invert_rtvecs(rvecs_c2w, tvecs_c2w)

    # Create (from_cam, to_cam) pairs using index
    cam_indices = list(range(num_cams))
    pairs = [p for p in itertools.product(cam_indices, repeat=2) if p[0] != p[1]]

    K_pairs = jnp.asarray([(Ks[i], Ks[j]) for i, j in pairs])
    rvecs_w2c_pairs = jnp.asarray([(rvecs_w2c[i], rvecs_w2c[j]) for i, j in pairs])
    tvecs_w2c_pairs = jnp.asarray([(tvecs_w2c[i], tvecs_w2c[j]) for i, j in pairs])

    F_matrices_batched = transforms.batched_fundamental_matrices(K_pairs, rvecs_w2c_pairs, tvecs_w2c_pairs)

    for idx, (i, j) in enumerate(pairs):
        f_mats[(i, j)] = np.asarray(F_matrices_batched[idx])

    return f_mats


# ============================================================================
# Videos IO
# ============================================================================
# TODO: Port this logic to mokap


def load_and_match_videos(data_folder: Path, video_format: str) -> Tuple[
    List[str], List[str], List[str], CalibrationDict]:
    """
    Load videos and calibration with smart names matching.

    Returns:
        Tuple of (video_paths, video_filenames, camera_names, calibration_dict)
        where calibration_dict is in the mokap-style Dict[CamName, CamParams]
        with keys: 'camera_matrix', 'dist_coeffs', 'rvec', 'tvec'.
    """

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

    # Build ordered lists based on the sorted toml_names order (which is the canonical order)
    ordered_paths = ["" for _ in range(n)]
    ordered_filenames = ["" for _ in range(n)]
    ordered_toml_names = toml_names

    # Create a map from the toml_index (from the assignment) back to the video_index
    toml_to_video_map = {ti: vi for ti, vi in zip(toml_indices, video_indices)}

    for i in range(n):  # i is the index in the sorted toml_names list
        matched_video_idx = toml_to_video_map[i]
        ordered_paths[i] = str(video_paths[matched_video_idx])
        ordered_filenames[i] = video_filenames[matched_video_idx]

    # Build calibration dictionary (format {cam_name: cam_params})
    calibration_dict: CalibrationDict = {}
    for toml_name in ordered_toml_names:
        cam = calib_data[toml_name]
        calibration_dict[toml_name] = {
            'camera_matrix': np.array(cam['camera_matrix'], dtype=np.float32),
            'dist_coeffs': np.array(cam['dist_coeffs'], dtype=np.float32),
            'rvec': np.array(cam['rvec'], dtype=np.float32),
            'tvec': np.array(cam['tvec'], dtype=np.float32),
        }

    return ordered_paths, ordered_filenames, ordered_toml_names, calibration_dict


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