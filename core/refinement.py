from typing import Dict, Any
import numpy as np
from jax import numpy as jnp

from state.calibration_state import CalibrationState

from mokap.geometry import compose_transform_matrix, decompose_transform_matrix
from mokap.calibration import bundle_adjustment


def prepare_refinement(snapshot: Dict[str, Any], is_independent: bool) -> Dict[str, Any]:
    """
    Prepares data for Bundle Adjustment.
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
            triangulated = temp_calib_state.triangulate(
                annots_in_calib_frames[i],
                weights=None
            )
            initial_structure[i] = triangulated
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
            points_3d = temp_calib_state.triangulate(
                frame_annots,
                weights=None
            )
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
        ba_data = prepare_refinement(snapshot, is_independent_mode)
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
    T_init = compose_transform_matrix(r_init, t_init)
    success, results = bundle_adjustment.run_bundle_adjustment(
        camera_matrices_initial=K_init,
        distortion_coeffs_initial=D_init,
        cam_poses_initial=T_init,
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
    K_opt, D_opt, T_opt = results['K_opt'], results['D_opt'], results['cam_poses_opt']

    r_opt, t_opt = decompose_transform_matrix(T_opt)

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
