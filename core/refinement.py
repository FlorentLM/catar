from typing import Dict, Any, Tuple
import numpy as np

from lucida import CameraRig
from lucida.calibration import bundle_adjustment


def prepare_refinement(rig: CameraRig, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Prepares data for Bundle Adjustment.
    Returns dictionaries of numpy arrays ready for JAX.
    """

    all_annots = snapshot["annotations"]
    calib_frames = snapshot["calibration_frames"]

    # Determine anchor camera index
    if rig.anchor_camera is not None:
        anchor_idx = rig.get_index(rig.anchor_camera.name)
    else:
        anchor_idx = 0

    print(f"[BA] Using camera '{rig.names[anchor_idx]}' (index {anchor_idx}) as anchor")

    # Prepare 2D observations
    # (snapshot annotations is (F_total, C, N, 3))
    annots_subset = all_annots[calib_frames]  # (P, C, N, 3)

    # BA expects (C, P, N, 2)
    annots_subset = np.transpose(annots_subset, (1, 0, 2, 3))

    points_2d = annots_subset[..., :2]
    visibility = ~np.isnan(points_2d[..., 0])

    C, P, N, _ = points_2d.shape

    print("[BA] Performing initialisation check...")
    metrics = rig.compute_metrics(
        points2d=points_2d,
        weights=visibility.astype(np.float32),
        per_point=True,
        commit=False
    )

    # Initial 3D points
    points_3d_flat = metrics['points3d']
    mre_per_point = metrics['mre_per_point']

    # Points with high initial error are mislabeled or bad LK tracks
    outlier_threshold = 20.0  # pixels

    # Fill nans with 0 (masked by visibility anyway)
    points_2d = np.nan_to_num(points_2d, nan=0.0)

    # Find points that were actually triangulated (seen in >= 2 cameras)
    successfully_triangulated = np.all(np.isfinite(points_3d_flat), axis=-1)

    # Filter for high error ONLY on the points that exist
    good_error = mre_per_point < outlier_threshold

    # Combine masks
    valid_mask = successfully_triangulated & good_error

    # Reporting
    num_high_error = np.sum(successfully_triangulated & ~good_error)

    print(f"[BA] Init: {np.sum(valid_mask)} valid points ready for BA.")
    if num_high_error > 0:
        print(f"[BA] Rejected {num_high_error} outliers (Error > {outlier_threshold:.2f}px)")

    # Set invalid points to the centre of the valid cloud just to be safe for JAX
    if np.any(valid_mask):
        center = np.mean(points_3d_flat[valid_mask], axis=0)
    else:
        center = np.zeros(3)

    points_3d_safe = np.nan_to_num(points_3d_flat, nan=center)

    # Covariance priors to prevent the calibration from drifting if the animal data is sparse (which is likely)
    cov_intr_list = []
    cov_extr_list = []

    for cam in rig:
        # Intrinsics covariance
        cov = None
        if hasattr(cam.intrinsics, 'covariance') and cam.intrinsics.covariance is not None:
            cov = np.array(cam.intrinsics.covariance)

            # Safety around dimension mismatch
            expected_dim = 4 + len(cam.intrinsics.D.flatten())  # 4 K params + whatever D is

            if cov.shape[0] != expected_dim:
                # Pad covariance matrix with high variance (low confidence) for missing params
                new_cov = np.eye(expected_dim) * 1000.0
                dim = min(cov.shape[0], expected_dim)
                new_cov[:dim, :dim] = cov[:dim, :dim]
                cov = new_cov

        cov_intr_list.append(cov)

        # Extrinsics covariance
        if hasattr(cam.extrinsics, 'covariance') and cam.extrinsics.covariance is not None:
            cov_extr_list.append(np.array(cam.extrinsics.covariance))
        else:
            cov_extr_list.append(None)

    cov_intr_stack = None
    if all(c is not None for c in cov_intr_list):
        cov_intr_stack = np.stack(cov_intr_list)

    cov_extr_stack = None
    if all(c is not None for c in cov_extr_list):
        # BA expects extrinsics covariance only for optimisable cameras (C-1)
        # Exclude the anchor camera
        cov_extr_stack = np.stack([
            cov_extr_list[i] for i in range(len(cov_extr_list)) if i != anchor_idx
        ])

    return {
        "points2d_observed": points_2d,
        "visibility_mask": visibility,
        "object_points": points_3d_safe,
        "covariance_intrinsics": cov_intr_stack,
        "covariance_extrinsics": cov_extr_stack,
        "shape_info": (C, P, N),
        "anchor_idx": anchor_idx
    }


def run_refinement(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs bundle adjustment.
    """
    print("[BA] Preparing data...")

    mode = snapshot.get("mode", "full_ba")
    initial_calib_dict = snapshot.get("best_individual")
    calib_frames = snapshot.get("calibration_frames")

    if not calib_frames or initial_calib_dict is None:
        return {"status": "error", "message": "Missing calibration frames or initial calibration."}

    # Configure optim flags
    if mode == "refine_cameras_only":
        fix_points = True  # trust points, move cameras
        fix_cameras = False
    elif mode == "refine_points_only":
        fix_points = False
        fix_cameras = True  # trust cameras, move points
    else:  # "full_ba"
        fix_points = False
        fix_cameras = False

    # Hydrate rig first so we can determine anchor
    rig = CameraRig.from_dict(initial_calib_dict)

    try:
        ba_data = prepare_refinement(rig, snapshot)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Data preparation failed: {str(e)}"}

    anchor_idx = ba_data["anchor_idx"]
    print(f"[BA] Starting optimisation engine (Mode: {mode})...")

    K_init = rig.K.copy()
    D_init = rig.D.copy()
    T_c2w_init = rig.T_c2w.copy()
    images_sizes = np.array([c.image_size for c in rig])

    dist_model = rig.distortion_model

    success, results = bundle_adjustment.run_bundle_adjustment(
        K=K_init,
        D=D_init,
        cameras_poses=T_c2w_init,
        images_sizes_wh=images_sizes,

        points2d_observed=ba_data["points2d_observed"],
        visibility_mask=ba_data["visibility_mask"],
        object_points=ba_data["object_points"],

        # Note: fix_object_poses=True *and* object_poses=None triggers 'Scaffolding' mode
        # which treats 'object_points' as a cloud of independent world-space points
        # (size P*N) rather than a single rigid object (size N) transformed by P poses.
        object_poses=None,
        fix_object_poses=True,

        fix_intrinsics=fix_cameras,
        fix_extrinsics=fix_cameras,
        fix_object_points=fix_points,

        origin_cam_idx=anchor_idx,
        distortion_model=dist_model,

        covariance_intrinsics=ba_data["covariance_intrinsics"],
        covariance_extrinsics=ba_data["covariance_extrinsics"],

        fix_aspect_ratio=False,
        max_nfev=500,
        f_scale=2.0
    )

    if not success:
        return {"status": "error", "message": "Bundle Adjustment failed to converge."}

    print("[BA] Optimisation successful!")

    K_opt = results['K_opt']
    D_opt = results['D_opt']
    T_opt = results['camera_poses_opt']  # c2w

    # Build index mapping for extrinsics covariance results
    # BA returns covariances for (C-1) cameras, excluding the anchor
    optimized_cam_indices = [i for i in range(len(rig)) if i != anchor_idx]

    for i, cam in enumerate(rig):
        if not fix_cameras:
            cam.intrinsics.K = K_opt[i]
            cam.intrinsics.D = D_opt[i]
            cam.extrinsics.T_c2w = T_opt[i]

            # Update the rig's covariance with the new posterior covariance
            if 'covariance_intrinsics' in results:
                cam.intrinsics.covariance = results['covariance_intrinsics'][i].tolist()

            if 'covariance_extrinsics' in results and i != anchor_idx:
                # Find the position of this camera in the optimized set
                opt_idx = optimized_cam_indices.index(i)
                cam.extrinsics.covariance = results['covariance_extrinsics'][opt_idx].tolist()

    refined_calib_dict = rig.to_dict()

    refined_points_to_return = None
    if not fix_points:
        C, P, N = ba_data['shape_info']
        refined_points_flat = results['object_points_opt']
        refined_points_to_return = refined_points_flat.reshape(P, N, 3)

    return {
        "status": 'success',
        "refined_calibration": refined_calib_dict,
        "refined_3d_points": refined_points_to_return,
        "calibration_frame_indices": calib_frames
    }