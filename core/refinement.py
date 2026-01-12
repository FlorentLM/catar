from typing import Dict, Any
import numpy as np

from lucida import CameraRig
from lucida.calibration import bundle_adjustment


def prepare_refinement(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares data for Bundle Adjustment.
    Returns dictionaries of numpy arrays ready for JAX.
    """

    all_annots = snapshot["annotations"]
    initial_calib_dict = snapshot["best_individual"]
    calib_frames = snapshot["calibration_frames"]

    # Re-hydrate rig
    rig = CameraRig.from_dict(initial_calib_dict)

    # Prepare 2D observations
    # (snapshot annotations is (F_total, C, N, 3))
    annots_subset = all_annots[calib_frames]  # (P, C, N, 3)

    # BA expects (C, P, N, 2)
    annots_subset = np.transpose(annots_subset, (1, 0, 2, 3))

    points_2d = annots_subset[..., :2]
    visibility = ~np.isnan(points_2d[..., 0])

    # Fill nans with 0 (masked by visibility anyway)
    points_2d = np.nan_to_num(points_2d, nan=0.0)

    # Initial triangulation to find outliers and nuke them
    C, P, N, _ = points_2d.shape


    # Scaffolding mode so treat every frame's points as unique in time
    points_2d_flat = points_2d.reshape(C, P * N, 2)

    print("[BA] Performing initialisation check...")
    metrics = rig.compute_metrics(
        points2d=points_2d_flat,
        per_point=True,
        commit=False  # don't overwrite yet
    )

    # Initial 3D points
    points_3d_flat = metrics['points3d']
    mre_per_point = metrics['mre_per_point']

    # Points with high initial error are mislabeled or bad LK tracks
    outlier_threshold = 20.0  # pixels
    valid_mask = mre_per_point < outlier_threshold

    # Also ignore points that failed triangulation
    valid_mask &= np.all(np.isfinite(points_3d_flat), axis=-1)

    num_outliers = (P * N) - np.sum(valid_mask)
    if num_outliers > 0:
        print(f"[BA] Init: Ignoring {num_outliers} outliers (Error > {outlier_threshold:.2f}px)")
        mask_reshaped = valid_mask.reshape(1, P, N)
        visibility &= mask_reshaped

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
            expected_dim = 4 + len(cam.intrinsics.D.flatten()) # 4 K params + whatever D is

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
        # BA expects extrinsics covariance only for optimisable camrras (C-1)
        # TODO: DO NOT drop the first camera but the ACTUAL anchor!!!
        cov_extr_stack = np.stack(cov_extr_list)[1:]

    return {
        "points2d_observed": points_2d,
        "visibility_mask": visibility,
        "object_points": points_3d_safe,
        "covariance_intrinsics": cov_intr_stack,
        "covariance_extrinsics": cov_extr_stack,
        "shape_info": (C, P, N)
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
        fix_cameras = True  # trust cameras, move points
        fix_points = False
    else:  # "full_ba"
        fix_cameras = False
        fix_points = False

    try:
        ba_data = prepare_refinement(snapshot)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Data preparation failed: {str(e)}"}

    print(f"[BA] Starting optimisation engine (Mode: {mode})...")

    rig = CameraRig.from_dict(initial_calib_dict)

    K_init = np.array([c.intrinsics.K for c in rig])
    D_init = np.array([c.intrinsics.D for c in rig])
    T_c2w_init = np.array([c.extrinsics.T_c2w for c in rig])
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

        origin_cam_idx=0,
        distortion_model=dist_model,

        covariance_intrinsics=ba_data["covariance_intrinsics"],
        covariance_extrinsics=ba_data["covariance_extrinsics"],

        fix_aspect_ratio=False,
        max_nfev=200,
        f_scale=2.0
    )

    if not success:
        return {"status": "error", "message": "Bundle Adjustment failed to converge."}

    print("[BA] Optimisation successful!")

    K_opt = results['K_opt']
    D_opt = results['D_opt']
    T_opt = results['camera_poses_opt']  # c2w

    for i, cam in enumerate(rig):
        if not fix_cameras:
            cam.intrinsics.K = K_opt[i]
            cam.intrinsics.D = D_opt[i]
            cam.extrinsics.T_c2w = T_opt[i]

            # Update the rig's covariance with the new posterior covariance
            if 'covariance_intrinsics' in results:
                cam.intrinsics.covariance = results['covariance_intrinsics'][i].tolist()

            if 'covariance_extrinsics' in results:
                # Results return (C-1, 6, 6) for optimisable cams
                # TODO: Anchor is not necessarily cam 0!!!!
                if i > 0:
                    cam.extrinsics.covariance = results['covariance_extrinsics'][i - 1].tolist()

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