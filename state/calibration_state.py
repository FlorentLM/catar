"""
This module centralizes all calibration data and provides methods for projection/reprojection.
"""
import itertools
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from mokap.geometry import (invert_vectors, projection_matrix, compose_transform_matrix, project,
                            project_to_cameras_multi, undistort, triangulate_from_projections,
                            fundamental_matrix)

if TYPE_CHECKING:
    from utils import CameraParameters, CalibrationDict


class CalibrationState:
    """
    Manages camera calibration with cached access and built-in transforms.
    
    Provides:
    - Cached arrays for batch operations (K, dist, rvecs, tvecs, P matrices)
    - Both camera-to-world (c2w) and world-to-camera (w2c) transforms
    - Direct projection/reprojection methods
    - Fundamental matrices for epipolar geometry
    
    All coordinate transforms are cached and reused across the application.
    """
    
    def __init__(self, initial_calibration: 'CalibrationDict', camera_names: List[str]):
        """
        Initialize calibration state.
        
        Args:
            initial_calibration: Dict mapping camera names to parameter dicts
                Each parameter dict must contain:
                - 'camera_matrix': (3, 3) intrinsic matrix
                - 'dist_coeffs': (D,) distortion coefficients
                - 'rvec': (3,) rotation vector (camera-to-world)
                - 'tvec': (3,) translation vector (camera-to-world)
            camera_names: Ordered list of camera names (defines index order)
        """
        if len(camera_names) != len(initial_calibration):
            raise ValueError(
                f"Mismatch: {len(camera_names)} camera names but "
                f"{len(initial_calibration)} calibrations"
            )
        
        for name in camera_names:
            if name not in initial_calibration:
                raise ValueError(f"Missing calibration for camera '{name}'")
        
        self._calibrations: 'CalibrationDict' = initial_calibration

        self._camera_names = tuple(camera_names)
        self.camera_nti = {name: i for i, name in enumerate(self._camera_names)}
        self.camera_itn = {i: name for name, i in self.camera_nti.items()}

        self._cache: Dict[str, np.ndarray] = {}
        self._f_mats_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        
        # Optimization state (for GA/BA workflows)
        self.best_fitness: float = float('inf')
        self.calibration_frames: List[int] = []
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    def _invalidate_cache(self):
        """Clears cached arrays when calibration changes."""
        self._cache.clear()
        self._f_mats_cache = None
    
    def update_calibration(self, new_calibration: 'CalibrationDict'):
        """
        Updates the entire calibration state and invalidates caches.
        
        Args:
            new_calibration: New calibration dictionary
        """
        for name in self._camera_names:
            if name not in new_calibration:
                raise ValueError(f"Missing calibration for camera '{name}'")
        
        self._calibrations = new_calibration
        self.best_fitness = float('inf')  # Will be updated by next GA evaluation
        self._invalidate_cache()
        
        print("Successfully applied new camera calibration.")
    
    def get_camera_params_c2w(self, camera_name: str) -> 'CameraParameters':
        """
        Get camera parameters in camera-to-world format.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Dictionary with keys: camera_matrix, dist_coeffs, rvec, tvec
            where rvec and tvec are in camera-to-world format
        """
        return self._calibrations[camera_name].copy()
    
    def get_camera_params_w2c(self, camera_name: str) -> 'CameraParameters':
        """
        Get camera parameters in world-to-camera format (for projection).
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Dictionary with keys: camera_matrix, dist_coeffs, rvec, tvec
            where rvec and tvec are in world-to-camera format
        """
        cam_idx = self.camera_nti[camera_name]
        
        params = self._calibrations[camera_name].copy()
        params['rvec'] = self.rvecs_w2c[cam_idx]
        params['tvec'] = self.tvecs_w2c[cam_idx]
        
        return params
    
    @property
    def camera_names(self) -> List[str]:
        return list(self._camera_names)
    
    @property
    def n_cameras(self) -> int:
        return len(self._camera_names)
    
    @property
    def best_calibration(self) -> Optional['CalibrationDict']:
        # TODO: Maybe get rid of this
        return self._calibrations

    def _get_or_compute(self, key: str, dtype=np.float32) -> np.ndarray:
        """
        Helper to get an array from cache or compute and store it.
        
        Args:
            key: Parameter key (e.g., 'camera_matrix', 'rvec')
            dtype: Numpy dtype for the output array
            
        Returns:
            Stacked array of shape (num_cameras, ...)
        """
        if key not in self._cache:
            data_list = [self._calibrations[name][key] for name in self._camera_names]
            self._cache[key] = np.array(data_list, dtype=dtype)
        
        return self._cache[key]

    def _compute_fundamental_matrices(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute fundamental matrices for all camera pairs.

        Returns:
            Dictionary mapping (cam_i, cam_j) to F matrix
        """
        if self.n_cameras < 2:
            return {}

        # Create camera pair indices
        cam_indices = list(range(self.n_cameras))
        pairs = [p for p in itertools.product(cam_indices, repeat=2) if p[0] != p[1]]

        idx_i = [p[0] for p in pairs]
        idx_j = [p[1] for p in pairs]

        F_matrices = fundamental_matrix(
            (self.K_mats[idx_i], self.K_mats[idx_j]),
            (self.rvecs_w2c[idx_i], self.rvecs_w2c[idx_j]),
            (self.tvecs_w2c[idx_i], self.tvecs_w2c[idx_j])
        )

        # Build result dictionary
        f_mats = {}
        for idx, (i, j) in enumerate(pairs):
            f_mats[(i, j)] = np.asarray(F_matrices[idx])

        return f_mats

    @property
    def K_mats(self) -> np.ndarray:
        return self._get_or_compute('camera_matrix')
    
    @property
    def dist_coeffs(self) -> np.ndarray:
        return self._get_or_compute('dist_coeffs')
    
    @property
    def rvecs_c2w(self) -> np.ndarray:
        return self._get_or_compute('rvec')
    
    @property
    def tvecs_c2w(self) -> np.ndarray:
        return self._get_or_compute('tvec')

    @property
    def T_c2w(self) -> np.ndarray:
        if 'T_c2w' not in self._cache:
            T_c2w = compose_transform_matrix(self.rvecs_c2w, self.tvecs_c2w)
            self._cache['T_c2w'] = np.asarray(T_c2w)
        return self._cache['T_c2w']

    @property
    def T_w2c(self) -> np.ndarray:
        if 'T_w2c' not in self._cache:
            T_w2c = compose_transform_matrix(self.rvecs_w2c, self.tvecs_w2c)
            self._cache['T_w2c'] = np.asarray(T_w2c)
        return self._cache['T_w2c']

    @property
    def rvecs_w2c(self) -> np.ndarray:
        if 'rvecs_w2c' not in self._cache:
            r_inv, _ = invert_vectors(self.rvecs_c2w, self.tvecs_c2w)
            self._cache['rvecs_w2c'] = np.asarray(r_inv)
        return self._cache['rvecs_w2c']
    
    @property
    def tvecs_w2c(self) -> np.ndarray:
        if 'tvecs_w2c' not in self._cache:
            _, t_inv = invert_vectors(self.rvecs_c2w, self.tvecs_c2w)
            self._cache['tvecs_w2c'] = np.asarray(t_inv)
        return self._cache['tvecs_w2c']
    
    @property
    def P_mats(self) -> np.ndarray:
        if 'P_mats' not in self._cache:
            P = projection_matrix(self.K_mats, self.T_w2c)
            self._cache['P_mats'] = np.asarray(P)
        return self._cache['P_mats']
    
    @property
    def F_mats(self) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
        if self._f_mats_cache is None and self._calibrations:
            self._f_mats_cache = self._compute_fundamental_matrices()
        return self._f_mats_cache

    def reproject_to_one(
        self,
        points3d: np.ndarray,
        camera_name: str
    ) -> np.ndarray:
        """
        Reproject 3D points to 2D in a specific camera.
        
        Args:
            points3d: Array of shape (N, 3) or (N, 4) with 3D coordinates
                      (4th channel is confidence/score and is ignored)
            camera_name: Name of the camera to project into
            
        Returns:
            Array of shape (N, 2) with 2D coordinates
        """
        if points3d.size == 0:
            return np.array([])
        
        # Handle both (N, 3) and (N, 4) inputs
        if points3d.shape[-1] == 4:
            points3d = points3d[..., :3]
        
        cam_idx = self.camera_nti[camera_name]

        reprojected, _ = project(
            points3d=points3d,
            T=self.T_w2c[cam_idx],
            K=self.K_mats[cam_idx],
            D=self.dist_coeffs[cam_idx]
        )
        
        return np.asarray(reprojected).reshape(-1, 2)
    
    def reproject_to_all(
        self,
        points3d: np.ndarray
    ) -> np.ndarray:
        """
        Reproject 3D points to 2D in all cameras.
        
        Args:
            points3d: Array of shape (N, 3) or (N, 4) with 3D coordinates
            
        Returns:
            Array of shape (C, N, 2) with 2D coordinates for all cameras
        """
        if points3d.size == 0:
            return np.array([])
        
        # Handle both (N, 3) and (N, 4) inputs
        if points3d.shape[-1] == 4:
            points3d = points3d[..., :3]
        
        # Project to all cameras at once
        reprojected, _ = project_to_cameras_multi(
            points3d=points3d,
            T_w2c=self.T_w2c,
            K=self.K_mats,
            D=self.dist_coeffs
        )
        
        return np.asarray(reprojected).squeeze(1)  # Shape: (C, N, 2)
    
    def undistort(
        self,
        points2d: np.ndarray,
        camera_name: str
    ) -> np.ndarray:
        """
        Undistort 2D points in a specific camera.
        
        Args:
            points2d: Array of shape (N, 2) with distorted 2D coordinates
            camera_name: Name of the camera
            
        Returns:
            Array of shape (N, 2) with undistorted 2D coordinates
        """
        valid_mask = ~np.isnan(points2d).any(axis=-1)
        if not np.any(valid_mask):
            return np.full_like(points2d, np.nan)
        
        cam_idx = self.camera_nti[camera_name]
        
        valid_points = points2d[valid_mask]
        
        undistorted = undistort(
            points2d=valid_points,
            camera_matrix=self.K_mats[cam_idx],
            dist_coeffs=self.dist_coeffs[cam_idx]
        )
        
        result = np.full_like(points2d, np.nan)
        result[valid_mask] = np.asarray(undistorted).reshape(-1, 2)
        
        return result
    
    def triangulate(
        self,
        points2d: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Triangulate 3D points from 2D observations across cameras.
        
        Args:
            points2d: Array of shape (C, N, 2) with 2D coordinates
            weights: Optional array of shape (C, N) with observation weights/confidences
            
        Returns:
            Array of shape (N, 3) with 3D coordinates
        """
        if weights is None:
            weights = np.ones(points2d.shape[:2], dtype=np.float32)
        
        points3d = triangulate_from_projections(
            points2d=points2d,
            P=self.P_mats,
            weights=weights
        )
        
        return np.asarray(points3d)
    
    def triangulate_subset(
        self,
        points2d: np.ndarray,
        camera_indices: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Triangulate 3D points from 2D observations using only a subset of cameras.

        This is useful when not all cameras have valid observations for a point,
        or when you want to triangulate based on consensus from specific cameras.

        Args:
            points2d: Array of shape (N_subset, M, 2) with 2D coordinates from subset cameras
            camera_indices: Array of camera indices to use (e.g., [0, 2, 3])
            weights: Optional array of shape (N_subset, M) with observation weights/confidences

        Returns:
            Array of shape (M, 3) with 3D coordinates
        """
        if weights is None:
            weights = np.ones(points2d.shape[:2], dtype=np.float32)

        # Get projection matrices for the subset of cameras
        proj_matrices = self.P_mats[camera_indices]

        points3d = triangulate_from_projections(
            points2d=points2d,
            P=proj_matrices,
            weights=weights
        )

        return np.asarray(points3d)

    def undistort_all(self, points2d: np.ndarray) -> np.ndarray:
        """
        Undistort 2D points for all cameras at once (batch operation).

        This is more efficient than calling undistort() for each camera separately
        when you need to undistort points for all cameras.

        Args:
            points2d: Array of shape (C, N, 2) with distorted 2D coordinates for all cameras

        Returns:
            Array of shape (C, N, 2) with undistorted 2D coordinates
        """

        undistorted = undistort(
            points2d,
            self.K_mats,
            self.dist_coeffs
        )

        return np.asarray(undistorted)

    def compute_reprojection_errors(
        self,
        points3d: np.ndarray,
        points2d: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection errors for 3D points.
        
        Args:
            points3d: Array of shape (N, 3) or (N, 4) with 3D coordinates
            points2d: Array of shape (C, N, 2) with observed 2D coordinates
            
        Returns:
            Array of shape (C, N) with reprojection errors in pixels
        """
        if points3d.shape[-1] == 4:
            points3d = points3d[..., :3]
        
        reprojected = self.reproject_to_all(points3d)  # (C, N, 2)
        
        # Euclidean distance
        diffs = reprojected - points2d
        errors = np.sqrt(np.sum(diffs ** 2, axis=-1))  # (C, N)
        
        return errors