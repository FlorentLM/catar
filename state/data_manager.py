"""
CATAR data manager for 2D annotations and 3D reconstructed points (wraps mokap's Polars backend).
"""
import threading
from pathlib import Path
from typing import Optional, List, Union, Sequence
import numpy as np
import polars as pl

from mokap.mokap_io.schemas import SCHEMAS, add_optional_columns, empty_dataframe
from mokap.mokap_io.loaders import load_dataframe
from mokap.mokap_io.savers import save_dataframe

IndexLike = Union[int, np.integer, str, slice, Sequence[Union[int, str]], None]


class DataManager:
    def __init__(self, nb_frames: int, camera_names: List[str], keypoint_names: List[str]):
        self.camera_names = list(camera_names)
        self.keypoint_names = list(keypoint_names)

        self.nb_frames = nb_frames
        self.nb_cameras = len(camera_names)
        self.nb_points = len(keypoint_names)

        self._cam_to_idx = {name: i for i, name in enumerate(camera_names)}
        self._kp_to_idx = {name: i for i, name in enumerate(keypoint_names)}

        self._points2d: pl.DataFrame = empty_dataframe('Points2D')
        self._points3d: pl.DataFrame = empty_dataframe('Tracks3D')

        self._manual_flags = np.zeros((nb_frames, self.nb_cameras, self.nb_points), dtype=bool)

        self._lock = threading.RLock()

        self._numpy_cache_valid = False
        self._cached_annotations = None
        self._cached_points3d = None

    # Context managers and resolvers

    class _ReadLock:
        def __init__(self, lock):
            self._lock = lock
        def __enter__(self):
            self._lock.acquire()
            return self
        def __exit__(self, *args): self._lock.release()

    class _BulkLock:
        def __init__(self, manager: 'DataManager'):
            self._manager = manager
        def __enter__(self):
            self._manager._lock.acquire()
            return self
        def __exit__(self, *args):
            self._manager._invalidate_cache()
            self._manager._lock.release()

    def read_lock(self):
        return self._ReadLock(self._lock)

    def bulk_lock(self):
        return self._BulkLock(self)

    def _resolve(self, val: IndexLike, max_len: int, name_map: Optional[dict] = None) -> np.ndarray:
        """Turns indices into an array of integers for indexing."""

        if val is None:
            return np.arange(max_len)

        if isinstance(val, (np.integer, int)):
            return np.array([val])

        if isinstance(val, str):
            return np.array([name_map[val]])

        if isinstance(val, slice):
            return np.arange(*val.indices(max_len))

        if isinstance(val, (list, tuple, np.ndarray, Sequence)):
            # Flatten potential nested sequences
            indices = []
            for item in val:
                indices.extend(self._resolve(item, max_len, name_map))
            return np.array(indices)

        raise TypeError(f"Unsupported type: {type(val)}")

    def _invalidate_cache(self):

        self._numpy_cache_valid = False
        self._cached_annotations = None
        self._cached_points3d = None

    def _ensure_numpy_cache(self):

        if self._numpy_cache_valid:
            return

        self._cached_annotations = np.full((self.nb_frames, self.nb_cameras, self.nb_points, 3), np.nan, dtype=np.float32)
        self._cached_points3d = np.full((self.nb_frames, self.nb_points, 4), np.nan, dtype=np.float32)

        if len(self._points2d) > 0:
            for row in self._points2d.iter_rows(named=True):
                f, c, p = row['frame'], self._cam_to_idx.get(row['camera']), self._kp_to_idx.get(row['keypoint'])

                if c is not None and p is not None:
                    self._cached_annotations[f, c, p] = [row['x'], row['y'], row['score']]

        if len(self._points3d) > 0:
            for row in self._points3d.iter_rows(named=True):
                f, p = row['frame'], self._kp_to_idx.get(row['keypoint'])

                if p is not None:
                    self._cached_points3d[f, p] = [row['x'], row['y'], row['z'], row['confidence']]

        self._numpy_cache_valid = True

    # Public access

    def get_2d(self, frame: IndexLike = None, camera: IndexLike = None, keypoint: IndexLike = None,
               copy: bool = False) -> np.ndarray:
        """
        2D point data getter.

        Args:
           frame, camera, keypoint: Identifiers (int, str, slice, sequence)
           copy: Whether to copy the data
        """

        with self._lock:
            self._ensure_numpy_cache()
            f_idx = self._resolve(frame, self.nb_frames)
            c_idx = self._resolve(camera, self.nb_cameras, self._cam_to_idx)
            p_idx = self._resolve(keypoint, self.nb_points, self._kp_to_idx)

            res = self._cached_annotations[np.ix_(f_idx, c_idx, p_idx)]
            res = np.squeeze(res)
            return res.copy() if copy else res

    def set_2d(self, frame: IndexLike = None, camera: IndexLike = None, keypoint: IndexLike = None,
               data: np.ndarray = None, is_manual: Union[bool, np.ndarray] = False):
        """
        2D point data setter.

        Args:
            frame, camera, keypoint: Identifiers (int, str, slice, sequence)
            data: np.ndarray of shape (len(f), len(c), len(p), 2 or 3) or (2 or 3,) to broadcast
            is_manual: bool or np.ndarray of shape (len(f), len(c), len(p)) or broadcastable
        """
        f_idx = self._resolve(frame, self.nb_frames)
        c_idx = self._resolve(camera, self.nb_cameras, self._cam_to_idx)
        p_idx = self._resolve(keypoint, self.nb_points, self._kp_to_idx)
        target_shape = (len(f_idx), len(c_idx), len(p_idx))

        if data is None:
            data = np.full(target_shape + (3,), np.nan, dtype=np.float32)
        else:
            data = np.asanyarray(data, dtype=np.float32)

            # Pad score if missing
            if data.shape[-1] == 2:
                padding = np.full(data.shape[:-1] + (1,), np.nan, dtype=np.float32)
                data = np.concatenate([data, padding], axis=-1)

            try:
                data = np.broadcast_to(data, target_shape + (3,))
            except ValueError:
                raise ValueError(f"2D Data shape mismatch. {data.shape} cannot broadcast to {target_shape + (3,)}")

        if isinstance(is_manual, (bool, np.bool_)):
            is_manual_arr = np.full(target_shape, is_manual, dtype=bool)
        else:
            is_manual_arr = np.broadcast_to(np.asanyarray(is_manual, dtype=bool), target_shape)

        with self.bulk_lock():
            c_names = [self.camera_names[i] for i in c_idx]
            p_names = [self.keypoint_names[i] for i in p_idx]

            self._points2d = self._points2d.filter(
                ~((pl.col('frame').is_in(f_idx)) &
                  (pl.col('camera').is_in(c_names)) &
                  (pl.col('keypoint').is_in(p_names)))
            )

            valid_mask = ~np.isnan(data[..., 0])

            if np.any(valid_mask):
                grid_f, grid_c, grid_p = np.meshgrid(f_idx, c_names, p_names, indexing='ij')

                new_df = pl.DataFrame({
                    "frame": grid_f[valid_mask],
                    "camera": grid_c[valid_mask],
                    "keypoint": grid_p[valid_mask],
                    "x": data[..., 0][valid_mask],
                    "y": data[..., 1][valid_mask],
                    "score": data[..., 2][valid_mask],
                    "is_h": is_manual_arr[valid_mask]
                })

                new_df = new_df.with_columns(
                    pl.when(pl.col("is_h"))
                    .then(pl.lit("catar_manual"))
                    .otherwise(pl.lit("catar"))
                    .alias("source")
                ).drop("is_h")

                new_df = add_optional_columns(new_df, 'Points2D')
                cast_ops = [pl.col(c.name).cast(c.polars_dtype) for c in SCHEMAS['Points2D'] if c.name in new_df.columns]
                self._points2d = pl.concat([self._points2d, new_df.with_columns(cast_ops)], how='diagonal')

            # Cache & flag update
            target_slice = np.ix_(f_idx, c_idx, p_idx)
            if self._numpy_cache_valid:
                self._cached_annotations[target_slice] = data

            self._manual_flags[target_slice] = is_manual_arr

    def get_3d(self, frame: IndexLike = None, keypoint: IndexLike = None, copy: bool = False) -> np.ndarray:
        """
        3D point data getter.

        Args:
            frame, keypoint: Identifiers (int, str, slice, sequence)
            copy: Whether to copy the data
        """

        with self._lock:
            self._ensure_numpy_cache()
            f_idx = self._resolve(frame, self.nb_frames)
            p_idx = self._resolve(keypoint, self.nb_points, self._kp_to_idx)
            res = self._cached_points3d[np.ix_(f_idx, p_idx)]
            res = np.squeeze(res)
            return res.copy() if copy else res

    def set_3d(self, frame: IndexLike = None, keypoint: IndexLike = None, data: np.ndarray = None):
        """
        3D point data setter.

        Args:
            frame, keypoint: Identifiers (int, str, slice, sequence)
            data: np.ndarray of shape (len(f), len(c), len(p), 3 or 4) or (3 or 4,) to broadcast
        """

        f_idx = self._resolve(frame, self.nb_frames)
        p_idx = self._resolve(keypoint, self.nb_points, self._kp_to_idx)
        target_shape = (len(f_idx), len(p_idx))

        if data is None:
            data = np.full(target_shape + (4,), np.nan, dtype=np.float32)
        else:
            data = np.asanyarray(data, dtype=np.float32)

            # Pad confidence if missing
            if data.shape[-1] == 3:
                padding = np.full(data.shape[:-1] + (1,), np.nan, dtype=np.float32)
                data = np.concatenate([data, padding], axis=-1)

            try:
                data = np.broadcast_to(data, target_shape + (4,))
            except ValueError:
                raise ValueError(f"3D Data shape mismatch. {data.shape} cannot broadcast to {target_shape + (4,)}")

        with self.bulk_lock():
            p_names = [self.keypoint_names[i] for i in p_idx]

            self._points3d = self._points3d.filter(
                ~((pl.col('frame').is_in(f_idx)) & (pl.col('keypoint').is_in(p_names)))
            )

            valid_mask = ~np.isnan(data[..., 0])
            if np.any(valid_mask):
                grid_f, grid_p = np.meshgrid(f_idx, p_names, indexing='ij')

                new_df = pl.DataFrame({
                    "frame": grid_f[valid_mask],
                    "keypoint": grid_p[valid_mask],
                    "x": data[..., 0][valid_mask],
                    "y": data[..., 1][valid_mask],
                    "z": data[..., 2][valid_mask],
                    "confidence": data[..., 3][valid_mask],
                    "track_id": 0
                })

                new_df = add_optional_columns(new_df, 'Tracks3D')
                cast_ops = [pl.col(c.name).cast(c.polars_dtype) for c in SCHEMAS['Tracks3D'] if
                            c.name in new_df.columns]
                self._points3d = pl.concat([self._points3d, new_df.with_columns(cast_ops)], how='diagonal')

            # Cache sync
            target_slice = np.ix_(f_idx, p_idx)
            if self._numpy_cache_valid: self._cached_points3d[target_slice] = data


    def is_manual(self, frame: IndexLike = None, camera: IndexLike = None,
                     keypoint: IndexLike = None, value: Optional[bool] = None) -> Union[np.ndarray, None]:
        """
        Accessor for "manual annotation" flags.

        Args:
            frame, camera, keypoint: Identifiers (int, str, slice, sequence)
            data: np.ndarray of shape (len(f), len(c), len(p), 3 or 4) or (3 or 4,) to broadcast
        """
        f_idx = self._resolve(frame, self.nb_frames)
        c_idx = self._resolve(camera, self.nb_cameras, self._cam_to_idx)
        p_idx = self._resolve(keypoint, self.nb_points, self._kp_to_idx)

        with self._lock:
            target_slice = np.ix_(f_idx, c_idx, p_idx)
            if value is None:
                return np.squeeze(self._manual_flags[target_slice])

            self._manual_flags[target_slice] = value
            return None

    # Persistence

    def save(self, directory: Path, prefix: str = 'catar'):
        """
        Save data to disk.

        Creates:
        - {prefix}_points2d.parquet (Points2D schema)
        - {prefix}_points3d.parquet (Tracks3D schema)
        - {prefix}_manual_flags.npy (numpy array)    # TODO: Might deprecate this
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if len(self._points2d) > 0:
                save_dataframe(
                    self._points2d,
                    directory / f'{prefix}_points2d.parquet',
                    schema_name='Points2D'
                )

            if len(self._points3d) > 0:
                save_dataframe(
                    self._points3d,
                    directory / f'{prefix}_points3d.parquet',
                    schema_name='Tracks3D'
                )

            np.save(directory / f'{prefix}_manual_flags.npy', self._manual_flags)

    def load(self, directory: Path, prefix: str = 'catar'):
        """
        Load data from disk.

        Args:
            directory: Directory containing saved files
            prefix: File prefix used when saving
        """
        directory = Path(directory)

        with self._lock:
            
            # Load Points2D
            points2d_path = directory / f'{prefix}_points2d.parquet'
            if points2d_path.exists():
                self._points2d = load_dataframe(
                    points2d_path,
                    schema_name='Points2D',
                    validate=True
                )
            else:
                self._points2d = empty_dataframe('Points2D')

            # Load Tracks3D
            points3d_path = directory / f'{prefix}_points3d.parquet'
            if points3d_path.exists():
                self._points3d = load_dataframe(
                    points3d_path,
                    schema_name='Tracks3D',
                    validate=True
                )
            else:
                self._points3d = empty_dataframe('Tracks3D')

            # Load "is manual" flags
            flags_path = directory / f'{prefix}_manual_flags.npy'
            if flags_path.exists():
                loaded_flags = np.load(flags_path)
                if loaded_flags.shape == self._manual_flags.shape:
                    self._manual_flags = loaded_flags

                else:
                    print(f'[WARN] Flag "manual" shape mismatch: {loaded_flags.shape} vs. {self._manual_flags.shape}')
                    # Copy what we can
                    min_f = min(loaded_flags.shape[0], self._manual_flags.shape[0])
                    min_c = min(loaded_flags.shape[1], self._manual_flags.shape[1])
                    min_p = min(loaded_flags.shape[2], self._manual_flags.shape[2])
                    self._manual_flags[:min_f, :min_c, :min_p] = loaded_flags[:min_f, :min_c, :min_p]

            self._invalidate_cache()
