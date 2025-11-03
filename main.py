"""
Main entry point for CATAR.
"""
import sys
import cv2
import multiprocessing
import numpy as np
import tomllib
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import Levenshtein
import dearpygui.dearpygui as dpg

import config
from state import AppState, Queues
from core import create_camera_visual, update_3d_reconstruction
from gui import create_ui, update_ui, resize_video_widgets
from workers import VideoReaderWorker, TrackingWorker, RenderingWorker, GAWorker
from viz_3d import SceneVisualizer, SceneObject


def load_and_match_videos(data_folder: Path, video_format: str):
    """
    Load videos and calibration with smart matching
    Returns: video_paths, video_names, calibration
    """
    # TODO: the smart matching should be moved to mokap too

    # Load calibration TOML
    calib_file = data_folder / 'parameters.toml'
    if not calib_file.exists():
        print(f"ERROR: 'parameters.toml' not found in '{data_folder}'")
        sys.exit(1)

    with calib_file.open("rb") as f:
        calib_data = tomllib.load(f)

    toml_names = sorted(calib_data.keys())
    print(f"Found {len(toml_names)} cameras in TOML")

    # Find video files
    video_paths = sorted(data_folder.glob(video_format))
    if not video_paths:
        print(f"ERROR: No videos matching '{video_format}' found in '{data_folder}'")
        sys.exit(1)

    video_filenames = [p.name for p in video_paths]
    print(f"Found {len(video_filenames)} video files")

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


def _build_scene_for_frame(app_state: AppState):
    """Build 3D scene for current frame."""

    scene = []

    with app_state.lock:
        # Add camera visualisations
        if app_state.show_cameras_in_3d and app_state.best_individual:
            for i, cam_params in enumerate(app_state.best_individual):
                scene.extend(
                    create_camera_visual(cam_params, app_state.video_names[i])
                )

        # Add reconstructed points and skeleton
        points_3d = app_state.reconstructed_3d_points[app_state.frame_idx]
        point_names = app_state.point_names
        point_colors = app_state.point_colors
        skeleton = app_state.skeleton

    # Draw points
    valid_points = 0
    for i, point in enumerate(points_3d):
        if not np.isnan(point).any():
            valid_points += 1
            color = tuple(point_colors[i])
            scene.append(SceneObject(
                type='point',
                coords=point,
                color=color,
                label=point_names[i]
            ))

            # Debug first point
            if valid_points == 1:
                print(f"Adding point '{point_names[i]}' at {point} with color {color}")

            # Draw skeleton
            for connected_name in skeleton.get(point_names[i], []):
                try:
                    j = point_names.index(connected_name)
                    if not np.isnan(points_3d[j]).any():
                        scene.append(SceneObject(
                            type='line',
                            coords=np.array([point, points_3d[j]]),
                            color=(128, 128, 128),
                            label=None
                        ))
                except ValueError:
                    pass

    if valid_points > 0:
        print(f"3D Scene: {len(scene)} objects, {valid_points} valid keypoints")

    return scene


def main_loop(app_state: AppState, queues: Queues, scene_viz: SceneVisualizer):
    """Main GUI update loop."""

    initial_resize_counter = 5

    while dpg.is_dearpygui_running():

        # initial resize
        if initial_resize_counter > 0:
            resize_video_widgets(None, None, {"app_state": app_state})
            initial_resize_counter -= 1

        # Check if 3D reconstruction is needed (like after a new annotation while paused)
        with app_state.lock:
            needs_reconstruction = app_state.needs_3d_reconstruction
            best_individual = app_state.best_individual

        if needs_reconstruction and best_individual:
            update_3d_reconstruction(app_state)
            with app_state.lock:
                app_state.needs_3d_reconstruction = False

            # Build and update 3D scene
            scene = _build_scene_for_frame(app_state)
            scene_viz.draw_scene(scene)

        # 3D visualisation updates (must be done from main thread)
        if not scene_viz.process_3d_updates():
            # 3D window was closed, try refresh on next update
            pass

        # Process rendered frames from rendering worker
        try:
            processed = queues.results.get_nowait()

            # Update video textures
            for i, frame_bgr in enumerate(processed['video_frames_bgr']):
                rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", rgba.ravel())

        except Exception:
            pass

        # Process batch tracking progress
        try:
            progress = queues.tracking_progress.get_nowait()
            if progress['status'] == 'running':
                dpg.configure_item("batch_track_progress", default_value=progress['progress'])
                dpg.set_value(
                    "batch_track_status_text",
                    f"Tracking... Frame {progress['current_frame']}/{progress['total_frames']}"
                )
            elif progress['status'] == 'complete':
                dpg.hide_item("batch_track_popup")
                app_state.frame_idx = progress['final_frame']
        except Exception:
            pass

        # Update UI with current state
        update_ui(app_state)

        dpg.render_dearpygui_frame()


def main():

    # Ensure data folder exists
    if not config.DATA_FOLDER.exists():
        config.DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{config.DATA_FOLDER}' directory. Please add videos and calibration.")
        sys.exit(0)

    # Load videos and calibration
    video_paths, video_names, calibration = load_and_match_videos(
        config.DATA_FOLDER,
        config.VIDEO_FORMAT
    )

    # Get video metadata
    metadata = get_video_metadata(video_paths[0])
    metadata['num_videos'] = len(video_paths)

    # Initialise application state
    app_state = AppState(metadata, config.SKELETON_CONFIG)
    app_state.video_names = video_names
    app_state.set_calibration(calibration)
    app_state.load_from_disk(config.DATA_FOLDER)

    # Initialise communication queues
    queues = Queues()

    # Initialise 3D scene visualiser
    scene_viz = SceneVisualizer(frame_size=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

    # Create worker threads/processes
    workers = [
        VideoReaderWorker(
            app_state,
            video_paths,
            queues.command,
            [queues.frames_for_tracking, queues.frames_for_rendering]
        ),
        TrackingWorker(
            app_state,
            queues.frames_for_tracking,
            queues.tracking_progress,
            video_paths,
            queues.tracking_command
        ),
        RenderingWorker(
            app_state,
            scene_viz,
            queues.frames_for_rendering,
            queues.results
        ),
        GAWorker(queues.ga_command, queues.ga_progress)
    ]

    # Start all workers
    for worker in workers:
        worker.start()

    # Create UI
    create_ui(app_state, queues, scene_viz)

    # Run main loop
    try:
        main_loop(app_state, queues, scene_viz)
    finally:
        # Shutdown
        print("Shutting down...")
        queues.shutdown_all()

        for worker in workers:
            worker.join(timeout=2)

        dpg.destroy_context()
        print("Shutdown complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()