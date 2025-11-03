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
from gui import create_ui, update_ui
from workers import VideoReaderWorker, TrackingWorker, RenderingWorker, GAWorker
from old.viz_3d import SceneVisualizer


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


def main_loop(app_state: AppState, queues: Queues):
    """Main GUI update loop."""

    initial_resize_counter = 5

    while dpg.is_dearpygui_running():

        # initial resize
        if initial_resize_counter > 0:
            from gui import resize_video_widgets
            resize_video_widgets(None, None, {"app_state": app_state})
            initial_resize_counter -= 1

        # Process rendered frames from rendering worker
        try:
            processed = queues.results.get_nowait()

            # Update video textures
            for i, frame_bgr in enumerate(processed['video_frames_bgr']):
                rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", rgba.ravel())

            # Update 3D visualisation texture
            rgba_3d = cv2.cvtColor(processed['3d_frame_bgr'], cv2.COLOR_BGR2RGBA)
            rgba_3d = rgba_3d.astype(np.float32) / 255.0
            dpg.set_value("3d_texture", rgba_3d.ravel())

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

    for worker in workers:
        worker.start()

    create_ui(app_state, queues, scene_viz)

    # Run main loop
    try:
        main_loop(app_state, queues)
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