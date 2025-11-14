"""
Main entry point for CATAR.
"""
import queue
import sys
import cv2
import multiprocessing
import numpy as np
import dearpygui.dearpygui as dpg

import config
from state import AppState, Queues
from gui import create_ui, update_ui, resize_video_widgets
from utils import load_and_match_videos, probe_video, calculate_fundamental_matrices
from workers import VideoReaderWorker, TrackingWorker, RenderingWorker, GAWorker
from viz_3d import Open3DVisualizer
from video_cache import VideoCacheBuilder, VideoCacheReader

from mokap.reconstruction.config import PipelineConfig
from mokap.reconstruction.anatomy import StatsBootstrapper
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.tracking import SkeletonAssembler, MultiObjectTracker
from mokap.utils import fileio


def main_loop(app_state: AppState, queues: Queues, open3d_viz: Open3DVisualizer):
    """Main GUI update loop."""

    initial_resize_counter = 5

    while dpg.is_dearpygui_running():

        # initial resize
        if initial_resize_counter > 0:
            resize_video_widgets(None, None, {"app_state": app_state})
            initial_resize_counter -= 1

        # 3D visualisation updates (this has to be done from main thread)
        open3d_viz.process_updates()

        # Process rendered frames from rendering worker
        try:
            processed = queues.results.get_nowait()
            for i, frame_bgr in enumerate(processed['video_frames_bgr']):
                rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", rgba.ravel())
        except queue.Empty:
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
        except queue.Empty:
            pass

        # Process cache build progress
        try:
            progress = queues.cache_progress.get_nowait()
            msg_type = progress.get("type")

            if msg_type == "overall":
                if dpg.does_item_exist("cache_progress_bar_total"):
                    dpg.configure_item("cache_progress_bar_total", default_value=progress['progress'])

                if dpg.does_item_exist("cache_progress_status"):
                    dpg.set_value("cache_progress_status", progress['status_text'])

            elif msg_type == "video":
                video_idx = progress['video_idx']
                total_videos = progress['total_videos']
                bar_tag = f"video_progress_bar_{video_idx}"
                text_tag = f"video_progress_text_{video_idx}"

                # Create widgets if they don't exist
                if not dpg.does_item_exist(bar_tag) and dpg.does_item_exist("video_progress_container"):
                    with dpg.group(parent="video_progress_container", horizontal=False):
                        dpg.add_text(f"Video {video_idx + 1}/{total_videos}:", tag=text_tag)
                        dpg.add_progress_bar(tag=bar_tag, width=-1, default_value=0.0)
                        dpg.add_spacer(height=2)

                # Update progress
                if dpg.does_item_exist(bar_tag):
                    dpg.configure_item(bar_tag, default_value=progress['progress_pct'] / 100.0)

                if dpg.does_item_exist(text_tag):
                    dpg.set_value(text_tag, f"Video {video_idx + 1}/{total_videos}: {progress['progress_pct']:.0f}%")

            elif msg_type == "complete":
                if dpg.does_item_exist("cache_progress_status"):
                    dpg.set_value("cache_progress_status", progress['status_text'])

                if dpg.does_item_exist("cache_progress_bar_total"):

                    dpg.configure_item("cache_progress_bar_total", default_value=1.0)
                if dpg.does_item_exist("cache_cancel_button"):
                    dpg.delete_item("cache_cancel_button")

                if dpg.does_item_exist("cache_progress_dialog"):
                    dpg.add_button(
                        label="Close", width=-1, parent="cache_progress_dialog",
                        callback=lambda: dpg.delete_item("cache_progress_dialog")
                    )

            elif msg_type == "error":
                # cache is invalid or deleted: update app state
                app_state.cache_reader = None

                # VideoReaderWorker should stop using the cache reader
                queues.command.put({"action": "reload_cache", "cache_reader": None})
                print("Cache invalidated due to build error/cancellation.")

                if dpg.does_item_exist("cache_progress_status"):
                    dpg.set_value("cache_progress_status", progress['status_text'])

                if dpg.does_item_exist("cache_progress_bar_total"):
                    dpg.configure_item("cache_progress_bar_total", overlay="ERROR")

                if dpg.does_item_exist("cache_cancel_button"):
                    dpg.delete_item("cache_cancel_button")

                if dpg.does_item_exist("cache_progress_dialog"):
                    dpg.add_button(
                        label="Close", width=-1, parent="cache_progress_dialog",
                        callback=lambda: dpg.delete_item("cache_progress_dialog")
                    )

        except queue.Empty:
            pass

        try:
            ga_progress = queues.ga_progress.get_nowait()
            status = ga_progress.get("status")

            if status == "running":
                # new best individual has been found: update main app state
                if ga_progress["best_fitness"] < app_state.best_fitness:
                    with app_state.lock:
                        app_state.best_fitness = ga_progress["best_fitness"]
                        app_state.best_individual = ga_progress["new_best_individual"]
                        # Recalculate fundamental matrices for epipolar lines
                        app_state.fundamental_matrices = calculate_fundamental_matrices(app_state.best_individual)
                        app_state.needs_3d_reconstruction = True

                # Update GUI popup with latest generation stats
                dpg.set_value("ga_generation_text", f"Generation: {ga_progress['generation']}")
                dpg.set_value("ga_fitness_text", f"Best Fitness: {ga_progress['best_fitness']:.4f}")
                dpg.set_value("ga_mean_fitness_text", f"Mean Fitness: {ga_progress['mean_fitness']:.4f}")

        except (queue.Empty, AttributeError):
            pass

        # Update UI with current state
        update_ui(app_state)

        dpg.render_dearpygui_frame()


def main():

    if not config.DATA_FOLDER.exists():
        config.DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{config.DATA_FOLDER}' directory. Please add videos and calibration.")
        sys.exit(0)

    # Load videos and calibration first (with proper matching)
    print("Loading videos and calibration...")
    video_paths, video_names, calibration = load_and_match_videos(
        config.DATA_FOLDER,
        config.VIDEO_FORMAT
    )

    # Check for video cache
    print("Checking for video cache...")
    cache_dir = config.DATA_FOLDER / 'video_cache'

    builder = VideoCacheBuilder(
        video_paths=video_paths,
        cache_dir=str(cache_dir),
        ram_budget_gb=2.0
    )
    cache_exists, cache_metadata = builder.check_cache_exists()

    # Decide on cache without blocking UI init
    cache_reader = None

    if cache_exists:
        # Cache exists, try to load it immediately
        try:
            cache_reader = VideoCacheReader(cache_dir=str(cache_dir))
            use_cache = True
            print(f"Using video cache: {cache_reader}")
        except Exception as e:
            print(f"WARNING: Could not load cache: {e}")
            print("Continuing without cache.")
            cache_reader = None
    else:
        print("No video cache found. Will use direct (slower) video files access.")
        print("You can build cache later via Tools > Build Video Cache...")

    # Get video metadata
    metadata = probe_video(video_paths[0])
    metadata['num_videos'] = len(video_paths)

    # Initialise mokap config
    print("Initializing mokap pipeline configuration...")
    mokap_config = PipelineConfig()

    # Load skeleton definition
    bones_list = [(k, v_i) for k, v in config.SKELETON_CONFIG['skeleton'].items() for v_i in v]

    # Load (or bootstrap) anatomical stats
    print("Loading/Bootstrapping anatomical stats...")
    stats_output_file = config.DATA_FOLDER / 'bone_stats.json'
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones_list,
        symmetry_map=config.SKELETON_CONFIG['symmetry_map'],
        bootstrap_data=None,  # we are not bootstrapping from data yet...
        config=mokap_config.anatomy
    )

    try:
        bone_stats = bootstrapper.get_initial_stats()
        print(f"Successfully loaded bone stats. Reference bone: {bone_stats['reference_bone']}")
    except ValueError as e:
        print(f"\n[ERROR] Could not get bone statistics: {e}")
        print("Please provide a 'bone_stats.json' or a prior file (bone_lengths.csv) in tthe 'data' folder.")

        # make a dummy bone_lengths.csv if it doesn't exist
        prior_file = config.DATA_FOLDER / 'bone_lengths.csv'
        if not prior_file.exists():
            with open(prior_file, 'w') as f:
                f.write("bone,length\n")
                f.write("thorax-neck,1.0\n")  # dummy bone
            print(f"Created a dummy bone stats file at '{prior_file}'. Please edit it with your measurements.")
        sys.exit(1)

    # Define the 3D volume of interest (same units as the calibration, like mm)
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}
    scene_centre = np.vstack([b for b in volume_bounds.values()]).mean(axis=1)
    print(f"Using 3D volume bounds: {volume_bounds}")

    # Mokap expects the camera parameters in a certain way
    mokap_calibration = {}
    for cam_name, catar_cal in zip(video_names, calibration):
        K = np.array([
            [catar_cal['fx'], 0.0, catar_cal['cx']],
            [0.0, catar_cal['fy'], catar_cal['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        mokap_calibration[cam_name] = {
            'camera_matrix': K,
            'dist_coeffs': np.array(catar_cal['dist'], dtype=np.float32),
            'rvec': np.array(catar_cal['rvec'], dtype=np.float32),
            'tvec': np.array(catar_cal['tvec'], dtype=np.float32)
        }

    # Instantiate the Reconstructor
    print("Initializing 3D Reconstructor...")
    reconstructor = Reconstructor(
        camera_parameters=mokap_calibration,
        volume_bounds=volume_bounds,
        config=mokap_config.reconstruction
    )

    # Instantiate the Skeleton Assembler and Multi-Object Tracker
    print("Initializing Skeleton Assembler and Tracker...")
    assembler = SkeletonAssembler(
        bones_list=bones_list,
        bone_stats=bone_stats,
        assembler_config=mokap_config.assembler,
        tracker_config=mokap_config.tracker
    )
    tracker = MultiObjectTracker(
        assembler=assembler,
        config=mokap_config.tracker
    )

    # Initialise application state
    app_state = AppState(metadata, config.SKELETON_CONFIG)
    app_state.video_names = video_names
    app_state.set_calibration(calibration)
    app_state.load_from_disk(config.DATA_FOLDER)
    app_state.cache_reader = cache_reader

    # Initialise communication queues
    queues = Queues()

    # Initialise 3D scene visualiser
    open3d_viz = Open3DVisualizer()

    # Create worker threads/processes
    workers = [
        VideoReaderWorker(
            app_state,
            video_paths,
            queues.command,
            [queues.frames_for_tracking, queues.frames_for_rendering],
            cache_reader=cache_reader
        ),
        TrackingWorker(
            app_state,
            reconstructor,
            tracker,
            queues.frames_for_tracking,
            queues.tracking_progress,
            video_paths,
            queues.tracking_command
        ),
        RenderingWorker(
            app_state,
            open3d_viz,
            scene_centre,
            reconstructor,
            tracker,
            queues.frames_for_rendering,
            queues.results
        ),
        GAWorker(queues.ga_command, queues.ga_progress)
    ]

    # Start all workers
    for worker in workers:
        worker.start()

    # Create all UI
    create_ui(app_state, queues, open3d_viz)

    # all done, run main loop:
    try:
        main_loop(app_state, queues, open3d_viz)

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