"""
Main entry point for CATAR.
"""
import queue
import sys
import cv2
import multiprocessing
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Dict, Any

import config
from state import AppState, Queues, VideoState, CalibrationState
from gui import create_ui, update_ui, resize_video_widgets
from utils import load_and_match_videos
from workers import VideoReaderWorker, TrackingWorker, RenderingWorker, GAWorker, BAWorker
from viz_3d import Open3DVisualizer
from cache_utils import DiskCacheBuilder, DiskCacheReader

from mokap.reconstruction.config import PipelineConfig
from mokap.reconstruction.anatomy import StatsBootstrapper
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.tracking import SkeletonAssembler, MultiObjectTracker


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

        # Check for progress from the GA worker
        try:
            ga_progress = queues.ga_progress.get_nowait()

            if ga_progress.get("status") == "running":
                new_best_fitness = ga_progress["best_fitness"]
                current_best_fitness = app_state.calibration.best_fitness

                # Check if a better individual was found
                if new_best_fitness < current_best_fitness:
                    new_individual = ga_progress["new_best_individual"]

                    with app_state.lock:
                        app_state.calibration.update_calibration(new_individual)
                        app_state.calibration.best_fitness = new_best_fitness

                    # Notify the TrackingWorker
                    queues.tracking_command.put({
                        "action": "update_calibration",
                        "calibration": new_individual
                    })

                # Update the GUI popup text
                dpg.set_value("ga_generation_text", f"Generation: {ga_progress['generation']}")
                dpg.set_value("ga_fitness_text", f"Best Fitness: {app_state.calibration.best_fitness:.2f}")
                dpg.set_value("ga_mean_fitness_text", f"Mean Fitness: {ga_progress['mean_fitness']:.2f}")

        except queue.Empty:
            pass

        # Process bundle adjustment results
        try:
            ba_result = queues.ba_results.get_nowait()
            dpg.hide_item("ba_progress_popup")

            if ba_result['status'] == 'success':
                print("Applying refined calibration from BA.")
                refined_calib: Dict[str, Dict[str, Any]] = ba_result['refined_calibration']

                with app_state.lock:
                    # Always update the main calibration
                    app_state.calibration.update_calibration(refined_calib)

                    # Notify the TrackingWorker
                    queues.tracking_command.put({
                        "action": "update_calibration",
                        "calibration": refined_calib
                    })

                    # Conditionally update the 3D points if they were returned
                    refined_3d_points = ba_result.get('refined_3d_points')
                    if refined_3d_points is not None:
                        from core import compute_3d_scores

                        calib_indices = ba_result['calibration_frame_indices']

                        # Get annotations for these frames to score against
                        with app_state.lock:
                            ba_annotations = app_state.annotations[calib_indices]

                        for i, frame_idx in enumerate(calib_indices):
                            pts_3d = refined_3d_points[i]
                            frame_annots = ba_annotations[i]

                            # Calculate valid scores for the refined points
                            scores = compute_3d_scores(pts_3d, frame_annots, app_state.calibration)

                            pts_4d = np.full((pts_3d.shape[0], 4), np.nan, dtype=np.float32)
                            pts_4d[:, :3] = pts_3d
                            pts_4d[:, 3] = scores

                            app_state.reconstructed_3d_points[frame_idx] = pts_4d
                    else:
                        print("BA was run in scaffolding mode; 3D points were not updated.")

                    app_state.needs_3d_reconstruction = True

            elif ba_result['status'] == 'error':
                print(f"BA ERROR: {ba_result['message']}")

            elif ba_result['status'] == 'cancelled':
                print(f"BA cancelled: {ba_result['message']}")

        except queue.Empty:
            pass

        # Update UI with current state
        update_ui(app_state)

        dpg.render_dearpygui_frame()


def main():

    if not config.DATA_FOLDER.exists():
        config.DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{config.DATA_FOLDER}' directory. Please add videos and calibration.")
        sys.exit(0)

    # Load videos and calibration first
    print("Loading videos and calibration...")
    video_paths, _, camera_names, mokap_calibration = load_and_match_videos(
        config.DATA_FOLDER,
        config.VIDEO_FORMAT
    )

    # Create dedicated state objects
    print("Initializing main application state...")
    video_state = VideoState(camera_names, video_paths)
    calib_state = CalibrationState(mokap_calibration, camera_names)
    app_state = AppState(video_state, calib_state, config.SKELETON_CONFIG)

    # Check for video cache
    print("Checking for video cache...")
    cache_dir = config.DATA_FOLDER / 'video_cache'

    builder = DiskCacheBuilder(
        video_paths=app_state.videos.filepaths,
        cache_dir=str(cache_dir),
        ram_budget_gb=2.0
    )
    cache_exists, cache_metadata = builder.check_cache_exists()

    # Decide on cache without blocking UI init
    cache_reader = None

    if cache_exists:
        # Cache exists, try to load it immediately
        try:
            cache_reader = DiskCacheReader(cache_dir=str(cache_dir))
            use_cache = True
            print(f"Using video cache: {cache_reader}")

        except Exception as e:
            print(f"WARNING: Could not load cache: {e}")
            print("Continuing without cache.")
            cache_reader = None
    else:
        print("No video cache found. Will use direct (slower) video files access.")
        print("You can build cache later via Tools > Build Video Cache...")

    # Set the cache reader in the app state (used by VideoReaderWorker)
    app_state.cache_reader = cache_reader

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

    # Define 3D volume of interest (same units as the calibration, like mm)
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}   # TODO: should be loaded from disk
    scene_centre = np.vstack([b for b in volume_bounds.values()]).mean(axis=1)
    print(f"Using 3D volume bounds: {volume_bounds}")

    # Instantiate the Reconstructor
    print("Initializing 3D Reconstructor...")
    reconstructor = Reconstructor(
        camera_parameters=calib_state.best_calibration,  # TODO: Urgent - this needs to have access to the updated calibration data
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

    # Load any saved annotation/calibration data from disk
    app_state.load_from_disk(config.DATA_FOLDER)
    app_state.scene_centre = scene_centre

    # Initialise communication queues
    queues = Queues()

    # Initialise 3D scene visualiser
    open3d_viz = Open3DVisualizer()

    # Create worker threads/processes
    workers = [
        VideoReaderWorker(
            app_state,
            queues.command,
            [queues.frames_for_tracking, queues.frames_for_rendering],
            diskcache_reader=app_state.cache_reader
        ),
        TrackingWorker(
            app_state,
            reconstructor,
            tracker,
            queues.frames_for_tracking,
            queues.tracking_progress,
            queues.tracking_command,
            queues.stop_batch_track
        ),
        RenderingWorker(
            app_state,
            open3d_viz,
            reconstructor,
            tracker,
            queues.frames_for_rendering,
            queues.results
        ),
        GAWorker(queues.ga_command, queues.ga_progress),
        BAWorker(queues.ba_command, queues.ba_results, queues.stop_bundle_adjustment)
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