"""
Main entry point for CATAR.
"""
import queue
import sys
import cv2
import multiprocessing
import numpy as np
from pathlib import Path
import dearpygui.dearpygui as dpg

import config
from state import AppState, Queues, CalibrationState
from gui import create_ui, update_ui, resize_video_widgets
from gui.rendering import Viewer3D
from utils import load_and_match_videos, compute_3d_scores
from video import create_video_backend, DiskCacheBuilder, VideoReaderWorker
from workers import GAWorker, BAWorker, TrackingWorker, RenderingWorker

from mokap.reconstruction.config import PipelineConfig
from mokap.reconstruction.anatomy import StatsBootstrapper
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.tracking import SkeletonAssembler, MultiObjectTracker


def handle_rendered_frames(new_frames: dict, app_state: 'AppState', queues: 'Queues'):
    for i, frame_bgr in enumerate(new_frames['video_frames_bgr']):
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value(f"video_texture_{i}", rgba.ravel())


def handle_cache_progress(progress: dict, app_state: 'AppState', queues: 'Queues'):
    """
    Handle cache building progress and backend hot-swapping.
    """
    msg_type = progress.get("type")

    # Handle overall progress
    if msg_type == "overall":
        if dpg.does_item_exist("cache_progress_bar_total"):
            dpg.configure_item("cache_progress_bar_total", default_value=progress['progress'])

        if dpg.does_item_exist("cache_progress_status"):
            dpg.set_value("cache_progress_status", progress['status_text'])

    # Handle per-video progress
    elif msg_type == "video":
        video_idx = progress['video_idx']
        total_videos = progress['total_videos']
        bar_tag = f"video_progress_bar_{video_idx}"
        text_tag = f"video_progress_text_{video_idx}"

        if not dpg.does_item_exist(bar_tag) and dpg.does_item_exist("video_progress_container"):

            with dpg.group(parent="video_progress_container", horizontal=False):
                dpg.add_text(f"Video {video_idx + 1}/{total_videos}:", tag=text_tag)
                dpg.add_progress_bar(tag=bar_tag, width=-1, default_value=0.0)
                dpg.add_spacer(height=2)

        if dpg.does_item_exist(bar_tag):
            dpg.configure_item(bar_tag, default_value=progress['progress_pct'] / 100.0)

        if dpg.does_item_exist(text_tag):
            dpg.set_value(text_tag, f"Video {video_idx + 1}/{total_videos}: {progress['progress_pct']:.0f}%")

    # Handle completion
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

        # Hotswap backend
        cache_dir = progress.get("cache_dir")

        if cache_dir:
            try:
                new_backend = create_video_backend(
                    video_paths=app_state.video_paths,
                    video_metadata=app_state.video_metadata,
                    cache_dir=cache_dir,
                    backend_type='cached',
                    ram_budget_gb=config.RAM_MAX_BUDGET_GB
                )

                # Update VideoReaderWorker
                queues.command.put({"action": "update_backend", "backend": new_backend})
                # Update app_state (TrackingWorker will automatically use this for its next batch operation)
                old_backend = app_state.video_backend
                app_state.video_backend = new_backend

                # Close old backend
                if old_backend is not None:
                    old_backend.close()

                print("Backend hot-swapped to use new cache.")

            except Exception as e:
                print(f"ERROR loading new cache: {e}")

    # Handle error
    elif msg_type == "error":
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


def handle_batch_tracking_results(tracking_result: dict, app_state: 'AppState', queues: 'Queues'):
    if tracking_result['status'] == 'running':
        dpg.configure_item("batch_track_progress", default_value=tracking_result['progress'])
        dpg.set_value(
            "batch_track_status_text",
            f"Tracking... Frame {tracking_result['current_frame']}/{tracking_result['total_frames']}"
        )

    elif tracking_result['status'] == 'complete':
        dpg.hide_item("batch_track_popup")
        app_state.frame_idx = tracking_result['final_frame']


def handle_ga_progress(ga_progress: dict, app_state: 'AppState'):
    """Handle genetic algorithm progress updates."""

    if ga_progress.get("status") == "running":
        dpg.set_value("ga_generation_text", f"Generation: {ga_progress['generation']}")
        dpg.set_value("ga_fitness_text", f"Best Fitness: {ga_progress['best_fitness']:.2f}")
        dpg.set_value("ga_mean_fitness_text", f"Mean Fitness: {ga_progress['mean_fitness']:.2f}")

        new_best_calib = ga_progress.get("new_best_individual")
        if new_best_calib:
            with app_state.lock:
                app_state.calibration.update_calibration(new_best_calib)
                app_state.calibration.best_fitness = ga_progress['best_fitness']


def handle_ba_results(ba_result: dict, app_state: 'AppState', queues: 'Queues'):
    """Handle bundle adjustment results."""

    dpg.hide_item("ba_progress_popup")

    if ba_result['status'] == 'success':
        print("BA completed successfully!")

        refined_calib = ba_result['refined_calibration']
        with app_state.lock:
            app_state.calibration.update_calibration(refined_calib)

        queues.tracking_command.put({
            "action": "update_calibration",
            "calibration": refined_calib
        })

        refined_3d_points = ba_result.get('refined_3d_points')
        if refined_3d_points is not None:
            calib_indices = ba_result['calibration_frame_indices']

            with app_state.data.bulk_lock():
                ba_annotations = app_state.data.annotations[calib_indices]

            for i, frame_idx in enumerate(calib_indices):
                pts_3d = refined_3d_points[i]
                frame_annots = ba_annotations[i]

                scores = compute_3d_scores(pts_3d, frame_annots, app_state.calibration)

                pts_4d = np.full((pts_3d.shape[0], 4), np.nan, dtype=np.float32)
                pts_4d[:, :3] = pts_3d
                pts_4d[:, 3] = scores

                app_state.data.set_frame_points3d(frame_idx, pts_4d)

        app_state.needs_3d_reconstruction = True

    elif ba_result['status'] == 'error':
        print(f"BA ERROR: {ba_result['message']}")


def main_loop(app_state: 'AppState', queues: 'Queues', viewer_3d: 'Viewer3D'):
    """Main GUI update loop."""

    initial_resize_counter = 5  # TODO: get rid of this awful thing

    while dpg.is_dearpygui_running():
        # Initial resize
        if initial_resize_counter > 0:
            resize_video_widgets(None, None, {"app_state": app_state})
            initial_resize_counter -= 1

        # 3D visualisation updates
        viewer_3d.process_updates()

        # Process rendered frames
        try:
            processed = queues.results.get_nowait()
            handle_rendered_frames(processed, app_state, queues)
        except queue.Empty:
            pass

        # Process cache build progress
        try:
            progress = queues.cache_progress.get_nowait()
            handle_cache_progress(progress, app_state, queues)
        except queue.Empty:
            pass

        # Process batch tracking progress
        try:
            progress = queues.tracking_progress.get_nowait()
            handle_batch_tracking_results(progress, app_state, queues)
        except queue.Empty:
            pass

        # Process GA progress
        try:
            ga_progress = queues.ga_progress.get_nowait()
            handle_ga_progress(ga_progress, app_state)
        except queue.Empty:
            pass

        # Process BA results
        try:
            ba_result = queues.ba_results.get_nowait()
            handle_ba_results(ba_result, app_state, queues)
        except queue.Empty:
            pass

        # Update UI
        update_ui(app_state)
        dpg.render_dearpygui_frame()


def main():
    """Main entry point."""

    data_folder = config.DATA_FOLDER if hasattr(config, 'DATA_FOLDER') else Path.cwd() / 'data'
    if not data_folder.is_dir():
        data_folder.mkdir(parents=True)
        print(f"Created '{data_folder}' directory. Please add videos and calibration.")
        sys.exit(0)

    print("Loading videos and calibration...")
    video_paths, _, camera_names, mokap_calibration = load_and_match_videos(
        data_folder,
        config.VIDEO_FORMAT
    )

    # Create state objects
    print("Initializing application state...")
    calib_state = CalibrationState(mokap_calibration, camera_names)
    app_state = AppState(data_folder, camera_names, video_paths, calib_state, config.SKELETON_CONFIG)

    print("Initialising video backend...")

    # Check for disk cache
    builder = DiskCacheBuilder(
        video_paths=app_state.video_paths,
        cache_dir=app_state.video_cache_dir
    )
    cache_exists, cache_metadata = builder.check_cache_exists()

    # Create the video backend
    video_backend = create_video_backend(
        video_paths=app_state.video_paths,
        video_metadata=app_state.video_metadata,
        cache_dir=builder.cache_dir if cache_exists else None,
        backend_type='auto',
        ram_budget_gb=config.RAM_MAX_BUDGET_GB
    )

    print(f"Using backend: {type(video_backend).__name__}")

    # Store backend in app_state for cache management callbacks
    app_state.video_backend = video_backend

    print("Initialising mokap pipeline...")     # TODO: Maybe run a dummy tracking frame to warm up JAX compile
    mokap_config = PipelineConfig()

    bones_list = [(k, v_i) for k, v in config.SKELETON_CONFIG['skeleton'].items() for v_i in v]

    stats_output_file = config.DATA_FOLDER / 'bone_stats.json'
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones_list,
        symmetry_map=config.SKELETON_CONFIG['symmetry_map'],
        bootstrap_data=None,
        config=mokap_config.anatomy
    )

    try:
        bone_stats = bootstrapper.get_initial_stats()
        print(f"Loaded bone stats. Reference bone: {bone_stats['reference_bone']}")
    except ValueError as e:
        print(f"\n[ERROR] Could not get bone statistics: {e}")
        sys.exit(1)

    # Define 3D volume
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}   # TODO: Load this from disk
    scene_centre = np.vstack([b for b in volume_bounds.values()]).mean(axis=1)

    reconstructor = Reconstructor(
        camera_parameters=calib_state.best_calibration,
        volume_bounds=volume_bounds,
        config=mokap_config.reconstruction
    )

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

    # Load saved data
    app_state.load_from_disk(config.DATA_FOLDER)
    app_state.scene_centre = scene_centre

    print("Initialising workers...")

    queues = Queues()
    open3d_viz = Viewer3D()

    workers = [

        VideoReaderWorker(
            app_state=app_state,
            video_backend=video_backend,
            command_queue=queues.command,
            output_queues=[queues.frames_for_tracking, queues.frames_for_rendering]
        ),

        TrackingWorker(
            app_state=app_state,
            video_backend=video_backend,
            reconstructor=reconstructor,
            tracker=tracker,
            frames_in_queue=queues.frames_for_tracking,
            progress_out_queue=queues.tracking_progress,
            command_queue=queues.tracking_command,
            stop_batch_track=queues.stop_batch_track
        ),

        RenderingWorker(
            app_state,
            open3d_viz,
            reconstructor,
            tracker,
            queues.frames_for_rendering,
            queues.results
        ),

        GAWorker(
            queues.ga_command,
            queues.ga_progress
        ),

        BAWorker(
            queues.ba_command,
            queues.ba_results,
            queues.stop_bundle_adjustment
        )
    ]

    # Start workers
    for worker in workers:
        worker.start()

    # Create UI
    create_ui(app_state, queues, open3d_viz)

    # Run main loop
    try:
        main_loop(app_state, queues, open3d_viz)
    finally:
        print("Shutting down...")
        queues.shutdown_all()

        # Close video backend
        video_backend.close()

        for worker in workers:
            worker.join(timeout=2)

        dpg.destroy_context()
        print("Shutdown complete")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()