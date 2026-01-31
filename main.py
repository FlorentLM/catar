"""
Main entry point for CATAR.
"""
import queue
import cv2
import multiprocessing
import numpy as np
import dearpygui.dearpygui as dpg

import config
from state import AppState, Queues
from gui import create_ui, update_ui, resize_video_widgets
from utils import compute_3d_scores
from video import create_video_backend, VideoReaderWorker
from workers import GAWorker, BAWorker, RenderingWorker
from workers.tracking_worker import TrackingWorker

from lucida import CameraRig

from mokap.pose_reconstruction.configs import TrackerConfig, AssemblerConfig
from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonStats
from mokap.pose_reconstruction.soup import Reconstructor
from mokap.pose_reconstruction.assembly import SkeletonAssembler, MultiObjectTracker


def handle_rendered_frames(new_frames: dict, app_state: 'AppState', queues: 'Queues'):
    for cam_name, frame_bgr in new_frames['video_frames_bgr'].items():
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value(f"video_texture_{cam_name}", rgba.ravel())


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
        cam_name = progress['camera_name']
        bar_tag = f"video_progress_bar_{cam_name}"
        text_tag = f"video_progress_text_{cam_name}"

        if not dpg.does_item_exist(bar_tag) and dpg.does_item_exist("video_progress_container"):
            with dpg.group(parent="video_progress_container", horizontal=False):
                dpg.add_text(f"{cam_name}:", tag=text_tag)
                dpg.add_progress_bar(tag=bar_tag, width=-1, default_value=0.0)
                dpg.add_spacer(height=2)

        if dpg.does_item_exist(bar_tag):
            dpg.configure_item(bar_tag, default_value=progress['progress_pct'] / 100.0)

        if dpg.does_item_exist(text_tag):
            dpg.set_value(text_tag, f"{cam_name}: {progress['progress_pct']:.0f}%")

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
                    video_metadata=app_state.video_info,
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

        new_best_calib_dict = ga_progress.get("new_best_individual")
        if new_best_calib_dict:
            with app_state.lock:
                # Re-hydrate rig from dict
                app_state.rig = CameraRig.from_dict(new_best_calib_dict)
                app_state.ga_best_fitness = ga_progress['best_fitness']


def handle_ba_results(ba_result: dict, app_state: 'AppState', queues: 'Queues'):
    """Handle bundle adjustment results."""

    dpg.hide_item("ba_progress_popup")

    if ba_result['status'] == 'success':
        print("BA completed successfully!")

        refined_calib_dict = ba_result['refined_calibration']
        with app_state.lock:
            # Re-hydrate rig from dict
            app_state.rig = CameraRig.from_dict(refined_calib_dict)

        # Notify workers of calibration change
        queues.tracking_command.put({"action": "update_calibration"})

        refined_3d_points = ba_result.get('refined_3d_points')
        if refined_3d_points is not None:
            calib_indices = ba_result['calibration_frame_indices']

            with app_state.data.bulk_lock():
                for i, frame_idx in enumerate(calib_indices):
                    pts_3d = refined_3d_points[i]

                    frame_annots = app_state.data.get_2d(frame=frame_idx, copy=True)

                    scores = compute_3d_scores(pts_3d, frame_annots, app_state.rig)

                    pts_4d = np.full((pts_3d.shape[0], 4), np.nan, dtype=np.float32)
                    pts_4d[:, :3] = pts_3d
                    pts_4d[:, 3] = scores

                    app_state.data.set_3d(frame=frame_idx, data=pts_4d)

        app_state.needs_reconstruction = True

    elif ba_result['status'] == 'error':
        print(f"BA ERROR: {ba_result['message']}")


def main_loop(app_state: 'AppState', queues: 'Queues'):
    """Main GUI update loop."""

    initial_resize_counter = 3

    while dpg.is_dearpygui_running():
        if initial_resize_counter > 0:
            resize_video_widgets(None, None, {"app_state": app_state})
            initial_resize_counter -= 1

        # 3D visualisation updates
        if app_state.viewer_3d is not None:
            app_state.viewer_3d.process_updates()

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

    app_state = AppState()
    app_state.load()    # defaults to data folder

    # print("Initialising mokap pipeline...")
    # reconstructor = Reconstructor(
    #     rig=app_state.rig,
    #     keypoint_names=app_state.skeleton.keypoints,
    #     min_views=2,
    #     epipolar_threshold=10.0,
    #     reprojection_threshold=5.0
    # )
    #
    # assembler = SkeletonAssembler(
    #     skeleton=app_state.skeleton,
    #     stats=app_state.skeleton_stats,
    #     config=AssemblerConfig(),
    # )
    #
    # mot_tracker = MultiObjectTracker(
    #     assembler=assembler,
    #     skeleton=app_state.skeleton,
    #     stats=app_state.skeleton_stats,
    #     config=TrackerConfig()
    # )

    queues = Queues()

    workers = [
        VideoReaderWorker(
            app_state=app_state,
            command_queue=queues.command,
            output_queues=[queues.frames_for_tracking, queues.frames_for_rendering]
        ),

        # TrackingWorker(
        #     app_state=app_state,
        #     reconstructor=reconstructor,
        #     mot_tracker=mot_tracker,
        #     in_queue=queues.frames_for_tracking,
        #     progress_queue=queues.tracking_progress,
        #     command_queue=queues.tracking_command,
        #     stop_event=queues.stop_batch_track,
        # ),

        RenderingWorker(
            app_state=app_state,
            in_queue=queues.frames_for_rendering,
            results_queue=queues.results
        ),

        GAWorker(
            command_queue=queues.ga_command,
            progress_queue=queues.ga_progress
        ),

        BAWorker(
            command_queue=queues.ba_command,
            results_queue=queues.ba_results,
            stop_event=queues.stop_bundle_adjustment
        )
    ]

    for worker in workers:
        worker.start()

    create_ui(app_state, queues)

    # Run main loop
    try:
        main_loop(app_state, queues)

    finally:
        print("Shutting down...")

        queues.shutdown_all()
        app_state.video_backend.close()
        for worker in workers:
            worker.join(timeout=2)
        dpg.destroy_context()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()