import sys
import cv2
from pathlib import Path
import queue
import multiprocessing
import dearpygui.dearpygui as dpg
import numpy as np
from state import AppState
from gui import create_dpg_ui, resize_video_widgets, update_annotation_overlays
from workers import VideoReaderWorker, TrackingWorker, GAWorker, RenderingWorker
from viz_3d import SceneVisualizer


DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

DATA_FOLDER = Path.cwd() / 'data'
VIDEO_FORMAT = '*.mp4'

SKELETON_CONFIG = {
    "point_names": [
        "thorax", "neck", "eye_L", "eye_R", "a_L0", "a_L1", "a_L2", "a_R0", "a_R1", "a_R2",
        "leg_f_L0", "leg_f_L1", "leg_f_L2", "leg_f_R0", "leg_f_R1", "leg_f_R2",
        "leg_m_L0", "leg_m_L1", "leg_m_L2", "leg_m_R0", "leg_m_R1", "leg_m_R2",
        "m_L0", "m_L1", "m_R0", "m_R1", "s_small", "s_large"
    ],
    "skeleton": {
        "thorax": ["neck", "leg_f_L0", "leg_f_R0", "leg_m_L0", "leg_m_R0"],
        "neck": ["thorax", "a_R0", "a_L0", "eye_L", "eye_R", "m_L0", "m_R0"],
        "eye_L": ["neck"], "eye_R": ["neck"],
        "a_L0": ["neck", "a_L1"], "a_L1": ["a_L2", "a_L0"], "a_L2": ["a_L1"],
        "a_R0": ["neck", "a_R1"], "a_R1": ["a_R2", "a_R0"], "a_R2": ["a_R1"],
        "leg_f_L0": ["thorax", "leg_f_L1"], "leg_f_L1": ["leg_f_L2", "leg_f_L0"], "leg_f_L2": ["leg_f_L1"],
        "leg_f_R0": ["thorax", "leg_f_R1"], "leg_f_R1": ["leg_f_R2", "leg_f_R0"], "leg_f_R2": ["leg_f_R1"],
        "leg_m_L0": ["thorax", "leg_m_L1"], "leg_m_L1": ["leg_m_L2", "leg_m_L0"], "leg_m_L2": ["leg_m_L1"],
        "leg_m_R0": ["thorax", "leg_m_R1"], "leg_m_R1": ["leg_m_R2", "leg_m_R0"], "leg_m_R2": ["leg_m_R1"],
        "m_L0": ["neck", "m_L1"], "m_L1": ["m_L0"],
        "m_R0": ["neck", "m_R1"], "m_R1": ["m_R0"],
        "s_small": ["s_large"], "s_large": []
    },
    "point_colors": np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
        [192, 192, 192], [255, 128, 0], [128, 0, 255], [255, 128, 128], [128, 128, 0],
        [0, 128, 128], [128, 0, 128], [192, 128, 128], [128, 192, 128], [128, 128, 192],
        [192, 192, 128], [192, 128, 192], [128, 192, 192], [255, 255, 255], [0, 0, 0],
        [128, 128, 128], [255, 128, 64], [128, 64, 255], [210, 105, 30], [128, 255, 64],
        [128, 64, 0], [64, 128, 255]
    ], dtype=np.uint8)
}


def load_videos(data_folder: Path, video_format: str):
    video_paths = sorted(data_folder.glob(video_format))
    if not video_paths:
        print(f"Error: No videos found in '{data_folder}/' with format '{video_format}'")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_paths[0]))
    if not cap.isOpened():
        print("Error: Could not open the first video file.")
        sys.exit(1)

    video_metadata = {
        'num_videos': len(video_paths),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()  # release it immediately

    video_names = [path.name for path in video_paths]
    return [str(p) for p in video_paths], video_metadata, video_names


def main():

    video_paths, video_metadata, video_names = load_videos(DATA_FOLDER, VIDEO_FORMAT)

    app_state = AppState(video_metadata, SKELETON_CONFIG)
    app_state.video_names = video_names
    app_state.load_data_from_files(DATA_FOLDER)  # Load any previously saved state

    # Communication queues
    # GUI -> VideoReaderWorker (for commands like forcing a frame jump)
    command_queue = queue.Queue()

    # VideoReaderWorker -> TrackingWorker (fan-out queue 1)
    frames_for_tracking_queue = queue.Queue(maxsize=2)

    # VideoReaderWorker -> RenderingWorker (fan-out queue 2)
    frames_for_rendering_queue = queue.Queue(maxsize=2)

    # RenderingWorker -> GUI (for final display frames)
    results_queue = queue.Queue(maxsize=2)

    # Queues for the GA process
    ga_command_queue = multiprocessing.Queue()
    ga_progress_queue = multiprocessing.Queue()

    scene_visualizer = SceneVisualizer(frame_size=(DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Start workers
    # Video Reader (Producer)
    video_reader = VideoReaderWorker(
        app_state,
        video_paths,
        command_queue,
        output_queues=[frames_for_tracking_queue, frames_for_rendering_queue]
    )
    video_reader.start()

    # Tracking Worker (consumer 1)
    tracking_worker = TrackingWorker(app_state, frames_for_tracking_queue)
    tracking_worker.start()

    # Rendering Worker (Cconsumer 2)
    rendering_worker = RenderingWorker(
        app_state,
        scene_visualizer,
        frames_for_rendering_queue,
        results_queue
    )
    rendering_worker.start()

    # Genetic Algorithm Worker
    ga_worker = GAWorker(ga_command_queue, ga_progress_queue)
    ga_worker.start()

    create_dpg_ui(app_state, command_queue, ga_command_queue, scene_visualizer)

    initial_resize_counter = 5

    # Main GUI loop
    while dpg.is_dearpygui_running():

        if initial_resize_counter > 0:
            resize_video_widgets(None, None, user_data={"app_state": app_state})
            initial_resize_counter -= 1

        try:
            # Get the final rendered frames from the RenderingWorker
            processed_data = results_queue.get_nowait()

            # Update video textures (these no longer have 2D annotations drawn on them)
            for i, frame_bgr in enumerate(processed_data['video_frames_bgr']):
                rgba_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", rgba_frame.ravel())

            # Update 3D texture
            rgba_3d = cv2.cvtColor(processed_data['3d_frame_bgr'], cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            dpg.set_value("3d_texture", rgba_3d.ravel())

        except queue.Empty:
            pass  # No new frame, continue

        update_annotation_overlays(app_state)

        # Update UI widgets from the app state
        with app_state.lock:
            dpg.set_value("frame_slider", app_state.frame_idx)
            dpg.set_value("frame_text", f"Frame: {app_state.frame_idx}/{video_metadata['num_frames']}")
            dpg.set_value("status_text", f"Status: {'Paused' if app_state.paused else 'Playing'}")
            dpg.configure_item("play_pause_button", label="Play" if app_state.paused else "Pause")
            dpg.set_value("tracking_text",f"Tracking: {'Enabled' if app_state.keypoint_tracking_enabled else 'Disabled'}")
            dpg.set_value("annotating_point_text", f"Annotating: {app_state.POINT_NAMES[app_state.selected_point_idx]}")
            dpg.set_value("fitness_text", f"Best Fitness: {app_state.best_fitness:.2f}")

        dpg.render_dearpygui_frame()

    # Shutdown
    print("Shutting down application...")
    command_queue.put({"action": "shutdown"})  # to VideoReader
    frames_for_tracking_queue.put({"action": "shutdown"})  # to TrackingWorker
    frames_for_rendering_queue.put({"action": "shutdown"})  # to RenderingWorker
    ga_command_queue.put({"action": "shutdown"})  # to GAWorker

    video_reader.join(timeout=2)
    tracking_worker.join(timeout=2)
    rendering_worker.join(timeout=2)
    ga_worker.join(timeout=2)

    dpg.destroy_context()
    print("Shutdown complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()    # this guard is importantt for multiprocessing
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos and data there.")
    main()