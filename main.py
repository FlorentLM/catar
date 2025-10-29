import sys
import cv2
from pathlib import Path
import queue
import multiprocessing
import dearpygui.dearpygui as dpg
import numpy as np
from state import AppState
from gui import create_dpg_ui, resize_video_widgets, update_annotation_overlays, update_histogram
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

    cap.release()
    video_names = [path.name for path in video_paths]
    return [str(p) for p in video_paths], video_metadata, video_names


def main():
    video_paths, video_metadata, video_names = load_videos(DATA_FOLDER, VIDEO_FORMAT)
    app_state = AppState(video_metadata, SKELETON_CONFIG)
    app_state.video_names = video_names
    app_state.load_data_from_files(DATA_FOLDER)

    # Queues
    command_queue = queue.Queue()
    frames_for_tracking_queue = queue.Queue(maxsize=2)
    frames_for_rendering_queue = queue.Queue(maxsize=2)
    results_queue = queue.Queue(maxsize=2)
    ga_command_queue = multiprocessing.Queue()
    ga_progress_queue = multiprocessing.Queue()
    scene_visualizer = SceneVisualizer(frame_size=(DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Workers
    video_reader = VideoReaderWorker(app_state, video_paths, command_queue,
                                     [frames_for_tracking_queue, frames_for_rendering_queue])
    tracking_worker = TrackingWorker(app_state, frames_for_tracking_queue)
    rendering_worker = RenderingWorker(app_state, scene_visualizer, frames_for_rendering_queue, results_queue)
    ga_worker = GAWorker(ga_command_queue, ga_progress_queue)

    video_reader.start()
    tracking_worker.start()
    rendering_worker.start()
    ga_worker.start()

    create_dpg_ui(app_state, command_queue, ga_command_queue, scene_visualizer)
    initial_resize_counter = 5

    # Main GUI loop
    while dpg.is_dearpygui_running():

        if initial_resize_counter > 0:
            resize_video_widgets(None, None, user_data={"app_state": app_state})
            initial_resize_counter -= 1

        try:
            processed_data = results_queue.get_nowait()

            for i, frame_bgr in enumerate(processed_data['video_frames_bgr']):
                rgba_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", rgba_frame.ravel())

            rgba_3d = cv2.cvtColor(processed_data['3d_frame_bgr'], cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            dpg.set_value("3d_texture", rgba_3d.ravel())

        except queue.Empty:
            pass

        # Update UI elements
        update_annotation_overlays(app_state)
        update_histogram(app_state)

        # Update UI widgets from the latest app state
        with app_state.lock:
            dpg.set_value("frame_slider", app_state.frame_idx)
            dpg.set_value("current_frame_line", float(app_state.frame_idx))
            dpg.configure_item("play_pause_button", label="Play" if app_state.paused else "Pause")
            dpg.set_value("point_combo", app_state.POINT_NAMES[app_state.selected_point_idx])
            dpg.set_value("focus_text", f"Focus Mode: {'Enabled' if app_state.focus_selected_point else 'Disabled'}")
            dpg.set_value("num_calib_frames_text", f"Calibration Frames: {len(app_state.calibration_frames)}")
            dpg.set_value("fitness_text", f"Best Fitness: {app_state.best_fitness:.2f}")

        dpg.render_dearpygui_frame()

    # Shutdown
    print("Shutting down application...")

    command_queue.put({"action": "shutdown"})
    frames_for_tracking_queue.put({"action": "shutdown"})
    frames_for_rendering_queue.put({"action": "shutdown"})
    ga_command_queue.put({"action": "shutdown"})

    video_reader.join(timeout=2)
    tracking_worker.join(timeout=2)
    rendering_worker.join(timeout=2)
    ga_worker.join(timeout=2)

    dpg.destroy_context()

    print("Shutdown complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos and data there.")

    main()