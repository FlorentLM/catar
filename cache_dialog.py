"""
Cache management UI dialogs for DearPyGUI.
"""
import multiprocessing
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import dearpygui.dearpygui as dpg

from state import Queues
from utils import load_and_match_videos, get_video_metadata
from video_cache import VideoCacheBuilder, VideoCacheReader


class CacheManagerDialog:
    """Manages the video cache management dialog and the build process."""

    def __init__(self, app_state, queues: Queues, data_folder: Path, video_format: str, on_complete: Optional[Callable]):
        self.app_state = app_state
        self.queues = queues
        self.data_folder = data_folder
        self.video_format = video_format
        self.on_complete = on_complete
        self.builder_thread: Optional[threading.Thread] = None
        self.dialog_tag = "cache_manager_dialog"
        self.manager: Optional[multiprocessing.Manager] = None
        self.cancel_build_event: Optional[multiprocessing.Event] = None

        self.video_paths = []
        self.video_load_error: Optional[str] = None
        try:
            self.video_paths, _, _ = load_and_match_videos(self.data_folder, self.video_format)
        except Exception as e:
            self.video_load_error = str(e)
            print(f"CacheManagerDialog Error: {self.video_load_error}")

    def show(self):
        """Creates and shows the main cache management dialog."""
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        cache_reader = getattr(self.app_state, 'cache_reader', None)

        with dpg.window(label="Manage video cache", tag=self.dialog_tag, modal=True, width=550, pos=(350, 300)):
            if cache_reader:
                self._render_cache_exists_view(cache_reader)
            else:
                self._render_no_cache_view()

    def _render_cache_exists_view(self, cache_reader: VideoCacheReader):
        """Renders the UI for when the cache is present."""

        info = cache_reader.get_cache_info()

        dpg.add_text("Video cache is Active.", color=(100, 255, 100))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        dpg.add_text("Cache details:", color=(200, 200, 255))
        dpg.add_text(f"  Location: {info['cache_dir']}", indent=10)
        dpg.add_text(f"  Size on disk: {info['total_size_gb']:.2f} GB", indent=10)
        dpg.add_text(f"  Created: {info['creation_time']}", indent=10)
        dpg.add_spacer(height=5)

        dpg.add_text("Video details:", color=(200, 200, 255))
        dpg.add_text(f"  Videos: {info['num_videos']}", indent=10)
        dpg.add_text(f"  Frames: {info['frame_count']}", indent=10)
        dpg.add_text(f"  Total chunks: {info['total_chunks']}", indent=10)
        dpg.add_spacer(height=5)

        dpg.add_text("Memory usage:", color=(200, 200, 255))
        dpg.add_text(f"  Chunks in RAM: {info['chunks_in_ram']} / {info['total_chunks']}", indent=10)
        ram_pct = (info['chunks_in_ram'] / info['total_chunks'] * 100) if info['total_chunks'] > 0 else 0
        dpg.add_progress_bar(default_value=ram_pct / 100.0, width=-1, overlay=f"{ram_pct:.1f}%")
        dpg.add_spacer(height=15)

        dpg.add_text("Actions:", color=(200, 200, 255))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear RAM Cache", width=160, callback=lambda: self._clear_ram_chunks(cache_reader))
            dpg.add_button(label="Recreate video cache", width=160, callback=self._on_build_cache)
        dpg.add_spacer(height=10)

        dpg.add_button(label="Close", width=-1, callback=lambda: dpg.delete_item(self.dialog_tag))

    def _render_no_cache_view(self):
        """Renders the UI for when the cache is absent."""

        dpg.add_text("No video cache available.", color=(255, 100, 100))
        dpg.add_separator()
        dpg.add_spacer(height=5)
        dpg.add_text(
            "Building a cache pre-processes the videos into a format that allows for instant, "
            "smooth frame seeking. Without a cache, seeking will be slow.",
            wrap=500
        )
        dpg.add_spacer(height=10)

        if self.video_load_error:
            dpg.add_text(f"Could not load video information: {self.video_load_error}", color=(255, 150, 150), wrap=500)

        elif self.video_paths:
            try:
                metadata = get_video_metadata(self.video_paths[0])
                dpg.add_text("Video information:", color=(200, 200, 255))
                dpg.add_text(f"  Videos found: {len(self.video_paths)}", indent=10)
                dpg.add_text(f"  Frames per video: {metadata['num_frames']}", indent=10)
                dpg.add_text(f"  Resolution: {metadata['width']}x{metadata['height']}", indent=10)
            except Exception as e:
                dpg.add_text(f"Could not load video metadata: {e}", color=(255, 150, 150), wrap=500)

        dpg.add_spacer(height=15)

        dpg.add_button(
            label="Create Cache",
            callback=self._on_build_cache,
            width=-1,
            enabled=bool(self.video_paths)
        )
        dpg.add_spacer(height=10)
        dpg.add_button(label="Close", width=-1, callback=lambda: dpg.delete_item(self.dialog_tag))

    def _on_build_cache(self):
        """Handles the creation button click."""

        if not self.video_paths:
            print("Build cache called but no video paths are loaded. Aborting.")
            return

        builder = VideoCacheBuilder(video_paths=self.video_paths, cache_dir=str(self.data_folder / 'video_cache'))
        cache_exists, _ = builder.check_cache_exists()

        if cache_exists:
            confirm_dialog_tag = "confirm_recreate_dialog"
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)

            with dpg.window(label="Confirm Recreate", tag=confirm_dialog_tag, modal=False, width=400, pos=(400, 400)):
                dpg.add_text("A valid cache already exists for this video set.", wrap=380)
                dpg.add_text("Recreating it is not necessary. Are you sure you want to proceed?", wrap=380,
                             color=(255, 255, 100))
                dpg.add_spacer(height=10)

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Recreate Anyway",
                        callback=self._confirm_and_recreate,
                        user_data={"dialog_tag": confirm_dialog_tag},
                        width=190
                    )
                    dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item(confirm_dialog_tag), width=190)

            dpg.focus_item(confirm_dialog_tag)
        else:
            self._start_build_process()

    def _confirm_and_recreate(self, sender, app_data, user_data):
        confirm_dialog_tag = user_data["dialog_tag"]
        dpg.delete_item(confirm_dialog_tag)
        self._start_build_process()

    def _start_build_process(self):
        """Closes the manager and opens the progress dialog to start the build."""

        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)
        time.sleep(0.05)

        num_videos = len(self.video_paths) if self.video_paths else 4
        self._show_build_progress_dialog(num_videos)

        self.builder_thread = threading.Thread(target=self._build_cache_thread, daemon=True)
        self.builder_thread.start()

    def _show_build_progress_dialog(self, num_videos: int):
        """Shows the progress dialog during the cache build."""

        dialog_tag = "cache_progress_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)

        child_height = min(250, num_videos * 32)

        with dpg.window(label="Building Video Cache", tag=dialog_tag, modal=True, no_close=True, width=550, pos=(350, 300)):
            dpg.add_text("Building video cache...", tag="cache_progress_status")
            dpg.add_spacer(height=10)
            dpg.add_text("Overall progress:", color=(200, 200, 255))
            dpg.add_progress_bar(tag="cache_progress_bar_total", width=-1)
            dpg.add_spacer(height=15)
            dpg.add_text("Per-video progress:", color=(200, 200, 255))

            with dpg.child_window(tag="video_progress_container", height=child_height, border=True):
                pass

            dpg.add_spacer(height=10)
            dpg.add_button(label="Cancel", tag="cache_cancel_button", width=-1, callback=self._on_cancel_build)

    def _on_cancel_build(self):
        """Callback for the cancel button."""

        print("Cancel build requested.")

        if self.cancel_build_event:
            self.cancel_build_event.set()

        if dpg.does_item_exist("cache_cancel_button"):
            dpg.configure_item("cache_cancel_button", label="Cancelling...", enabled=False)
        if dpg.does_item_exist("cache_progress_status"):
            dpg.set_value("cache_progress_status", "Cancellation requested, please wait...")

    def _build_cache_thread(self):
        """Background thread that runs the cache builder."""

        builder = None
        self.manager = multiprocessing.Manager()
        self.cancel_build_event = self.manager.Event()
        cache_dir = str(self.data_folder / 'video_cache')

        try:

            builder = VideoCacheBuilder(
                video_paths=self.video_paths,
                cache_dir=cache_dir,
                ram_budget_gb=0.5
            )

            def on_progress(completed, total):
                if total > 0:
                    progress = completed / total
                    self.queues.cache_progress.put({
                        "type": "overall", "progress": progress,
                        "status_text": f"Processing: {completed}/{total} frames ({progress*100:.1f}%)"
                    })

            def on_video_progress(video_idx, pct):
                self.queues.cache_progress.put({
                    "type": "video", "video_idx": video_idx, "progress_pct": pct, "total_videos": len(self.video_paths)
                })

            metadata = builder.build_cache(
                progress_callback=on_progress,
                video_progress_callback=on_video_progress,
                cancel_event=self.cancel_build_event
            )

            if self.cancel_build_event.is_set():
                self.queues.cache_progress.put({"type": "error", "status_text": "Cache build cancelled by user."})
                return

            self.queues.cache_progress.put({"type": "complete", "status_text": "Cache built successfully."})

            if self.on_complete:
                self.on_complete(metadata=metadata, cache_dir=cache_dir)

        except (Exception, InterruptedError) as e:
            if self.cancel_build_event and self.cancel_build_event.is_set():
                self.queues.cache_progress.put({"type": "error", "status_text": "Cache build cancelled."})
            else:
                print(f"ERROR building cache: {e}")
                import traceback
                traceback.print_exc()
                self.queues.cache_progress.put({"type": "error", "status_text": f"ERROR: {str(e)}"})
        finally:
            if self.cancel_build_event and self.cancel_build_event.is_set():
                print("Cleaning up cancelled cache build...")
                time.sleep(0.5)

                # Ensure builder was instantiated before trying to delete
                if builder:
                    builder.delete_cache()
                elif Path(cache_dir).exists(): # Fallback cleanup
                    shutil.rmtree(cache_dir)
                print("Cleanup complete.")
            if self.manager:
                self.manager.shutdown()

    def _clear_ram_chunks(self, cache_reader: VideoCacheReader):
        """Clears in-memory cache and shows a confirmation."""

        cache_reader.clear_cache()
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        confirm_tag = "cache_cleared_dialog"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)

        with dpg.window(label="Cache cleared", tag=confirm_tag, modal=True, width=300, pos=(500, 400)):
            dpg.add_text("RAM cache cleared successfully!")
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.delete_item(confirm_tag))