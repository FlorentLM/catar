"""
Cache management UI dialogs for DearPyGUI.
"""
import threading
from pathlib import Path
from typing import Optional, Callable
import time
import dearpygui.dearpygui as dpg

from video_cache import VideoCacheBuilder, VideoCacheReader
from state import Queues
from utils import load_and_match_videos


class CacheDialog:
    """Manages the cache build dialog and progress."""

    def __init__(self, data_folder: Path, video_format: str, queues: Queues, on_complete: Optional[Callable] = None):
        self.data_folder = data_folder
        self.video_format = video_format
        self.queues = queues
        self.on_complete = on_complete
        self.builder_thread: Optional[threading.Thread] = None
        self.is_building = False

    def cache_build_dialog(self):
        """Show dialog for building cache from menu."""

        if dpg.does_item_exist("cache_build_dialog"):
            dpg.delete_item("cache_build_dialog")

        with dpg.window(
            label="Build video cache",
            tag="cache_build_dialog",
            modal=True,
            width=500,
            height=280,
            pos=(400, 300)
        ):
            dpg.add_text("Building the video cache will enable instant seeking.")
            dpg.add_spacer(height=5)
            dpg.add_text("This process can take a few minutes.")
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Build Cache",
                    width=150,
                    callback=self._on_build_cache
                )
                dpg.add_button(
                    label="Cancel",
                    width=150,
                    callback=lambda: dpg.delete_item("cache_build_dialog")
                )

    def _on_build_cache(self):

        # Close menu dialog
        if dpg.does_item_exist("cache_build_dialog"):
            dpg.delete_item("cache_build_dialog")

        # Small delay to ensure dialog is properly closed
        time.sleep(0.05)

        # To calculate the required window height, we need to know the number of videos first.
        try:
            video_paths, _, _ = load_and_match_videos(self.data_folder, self.video_format)
            num_videos = len(video_paths)
        except Exception as e:
            print(f"Could not count videos before building cache, using default height. Error: {e}")
            num_videos = 4  # A reasonable default if video loading fails here

        self._show_build_progress_dialog(num_videos)

        self.is_building = True
        self.builder_thread = threading.Thread(target=self._build_cache_thread, daemon=True)
        self.builder_thread.start()

    def _show_build_progress_dialog(self, num_videos: int):
        """Show progress dialog during cache build with per-video progress bars."""

        if dpg.does_item_exist("cache_progress_dialog"):
            dpg.delete_item("cache_progress_dialog")

        # Calculate height for the progress bar area
        # Approx 32px per video (text + bar + smaller spacer)
        # Max height of 250px, after which it will scroll
        child_height = min(250, num_videos * 32)

        with dpg.window(
            label="Building video cache",
            tag="cache_progress_dialog",
            modal=True,
            no_close=True,
            no_collapse=True,
            width=550,  # Narrower width
            autosize=True,  # Auto-resize vertically
            pos=(350, 300)
        ):
            dpg.add_text("Building compressed video cache...", tag="cache_progress_status")
            dpg.add_spacer(height=10)

            # overall progress bar
            dpg.add_text("Overall Progress:", color=(200, 200, 255))
            dpg.add_progress_bar(tag="cache_progress_bar_total", width=-1, default_value=0.0)
            dpg.add_spacer(height=15)

            # per-video progressbars
            dpg.add_text("Per-Video Progress:", color=(200, 200, 255))
            with dpg.child_window(tag="video_progress_container", height=child_height, border=True):
                pass  # Items are added dynamically from main thread

            # Cancel button
            # TODO: Cancellation is a more complex topic, focusing on progress first.
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Building...",
                tag="cache_cancel_button",
                width=-1,
                enabled=False
            )

    def _build_cache_thread(self):
        """Background thread that builds the cache."""

        try:
            video_paths, _, _ = load_and_match_videos(self.data_folder, self.video_format)

            builder = VideoCacheBuilder(
                video_paths=video_paths,
                cache_dir=str(self.data_folder / 'video_cache'),
                ram_budget_gb=0.5
            )

            # Gather info
            video_info = builder.gather_video_info()
            total_videos = len(video_info)

            # Callback to send progress to the main thread via a queue
            def on_progress(completed_frames, total):
                if total > 0:
                    overall_progress = completed_frames / total
                    status_text = f"Processing: {completed_frames}/{total} frames ({overall_progress*100:.1f}%)"
                    self.queues.cache_progress.put({
                        "type": "overall",
                        "progress": overall_progress,
                        "status_text": status_text
                    })

            # This callback is for per-video progress reporting from the worker process
            def on_video_progress(video_idx, progress_pct):
                self.queues.cache_progress.put({
                    "type": "video",
                    "video_idx": video_idx,
                    "progress_pct": progress_pct,
                    "total_videos": total_videos
                })

            print("Starting cache build...")
            metadata = builder.build_cache(
                progress_callback=on_progress,
                video_progress_callback=on_video_progress
            )
            print("Cache build complete!")

            # Final success message
            self.queues.cache_progress.put({
                "type": "complete",
                "status_text": "Cache built successfully."
            })

            # Notify main application thread of completion
            if self.on_complete:
                self.on_complete(metadata=metadata, cache_dir=str(self.data_folder / 'video_cache'))

        except Exception as e:
            print(f"ERROR building cache: {e}")
            import traceback
            traceback.print_exc()

            # Send error message to main thread
            self.queues.cache_progress.put({
                "type": "error",
                "status_text": f"ERROR: {str(e)}"
            })
        finally:
            self.is_building = False


def cache_info_dialog(cache_reader: Optional[VideoCacheReader]):
    """Show information about the current cache."""

    if dpg.does_item_exist("cache_info_dialog"):
        dpg.delete_item("cache_info_dialog")

    with dpg.window(
        label="Video Cache Information",
        tag="cache_info_dialog",
        modal=True,
        width=500,
        pos=(400, 300)
    ):
        if cache_reader is None:
            dpg.add_text("No cache currently loaded", color=(255, 100, 100))
            dpg.add_spacer(height=10)
            dpg.add_text("Build a cache via Tools > Build Video Cache")
            dpg.add_text("to enable instant seeking.")
        else:
            info = cache_reader.get_cache_info()

            dpg.add_text("Cache Active.", color=(100, 255, 100))
            dpg.add_separator()
            dpg.add_spacer(height=5)

            dpg.add_text("Cache Details:", color=(200, 200, 255))
            dpg.add_text(f"  Location: {info['cache_dir']}", indent=10)
            dpg.add_text(f"  Size on disk: {info['total_size_gb']:.2f} GB", indent=10)
            dpg.add_text(f"  Created: {info['creation_time']}", indent=10)
            dpg.add_spacer(height=5)

            dpg.add_text("Video Details:", color=(200, 200, 255))
            dpg.add_text(f"  Videos: {info['num_videos']}", indent=10)
            dpg.add_text(f"  Frames: {info['frame_count']}", indent=10)
            dpg.add_text(f"  Total chunks: {info['total_chunks']}", indent=10)
            dpg.add_spacer(height=5)

            dpg.add_text("Memory Usage:", color=(200, 200, 255))
            dpg.add_text(f"  Chunks in RAM: {info['chunks_in_ram']} / {info['total_chunks']}", indent=10)
            dpg.add_text(f"  Frames per chunk: {info['frames_per_chunk']}", indent=10)

            ram_pct = (info['chunks_in_ram'] / info['total_chunks'] * 100) if info['total_chunks'] > 0 else 0
            dpg.add_progress_bar(default_value=ram_pct / 100.0, width=-1, overlay=f"{ram_pct:.1f}%")

            dpg.add_spacer(height=15)
            dpg.add_text("Actions:", color=(200, 200, 255))

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Clear chunks from RAM",
                    width=230,
                    callback=lambda: _clear_ram_chunks(cache_reader)
                )

        dpg.add_spacer(height=10)
        dpg.add_button(
            label="Close",
            width=-1,
            callback=lambda: dpg.delete_item("cache_info_dialog")
        )


def _clear_ram_chunks(cache_reader: VideoCacheReader):
    """Clear in-RAM chunks."""

    cache_reader.clear_cache()

    # Close info dialog
    if dpg.does_item_exist("cache_info_dialog"):
        dpg.delete_item("cache_info_dialog")

    # Delete success dialog if it already exists
    if dpg.does_item_exist("cache_cleared_dialog"):
        dpg.delete_item("cache_cleared_dialog")

    # Show confirmation
    with dpg.window(
        label="Cache Cleared",
        tag="cache_cleared_dialog",
        modal=True,
        width=300,
        height=120,
        pos=(500, 400)
    ):
        dpg.add_text("✓ RAM cache cleared successfully!")
        dpg.add_spacer(height=10)
        dpg.add_button(
            label="OK",
            width=-1,
            callback=lambda: dpg.delete_item("cache_cleared_dialog")
        )