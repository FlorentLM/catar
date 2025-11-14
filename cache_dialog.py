"""
Cache management UI dialogs for DearPyGUI.
"""
import threading
from pathlib import Path
from typing import Optional, Callable
import time
import dearpygui.dearpygui as dpg

from video_cache import VideoCacheBuilder, VideoCacheReader


class CacheDialog:
    """Manages the cache build dialog and progress."""

    def __init__(self, data_folder: Path, video_format: str, on_complete: Optional[Callable] = None):
        self.data_folder = data_folder
        self.video_format = video_format
        self.on_complete = on_complete
        self.builder_thread: Optional[threading.Thread] = None
        self.is_building = False
        self.build_progress = {
            'video_progress': {},
            'total_progress': 0.0,
            'status': ''
        }

    def show_menu_build_dialog(self):
        """Show dialog for building cache from menu."""

        if dpg.does_item_exist("cache_menu_dialog"):
            dpg.delete_item("cache_menu_dialog")

        with dpg.window(
            label="Build Video Cache",
            tag="cache_menu_dialog",
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
                    width=230,
                    callback=self._on_build_cache
                )
                dpg.add_button(
                    label="Cancel",
                    width=230,
                    callback=lambda: dpg.delete_item("cache_menu_dialog")
                )

    def _on_build_cache(self):

        # Close menu dialog
        if dpg.does_item_exist("cache_menu_dialog"):
            dpg.delete_item("cache_menu_dialog")

        # Small delay to ensure dialog is properly closed
        time.sleep(0.05)

        self._show_build_progress_dialog()

        self.is_building = True
        self.builder_thread = threading.Thread(target=self._build_cache_thread, daemon=True)
        self.builder_thread.start()

    def _show_build_progress_dialog(self):
        """Show progress dialog during cache build with per-video progress bars."""
        # TODO: This is not working

        if dpg.does_item_exist("cache_progress_dialog"):
            dpg.delete_item("cache_progress_dialog")

        with dpg.window(
            label="Building Video Cache",
            tag="cache_progress_dialog",
            modal=True,
            no_close=True,
            no_collapse=True,
            width=650,
            height=400,
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
            with dpg.child_window(tag="video_progress_container", height=180, border=True):
                pass

            # Cancel button
            # TODO: This is not working either
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Building...",
                tag="cache_cancel_button",
                width=-1,
                enabled=True
            )

    def _update_video_progress_ui(self, video_idx: int, progress_pct: float, total_videos: int):

        bar_tag = f"video_progress_bar_{video_idx}"
        text_tag = f"video_progress_text_{video_idx}"

        if not dpg.does_item_exist(bar_tag):

            if dpg.does_item_exist("video_progress_container"):
                with dpg.group(parent="video_progress_container", horizontal=False):
                    dpg.add_text(f"Video {video_idx + 1}/{total_videos}:", tag=text_tag)
                    dpg.add_progress_bar(tag=bar_tag, width=-1, default_value=0.0)
                    dpg.add_spacer(height=5)

        # Update progress
        if dpg.does_item_exist(bar_tag):
            dpg.configure_item(bar_tag, default_value=progress_pct / 100.0)
        if dpg.does_item_exist(text_tag):
            dpg.set_value(text_tag, f"Video {video_idx + 1}/{total_videos}: {progress_pct:.0f}%")

    def _build_cache_thread(self):
        """Background thread that builds the cache."""

        try:
            from utils import load_and_match_videos
            video_paths, _, _ = load_and_match_videos(self.data_folder, self.video_format)

            builder = VideoCacheBuilder(
                video_paths=video_paths,
                cache_dir=str(self.data_folder / 'video_cache'),
                ram_budget_gb=0.5
            )

            # Gather info
            video_info = builder.gather_video_info()
            total_videos = len(video_info)

            # Track progress per video
            video_completion = {}

            def on_progress(completed_frames, total):

                # estimate which video we're on based on completed frames
                frames_per_video = video_info[0]['frame_count']
                current_video = min(completed_frames // frames_per_video, total_videos - 1)

                # Update current video progress
                frames_in_current = completed_frames % frames_per_video
                if frames_in_current == 0 and completed_frames > 0:
                    frames_in_current = frames_per_video
                    if current_video > 0:
                        current_video -= 1

                progress_pct = (frames_in_current / frames_per_video) * 100

                # Mark completed videos as 100%
                for i in range(current_video):
                    if i not in video_completion:
                        video_completion[i] = 100.0
                        self._update_video_progress_ui(i, 100.0, total_videos)

                # Update current video
                if current_video < total_videos:
                    video_completion[current_video] = progress_pct
                    self._update_video_progress_ui(current_video, progress_pct, total_videos)

                # Update overall progress
                overall_progress = completed_frames / total if total > 0 else 0
                if dpg.does_item_exist("cache_progress_bar_total"):
                    dpg.configure_item("cache_progress_bar_total", default_value=overall_progress)
                if dpg.does_item_exist("cache_progress_status"):
                    dpg.set_value(
                        "cache_progress_status",
                        f"Processing: {completed_frames}/{total} frames ({overall_progress*100:.1f}%)"
                    )

            print("Starting cache build...")
            metadata = builder.build_cache(progress_callback=on_progress)
            print("Cache build complete!")

            # Mark all videos complete
            for i in range(total_videos):
                self._update_video_progress_ui(i, 100.0, total_videos)

            # Update UI on completion
            if dpg.does_item_exist("cache_progress_status"):
                dpg.set_value("cache_progress_status", "Cache built successfully.")
            if dpg.does_item_exist("cache_progress_bar_total"):
                dpg.configure_item("cache_progress_bar_total", default_value=1.0)

            # Replace cancel button with close button
            if dpg.does_item_exist("cache_cancel_button"):
                dpg.delete_item("cache_cancel_button")

            if dpg.does_item_exist("cache_progress_dialog"):
                dpg.add_spacer(height=10, parent="cache_progress_dialog")
                dpg.add_button(
                    label="Close",
                    width=-1,
                    callback=lambda: dpg.delete_item("cache_progress_dialog"),
                    parent="cache_progress_dialog"
                )

            # Notify completion
            if self.on_complete:
                self.on_complete(metadata=metadata, cache_dir=str(self.data_folder / 'video_cache'))

        except Exception as e:
            print(f"ERROR building cache: {e}")
            import traceback
            traceback.print_exc()

            if dpg.does_item_exist("cache_progress_status"):
                dpg.set_value("cache_progress_status", f"ERROR: {str(e)}")
            if dpg.does_item_exist("cache_progress_bar_total"):
                dpg.configure_item("cache_progress_bar_total", default_value=0.0, overlay="ERROR")

            # Show close button on error too
            if dpg.does_item_exist("cache_cancel_button"):
                dpg.delete_item("cache_cancel_button")
            if dpg.does_item_exist("cache_progress_dialog"):
                dpg.add_button(
                    label="Close",
                    width=-1,
                    callback=lambda: dpg.delete_item("cache_progress_dialog"),
                    parent="cache_progress_dialog"
                )
        finally:
            self.is_building = False


def show_cache_info_dialog(cache_reader: Optional[VideoCacheReader]):
    """Show information about the current cache."""

    if dpg.does_item_exist("cache_info_dialog"):
        dpg.delete_item("cache_info_dialog")

    with dpg.window(
        label="Video Cache Information",
        tag="cache_info_dialog",
        modal=True,
        width=500,
        height=380,
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