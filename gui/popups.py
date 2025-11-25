import multiprocessing
import shutil
import threading
import time
from typing import TYPE_CHECKING

from dearpygui import dearpygui as dpg

from video import DiskCacheBuilder

from gui.callbacks.calibration import (
    stop_ga_callback,
    ba_mode_change_callback,
    start_ba_callback,
    stop_ba_callback,

)
from gui.callbacks.tracking import stop_batch_tracking_callback

from mokap.utils.fileio import probe_video

from video.backends import create_video_backend

if TYPE_CHECKING:
    from state import AppState, Queues


def create_ga_popup(app_state: 'AppState', queues: 'Queues'):
    """Create genetic algorithm progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
            label="Calibration Progress",
            modal=True,
            show=False,
            tag="ga_popup",
            width=400,
            height=150,
            no_close=True
    ):
        dpg.add_text("Running Genetic Algorithm...", tag="ga_status_text")
        dpg.add_text("Generation: 0", tag="ga_generation_text")
        dpg.add_text("Best Fitness: inf", tag="ga_fitness_text")
        dpg.add_text("Mean Fitness: inf", tag="ga_mean_fitness_text")
        dpg.add_button(
            label="Stop Calibration",
            callback=stop_ga_callback,
            user_data=user_data,
            width=-1
        )


def create_ba_config_popup(app_state: 'AppState', queues: 'Queues'):
    """Creates the popup for configuring the BA run before starting."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(label="Bundle Adjustment Settings", modal=True, show=False, tag="ba_config_popup",
                    width=450, height=270, no_close=False):
        dpg.add_text("Workflow selection:")
        dpg.add_radio_button(
            tag="ba_mode_radio",
            items=[
                "Refine Cameras",
                "Refine Cameras and 3D Points",
                "Refine 3D Points only"
            ],
            default_value="Refine Cameras",
            callback=ba_mode_change_callback
        )

        dpg.add_separator()
        with dpg.group():
            dpg.add_text("", tag="ba_help_text", wrap=410)

        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Refinement", callback=start_ba_callback,
                           user_data=user_data, width=-1)
            dpg.add_button(label="Cancel", callback=lambda: dpg.hide_item("ba_config_popup"), width=80)

    ba_mode_change_callback(None, dpg.get_value("ba_mode_radio"), None)


def create_ba_progress_popup(app_state: 'AppState', queues: 'Queues'):
    """Create bundle adjustment progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
            label="Refining Calibration", modal=True, show=False, tag="ba_progress_popup",
            width=400, height=100, no_close=True, no_move=False
    ):
        dpg.add_text("Running Bundle Adjustment...", tag="ba_status_text")
        dpg.add_text("This may take a few minutes...")
        dpg.add_separator()
        dpg.add_button(
            label="Cancel Refinement",
            callback=stop_ba_callback,
            user_data=user_data,
            width=-1
        )


def create_batch_tracking_popup(app_state: 'AppState', queues: 'Queues'):
    """Create batch tracking progress popup."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.window(
            label="Tracking Progress",
            modal=True,
            show=False,
            tag="batch_track_popup",
            width=400,
            no_close=True
    ):
        dpg.add_text("Processing frames...", tag="batch_track_status_text")
        dpg.add_progress_bar(tag="batch_track_progress", width=-1)
        dpg.add_button(
            label="Stop",
            callback=stop_batch_tracking_callback,
            user_data=user_data,
            width=-1
        )


def create_loupe():
    """Creates the floating, borderless window for the loupe."""
    loupe_size = 128

    with dpg.window(
            tag="loupe_window",
            show=False,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            width=loupe_size,
            height=loupe_size,
            no_scrollbar=True,
    ):
        with dpg.drawlist(
                width=loupe_size,
                height=loupe_size,
                tag="loupe_drawlist"
        ):
            # Layer for the zoomed video image
            dpg.draw_image(
                "loupe_texture",
                pmin=(0, 0),
                pmax=(loupe_size, loupe_size),
                tag="loupe_image"
            )
            # Layer for drawing annotations and lines on top
            dpg.add_draw_layer(tag="loupe_overlay_layer")

    dpg.bind_item_theme("loupe_window", "loupe_theme")


def cache_manager_callback(sender, app_data, user_data):
    """Show the cache manager dialog."""

    app_state = user_data["app_state"]
    queues = user_data["queues"]

    def on_cache_complete(metadata, cache_dir):
        """
        Called when cache build completes. Hotswap to new backend.

        Args:
            metadata: Cache metadata from builder
            cache_dir: Path to cache directory
        """
        print(f"Cache built successfully: {cache_dir}")

        try:
            new_backend = create_video_backend(
                video_paths=app_state.video_paths,
                video_metadata=app_state.video_metadata,
                cache_dir=cache_dir,
                backend_type='cached'
            )
            queues.command.put({"action": "update_backend", "backend": new_backend})

            old_backend = app_state.video_backend
            app_state.video_backend = new_backend  # TrackingWorker reads from here, so no need to notify it

            # Close old backend
            if old_backend is not None:
                old_backend.close()

            print("Backend successfully hot-swapped to use new cache!")

        except Exception as e:
            print(f"ERROR loading cache after build: {e}")
            import traceback
            traceback.print_exc()

    # Instantiate and show the dialog
    dialog = CacheManagerDialog(
        app_state=app_state,
        queues=queues,
        on_complete=on_cache_complete
    )
    dialog.show()


class CacheManagerDialog:
    """Manages the video cache management dialog and the build process."""

    def __init__(
        self,
        app_state: 'AppState',
        queues: 'Queues',
        on_complete
    ):
        self.app_state = app_state
        self.queues = queues
        self.on_complete = on_complete
        self.dialog_tag = "cache_manager_dialog"

        # Get video info
        self.video_paths = app_state.video_paths
        self.cache_dir = app_state.video_cache_dir

    def show(self):
        """Show the cache manager dialog."""

        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        self._render_initial_view()

    def _render_initial_view(self):
        """Render the main cache manager view."""
        with dpg.window(
                label="Video Cache Manager",
                tag=self.dialog_tag,
                modal=True,
                width=550,
                pos=(350, 250)
        ):
            # Check if backend has cache
            backend = self.app_state.video_backend
            has_cache = False
            cache_info = None

            if backend is not None:
                stats = backend.get_stats()
                backend_type = stats.get('backend_type', 'unknown')

                if backend_type == 'cached':
                    has_cache = True
                    cache_info = stats
                elif backend_type == 'hybrid':
                    has_cache = stats.get('cache_enabled', False)
                    if has_cache:
                        cache_info = stats

            if has_cache and cache_info:
                self._render_cache_exists_view(cache_info)
            else:
                self._render_no_cache_view()

    def _render_cache_exists_view(self, cache_info: dict):
        """Render view when cache exists."""
        dpg.add_text("Video cache is active", color=(100, 255, 100), parent=self.dialog_tag)
        dpg.add_separator(parent=self.dialog_tag)
        dpg.add_spacer(height=10, parent=self.dialog_tag)

        # Show cache statistics
        dpg.add_text("Cache Statistics:", color=(200, 200, 255), parent=self.dialog_tag)

        if 'cache_dir' in cache_info:
            dpg.add_text(f"  Location: {str(cache_info['cache_dir'])}", indent=10, parent=self.dialog_tag)

        if 'total_size_gb' in cache_info:
            dpg.add_text(f"  Size on disk: {cache_info['total_size_gb']:.2f} GB", indent=10, parent=self.dialog_tag)

        if 'chunks_in_ram' in cache_info:
            dpg.add_text(f"  Chunks in RAM: {cache_info['chunks_in_ram']}/{cache_info.get('ram_limit_chunks', '?')}",
                         indent=10, parent=self.dialog_tag)

        if 'frames_per_chunk' in cache_info:
            dpg.add_text(f"  Frames per chunk: {cache_info['frames_per_chunk']}", indent=10, parent=self.dialog_tag)

        dpg.add_spacer(height=15, parent=self.dialog_tag)

        # Action buttons
        dpg.add_button(
            label="Clear RAM Cache",
            callback=self._clear_ram_cache,
            width=-1,
            parent=self.dialog_tag
        )
        dpg.add_spacer(height=5, parent=self.dialog_tag)
        dpg.add_button(
            label="Rebuild Cache",
            callback=self._on_rebuild_cache,
            width=-1,
            parent=self.dialog_tag
        )
        dpg.add_spacer(height=5, parent=self.dialog_tag)
        dpg.add_button(
            label="Delete Cache",
            callback=self._on_delete_cache,
            width=-1,
            parent=self.dialog_tag
        )
        dpg.add_spacer(height=10, parent=self.dialog_tag)
        dpg.add_button(
            label="Close",
            width=-1,
            callback=lambda: dpg.delete_item(self.dialog_tag),
            parent=self.dialog_tag
        )

    def _render_no_cache_view(self):
        """Render view when no cache exists."""

        dpg.add_text("No video cache available", color=(255, 100, 100), parent=self.dialog_tag)
        dpg.add_separator(parent=self.dialog_tag)
        dpg.add_spacer(height=5, parent=self.dialog_tag)
        dpg.add_text(
            "Building a cache pre-processes the videos into a format that allows for instant, "
            "smooth frame seeking. Without a cache, seeking will be slower.",
            wrap=500, parent=self.dialog_tag
        )
        dpg.add_spacer(height=10, parent=self.dialog_tag)

        # Show video info
        try:
            metadata = probe_video(self.video_paths[0])
            dpg.add_text("Video information:", color=(200, 200, 255), parent=self.dialog_tag)
            dpg.add_text(f"  Videos found: {len(self.video_paths)}", indent=10, parent=self.dialog_tag)
            dpg.add_text(f"  Frames per video: {metadata['num_frames']}", indent=10, parent=self.dialog_tag)
            dpg.add_text(f"  Resolution: {metadata['width']}x{metadata['height']}", indent=10,
                         parent=self.dialog_tag)
        except Exception as e:
            dpg.add_text(f"Could not load video metadata: {e}", color=(255, 150, 150), wrap=500,
                         parent=self.dialog_tag)

        dpg.add_spacer(height=15, parent=self.dialog_tag)

        # Action button
        dpg.add_button(
            label="Build Cache",
            callback=self._on_build_cache,
            width=-1,
            enabled=bool(self.video_paths),
            parent=self.dialog_tag
        )
        dpg.add_spacer(height=10, parent=self.dialog_tag)
        dpg.add_button(
            label="Close",
            width=-1,
            callback=lambda: dpg.delete_item(self.dialog_tag),
            parent=self.dialog_tag
        )

    def _clear_ram_cache(self):
        """Clear the RAM cache of the current backend."""

        backend = self.app_state.video_backend
        if backend is not None:
            backend.clear_cache()
            print("RAM cache cleared")

        # Close and re-show dialog
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        # Show confirmation
        confirm_tag = "cache_cleared_dialog"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)

        with dpg.window(label="Cache Cleared", tag=confirm_tag, modal=True, width=300, pos=(500, 400)):
            dpg.add_text("RAM cache cleared successfully!")
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.delete_item(confirm_tag))

    def _on_rebuild_cache(self):
        """Handle rebuild cache button click."""

        # Show confirmation dialog
        confirm_tag = "rebuild_cache_confirm"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)

        with dpg.window(label="Confirm Rebuild", tag=confirm_tag, modal=True, width=400, pos=(450, 350)):
            dpg.add_text("Are you sure you want to rebuild the cache?", wrap=380)
            dpg.add_text("This will delete the existing cache and create a new one.", wrap=380, color=(255, 255, 100))
            dpg.add_spacer(height=15)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Yes, Rebuild",
                    callback=lambda: [dpg.delete_item(confirm_tag), self._start_build_process()],
                    width=190
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item(confirm_tag),
                    width=190
                )

    def _on_delete_cache(self):
        """Handle delete cache button click."""

        # Show confirmation dialog
        confirm_tag = "delete_cache_confirm"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)

        with dpg.window(label="Confirm Delete", tag=confirm_tag, modal=True, width=400, pos=(450, 350)):
            dpg.add_text("Are you sure you want to delete the cache?", wrap=380)
            dpg.add_text("You will need to rebuild it or use slower direct file access.", wrap=380,
                         color=(255, 255, 100))
            dpg.add_spacer(height=15)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Yes, Delete",
                    callback=lambda: [dpg.delete_item(confirm_tag), self._delete_cache()],
                    width=190
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item(confirm_tag),
                    width=190
                )

    def _delete_cache(self):
        """Delete the cache and switch to direct backend."""

        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir)
            print(f"Cache deleted: {self.cache_dir}")

        # Switch to direct backend
        new_backend = create_video_backend(
            video_paths=self.app_state.video_paths,
            video_metadata=self.app_state.video_metadata,
            cache_reader=None,
            backend_type='direct',
            ram_budget_gb=1.5
        )

        # Update VideoReaderWorker
        self.queues.command.put({"action": "update_backend", "backend": new_backend})

        # Store and cleanup
        old_backend = self.app_state.video_backend
        self.app_state.video_backend = new_backend
        if old_backend is not None:
            old_backend.close()

        print("Switched to direct backend")

        # Close dialog
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

    def _on_build_cache(self):
        """Handle build cache button click."""

        # Check if cache exists
        builder = DiskCacheBuilder(
            video_paths=self.video_paths,
            cache_dir=self.cache_dir
        )
        cache_exists, _ = builder.check_cache_exists()

        if cache_exists:
            self._show_rebuild_confirmation()
        else:
            self._start_build_process()

    def _show_rebuild_confirmation(self):
        """Show confirmation for rebuilding existing cache."""

        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag, children_only=True)

        dpg.configure_item(self.dialog_tag, label="Confirm Rebuild")

        dpg.add_text("A valid cache already exists for this video set.", wrap=530, parent=self.dialog_tag)
        dpg.add_text("Recreating it is not necessary. Are you sure you want to proceed?", wrap=530,
                     color=(255, 255, 100), parent=self.dialog_tag)
        dpg.add_spacer(height=20, parent=self.dialog_tag)

        with dpg.group(horizontal=True, parent=self.dialog_tag):
            dpg.add_button(
                label="Recreate Anyway",
                callback=self._start_build_process,
                width=260
            )
            dpg.add_button(
                label="Cancel",
                callback=self.show,
                width=260
            )

    def _start_build_process(self):
        """Start the cache building process."""

        # Close current dialog
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        time.sleep(0.05)  # let UI update

        # Show progress dialog
        num_videos = len(self.video_paths) if self.video_paths else 4
        self._show_build_progress_dialog(num_videos)

        # Start background thread
        self.builder_thread = threading.Thread(target=self._build_cache_thread, daemon=True)
        self.builder_thread.start()

    def _show_build_progress_dialog(self, num_videos: int):
        """Show the progress dialog during cache build."""

        dialog_tag = "cache_progress_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)

        child_height = min(250, num_videos * 32)

        with dpg.window(
                label="Building Video Cache",
                tag=dialog_tag,
                modal=True,
                no_close=True,
                width=550,
                pos=(350, 300)
        ):
            dpg.add_text("Building video cache...", tag="cache_progress_status")
            dpg.add_spacer(height=10)
            dpg.add_text("Overall progress:", color=(200, 200, 255))
            dpg.add_progress_bar(tag="cache_progress_bar_total", width=-1)
            dpg.add_spacer(height=15)
            dpg.add_text("Per-video progress:", color=(200, 200, 255))

            with dpg.child_window(tag="video_progress_container", height=child_height, border=True):
                pass

            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Cancel",
                tag="cache_cancel_button",
                width=-1,
                callback=self._on_cancel_build
            )

    def _on_cancel_build(self):
        """Handle cancel button click during build."""

        print("Cancel build requested")

        if hasattr(self, 'cancel_build_event') and self.cancel_build_event:
            self.cancel_build_event.set()

        if dpg.does_item_exist("cache_cancel_button"):
            dpg.configure_item("cache_cancel_button", label="Cancelling...", enabled=False)
        if dpg.does_item_exist("cache_progress_status"):
            dpg.set_value("cache_progress_status", "Cancellation requested, please wait...")

    def _build_cache_thread(self):
        """Background thread that builds the cache."""

        builder = None
        manager = multiprocessing.Manager()
        self.cancel_build_event = manager.Event()

        try:
            # Create builder
            builder = DiskCacheBuilder(
                video_paths=self.video_paths,
                cache_dir=self.cache_dir,
                ram_budget_gb=0.5
            )

            # Progress callbacks
            def on_progress(completed, total):
                if total > 0:
                    progress = completed / total
                    self.queues.cache_progress.put({
                        "type": "overall",
                        "progress": progress,
                        "status_text": f"Processing: {completed}/{total} frames ({progress * 100:.1f}%)"
                    })

            def on_video_progress(video_idx, pct):
                self.queues.cache_progress.put({
                    "type": "video",
                    "video_idx": video_idx,
                    "progress_pct": pct,
                    "total_videos": len(self.video_paths)
                })

            # Build cache
            metadata = builder.build_cache(
                progress_callback=on_progress,
                video_progress_callback=on_video_progress,
                cancel_event=self.cancel_build_event,
                manager=manager
            )

            # Check if cancelled
            if self.cancel_build_event.is_set():
                self.queues.cache_progress.put({
                    "type": "error",
                    "status_text": "Cache build cancelled by user."
                })
                return

            # Success
            self.queues.cache_progress.put({
                "type": "complete",
                "status_text": "Cache built successfully.",
                "cache_dir": self.cache_dir  # include cache_dir for hotswap
            })

            # Call completion callback
            if self.on_complete:
                self.on_complete(metadata=metadata, cache_dir=self.cache_dir)

        except (Exception, InterruptedError) as e:

            if self.cancel_build_event and self.cancel_build_event.is_set():
                self.queues.cache_progress.put({
                    "type": "error",
                    "status_text": "Cache build cancelled."
                })
            else:
                print(f"ERROR building cache: {e}")
                import traceback
                traceback.print_exc()
                self.queues.cache_progress.put({
                    "type": "error",
                    "status_text": f"ERROR: {str(e)}"
                })

        finally:
            # Cleanup on cancel
            if self.cancel_build_event and self.cancel_build_event.is_set():
                print("Cleaning up cancelled cache build...")
                time.sleep(0.5)

                if builder:
                    builder.delete_cache()
                elif self.cache_dir.is_dir():
                    shutil.rmtree(self.cache_dir)
                print("Cleanup complete")

            # Shutdown manager
            if manager:
                manager.shutdown()