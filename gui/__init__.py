"""
GUI components for CATAR (interface layout, rendering, popups, and event handling).
"""

# Main GUI components
from gui.layout import create_ui
from gui.rendering import update_ui, resize_video_widgets
from gui.popups import (
    create_ga_popup,
    create_ba_config_popup,
    create_ba_progress_popup,
    create_batch_tracking_popup,
    create_loupe,
    cache_manager_callback
)

__all__ = [
    # Layout
    'create_ui',

    # Rendering
    'update_ui',
    'resize_video_widgets',

    # Popups
    'create_ga_popup',
    'create_ba_config_popup',
    'create_ba_progress_popup',
    'create_batch_tracking_popup',
    'create_loupe',
    'cache_manager_callback',
]