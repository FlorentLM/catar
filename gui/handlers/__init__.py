from gui.handlers.mouse import *
from gui.handlers.keyboard import *
from gui.callbacks.playback import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state import AppState, Queues

def register_event_handlers(app_state: 'AppState', queues: 'Queues'):
    """Register event handlers (global)."""

    user_data = {"app_state": app_state, "queues": queues}

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press, user_data=user_data)

        dpg.add_mouse_drag_handler(
            button=dpg.mvMouseButton_Left,
            callback=image_mousedrag_callback,
            user_data=user_data
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Left,
            callback=leftclick_release_callback,
            user_data=user_data
        )
        dpg.add_key_down_handler(
            key=dpg.mvKey_LAlt,
            callback=on_alt_down,
            user_data=user_data
        )
        dpg.add_key_release_handler(
            key=dpg.mvKey_LAlt,
            callback=on_alt_up,
            user_data=user_data
        )