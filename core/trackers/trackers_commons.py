from typing import TYPE_CHECKING
# from core.trackers.tracker_IMM import IMMTracker
from core.trackers.tracker_mokap import MokapTracker
from core.trackers.tracker_kalman_simple import SimpleTracker

if TYPE_CHECKING:
    from core.trackers.trackers_base import BaseTracker


def create_tracker(algorithm_type: str, app_state, **kwargs) -> 'BaseTracker':
    if algorithm_type == "simple_kalman":
        return SimpleTracker(app_state)

    # elif algorithm_type == "imm":
    #     config = kwargs.get('config')
    #     bone_stats = kwargs.get('bone_stats')
    #     return IMMTracker(app_state, config=config, bone_stats=bone_stats)

    if algorithm_type == "imm":
        print('Not implemented yet')
        return SimpleTracker(app_state)

    elif algorithm_type == "mokap":
        # Extract the specific objects required for the mokap pipeline
        reconstructor = kwargs.get('reconstructor')
        mot_tracker = kwargs.get('mot_tracker')

        if not reconstructor or not mot_tracker:
            raise ValueError("Mokap tracker requires 'reconstructor' and 'mot_tracker'")

        return MokapTracker(app_state, reconstructor, mot_tracker)

    raise ValueError(f"Unknown tracker type: {algorithm_type}")
