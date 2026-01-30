"""
Centralised configuration for CATAR.
All constants and default values are defined here.
"""
import cv2
from pathlib import Path

# Debug prints
VERBOSE = True
DISABLE_3D_VIEW = True

# Display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# UI layout
CONTROL_PANEL_WIDTH = 300
BOTTOM_PANEL_HEIGHT_FULL = 220
BOTTOM_PANEL_HEIGHT_COLLAPSED = 75
PADDING = 20

# Data inputs
DATA_FOLDER = Path.cwd() / 'data'
VIDEO_FORMAT = '*.mp4'

# Video cache settings
VIDEO_CACHE_FOLDER = DATA_FOLDER / 'video_cache'
RAM_MAX_BUDGET_GB = 1.5

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(9, 9),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)

# Tracking parameters

# Tracker confidence half-life:
# After how much (real-life) time its certainty is 50% compared to when it started.
# This is automatically combined with the videos framerate.
TRACKER_HALF_LIFE_CONFIDENCE_DECAY = 0.5  # in seconds

# Fusion precedence ratio:
# During tracking, if a source is > 2.0x more confident than an other source,
# the weaker source is ignored.
#
# This creates a "winner-takes-all" to silence weak signals (e.g. drifted tracks),
# while falling back to "weighted averaging" when signals have comparable confidence.
#
# Human annotations are EXEMPT from this check and are never ignored.
#
# Example (assuming fresh track confidence ~1.0):
# 2.0 = discard old tracks decayed below 50% (1 half-life)
# 4.0 = discard old tracks decayed below 25% (2 half-lives)
# inf = precedence ratio disabled
FUSION_PRECEDENCE_RATIO = 2.0

# The maximum distance for two sources to be considered in agreement.
# This defines the tolerance for consensus and the boundary for the confidence bonus falloff.
FUSION_AGREEMENT_RADIUS = 5.0  # in pixels

# The maximum confidence bonus awarded when sources agree perfectly (distance = 0).
# The actual bonus decreases linearly to 0 as the distance approaches FUSION_AGREEMENT_RADIUS.
FUSION_AGREEMENT_BONUS = 0.15

# A hard ceiling on the confidence of any machine-generated annotation.
# Ensures that user-provided annotations can always remain the highest-trust source.
FUSION_MAX_AUTO_CONFIDENCE = 0.98

# The initial confidence assigned to an annotation when it is first created or moved by the user.
# This value establishes it as a high-trust point in the fusion process.
FUSION_HUMAN_CONFIDENCE = 1.0

# The maximum confidence assigned to a single-view LK track that is geometrically unverified.
# This value is achieved when the forward-backward tracking error is zero and scales down from there.
MAX_SINGLE_VIEW_CONFIDENCE = 0.5

# If an LK-tracked point deviates from the multi-view geometric consensus by more
# than this many pixels, it is considered to be drifting and is invalidated.
LK_CONFIDENCE_MAX_ERROR = 7.5  # in pixels

# Threshold for the forward-backward check. If the endpoint of a backward track
# is further than this many pixels from the start point, the track is unreliable.
FORWARD_BACKWARD_THRESHOLD = 5.0  # in pixels


# NCC settings for visual consistency checks
NCC_THRESHOLD_WARNING = 0.55
NCC_THRESHOLD_KILL = 0.3
NCC_PATCH_SIZE = 11


# Camera calibration
NUM_DIST_COEFFS = 14

# Genetic Algorithm parameters
GA_POPULATION_SIZE = 200
GA_ELITISM_RATE = 0.1
GA_MUTATION_RATE = 0.8
GA_MUTATION_STRENGTH = 0.05  # General mutation strength
GA_MUTATION_STRENGTH_INIT = 0.01 # Strength for seeding initial population

# UI interaction
ANNOTATION_DRAG_THRESHOLD = 15  # in pixels


# Camera visualisation colors
CAMERA_COLORS = [
    (255, 185, 20), (255, 0, 255), (0, 255, 128), (255, 55, 55),
    (0, 128, 128), (0, 120, 255), (128, 0, 128)
]
