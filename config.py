"""
Centralised configuration for CATAR.
All constants and default values are defined here.
"""
import cv2
from pathlib import Path
import numpy as np

# Display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# UI layout
CONTROL_PANEL_WIDTH = 300
BOTTOM_PANEL_HEIGHT_FULL = 220
BOTTOM_PANEL_HEIGHT_COLLAPSED = 75
PADDING = 20

# Video settings
DATA_FOLDER = Path.cwd() / 'data'
VIDEO_FORMAT = '*.mp4'

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(9, 9),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)

# Tracking parameters

# Decaying confidence with time from last human-annotated frame
CONFIDENCE_TIME_DECAY = 0.96

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

# Skeleton configuration
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
    "symmetry_map": [('a_L1', 'a_R1'),
                     ('a_L2', 'a_R2'),
                     ('a_L1', 'a_R1'),
                     ('a_L2', 'a_R2'),
                     ('leg_f_L1', 'leg_f_R1'),
                     ('leg_f_L2', 'leg_f_R2'),
                     ('leg_f_L1', 'leg_f_R1'),
                     ('leg_f_L2', 'leg_f_R2'),
                     ('leg_m_L1', 'leg_m_R1'),
                     ('leg_m_L2', 'leg_m_R2'),
                     ('leg_m_L1', 'leg_m_R1'),
                     ('leg_m_L2', 'leg_m_R2'),
                     ('a_R0', 'a_L0'),
                     ('a_R0', 'a_L0'),
                     ('leg_f_R0', 'leg_f_L0'),
                     ('leg_f_R0', 'leg_f_L0'),
                     ('leg_m_R0', 'leg_m_L0'),
                     ('leg_m_R0', 'leg_m_L0'),
                     ('eye_L', 'eye_R'),
                     ('eye_L', 'eye_R'),
                     ('m_L0', 'm_R0'),
                     ('m_L0', 'm_R0'),
                     ('m_L1', 'm_R1'),
                     ('m_L1', 'm_R1')],
    "point_colors": np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
        [192, 192, 192], [255, 128, 0], [128, 0, 255], [255, 128, 128], [128, 128, 0],
        [0, 128, 128], [128, 0, 128], [192, 128, 128], [128, 192, 128], [128, 128, 192],
        [192, 192, 128], [192, 128, 192], [128, 192, 192], [255, 255, 255], [0, 0, 0],
        [128, 128, 128], [255, 128, 64], [128, 64, 255], [210, 105, 30], [128, 255, 64],
        [128, 64, 0], [64, 128, 255]
    ], dtype=np.uint8)
}

# Camera visualisation colors
CAMERA_COLORS = [
    (255, 185, 20), (255, 0, 255), (0, 255, 128), (255, 55, 55),
    (0, 128, 128), (0, 120, 255), (128, 0, 128)
]
