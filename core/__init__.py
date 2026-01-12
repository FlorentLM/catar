"""
Core algorithms for CATAR (annotation, calibration, and tracking).
"""
from core.genetic_algorithm import compute_fitness, run_genetic_step
from core.refinement import prepare_refinement, run_refinement
from core.tracking_and_annotations import process_frame, track_points, compute_patch_ncc, snap_annotation, \
    fuse_annotations

__all__ = [
    # Annotation
    'snap_annotation',
    'fuse_annotations',

    # Genetic Algorithm
    'compute_fitness',
    'run_genetic_step',

    # Refinement (Bundle Adjustment)
    'prepare_refinement',
    'run_refinement',

    # Tracking
    'process_frame',
    'track_points',
    'compute_patch_ncc',
]