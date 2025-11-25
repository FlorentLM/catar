"""
Core algorithms for CATAR (annotation, calibration, and tracking).
"""
from core.annotation import snap_annotation, fuse_annotations
from core.genetic_algorithm import create_individual, compute_fitness, run_genetic_step
from core.refinement import prepare_refinement, run_refinement
from core.tracking import process_frame, track_points, compute_patch_ncc

__all__ = [
    # Annotation
    'snap_annotation',
    'fuse_annotations',

    # Genetic Algorithm
    'create_individual',
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