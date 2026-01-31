from typing import Dict, List, Any

import numpy as np

import config
from lucida import CameraRig
from lucida.geometry import quaternion_average, rotation_vector, vector_from_quaternion, quaternion_from_vector


def create_individual(rig_template: CameraRig, scene_centre: np.ndarray) -> CameraRig:
    """
    Create a random camera calibration individual using Lucida.

    Uses the template rig to get camera names and image sizes, then randomizes
    intrinsics and extrinsics for genetic algorithm initialization.

    TODO: Use Lucida's CameraRig factory methods
    """
    cam_names = rig_template.names
    first_cam = rig_template[cam_names[0]]
    w, h = first_cam.image_size

    # Create circular layout as starting point
    # TODO: Expose more layout options via Lucida factories
    rig = CameraRig.from_circular_layout(
        cameras=cam_names,
        radius=100.0,
        center=scene_centre,
        target=scene_centre,
        plane='xz'
    )

    for cam in rig:
        # Randomise intrinsics
        fx = np.random.uniform(w * 0.8, w * 1.5)
        fy = np.random.uniform(h * 0.8, h * 1.5)
        cx = w / 2 + np.random.uniform(-w * 0.05, w * 0.05)
        cy = h / 2 + np.random.uniform(-h * 0.05, h * 0.05)

        cam.intrinsics.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Randomise distortion
        cam.intrinsics.D = np.random.normal(0.0, 0.001, size=config.NUM_DIST_COEFFS)

        # Perturb pose slightly
        cam.translate(np.random.normal(0, 5.0, 3), relative=True)

    return rig


def compute_fitness(
        individual: CameraRig,
        annotations: np.ndarray,
        calibration_frames: List[int]
) -> float:
    """Compute reprojection error fitness for a calibration."""

    if not calibration_frames:
        return float('inf')

    # Filter to calibration frames with valid data
    calib_mask = np.zeros(annotations.shape[0], dtype=bool)
    calib_mask[calibration_frames] = True
    valid_mask = np.any(~np.isnan(annotations[..., 0]), axis=(1, 2))
    combined_mask = calib_mask & valid_mask

    if not np.any(combined_mask):
        return float('inf')

    # Get annotations (x, y) for valid frames
    valid_annots = annotations[combined_mask][..., :2]  # (F_valid, C, P, 2)
    num_cams = len(individual)

    # Undistort observations
    # Flatten to (C, N_total, 2) for batch processing
    annots_flat = np.transpose(valid_annots, (1, 0, 2, 3)).reshape(num_cams, -1, 2)
    undistorted_flat = individual.undistort(annots_flat)

    # Reshape back to (F_valid, C, P, 2)
    F_valid, C, P, _ = valid_annots.shape
    undistorted_annots = undistorted_flat.reshape(C, F_valid, P, 2).transpose(1, 0, 2, 3)

    # Triangulate for each frame (requires rearranging dimensions)
    # Lucida expects (C, N, 2)

    points_3d_per_frame = []
    for f in range(F_valid):
        # (C, P, 2)
        frame_obs = undistorted_annots[f]
        p3d = individual.triangulate(frame_obs)
        points_3d_per_frame.append(p3d)

    points_3d = np.array(points_3d_per_frame)  # shape (F_valid, P, 3)

    nb_frames, num_points, _ = points_3d.shape
    points_3d_flat = points_3d.reshape(-1, 3)  # shape (F*P, 3)

    # Project all points into all cameras
    reprojected_flat = individual.project(points_3d_flat)  # (C, F*P, 2)

    # Reshape back to match annotation structure
    reprojected_unflat = reprojected_flat.reshape(num_cams, nb_frames, num_points, 2)
    reprojected_final = np.transpose(reprojected_unflat, (1, 0, 2, 3))  # shape (F, C, P, 2)

    # Calculate error
    errors = np.linalg.norm(reprojected_final - undistorted_annots, axis=-1)

    valid_mask = ~np.isnan(undistorted_annots[..., 0])
    total_error = np.sum(errors[valid_mask])
    total_points = np.sum(valid_mask)

    return total_error / total_points if total_points > 0 else float('inf')


def run_genetic_step(ga_state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one generation of the genetic algorithm."""

    population = ga_state.get("population")  # List of CameraRig objects
    best_fitness = ga_state.get("best_fitness", float('inf'))
    best_individual_dict = ga_state.get("best_individual")  # could be None for fresh start
    generation = ga_state.get("generation", 0)
    scene_centre = ga_state.get("scene_centre", np.zeros(3))
    stagnation_counter = ga_state.get("stagnation_counter", 0)

    # Re-hydrate best individual (to get camera names/sizes)
    best_individual = CameraRig.from_dict(best_individual_dict)

    # Re-hydrate population if needed (first run or from serialization)
    if population is None:
        if best_fitness < float('inf'):
            # We have a valid starting calibration - seed from it
            population = [best_individual]
            for _ in range(config.GA_POPULATION_SIZE - 1):
                mutated_rig = best_individual.copy()
                mutate_rig(mutated_rig, config.GA_MUTATION_STRENGTH_INIT)
                population.append(mutated_rig)
        else:
            # Fresh start - create random individuals
            population = [
                create_individual(best_individual, scene_centre)
                for _ in range(config.GA_POPULATION_SIZE)
            ]

    # Ensure all are CameraRig objects
    population = [p if isinstance(p, CameraRig) else CameraRig.from_dict(p) for p in population]

    # Evaluate fitness
    fitness_scores = np.array([
        compute_fitness(
            ind,
            ga_state['annotations'],
            ga_state['calibration_frames']
        )
        for ind in population
    ])
    sorted_indices = np.argsort(fitness_scores)

    # Elitism
    current_best_fit = fitness_scores[sorted_indices[0]]
    current_best_ind = population[sorted_indices[0]]

    if current_best_fit < best_fitness:
        best_fitness = current_best_fit
        best_individual_dict = current_best_ind.to_dict()
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    mutation_str = config.GA_MUTATION_STRENGTH
    if stagnation_counter > 20:
        mutation_str *= 2.5

    # Selection & Reproduction
    num_elites = int(config.GA_POPULATION_SIZE * config.GA_ELITISM_RATE)
    next_population = [population[i].copy() for i in sorted_indices[:num_elites]]

    while len(next_population) < config.GA_POPULATION_SIZE:
        # Tournament
        idx_a, idx_b = np.random.choice(len(population), 2, replace=False)
        p1 = population[idx_a] if fitness_scores[idx_a] < fitness_scores[idx_b] else population[idx_b]

        idx_c, idx_d = np.random.choice(len(population), 2, replace=False)
        p2 = population[idx_c] if fitness_scores[idx_c] < fitness_scores[idx_d] else population[idx_d]

        child = crossover_rigs(p1, p2)

        if np.random.rand() < config.GA_MUTATION_RATE:
            mutate_rig(child, mutation_str)

        next_population.append(child)

    return {
        "status": "running",
        "new_best_fitness": best_fitness,
        "new_best_individual": best_individual_dict,  # dict for serialization
        "generation": generation + 1,
        "mean_fitness": np.nanmean(fitness_scores),
        "std_fitness": np.nanstd(fitness_scores),
        "next_population": next_population,
        "stagnation_counter": stagnation_counter,
    }


def mutate_rig(rig: CameraRig, strength: float):
    for cam in rig:
        # K
        K = cam.intrinsics.K
        K[0, 0] += np.random.normal(0, strength * abs(K[0, 0]))
        K[1, 1] += np.random.normal(0, strength * abs(K[1, 1]))
        K[0, 2] += np.random.normal(0, strength * abs(K[0, 2]))
        K[1, 2] += np.random.normal(0, strength * abs(K[1, 2]))
        cam.intrinsics.K = K

        # D
        cam.intrinsics.D += np.random.normal(0, strength, size=cam.intrinsics.D.shape)

        # Pose
        cam.translate(np.random.normal(0, strength, 3), relative=True)

        # Rotate (small perturbations)
        rvec = np.random.normal(0, strength * 0.001, 3)
        cam.rotate(rotation_vector(rvec), relative=True)


def crossover_rigs(parent1: CameraRig, parent2: CameraRig) -> CameraRig:
    child_cams = []
    for c1, c2 in zip(parent1, parent2):
        new_cam = c1.copy()

        # Average Intrinsics
        new_cam.intrinsics.K = (c1.intrinsics.K + c2.intrinsics.K) / 2.0
        new_cam.intrinsics.D = (c1.intrinsics.D + c2.intrinsics.D) / 2.0

        # Average Pose (Linear position, Slerp rotation via quaternion average)
        new_cam.extrinsics.tvec_c2w = (c1.extrinsics.tvec_c2w + c2.extrinsics.tvec_c2w) / 2.0

        q1 = quaternion_from_vector(c1.extrinsics.rvec_c2w)
        q2 = quaternion_from_vector(c2.extrinsics.rvec_c2w)
        q_avg = quaternion_average(np.stack([q1, q2]))
        new_cam.extrinsics.rvec_c2w = vector_from_quaternion(q_avg)

        child_cams.append(new_cam)

    return CameraRig(child_cams)