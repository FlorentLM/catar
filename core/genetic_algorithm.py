import random
from typing import Dict, List, Any

import numpy as np

import config
from state.calibration_state import CalibrationState

from mokap.geometry import quaternion_average, rotation_vector, vector_from_quaternion, quaternion_from_vector

def create_individual(video_metadata: Dict, cam_names: List[str], scene_centre: np.ndarray) -> Dict[str, Dict]:
    """Create a random camera calibration individual."""

    w, h = video_metadata['width'], video_metadata['height']
    num_cameras = video_metadata['num_videos']

    radius = np.linalg.norm(scene_centre) if np.linalg.norm(scene_centre) > 1 else 100.0

    individual: Dict[str, Dict] = {}
    for i, cam_name in enumerate(cam_names):
        # Position cameras in a circle around the scene centre
        angle = (2 * np.pi / num_cameras) * i

        # Offset camera positions by the scene centre
        cam_pos_world = scene_centre + np.array([radius * np.cos(angle), 2.0, radius * np.sin(angle)])

        up_vector = np.array([0.0, 1.0, 0.0])
        forward = (scene_centre - cam_pos_world) / np.linalg.norm(scene_centre - cam_pos_world)

        right = np.cross(forward, up_vector)
        cam_up = np.cross(right, forward)

        R_w2c = np.array([-right, cam_up, -forward])
        R_c2w = R_w2c.T
        tvec_c2w = cam_pos_world
        rvec_c2w = rotation_vector(R_c2w)

        # Randomize K components
        # TODO: These factors are a bit large...
        fx = random.uniform(w * 0.8, w * 1.5)
        fy = random.uniform(h * 0.8, h * 1.5)
        cx = w / 2 + random.uniform(-w * 0.05, w * 0.05)
        cy = h / 2 + random.uniform(-h * 0.05, h * 0.05)

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        individual[cam_name] = {
            'camera_matrix': K,
            'dist_coeffs': np.random.normal(0.0, 0.001, size=config.NUM_DIST_COEFFS),
            'rvec': rvec_c2w.flatten(),
            'tvec': tvec_c2w.flatten()
        }

    return individual


def compute_fitness(
    individual: Dict[str, Dict],  # This is the candidate from the GA
    annotations: np.ndarray,
    calibration_frames: List[int],
    video_metadata: Dict,
    cam_names: List[str]
) -> float:
    """Compute reprojection error fitness for a calibration."""

    if not calibration_frames:
        return float('inf')

    # Temporary CalibrationState for candidate individual
    temp_calib_state = CalibrationState(individual, cam_names)

    # Filter to calibration frames with valid data
    calib_mask = np.zeros(annotations.shape[0], dtype=bool)
    calib_mask[calibration_frames] = True
    valid_mask = np.any(~np.isnan(annotations[..., 0]), axis=(1, 2))
    combined_mask = calib_mask & valid_mask

    if not np.any(combined_mask):
        return float('inf')

    # Get annotations (x, y) for valid frames
    valid_annots = annotations[combined_mask][..., :2]
    num_cams = video_metadata['num_videos']

    annots_for_undistort = np.transpose(valid_annots, (1, 2, 0, 3)).reshape(num_cams, -1, 2)
    undistorted_flat = temp_calib_state.undistort_all(annots_for_undistort)
    undistorted_annots = undistorted_flat.reshape(num_cams, -1, valid_annots.shape[2], 2)
    undistorted_annots = np.transpose(undistorted_annots, (1, 0, 2, 3))

    # Triangulate for each frame
    points_3d_per_frame = [
        temp_calib_state.triangulate(frame_annots, weights=None)
        for frame_annots in undistorted_annots
    ]
    points_3d = np.array(points_3d_per_frame)  # shape (F, P, 3)

    # Temporary state for candidate calibration
    temp_calib_state = CalibrationState(individual, cam_names)

    num_frames, num_points, _ = points_3d.shape
    points_3d_flat = points_3d.reshape(-1, 3)  # shape (F*P, 3)

    # Project all points into all cameras
    reprojected_flat = temp_calib_state.reproject_to_all(points_3d_flat)

    # Reshape back to match annotation structure
    reprojected_unflat = reprojected_flat.reshape(num_cams, num_frames, num_points, 2)
    reprojected_final = np.transpose(reprojected_unflat, (1, 0, 2, 3))  # shape (F, C, P, 2)

    # Calculate error across all points and cameras
    errors = np.linalg.norm(reprojected_final - undistorted_annots, axis=-1)

    # Mask out invalid points and calculate final fitness score
    valid_mask = ~np.isnan(undistorted_annots[..., 0])
    total_error = np.sum(errors[valid_mask])
    total_points = np.sum(valid_mask)

    return total_error / total_points if total_points > 0 else float('inf')


def run_genetic_step(ga_state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one generation of the genetic algorithm."""

    population = ga_state.get("population")
    best_fitness = ga_state.get("best_fitness", float('inf'))
    best_individual = ga_state.get("best_individual")
    generation = ga_state.get("generation", 0)
    scene_centre = ga_state.get("scene_centre", np.zeros(3))
    stagnation_counter = ga_state.get("stagnation_counter", 0)
    cam_names = ga_state["camera_names"]

    if population is None:
        if best_individual:
            # Seed initial population from best individual
            population = [best_individual]
            for _ in range(config.GA_POPULATION_SIZE - 1):
                mutated_ind = {}

                for cam_name, cam_params in best_individual.items():
                    # TODO: Are these copies really necessary?
                    mutated_cam = cam_params.copy()

                    # Mutate K matrix elements (fx, fy, cx, cy)
                    K = cam_params['camera_matrix'].copy()
                    K[0, 0] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[0, 0]))  # fx
                    K[1, 1] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[1, 1]))  # fy
                    K[0, 2] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[0, 2]))  # cx
                    K[1, 2] += np.random.normal(0, config.GA_MUTATION_STRENGTH_INIT * abs(K[1, 2]))  # cy
                    mutated_cam['camera_matrix'] = K

                    # Mutate other parameters
                    for key in ['tvec', 'dist_coeffs']:
                        mutated_cam[key] = np.asarray(mutated_cam[key]) + np.random.normal(0,
                                                                                           config.GA_MUTATION_STRENGTH_INIT,
                                                                                           size=mutated_cam[key].shape)

                    mutated_cam['rvec'] = np.asarray(mutated_cam['rvec']) + np.random.normal(0,
                                                                                             config.GA_MUTATION_STRENGTH_INIT * 0.001,
                                                                                             # Massively reduce mutation on rvec because radians
                                                                                             size=mutated_cam['rvec'].shape)
                    mutated_ind[cam_name] = mutated_cam
                population.append(mutated_ind)
        else:
            population = [create_individual(ga_state['video_metadata'], cam_names, scene_centre) for _ in
                          range(config.GA_POPULATION_SIZE)]

    # Evaluate fitness
    fitness_scores = np.array([
        compute_fitness(
            ind,
            ga_state['annotations'],
            ga_state['calibration_frames'],
            ga_state['video_metadata'],
            cam_names
        )
        for ind in population
    ])
    sorted_indices = np.argsort(fitness_scores)

    if fitness_scores[sorted_indices[0]] < best_fitness:
        best_fitness = fitness_scores[sorted_indices[0]]
        best_individual = population[sorted_indices[0]]
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    current_mutation_strength = config.GA_MUTATION_STRENGTH
    if stagnation_counter > 20:
        print("GA is stagnating, temporarily increasing mutation strength.")
        current_mutation_strength *= 2.5

    # Create next generation
    num_elites = int(config.GA_POPULATION_SIZE * config.GA_ELITISM_RATE)
    next_population = [population[i] for i in sorted_indices[:num_elites]]

    while len(next_population) < config.GA_POPULATION_SIZE:

        # Tournament selection
        # TODO: This is a bit simple, could be improved

        p1_idx, p2_idx = np.random.choice(len(population), 2, replace=False)
        parent1 = population[p1_idx] if fitness_scores[p1_idx] < fitness_scores[p2_idx] else population[p1_idx]
        p3_idx, p4_idx = np.random.choice(len(population), 2, replace=False)
        parent2 = population[p3_idx] if fitness_scores[p3_idx] < fitness_scores[p4_idx] else population[p3_idx]

        child = {}
        for cam_name in cam_names:
            p1_cam = parent1[cam_name]
            p2_cam = parent2[cam_name]

            child_cam = {}

            # rvec averaging (quaternion)
            q_batch = quaternion_from_vector(np.stack([p1_cam['rvec'], p2_cam['rvec']]))
            q_avg = quaternion_average(q_batch)

            rvec_avg = vector_from_quaternion(q_avg)
            child_cam['rvec'] = np.asarray(rvec_avg)

            # Linear Averaging for other parameters
            for key in p1_cam:
                if key != 'rvec':
                    p1_val = np.asarray(p1_cam[key])
                    p2_val = np.asarray(p2_cam[key])
                    child_cam[key] = (p1_val + p2_val) / 2.0

            # Mutation
            if np.random.rand() < config.GA_MUTATION_RATE:

                # Mutate K elements
                K_mut = child_cam['camera_matrix'].copy()
                K_mut[0, 0] += np.random.normal(0, current_mutation_strength * abs(K_mut[0, 0]))
                K_mut[1, 1] += np.random.normal(0, current_mutation_strength * abs(K_mut[1, 1]))
                K_mut[0, 2] += np.random.normal(0, current_mutation_strength * abs(K_mut[0, 2]))
                K_mut[1, 2] += np.random.normal(0, current_mutation_strength * abs(K_mut[1, 2]))
                child_cam['camera_matrix'] = K_mut

                # Mutate tvec and dist_coeffs
                for key in ['tvec', 'dist_coeffs']:
                    child_cam[key] = child_cam[key] + np.random.normal(0, current_mutation_strength,
                                                                       size=child_cam[key].shape)

                # Mutate rvec
                child_cam['rvec'] = child_cam['rvec'] + np.random.normal(0,
                                                                         current_mutation_strength * 0.001, # Massively reduce mutation on rvec because radians
                                                                         size=child_cam['rvec'].shape)

            child[cam_name] = child_cam
        next_population.append(child)

    mean_fitness = np.nanmean(fitness_scores)
    std_fitness = np.nanstd(fitness_scores)

    return {
        "status": "running",
        "new_best_fitness": best_fitness,
        "new_best_individual": best_individual,
        "generation": generation + 1,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "next_population": next_population,
        "stagnation_counter": stagnation_counter,
    }
