"""Generate various plots to visualise the 3D pose of ant and seed."""
import pathlib
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import cv2
import trimesh
import plotly.graph_objects as go
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from run import SKELETON, POINT_NAMES, DATA_FOLDER, point_colors


def load_3d_points():
    """Load the 3D reconstructed pose and align the points so that the estimated ground plane is y=0."""
    points_3d = np.load(DATA_FOLDER / 'reconstructed_3d_points.npy')
    seed_mesh_poses = np.load(DATA_FOLDER / 'seed_mesh_poses.npy')
    with open(DATA_FOLDER / 'ground_plane.pkl', 'rb') as f:
        ground_plane_data = pickle.load(f)
    with open(DATA_FOLDER / 'reconstructed_seed_mesh.pkl', 'rb') as f:
        reconstructed_seed_mesh = pickle.load(f)
    # Adjust 3d points for ground
    plane_model = ground_plane_data['plane_model']
    normal, d = plane_model['normal'], plane_model['d']
    # Ensure normal vector is a unit vector and points "up"
    normal = normal / np.linalg.norm(normal)
    if normal[1] < 0:
        normal = -normal
        d = -d
    new_y_axis = normal
    # Determine and apply rotation to the 2D array of vectors
    source_vector = new_y_axis
    target_vector = np.array([0, 1, 0])
    rotation_obj, _ = Rotation.align_vectors([target_vector], [source_vector])
    origin_on_plane = -d * normal
    points_translated = points_3d - origin_on_plane  # Translate points so new origin is at (0,0,0)
    original_shape = points_translated.shape
    points_as_list_of_vectors = points_translated.reshape(-1, 3)
    rotated_points = rotation_obj.apply(points_as_list_of_vectors)
    points_3d_aligned = rotated_points.reshape(original_shape)  # (num_frames, num_keypoints, 3)
    return points_3d_aligned, reconstructed_seed_mesh, seed_mesh_poses, rotation_obj, origin_on_plane

def plot_poses_for_frame(frame_num, points_3d_aligned):
    """3D interactive plot showing the skeleton keypoints, and
    estimated ground plane for a given frame."""
    # Obtain data to plot
    frame_data = points_3d_aligned[frame_num, :, :]
    valid_indices = ~np.isnan(frame_data).any(axis=1)
    frame_data_valid = frame_data[valid_indices]
    point_names_valid = [POINT_NAMES[i] for i, v in enumerate(valid_indices) if v]
    colors_for_plot = point_colors / 255.0
    colors_valid = colors_for_plot[valid_indices]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Plot ground plane (y=0)
    x_min, x_max = np.nanmin(frame_data[:, 0]) - 0.5, np.nanmax(frame_data[:, 0]) + 0.5
    z_min, z_max = np.nanmin(frame_data[:, 2]) - 0.5, np.nanmax(frame_data[:, 2]) + 0.5
    xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    # Plot 3D keypoints
    ax.scatter(frame_data_valid[:, 0], frame_data_valid[:, 1], frame_data_valid[:, 2],
            c=colors_valid, marker='o', s=60, edgecolor='black', depthshade=True)
    # Plot skeleton connections
    for i, p_name in enumerate(point_names_valid):
        start_point_idx = POINT_NAMES.index(p_name)
        start_point_coords = frame_data[start_point_idx]
        for end_point_name in SKELETON[p_name]:
            end_point_idx = POINT_NAMES.index(end_point_name)
            if valid_indices[end_point_idx]:
                end_point_coords = frame_data[end_point_idx]
                ax.plot(*zip(start_point_coords, end_point_coords), color='dimgray')
    # Add text labels
    for i, name in enumerate(point_names_valid):
        x, y, z = frame_data_valid[i, 0], frame_data_valid[i, 1], frame_data_valid[i, 2]
        ax.text(x, y, z, f'  {name}', color='black', fontsize=8)
    ax.set_title(f'3D poses for frame {frame_num}', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y (height from ground)', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.view_init(elev=30, azim=-75)
    ax.set_box_aspect([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    plt.show()

def plot_seed_pose_for_frame(frame_num, reconstructed_seed_mesh, seed_mesh_poses):
    """Plot 3D mesh of seed for a given frame."""
    base_seed_vertices = reconstructed_seed_mesh['points']
    faces = reconstructed_seed_mesh['faces']
    pose = seed_mesh_poses[frame_num]
    tvec, rvec = pose[:3], pose[3:]
    R, _ = cv2.Rodrigues(rvec)
    posed_seed_vertices = (R @ base_seed_vertices.T).T + tvec
    fig = go.Figure(data=[go.Mesh3d(
        x=posed_seed_vertices[:, 0],
        y=posed_seed_vertices[:, 1],
        z=posed_seed_vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=1.0,
        name=f'Seed at frame {frame_num}'
    )])
    fig.update_layout(
        title_text=f'Seed pose for frame {frame_num}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' 
        )
    )
    fig.show()

def plot_antennae_height(points_3d_aligned):
    """2D plot of the height of left and right antennae tips' from the ground."""
    left_antenna_idx = POINT_NAMES.index("a_L2")
    right_antenna_idx = POINT_NAMES.index("a_R2")
    height_L = points_3d_aligned[:, left_antenna_idx, 1]
    height_R = points_3d_aligned[:, right_antenna_idx, 1]
    frames = np.arange(points_3d_aligned.shape[0]) # x-axis
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(frames, height_L, label='Left tip (a_L2)', color='royalblue', alpha=0.9)
    ax.plot(frames, height_R, label='Right tip (a_R2)', color='crimson', alpha=0.9)
    ax.set_title('Height of antennae tips from ground')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Height')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, frames.shape[0])
    ax.axhline(0, color='gray', linestyle='-', linewidth=1) # Ground plane
    ax.text(ax.get_xlim()[1], -0.15, 'Estimated ground plane', 
            color='gray',
            verticalalignment='bottom', 
            horizontalalignment='right')    
    plt.show()

def plot_distance_between_mandibles(points_3d_aligned):
    """2D plot of Euclidean distance between the left and right mandibles."""
    left_mandible_idx = POINT_NAMES.index("m_L1")
    right_mandible_idx = POINT_NAMES.index("m_R1")
    coords_L = points_3d_aligned[:, left_mandible_idx, :]
    coords_R = points_3d_aligned[:, right_mandible_idx, :]
    distance = np.linalg.norm(coords_L - coords_R, axis=1)
    frames = np.arange(points_3d_aligned.shape[0]) # x-axis
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, distance, color='black', alpha=1)
    ax.set_title('Distance between mandibles')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, frames.shape[0])
    plt.show()

def plot_eye_distance(points_3d_aligned):
    """2D plot of Euclidean distance between left and right eyes."""
    left_eye_idx = POINT_NAMES.index("eye_L")
    right_eye_idx = POINT_NAMES.index("eye_R")
    coords_L = points_3d_aligned[:, left_eye_idx, :]
    coords_R = points_3d_aligned[:, right_eye_idx, :]
    distance = np.linalg.norm(coords_L - coords_R, axis=1)
    frames = np.arange(points_3d_aligned.shape[0]) # x-axis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, distance, color='black', alpha=1)
    ax.set_title('Distance between eyes')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, frames.shape[0])
    plt.show()

def plot_distance_between_antennae(points_3d_aligned):
    """2D plot of Euclidean distance between left and right antennae tips."""
    left_antenna_idx = POINT_NAMES.index("a_L2")
    right_antenna_idx = POINT_NAMES.index("a_R2")
    coords_L = points_3d_aligned[:, left_antenna_idx, :]
    coords_R = points_3d_aligned[:, right_antenna_idx, :]
    distance = np.linalg.norm(coords_L - coords_R, axis=1)
    frames = np.arange(points_3d_aligned.shape[0]) # x-axis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, distance, color='black', alpha=1)
    ax.set_title('Distance between antennae tips')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, frames.shape[0])
    plt.show()

def plot_ant_seed_orientation(points_3d_aligned):
    """2D plot of the relative angle between the ant's body axis
    (thorax-to-neck) and the seed's axis (s_large-to-s_small)."""
    thorax_coords = points_3d_aligned[:, POINT_NAMES.index("thorax"), :]
    neck_coords = points_3d_aligned[:, POINT_NAMES.index("neck"), :]
    s_large_coords = points_3d_aligned[:, POINT_NAMES.index("s_large"), :]
    s_small_coords = points_3d_aligned[:, POINT_NAMES.index("s_small"), :]

    # Creat direction vectors for each frame
    ant_vector = neck_coords - thorax_coords
    seed_vector = s_small_coords - s_large_coords
    # Calculate the angle between the two vectors for each frame
    dot_product = np.einsum('ij,ij->i', ant_vector, seed_vector)
    norm_ant = np.linalg.norm(ant_vector, axis=1)
    norm_seed = np.linalg.norm(seed_vector, axis=1)
    cos_angle = np.clip(dot_product / (norm_ant * norm_seed), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    frames = np.arange(points_3d_aligned.shape[0]) # x-axis

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, angle_deg, color='black', alpha=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], 0, ' Parallel', color='gray', va='bottom', ha='right')
    ax.axhline(90, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], 90, ' Perpendicular', color='gray', va='bottom', ha='right')
    ax.set_title("Angle between ant's heading and seed's axis")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, frames.shape[0])
    ax.set_ylim(0, 180)
    plt.show()

def plot_antennae_to_seed_surface_distance(points_3d_aligned, reconstructed_seed_mesh, seed_mesh_poses, rotation_object, origin_on_plane):
    """Calculate and plot the shortest distance from the antennae tips to the
    surface of the 3D seed mesh."""
    left_antenna_idx = POINT_NAMES.index("a_L2")
    right_antenna_idx = POINT_NAMES.index("a_R2")
    num_frames = points_3d_aligned.shape[0]
    base_seed_vertices = reconstructed_seed_mesh['points']
    distances_L, distances_R = np.full(num_frames, np.nan), np.full(num_frames, np.nan)
    for f in range(num_frames):
        ant_L_coord = points_3d_aligned[f, left_antenna_idx, :]
        ant_R_coord = points_3d_aligned[f, right_antenna_idx, :]
        if np.isnan(ant_L_coord).any() or np.isnan(ant_R_coord).any():
            continue # Skip frame where data is missing

        # Get seed mesh's position in the aligned coordinate system
        ### Get the seed's pose (tvec, rvec) in the original world system
        pose = seed_mesh_poses[f]
        tvec, rvec = pose[:3], pose[3:]
        R, _ = cv2.Rodrigues(rvec)
        ### Apply this pose to the base seed mesh to position it in the world
        posed_seed_vertices = (R @ base_seed_vertices.T).T + tvec
        ### Apply the ground-plane alignment to these posed vertices
        translated_seed_vertices = posed_seed_vertices - origin_on_plane
        aligned_seed_vertices = rotation_object.apply(translated_seed_vertices)
        
        # Calculate distance from antenna to the aligned mesh surface
        if aligned_seed_vertices.shape[0] == 0: continue
        seed_tree = KDTree(aligned_seed_vertices)  # Nearest neighbour search
        dist_L, _ = seed_tree.query(ant_L_coord)  # Find the closest vertex
        dist_R, _ = seed_tree.query(ant_R_coord)
        distances_L[f] = dist_L
        distances_R[f] = dist_R

    frames = np.arange(num_frames)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, distances_L, label='Left tip (a_L2)', color='royalblue')
    ax.plot(frames, distances_R, label='Right tip (a_R2)', color='crimson')
    ax.set_title('Distance from antennae tips to seed surface')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance to seed')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, num_frames)
    ax.set_ylim(bottom=0)
    plt.show()

def plot_ant_seed_interaction(
    points_3d_aligned,
    reconstructed_seed_mesh,
    seed_mesh_poses,
    rotation_obj,
    origin_on_plane,
    contact_threshold=0.3,
    frame_range=(0, -1),
):
    """3D plot showing the interaction between the ant and the seed:
        1. The seed mesh, oriented in its local coordinate frame.
        2. The trajectories of the left and right mandibles relative to the seed.
        3. The trajectories of the left and right antennae tips relative to the seed.
        4. The specific points on the seed surface where the mandibles made contact.
        5. The specific points on the seed surface where the antennae made contact.
    Data is colored by frame number, and the entire scene is shifted to rest on a y=0 ground plane."""
    if frame_range[1] == -1:
        frame_range = (frame_range[0], len(points_3d_aligned))
    start_idx, end_idx = frame_range
    indices = {
        name: POINT_NAMES.index(point_name)
        for name, point_name in [
            ("mand_L", "m_L1"), ("mand_R", "m_R1"),
            ("ant_L", "a_L2"), ("ant_R", "a_R2")
        ]
    }
    positions = { "mand_L": [], "mand_R": [], "ant_L": [], "ant_R": [] }
    frames = { "mand_L": [], "mand_R": [], "ant_L": [], "ant_R": [] }
    contacts = { "mand_L": {}, "mand_R": {}, "ant_L": {}, "ant_R": {} }
    base_seed_vertices, base_seed_faces = reconstructed_seed_mesh['points'], reconstructed_seed_mesh['faces']
    for i in range(start_idx, end_idx):
        pose = seed_mesh_poses[i]
        tvec_orig, rvec_orig = pose[:3], pose[3:]
        R_orig, _ = cv2.Rodrigues(rvec_orig)
        R_aligned = rotation_obj.as_matrix() @ R_orig
        T_aligned = rotation_obj.apply(tvec_orig - origin_on_plane)
        R_aligned_inv = R_aligned.T
        posed_seed_vertices_aligned = (R_aligned @ base_seed_vertices.T).T + T_aligned
        seed_mesh_trimesh = trimesh.Trimesh(vertices=posed_seed_vertices_aligned, faces=base_seed_faces)
        for part in ["mand_L", "mand_R", "ant_L", "ant_R"]:
            keypoint_aligned = points_3d_aligned[i, indices[part], :]
            if np.isnan(keypoint_aligned).any():
                continue
            relative_pos = R_aligned_inv @ (keypoint_aligned - T_aligned)
            positions[part].append(relative_pos)
            frames[part].append(i)
            closest_pt, dist, _ = seed_mesh_trimesh.nearest.on_surface([keypoint_aligned])
            if dist[0] < contact_threshold:
                contact_pt_base = R_aligned_inv @ (closest_pt[0] - T_aligned)
                key = tuple(np.round(contact_pt_base, decimals=4))
                contacts[part][key] = i
    if all(len(v) == 0 for v in positions.values()):
        print("No valid data found in the specified frame range to plot.")
        return

    min_y_seed = base_seed_vertices[:, 1].min()
    shift_vector = np.array([0, -min_y_seed, 0])
    shifted_seed_vertices = base_seed_vertices + shift_vector
    shifted_positions = {part: np.array(pos_list) + shift_vector for part, pos_list in positions.items()}
    shifted_contacts = {part: (np.array(list(cont_dict.keys())) + shift_vector if cont_dict else np.empty((0,3))) for part, cont_dict in contacts.items()}
    frames_contacts = {part: np.array(list(cont_dict.values())) for part, cont_dict in contacts.items()}

    fig = go.Figure()
    title = f"Ant-seed interaction (frames {start_idx}-{end_idx})"
    fig.add_trace(go.Mesh3d(
        x=shifted_seed_vertices[:, 0], y=shifted_seed_vertices[:, 1], z=shifted_seed_vertices[:, 2],
        i=base_seed_faces[:, 0], j=base_seed_faces[:, 1], k=base_seed_faces[:, 2],
        color='lightgray', opacity=0.6, name='Seed',
        hoverinfo='none'
    ))
    plot_props = {
        'mand_L': {'pos_color': 'Greens', 'cont_color': 'springgreen', 'name': 'Left mandible', 'line_color': 'green'},
        'mand_R': {'pos_color': 'Purples', 'cont_color': 'magenta', 'name': 'Right mandible', 'line_color': 'purple'},
        'ant_L': {'pos_color': 'Blues', 'cont_color': 'deepskyblue', 'name': 'Left antenna', 'line_color': 'blue'},
        'ant_R': {'pos_color': 'Reds', 'cont_color': 'red', 'name': 'Right antenna', 'line_color': 'darkred'}
    }
    hover_template = (
        '<b>Frame</b>: %{customdata}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>'
    )
    for part, props in plot_props.items():
        # Trajectory / position plot
        fig.add_trace(go.Scatter3d(
            x=shifted_positions[part][:, 0], y=shifted_positions[part][:, 1], z=shifted_positions[part][:, 2],
            mode='lines+markers',
            name=f'{props["name"]} position',
            customdata=frames[part],
            hovertemplate=hover_template,
            line=dict(color=props['line_color'], width=3),
            marker=dict(size=2.5, color=frames[part], colorscale=props['pos_color'], showscale=False)
        ))
        # Contact points plot
        fig.add_trace(go.Scatter3d(
            x=shifted_contacts[part][:, 0], y=shifted_contacts[part][:, 1], z=shifted_contacts[part][:, 2],
            mode='markers',
            name=f'{props["name"]} contact',
            customdata=frames_contacts[part],
            hovertemplate=hover_template,
            marker=dict(size=5, color=props['cont_color'], symbol='diamond',
                        line=dict(color='black', width=1))
        ))
    padding = 2.0
    all_x = np.concatenate([p[:, 0] for p in shifted_positions.values() if p.size > 0])
    all_z = np.concatenate([p[:, 2] for p in shifted_positions.values() if p.size > 0])
    x_min, x_max = np.min(all_x) - padding, np.max(all_x) + padding
    z_min, z_max = np.min(all_z) - padding, np.max(all_z) + padding
    ground_x, ground_z = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(z_min, z_max, 2))
    ground_y = np.zeros_like(ground_x)
    fig.add_trace(go.Surface(
        x=ground_x, y=ground_y, z=ground_z,
        colorscale=[[0, 'darkgray'], [1, 'darkgray']],
        showscale=False, opacity=0.3, name='Ground plane',
        hoverinfo='none'
    ))
    fig.update_layout(
        title_text=title,
        scene=dict(xaxis_title='X', yaxis_title='Y (height)', zaxis_title='Z', aspectmode='data', camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=1.5, z=1.5))),
        legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
    # fig.write_html("ant_seed_interactions.html")

if __name__ == "__main__":
    # Load data
    points_3d_aligned, reconstructed_seed_mesh, seed_mesh_poses, rotation_obj, origin_on_plane = load_3d_points()
    total_frames = len(points_3d_aligned)

    # Plots
    # plot_poses_for_frame(300, points_3d_aligned)
    # plot_seed_pose_for_frame(300, reconstructed_seed_mesh, seed_mesh_poses)
    # plot_antennae_height(points_3d_aligned)
    # plot_distance_between_mandibles(points_3d_aligned)
    # plot_eye_distance(points_3d_aligned)
    # plot_distance_between_antennae(points_3d_aligned)
    # plot_ant_seed_orientation(points_3d_aligned)
    # plot_antennae_to_seed_surface_distance(points_3d_aligned=points_3d_aligned,
    #                                        reconstructed_seed_mesh=reconstructed_seed_mesh,
    #                                        seed_mesh_poses=seed_mesh_poses,
    #                                        rotation_object=rotation_obj,
    #                                        origin_on_plane=origin_on_plane)
    plot_ant_seed_interaction(
        points_3d_aligned,
        reconstructed_seed_mesh,
        seed_mesh_poses,
        rotation_obj,
        origin_on_plane,
        contact_threshold=0.2,
        frame_range=(400, 1000),
    )    