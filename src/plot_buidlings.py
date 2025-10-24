import trimesh
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.collections import LineCollection
import alphashape
from shapely.geometry import Polygon, MultiPolygon


def load_models(model_paths):
    """Load 3D models from the given file paths."""
    return [trimesh.load(path) for path in model_paths]

def get_plane(slice_plane, slice_value):
    """Define the slicing plane normal and point."""
    if slice_plane == 'xy':
        normal = [0, 0, 1]
        point  = [0, 0, slice_value]
    elif slice_plane == 'yz':
        normal = [1, 0, 0]
        point  = [slice_value, 0, 0]
    elif slice_plane == 'xz':
        normal = [0, 1, 0]
        point  = [0, slice_value, 0]
    else:
        raise ValueError(f"Unknown slice_plane: {slice_plane}")
    return normal, point


def clean_and_slice_model(mesh_or_scene, plane_origin, plane_normal):
    """Clean the mesh and perform slicing."""

    # Handle Scene objects by extracting the mesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        # Get the first mesh from the scene
        if len(mesh_or_scene.geometry) == 0:
            raise ValueError("Scene contains no geometry")

        # If there are multiple meshes, combine them or take the first one
        if len(mesh_or_scene.geometry) == 1:
            mesh = list(mesh_or_scene.geometry.values())[0]
        else:
            # Combine multiple meshes into one
            meshes = list(mesh_or_scene.geometry.values())
            mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = mesh_or_scene

    # Rest of your original code...
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        print("No intersection found with the slicing plane.")
        return None

    filtered_paths = []
    for path in section.discrete:
        path = np.array(path)
        if len(path) > 2:
            filtered_paths.append(path)
        else:
            print(f"Skipping path with insufficient points: {path}")

    return filtered_paths


def plot_2d_slices(models, building, slice_plane, slice_value, names):
    """Plot 2D slices of the models on the specified plane using a colormap and custom legend names."""
    if len(models) != len(names):
        raise ValueError("The length of the names list must match the number of models.")
    print(len(models))

    normal, point = get_plane(slice_plane, slice_value)
    plt.figure(figsize=(8, 8))

    num_models = len(models)
    colormap = cm.batlowS  # Use the colormap
    colors = [colormap(i / (num_models - 1)) for i in range(num_models)]  # Map indices to colors

    for i, model in enumerate(models):
        paths = clean_and_slice_model(model, plane_origin=point, plane_normal=normal)
        if not paths:
            print(f"No valid intersection for model {i+1} at {slice_plane}={slice_value}")
            continue
        segments = []
        min_path_length = 10
        raw_segments = []

        for path in paths:
            if path.shape[0] < min_path_length:
                print(f"skipping path for model {i}")
                continue

            if slice_plane == 'xy':
                points = path[:, [0, 1]]
            elif slice_plane == 'yz':
                points = path[:, [1, 2]]
            elif slice_plane == 'xz':
                points = path[:, [0, 2]]
            else:
                continue

            raw_segments.append(points)

            segments = []
            seen = {}

            # Save index of last occurrence for each (start, end) pair
            for idx, seg in enumerate(raw_segments):
                key1 = (tuple(seg[0]), tuple(seg[-1]))
                key2 = (tuple(seg[-1]), tuple(seg[0]))  # reversed path
                seen[key1] = idx
                seen[key2] = idx  # update both directions

            # Keep only segments whose index matches the last occurrence
            for idx, seg in enumerate(raw_segments):
                key = (tuple(seg[0]), tuple(seg[-1]))
                if seen[key] == idx:
                    segments.append(seg)
                else:
                    print(f"Skipping earlier duplicate at index {idx}")

        # Only add if we have valid segments
        if segments:
            lc = LineCollection(segments, colors=[colors[i]], linewidths=0.5, label=names[i])
            plt.gca().add_collection(lc)


    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    ax = plt.gca()
    ax.autoscale_view()

    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    x_min, x_max = x_range
    plot_width = abs(x_max - x_min)

    plt.ylim(0, plot_width)

    y_max = y_range[1] / plot_width  # Used for axvline box calculation


    plt.axhline(y=y_range[1], color='black', linewidth=0.7)
    plt.axhline(y=0, color='black', linewidth=1.1)
    plt.axvline(x=x_range[0], ymin=0, ymax=y_max, color='black', linewidth=0.8)
    plt.axvline(x=x_range[1], ymin=0, ymax=y_max, color='black', linewidth=1.1)

    plt.title(f"Slice on {slice_plane.upper()} plane at {slice_value}")
    plt.xlabel(slice_plane[0].upper())
    plt.ylabel(slice_plane[1].upper())
    ax.yaxis.set_label_coords(-0.05, y_max/2)

    y_max = np.abs(x_range[1] - x_range[0])
    plt.ylim(0, y_max)
    plt.legend()
    plt.savefig(f'../figures/buildings/{building}/slice_{slice_plane}_{slice_value}.pdf', format='pdf', bbox_inches='tight')

    plt.show()

def plot_3d_models(models, building):
    """Plot 3D models using Plotly."""
    fig = go.Figure()

    num_models = len(models)
    if num_models == 1:
        colors = [cm.grayC(0.75)]
    else:
        colors = [cm.batlowS(i / (num_models - 1)) for i in range(num_models)]

    for i, model in enumerate(models):
        v = model.vertices
        faces = model.faces
        color = f"rgba({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)}, {colors[i][3]})"
        opacities = [1, 0.8, 0.5, 0.3]  # Define opacity for each building
        opacity = opacities[i] if i < len(opacities) else 1  # Use default opacity if not specified
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=opacity, color=color,  name=f'Level {i+1}'
        ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',  # Ensures the aspect ratio matches the data
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera position
            ),
            zaxis=dict(
                range=[0, None],  # Limit z-axis to 0 and above
                showbackground=True,
            )
        )
    )
    fig.show()

    pio.write_image(fig, f'../figures/buildings/{building}/3d_model.pdf', format="pdf", width=2000, height=1600)

from scipy.spatial import ConvexHull

def plot_topview_roof_convex(model, building, roof_color='lightgray', axis_lines=None, alpha=1.0):
    """
    Plot a clean top-view of the roof:
    - Use triangles connected to ridge vertices
    - Compute XY convex hull for roof outline
    - Fill convex hull with roof_color
    - Plot ridge line
    - No triangulated surfaces are shown
    """
    import matplotlib.pyplot as plt
    import numpy as np

    verts = model.vertices
    faces = model.faces

    # 1. Ridge vertices (top 5% Z)
    z_values = verts[:, 2]
    threshold_z = np.percentile(z_values, 99.7)  # 95th percentile
    ridge_indices = np.where(z_values >= threshold_z)[0]
    ridge_vertices = verts[ridge_indices][:, :2]  # XY projection

    # 2. Find faces connected to ridge vertices
    mask = np.any(np.isin(faces, ridge_indices), axis=1)
    roof_faces = faces[mask]

    # 3. Collect all vertices of these faces (project to XY)
    roof_vertices = verts[roof_faces].reshape(-1, 3)[:, :2]

    # 4. Compute concave hull of roof vertices
    if len(roof_vertices) >= 3:
        concave_hull = alphashape.alphashape(roof_vertices, alpha)
        if isinstance(concave_hull, Polygon):
            hull_points = np.array(concave_hull.exterior.coords)
        elif isinstance(concave_hull, MultiPolygon):
            hull_points = []
            for polygon in concave_hull.geoms:
                hull_points.append(np.array(polygon.exterior.coords))
    else:
        hull_points = roof_vertices

    plt.figure(figsize=(8, 8))

    # 5. Fill convex hull
    plt.fill(hull_points[:, 0], hull_points[:, 1], color=roof_color, alpha=0.7, zorder=1, edgecolor='black')

    # 6. Plot ridge line (sorted by X)
    if len(ridge_vertices) > 1:
        ridge_sorted = ridge_vertices[ridge_vertices[:, 0].argsort()]
        plt.plot(ridge_sorted[:, 0], ridge_sorted[:, 1], color='black', linewidth=2, zorder=2, label='Ridge line')

    # 7. Add custom axis lines
    if axis_lines:
        for line in axis_lines:
            orientation = line.get('orientation', 'x')
            position = line.get('position', 0)
            label = line.get('label', '')

            if orientation == 'x':
                # Place label at the right end of the x-range of the roof
                x_end = hull_points[:, 0].max() + 0.5  # or min() for left end
                plt.axhline(y=position, color='black', linestyle='--', linewidth=1.2, zorder=3)
                plt.text(x_end, position + 0.1, label, color='black', fontsize=12,
                         va='bottom', ha='right', zorder=4, weight='bold')

            elif orientation == 'y':
                # Place label at the top end of the y-range of the roof
                y_end = hull_points[:, 1].max() + 0.5  # or min() for bottom
                plt.axvline(x=position, color='black', linestyle='--', linewidth=1.2, zorder=3)
                plt.text(position + 0.1, y_end, label, color='black', fontsize=12,
                         va='top', ha='left', zorder=4, weight='bold')


    plt.gca().set_aspect('equal', adjustable='datalim')
    xmin, xmax = plt.gca().get_xlim()
    plt.xlim(xmin - 1, xmax + 1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Top-view Roof via Convex Hull and Ridge Line")
    plt.savefig(f'../figures/{building}/topview.pdf', dpi=300)
    plt.show()



def main():
    # Variables to modify
    slice_plane = 'xz'  # 'xy', 'yz', or 'xz'
    slice_value = 0   # Location of the slice along the perpendicular axis
    building = "A-40"
    model_paths = [
        f"../buildings/{building}/{building}_truth.obj",
        f"../buildings/{building}/{building}_1m.obj",
        f"../buildings/{building}/{building}_05m.obj",
        f"../buildings/{building}/{building}_025m.obj",
        f"../buildings/{building}/{building}_01m.obj",
        f"../buildings/{building}/{building}_005m.obj",
        f"../buildings/{building}/{building}_0025m.obj",
    ]
    names = ["Truth model", "1m voxel", "0.5m voxel", "0.25m voxel", "0.1m voxel", "0.05m voxel", "0.025m voxel"]

    # Load models
    models = load_models(model_paths)

    # Plot 2D slices
    plot_2d_slices(models, building, slice_plane, slice_value, names=names)

    # Plot 3D models
    plot_3d_models([models[3]], building)

    axis_lines = [
        {'orientation': 'x', 'position': 0, 'label': 'A'},
        {'orientation': 'y', 'position': 0, 'label': 'B'}
    ]

    #plot_topview_roof_convex(models[0], building, roof_color='lightgray', axis_lines=axis_lines)

if __name__ == "__main__":
    main()