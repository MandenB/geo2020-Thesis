import os
import numpy as np
import numpy.ma as ma
from collections import defaultdict
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
from cmcrameri import cm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import trimesh
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class ErrorMetrics:
    """Container for all error metrics"""
    coords: np.ndarray
    error_norm: np.ndarray
    relative_error: np.ndarray
    error_vec: np.ndarray
    normalized_error: np.ndarray
    mean_error: float


# === CONFIG ==
"""Configure the required parameters"""
buildings = ["A-40", "FZK_Haus", "Institute"]
data_dir = fr"..\Data"
default_obj_path = fr"..\buildings"
sizes = ["1m", "05m", "025m", "01m", "005m", "0025m", "truth"]
angles = ["90_degrees", "45_degrees", "22.5_degrees", "360_degrees"]

# === STORAGE ===
data = defaultdict(lambda: defaultdict(lambda: defaultdict(np.ndarray)))


def read_file(filepath):
    """Reads a file and returns numpy array of shape (N, 6)"""
    return np.loadtxt(filepath)


def get_filepath(building, size, angle):
    return os.path.join(data_dir, building, "results", size, angle, "postProcessing\\pointCloud\\2000\\pointCloud_U.xy")


def load_all_data():
    """Load all data files into memory"""
    for building in buildings:
        for size in sizes:
            for angle in angles:
                if os.path.isdir(os.path.join(data_dir, building, "results", size, angle)):
                    filepath = get_filepath(building, size, angle)
                    if os.path.exists(filepath):
                        print(f"Loading {filepath}")
                        data[building][angle][size] = read_file(filepath)
                    else:
                        print(f"Missing file: {filepath}")
                else:
                    print(f"Missing directory: {os.path.join(data_dir, building, 'results', size, angle)}")


def find_overlapping_points(truth_coords, truth_vel, pred_coords, pred_vel, decimal_round=9):
    """
    Find overlapping points between truth and prediction data.

    Returns:
        tuple: (common_coords, truth_velocities, pred_velocities)
    """
    # Create hashes for (x,y,z) coordinates
    truth_hash = {tuple(coord): i for i, coord in enumerate(truth_coords)}
    pred_hash = {tuple(coord): i for i, coord in enumerate(pred_coords)}

    # Find common coordinates
    common_keys = list(set(truth_hash.keys()) & set(pred_hash.keys()))

    if not common_keys:
        return None, None, None

    # Sort for consistency
    common_keys.sort()

    # Get indices and extract corresponding data
    indices_truth = [truth_hash[k] for k in common_keys]
    indices_pred = [pred_hash[k] for k in common_keys]

    coords_common = np.array(common_keys)
    true_vel = truth_vel[indices_truth]
    pred_vel = pred_vel[indices_pred]

    return coords_common, true_vel, pred_vel


def compute_error_metrics(true_vel, pred_vel, Uref=5):
    """
    Compute all error metrics given truth and predicted velocities.

    Returns:
        ErrorMetrics object containing all computed metrics
    """
    error_vec = pred_vel - true_vel
    error_norm = np.linalg.norm(error_vec, axis=1)
    truth_mag = np.linalg.norm(true_vel, axis=1)
    relative_error = error_norm / (truth_mag + 1e-8)

    error_mag_signed = (np.linalg.norm(pred_vel, axis=1) -
                        np.linalg.norm(true_vel, axis=1))
    normalized_error = np.clip(error_mag_signed / Uref, -1, 1)
    mean_error = np.mean(error_norm)

    return error_norm, relative_error, error_vec, normalized_error, mean_error


def compute_all_errors(building, z_value=None, decimal_round=9):
    """
    Compute errors for all angle/size combinations, optionally filtering by z-value.

    Returns:
        dict: Nested dict with structure [angle][size] = ErrorMetrics
    """
    all_errors = defaultdict(dict)

    for angle in angles:
        # Load truth data
        truth = data[building][angle]["truth"]
        if truth is None or len(truth) == 0:
            continue

        truth_coords = np.round(truth[:, :3], decimals=decimal_round)
        truth_vel = truth[:, 3:]

        # Filter by z-value if specified
        if z_value is not None:
            mask_truth = np.isclose(truth_coords[:, 2], z_value, atol=1e-6)
            truth_coords = truth_coords[mask_truth]
            truth_vel = truth_vel[mask_truth]

        for size in sizes:
            if size == "truth":
                continue

            pred = data[building][angle][size]
            if pred is None or len(pred) == 0:
                continue

            pred_coords = np.round(pred[:, :3], decimals=decimal_round)
            pred_vel = pred[:, 3:]

            # Filter by z-value if specified
            if z_value is not None:
                mask_pred = np.isclose(pred_coords[:, 2], z_value, atol=1e-6)
                pred_coords = pred_coords[mask_pred]
                pred_vel = pred_vel[mask_pred]

            # Find overlapping points
            coords_common, true_vel_common, pred_vel_common = find_overlapping_points(
                truth_coords, truth_vel, pred_coords, pred_vel, decimal_round
            )

            if coords_common is None:
                if z_value is not None:
                    print(f"⚠️ No overlap at z={z_value} in {angle}/{size}")
                else:
                    print(f"⚠️ No overlap in {angle}/{size}")
                continue

            # Compute all error metrics
            error_norm, relative_error, error_vec, normalized_error, mean_error = compute_error_metrics(
                true_vel_common, pred_vel_common
            )

            # Store in ErrorMetrics object
            all_errors[angle][size] = ErrorMetrics(
                coords=coords_common,
                error_norm=error_norm,
                relative_error=relative_error,
                error_vec=error_vec,
                normalized_error=normalized_error,
                mean_error=mean_error
            )

    return all_errors


def get_unique_z_values(building):
    """Extract all unique z-values from the loaded data."""
    z_values = set()
    for angle in angles:
        for size in sizes:
            if building in data and angle in data[building] and size in data[building][angle]:
                array = data[building][angle][size]
                if array is not None and len(array) > 0:
                    z_values.update(np.unique(array[:, 2]))
    return sorted(z_values)


def create_building_mask_from_obj(obj_path, z_value, resolution=0.1):
    """
    Load OBJ file and create a binary mask at a given z-slice.
    """
    mesh = trimesh.load(obj_path, force='mesh')

    if not mesh.is_watertight:
        print(f"Warning: Mesh at {obj_path} is not watertight. Results may be inaccurate.")

    bounds = mesh.bounds
    xmin, ymin, _ = bounds[0]
    xmax, ymax, _ = bounds[1]

    x_coords = np.arange(xmin, xmax, resolution)
    y_coords = np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z_value)]

    inside = mesh.contains(grid_points).reshape(X.shape)
    return inside.astype(int), (xmin, xmax, ymin, ymax)


def plot_error_contours(all_errors, z_value, levels=None):
    """Plot contour lines of error values for each voxel size, per angle."""
    color_map = {
        "025m": 'magenta',
        "05m": 'cyan',
        "1m": 'yellow',
    }

    for angle, angle_errors in all_errors.items():
        plt.figure(figsize=(10, 10))

        # Get building mask (assuming we can get obj_path)
        obj_path = os.path.join(default_obj_path, f"{buildings[0]}/{buildings[0]}_truth.obj")  # Adjust as needed
        building_mask, extent = create_building_mask_from_obj(obj_path, z_value)
        plt.imshow(building_mask, extent=extent, origin='lower', cmap='gray_r', alpha=1.0)

        for size, metrics in angle_errors.items():
            if size not in color_map:
                continue

            x, y = metrics.coords[:, 0], metrics.coords[:, 1]
            plt.tricontour(x, y, metrics.normalized_error,
                           levels=levels or 10, colors=color_map[size], linewidths=1.5)

        legend_elements = [
            Line2D([0], [0], color=color_map["025m"], lw=2, label='Voxel 0.25m'),
            Line2D([0], [0], color=color_map["05m"], lw=2, label='Voxel 0.5m'),
            Line2D([0], [0], color=color_map["1m"], lw=2, label='Voxel 1m'),
        ]
        plt.legend(handles=legend_elements, loc='lower left', frameon=True)
        plt.title(f"Error contours at Z={z_value} - {angle} with error at {levels}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.tight_layout()
        plt.savefig(f"figures/isolines/error_contours_{angle}_{z_value}_{levels}.pdf",
                    dpi=300, format="pdf")


def plot_error_isolines(coords, error, title, z_value, obj_path, vmax, vmin, output_dir):
    """Plot error isolines with building mask."""
    building_mask, extent = create_building_mask_from_obj(obj_path, z_value)
    masked_building_mask = ma.masked_where(building_mask == 0, building_mask)

    x, y = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(10.3, 8))
    plt.imshow(masked_building_mask, extent=extent, origin='lower',
               cmap='binary', alpha=1, zorder=2)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    step = (vmax - vmin) / ((vmax - vmin) * 20)
    levels = np.arange(vmin, vmax + step, step)

    tcf = plt.tricontourf(x, y, error, levels=levels, cmap=cm.vik, norm=norm, zorder=1)
    cbar = plt.colorbar(tcf, label='Normalized velocity difference')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(output_dir, dpi=300, format="pdf")
    plt.close()


def get_global_error_range(all_errors_by_z):
    """Calculate global min/max error values across all z-values."""
    all_errors = []
    for z_errors in all_errors_by_z.values():
        for angle_errors in z_errors.values():
            for metrics in angle_errors.values():
                all_errors.extend(metrics.normalized_error)

    if all_errors:
        return math.floor(min(all_errors)), math.ceil(max(all_errors))
    return 0, 1


def print_mean_errors(all_errors):
    """Print mean errors in a formatted way."""
    print("\n=== MEAN ERRORS ===")
    for angle, angle_errors in all_errors.items():
        print(f"\n{angle}:")
        for size, metrics in angle_errors.items():
            print(f"  mean for {size}: {metrics.mean_error:.6f}")
            print(f"  max for{size}: {np.max(metrics.error_norm):.6f}")
            print(f"  min for{size}: {np.min(metrics.error_norm):.6f}")


def load_timing_data(building, default_file_path=r"..\buildings"):
    """
    Load timing data for different model sizes.

    Args:
        building: Building name
        timing_file_path: Path to timing file. If None, tries default locations.

    Returns:
        dict: {size: time_in_seconds}
    """
    timing_file_path = os.path.join(default_file_path, building, f"times.txt")

    timing_data = {}
    if timing_file_path and os.path.exists(timing_file_path):
        try:
            with open(timing_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        parts = line.split()
                        if len(parts) >= 2:
                            size = parts[0]
                            time_val = float(parts[1])
                            timing_data[size] = time_val
        except Exception as e:
            print(f"Warning: Could not load timing data from {timing_file_path}: {e}")
    else:
        print(f"Warning: No timing data found for {building}")

    return timing_data


def plot_mean_errors_vs_size(buildings_data, timing_data_dict=None, output_dir="../figures/mean_errors",
                             angle_colormap='viridis', building_colormap='tab10'):
    """
    Plot mean error analysis with multiple buildings in combined plots.

    Args:
        buildings_data: List of dicts with structure [{"name": str, "errors": dict}]
                       where errors has structure [angle][size] = ErrorMetrics
        timing_data_dict: Dict with {building_name: {size: time_in_seconds}}, optional
        output_dir: Directory to save plots
        angle_colormap: Matplotlib colormap name for angle colors
        building_colormap: Matplotlib colormap name for building colors
    """
    # Use sizes from global config and create labels
    size_order = [s for s in sizes if s != "truth"]  # Exclude "truth"

    # Create size labels with proper decimal formatting
    size_labels = {}
    for size in size_order:
        if size.endswith('m'):
            # Remove 'm' and convert to proper decimal format
            size_num = size[:-1]  # Remove 'm'
            if size_num == "1":
                size_labels[size] = "1.0"
            elif size_num == "05":
                size_labels[size] = "0.5"
            elif size_num == "025":
                size_labels[size] = "0.25"
            elif size_num == "01":
                size_labels[size] = "0.1"
            elif size_num == "005":
                size_labels[size] = "0.05"
            else:
                # Fallback: try to parse as number
                try:
                    num = float(size_num) if '.' in size_num else float(size_num) / (10 ** (len(size_num) - 1))
                    size_labels[size] = f"{num}"
                except:
                    size_labels[size] = size.replace('m', '')
        else:
            size_labels[size] = size

    # Convert size labels to numeric values for proper logarithmic plotting
    size_values = {}
    for size in size_order:
        try:
            size_values[size] = float(size_labels[size])
        except:
            # Fallback if conversion fails
            size_values[size] = 1.0

    # Get colors for buildings
    building_cmap = plt.get_cmap(building_colormap)
    building_colors = {building['name']: building_cmap(i / max(1, len(buildings_data) - 1))
                       for i, building in enumerate(buildings_data)}

    # Get all unique angles across all buildings
    all_angles = set()
    for building in buildings_data:
        all_angles.update(building['errors'].keys())
    all_angles = sorted(list(all_angles))

    # Get colors for angles
    angle_cmap = plt.get_cmap(angle_colormap)
    angle_colors = {angle: angle_cmap(i / max(1, len(all_angles) - 1))
                    for i, angle in enumerate(all_angles)}

    os.makedirs(output_dir, exist_ok=True)

    # Find maximum error across all buildings and angles for consistent y-axis scaling
    max_error = 0
    for building in buildings_data:
        building_errors = building['errors']
        for angle, angle_errors in building_errors.items():
            for size, metrics in angle_errors.items():
                max_error = max(max_error, metrics.mean_error)

    # Add small buffer to max error
    y_max = max_error * 1.05

    # Plot 1: Mean Error vs Voxel Size (Split by Wind Direction)
    for angle in all_angles:
        plt.figure(figsize=(10, 6))

        angle_has_data = False

        for building in buildings_data:
            building_name = building['name']
            building_errors = building['errors']

            if angle not in building_errors:
                continue

            sizes_available = []
            errors = []
            size_nums = []

            for size in size_order:
                if size in building_errors[angle]:
                    sizes_available.append(size_labels[size])
                    size_nums.append(size_values[size])
                    errors.append(building_errors[angle][size].mean_error)

            if sizes_available:
                plt.plot(size_nums, errors, 'o-',
                         label=building_name,
                         color=building_colors[building_name],
                         linewidth=2, markersize=8)
                angle_has_data = True

        if angle_has_data:
            plt.xlabel('Voxel Size (m)', fontsize=12)
            plt.ylabel('Mean Velocity Difference (m/s)', fontsize=12)
            plt.title(f'Mean Velocity Difference vs Voxel Size - {angle.replace("_", " ").title()}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.ylim(0, y_max)  # Set consistent y-axis limits
            plt.tight_layout()
            plt.savefig(f"{output_dir}/all_buildings_{angle}_error_vs_size.pdf", dpi=300, format="pdf")

        plt.close()

    # Plot 2: Computation Time vs Voxel Size (All Buildings Combined)
    if timing_data_dict:
        plt.figure(figsize=(12, 8))

        for building in buildings_data:
            building_name = building['name']

            if building_name not in timing_data_dict:
                continue

            timing_data = timing_data_dict[building_name]

            # Plot truth time as horizontal background line if available
            if 'truth' in timing_data:
                plt.axhline(y=timing_data['truth'],
                            color=building_colors[building_name],
                            linestyle='-', linewidth=2, alpha=0.5,
                            zorder=1)

            sizes_with_timing = []
            times = []
            size_nums = []

            for size in size_order:
                if size in timing_data:
                    sizes_with_timing.append(size_labels[size])
                    size_nums.append(size_values[size])
                    times.append(timing_data[size])

            if sizes_with_timing:
                plt.plot(size_nums, times, 'o-',
                         label=building_name,
                         color=building_colors[building_name],
                         linewidth=2, markersize=8, zorder=2)

        plt.xlabel('Voxel Size (m)', fontsize=12)
        plt.ylabel('Computation Time (seconds)', fontsize=12)
        plt.title('Computation Time vs Voxel Size - All Buildings', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_buildings_time_vs_size.pdf", dpi=300, format="pdf")
        plt.close()

        # Plot 3: Mean Error vs Computation Time (keep individual building plots)
        for building in buildings_data:
            building_name = building['name']
            building_errors = building['errors']

            if building_name not in timing_data_dict:
                continue

            timing_data = timing_data_dict[building_name]

            plt.figure(figsize=(10, 6))

            for angle, angle_errors in building_errors.items():
                x_times = []  # computation times
                y_errors = []  # mean errors
                size_labels_plot = []

                for size in size_order:
                    if size in angle_errors and size in timing_data:
                        x_times.append(timing_data[size])
                        y_errors.append(angle_errors[size].mean_error)
                        size_labels_plot.append(size_labels[size])

                if x_times:
                    plt.plot(x_times, y_errors, 'o-',
                             label=f'{angle.replace("_", " ").title()}',
                             color=angle_colors.get(angle, 'gray'),
                             linewidth=2, markersize=8)

            plt.xlabel('Computation Time (seconds)', fontsize=12)
            plt.ylabel('Mean Error', fontsize=12)
            plt.title(f'Mean Error vs Computation Time - {building_name}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{building_name}_error_vs_time.pdf", dpi=300, format="pdf")
            plt.close()

    print(f"Mean error plots saved in {output_dir}/")


def main():
    load_all_data()

    # Collect data for all buildings
    buildings_data = []
    timing_data_dict = {}

    for building in buildings:
        # Get unique z-values
        unique_z_values = get_unique_z_values(building)
        print(f"\nAvailable z-values for {building}:", unique_z_values)

        # Get user input for z-values
        while True:
            try:
                user_input = input("Enter one or multiple z-values (comma-separated) from the list above: ")
                if user_input.strip() == "all":
                    user_values = unique_z_values
                    break
                else:
                    user_values = [float(value.strip()) for value in user_input.split(',')]
                    if all(value in unique_z_values for value in user_values):
                        break
                    else:
                        print("Invalid z-value(s). Please select values from the list.")
            except ValueError:
                print("Please enter valid numbers.")


        # Compute mean errors across all data (no z-filtering)
        print(f"Computing overall mean errors for {building}...")
        mean_errors = compute_all_errors(building)
        print_mean_errors(mean_errors)

        # Store building data for combined plotting
        buildings_data.append({
            "name": building,
            "errors": mean_errors
        })

        # Load timing data and store it
        print(f"Loading timing data for {building}...")
        timing_data = load_timing_data(building)
        timing_data_dict[building] = timing_data

        # Compute errors for each z-value (rest of your existing code)
        print("Computing errors for each z-value...")
        all_errors_by_z = {}
        for z_value in user_values:
            print(f"Processing z={z_value}...")
            all_errors_by_z[z_value] = compute_all_errors(building, z_value)

        # Get global error range for consistent plotting
        global_min, global_max = get_global_error_range(all_errors_by_z)

        # Generate plots for each z-value
        for z_value, z_errors in all_errors_by_z.items():
            print(f"Generating plots for z={z_value}...")

            # Plot contours (uncomment if needed)
            # plot_error_contours(z_errors, z_value, levels=[0.2])

            # Plot heatmaps
            for angle, angle_errors in z_errors.items():
                for size, metrics in angle_errors.items():
                    obj_path = os.path.join(default_obj_path, f"{building}/{building}_{size}.obj")
                    os.makedirs(f"figures/heatmap/{building}/{z_value}/{angle}", exist_ok=True)
                    output_dir = f"figures/heatmap/{building}/{z_value}/{angle}/{size}.pdf"

                    plot_error_isolines(
                        metrics.coords,
                        metrics.normalized_error,
                        f"{angle} - {size} error at z={z_value}",
                        z_value,
                        obj_path,
                        global_max,
                        global_min,
                        output_dir
                    )

        print(f"Plots saved in figures/heatmap/{building}/")


    # Generate combined mean error plots for all buildings
    print("\nGenerating combined mean error plots for all buildings...")
    plot_mean_errors_vs_size(buildings_data, timing_data_dict, building_colormap=cm.batlowS)

if __name__ == "__main__":
    main()