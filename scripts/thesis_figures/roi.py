"""Animate polygon of interest based on stick positions and displacements."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import xarray as xr


def create_roi_polygon(positions, displacements, frame_idx, keypoint_names):
    """Create polygon corners from positions and displacements.

    Parameters
    ----------
    positions : xr.DataArray
        Position data with dimensions (time, individuals, keypoints, space)
    displacements : xr.DataArray
        Displacement data with same dimensions
    frame_idx : int
        Frame index to extract data for
    keypoint_names : list
        List of keypoint names to use for polygon corners

    Returns
    -------
    corners : np.ndarray
        Array of polygon corners shape (4, 2) for x,y coordinates
    """
    corners = []

    for kp_name in keypoint_names:
        # Get position at this frame
        pos = positions.isel(time=frame_idx).sel(keypoints=kp_name)

        # Get displacement at this frame
        disp = displacements.isel(time=frame_idx).sel(keypoints=kp_name)

        # Add position + displacement as corner
        corner = np.array([
            pos.sel(space='x').values + disp.sel(space='x').values,
            pos.sel(space='y').values + disp.sel(space='y').values
        ])

        # Handle multiple individuals by taking mean
        if len(corner.shape) > 1:
            corner = corner.mean(axis=1)

        corners.append(corner)

    return np.array(corners)


def animation(ds, start_frame=0, n_frames=50,
                       keypoint_names=['stickTip', 'stickStripeProx'],
                       figsize=(15, 10)):
    """Create subplot animation of ROI polygons over time.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'position' and 'displacement' variables
    trial_idx : int
        Trial index to visualize
    start_frame : int
        Starting frame index
    n_frames : int
        Number of frames to animate
    keypoint_names : list
        Keypoint names to use for polygon corners
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Extract data for selected trial and frame range
    positions = ds['position'].isel(
        time=slice(start_frame, start_frame + n_frames)
    )

    displacements = ds['displacement'].isel(
        time=slice(start_frame, start_frame + n_frames)
    )

    # Create figure with subplots grid
    n_cols = 10
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Hide extra subplots
    for ax in axes[n_frames:]:
        ax.axis('off')

    # Get global min/max for consistent axis limits
    # Extract x and y data using xarray selection
    x_positions = positions.sel(space='x').values.flatten()
    y_positions = positions.sel(space='y').values.flatten()
    x_displacements = displacements.sel(space='x').values.flatten()
    y_displacements = displacements.sel(space='y').values.flatten()

    x_data = np.concatenate([
        x_positions,
        x_positions + x_displacements
    ])
    y_data = np.concatenate([
        y_positions,
        y_positions + y_displacements
    ])

    # Filter out NaN values
    x_data = x_data[~np.isnan(x_data)]
    y_data = y_data[~np.isnan(y_data)]

    if len(x_data) > 0 and len(y_data) > 0:
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        xlim = (x_min - x_padding, x_max + x_padding)
        ylim = (y_min - y_padding, y_max + y_padding)
    else:
        xlim = (0, 100)
        ylim = (0, 100)

    # Plot each frame
    for frame_idx in range(n_frames):
        ax = axes[frame_idx]

        # Create polygon corners
        corners = create_roi_polygon(
            positions, displacements, frame_idx, keypoint_names
        )

        # Duplicate keypoints to create 4 corners
        if len(corners) == 2:
            # Create rectangle from 2 diagonal points
            corners = np.array([
                corners[0],  # First corner
                [corners[1][0], corners[0][1]],  # Top right
                corners[1],  # Second corner (diagonal)
                [corners[0][0], corners[1][1]]   # Bottom left
            ])

        # Plot the polygon
        if not np.any(np.isnan(corners)):
            polygon = patches.Polygon(
                corners,
                closed=True,
                edgecolor='red',
                facecolor='yellow',
                alpha=0.3,
                linewidth=2,
                label='Nest region'
            )
            ax.add_patch(polygon)

            # Plot corner points
            ax.scatter(corners[:, 0], corners[:, 1],
                      c='blue', s=20, zorder=5)

            # Plot original positions
            for kp_name in keypoint_names:
                pos = positions.isel(time=frame_idx).sel(keypoints=kp_name)
                ax.scatter(pos.sel(space='x'), pos.sel(space='y'),
                          c='green', s=15, marker='^',
                          label=f'{kp_name} (original)')

        # Set axis properties
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'Frame {start_frame + frame_idx}', fontsize=8)
        ax.tick_params(labelsize=6)

        # Add legend to first subplot
        if frame_idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    plt.suptitle(f'ROI Polygon Animation - Frames {start_frame}-{start_frame+n_frames}', fontsize=12)
    plt.tight_layout()

    return fig, axes


def plot_roi_trajectory(ds, trial_idx=0, start_frame=0, n_frames=50,
                        keypoint_names=['sticktip', 'stickstripeprox']):
    """Plot the trajectory of ROI polygon over time in a single plot.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'position' and 'displacement' variables
    trial_idx : int
        Trial index to visualize
    start_frame : int
        Starting frame index
    n_frames : int
        Number of frames to show
    keypoint_names : list
        Keypoint names to use for polygon corners

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    # Extract data
    positions = ds['position'].isel(
        trials=trial_idx,
        time=slice(start_frame, start_frame + n_frames)
    )

    displacements = ds['displacement'].isel(
        trials=trial_idx,
        time=slice(start_frame, start_frame + n_frames)
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create colormap for time
    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))

    for frame_idx in range(n_frames):
        # Create polygon corners
        corners = create_roi_polygon(
            positions, displacements, frame_idx, keypoint_names
        )

        # Duplicate keypoints to create 4 corners if needed
        if len(corners) == 2:
            corners = np.array([
                corners[0],
                [corners[1][0], corners[0][1]],
                corners[1],
                [corners[0][0], corners[1][1]]
            ])

        if not np.any(np.isnan(corners)):
            # Plot polygon with transparency based on time
            alpha = 0.1 + 0.5 * (frame_idx / n_frames)
            polygon = patches.Polygon(
                corners,
                closed=True,
                edgecolor=colors[frame_idx],
                facecolor='none',
                alpha=alpha,
                linewidth=1
            )
            ax.add_patch(polygon)

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=start_frame, vmax=start_frame+n_frames)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Frame')

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'ROI Polygon Trajectory - Trial {trial_idx}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Load your dataset
    # ds = xr.open_dataset('your_dataset.nc')

    # Create example data for testing
    np.random.seed(42)
    n_times = 100
    n_trials = 3
    n_individuals = 1
    n_keypoints = 2

    # Create synthetic data
    time = np.arange(n_times)
    trials = np.arange(n_trials)
    individuals = ['individual1']
    keypoints = ['sticktip', 'stickstripeprox']
    space = ['x', 'y']

    # Generate random walk positions
    positions = np.cumsum(np.random.randn(n_times, n_trials, n_individuals, n_keypoints, 2) * 2, axis=0) + 50

    # Generate displacements
    displacements = np.random.randn(n_times, n_trials, n_individuals, n_keypoints, 2) * 5

    # Create dataset
    ds = xr.Dataset({
        'position': (['time', 'trials', 'individuals', 'keypoints', 'space'], positions),
        'displacement': (['time', 'trials', 'individuals', 'keypoints', 'space'], displacements)
    }, coords={
        'time': time,
        'trials': trials,
        'individuals': individuals,
        'keypoints': keypoints,
        'space': space
    })

    # Create animation plot
    fig1, axes1 = plot_roi_animation(ds, trial_idx=0, start_frame=0, n_frames=50)

    # Create trajectory plot
    fig2, ax2 = plot_roi_trajectory(ds, trial_idx=0, start_frame=0, n_frames=50)

    plt.show()