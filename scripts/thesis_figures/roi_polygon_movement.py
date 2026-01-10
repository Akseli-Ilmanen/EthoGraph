"""ROI Polygon visualization using movement library."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr
from movement.io import load_poses
import movement.kinematics as kin


def create_roi_polygon_from_movement(ds, frame_idx, keypoint_names=['stickTip', 'stickStripeProx']):
    """Create polygon corners from movement dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with position and displacement
    frame_idx : int
        Frame index to extract data for
    keypoint_names : list
        List of keypoint names to use for polygon corners

    Returns
    -------
    corners : np.ndarray
        Array of polygon corners shape (4, 2) for x,y coordinates
    original_positions : np.ndarray
        Array of original positions before displacement
    """
    # Get positions and displacements at this frame
    positions = ds['position'].isel(time=frame_idx)
    displacements = ds['displacement'].isel(time=frame_idx)

    corners = []
    original_positions = []

    for kp_name in keypoint_names:
        # Get position
        pos_x = positions.sel(keypoints=kp_name, space='x').values
        pos_y = positions.sel(keypoints=kp_name, space='y').values

        # Get displacement
        disp_x = displacements.sel(keypoints=kp_name, space='x').values
        disp_y = displacements.sel(keypoints=kp_name, space='y').values

        # Handle multiple individuals by taking mean
        if isinstance(pos_x, np.ndarray) and pos_x.size > 1:
            pos_x = np.nanmean(pos_x)
            pos_y = np.nanmean(pos_y)
            disp_x = np.nanmean(disp_x)
            disp_y = np.nanmean(disp_y)

        # Store original position
        original_positions.append([pos_x, pos_y])

        # Calculate corner (position + displacement)
        corner_x = pos_x + disp_x
        corner_y = pos_y + disp_y
        corners.append([corner_x, corner_y])

    corners = np.array(corners)
    original_positions = np.array(original_positions)

    # Create 4 corners from 2 diagonal points
    if len(corners) == 2:
        polygon_corners = np.array([
            corners[0],  # First corner
            [corners[1][0], corners[0][1]],  # Top right
            corners[1],  # Second corner (diagonal)
            [corners[0][0], corners[1][1]]   # Bottom left
        ])
    else:
        polygon_corners = corners

    return polygon_corners, original_positions


def plot_roi_animation_grid(ds, start_frame=0, n_frames=50,
                            keypoint_names=['stickTip', 'stickStripeProx'],
                            figsize=(20, 12), n_cols=10):
    """Create grid of subplots showing ROI polygon over time.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset containing position and displacement
    start_frame : int
        Starting frame index
    n_frames : int
        Number of frames to plot
    keypoint_names : list
        Keypoint names to use for polygon corners
    figsize : tuple
        Figure size
    n_cols : int
        Number of columns in subplot grid

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Extract data for frame range
    positions = ds['position'].isel(time=slice(start_frame, start_frame + n_frames))
    displacements = ds['displacement'].isel(time=slice(start_frame, start_frame + n_frames))

    # Create subplot grid
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Hide extra subplots
    for ax in axes[n_frames:]:
        ax.axis('off')

    # Calculate global axis limits
    x_pos = positions.sel(space='x').values.flatten()
    y_pos = positions.sel(space='y').values.flatten()
    x_disp = displacements.sel(space='x').values.flatten()
    y_disp = displacements.sel(space='y').values.flatten()

    # Remove NaNs
    valid_mask = ~(np.isnan(x_pos) | np.isnan(y_pos) | np.isnan(x_disp) | np.isnan(y_disp))
    x_pos = x_pos[valid_mask]
    y_pos = y_pos[valid_mask]
    x_disp = x_disp[valid_mask]
    y_disp = y_disp[valid_mask]

    if len(x_pos) > 0:
        x_all = np.concatenate([x_pos, x_pos + x_disp])
        y_all = np.concatenate([y_pos, y_pos + y_disp])

        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()

        # Add padding
        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15

        xlim = (x_min - x_padding, x_max + x_padding)
        ylim = (y_min - y_padding, y_max + y_padding)
    else:
        xlim = (0, 100)
        ylim = (0, 100)

    # Plot each frame
    for i in range(n_frames):
        ax = axes[i]

        # Create polygon
        corners, orig_pos = create_roi_polygon_from_movement(
            ds, start_frame + i, keypoint_names
        )

        if not np.any(np.isnan(corners)):
            # Plot polygon
            polygon = patches.Polygon(
                corners,
                closed=True,
                edgecolor='red',
                facecolor='yellow',
                alpha=0.3,
                linewidth=1.5,
                label='ROI' if i == 0 else ''
            )
            ax.add_patch(polygon)

            # Plot corner points
            ax.scatter(corners[:, 0], corners[:, 1],
                      c='blue', s=20, marker='o', zorder=5,
                      label='Corners' if i == 0 else '')

            # Plot original positions
            if not np.any(np.isnan(orig_pos)):
                ax.scatter(orig_pos[:, 0], orig_pos[:, 1],
                          c='green', s=15, marker='^', zorder=4,
                          label='Original' if i == 0 else '')

                # Draw displacement arrows
                for j in range(min(len(orig_pos), 2)):
                    ax.arrow(orig_pos[j, 0], orig_pos[j, 1],
                            corners[j, 0] - orig_pos[j, 0],
                            corners[j, 1] - orig_pos[j, 1],
                            head_width=1.5, head_length=1,
                            fc='gray', ec='gray', alpha=0.4, linewidth=0.5)

        # Set axis properties
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'Frame {start_frame + i}', fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Add legend to first subplot
        if i == 0:
            ax.legend(fontsize=5, loc='best')

    plt.suptitle(f'ROI Polygon Animation (Frames {start_frame}-{start_frame+n_frames-1})',
                fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_roi_trajectory_overlay(ds, start_frame=0, n_frames=50,
                               keypoint_names=['stickTip', 'stickStripeProx'],
                               skip_frames=2, figsize=(10, 10)):
    """Plot overlayed ROI polygons showing trajectory over time.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset
    start_frame : int
        Starting frame
    n_frames : int
        Number of frames to overlay
    keypoint_names : list
        Keypoint names for polygon corners
    skip_frames : int
        Plot every nth frame for clarity
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap for time
    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))

    for i in range(0, n_frames, skip_frames):
        frame_idx = start_frame + i

        # Get polygon corners
        corners, orig_pos = create_roi_polygon_from_movement(
            ds, frame_idx, keypoint_names
        )

        if not np.any(np.isnan(corners)):
            # Calculate alpha based on time (later frames more opaque)
            alpha = 0.1 + 0.6 * (i / n_frames)

            # Plot polygon
            polygon = patches.Polygon(
                corners,
                closed=True,
                edgecolor=colors[i],
                facecolor='none',
                alpha=alpha,
                linewidth=1
            )
            ax.add_patch(polygon)

            # Plot centroid for this frame
            centroid = corners.mean(axis=0)
            ax.scatter(centroid[0], centroid[1],
                      c=[colors[i]], s=10, alpha=alpha, zorder=5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=start_frame, vmax=start_frame+n_frames)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Frame')

    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f'ROI Trajectory (Frames {start_frame}-{start_frame+n_frames-1})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_roi_single_frame(ds, frame_idx, keypoint_names=['stickTip', 'stickStripeProx'],
                         figsize=(8, 8)):
    """Detailed plot of ROI polygon for a single frame.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset
    frame_idx : int
        Frame index to plot
    keypoint_names : list
        Keypoint names for polygon corners
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get polygon corners and original positions
    corners, orig_pos = create_roi_polygon_from_movement(
        ds, frame_idx, keypoint_names
    )

    if not np.any(np.isnan(corners)):
        # Plot polygon
        polygon = patches.Polygon(
            corners,
            closed=True,
            edgecolor='red',
            facecolor='yellow',
            alpha=0.3,
            linewidth=2,
            label='ROI (Nest region)'
        )
        ax.add_patch(polygon)

        # Plot and label each point
        for i, (corner, orig, kp_name) in enumerate(zip(corners[:2], orig_pos, keypoint_names)):
            # Original position
            ax.scatter(orig[0], orig[1], c='green', s=100, marker='^', zorder=5)
            ax.annotate(f'{kp_name}\n(original)',
                       xy=(orig[0], orig[1]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3))

            # Displaced position
            ax.scatter(corner[0], corner[1], c='blue', s=100, marker='o', zorder=5)
            ax.annotate(f'{kp_name}\n(+displacement)',
                       xy=(corner[0], corner[1]),
                       xytext=(5, -20),
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))

            # Displacement arrow
            ax.arrow(orig[0], orig[1],
                    corner[0] - orig[0], corner[1] - orig[1],
                    head_width=3, head_length=2,
                    fc='gray', ec='gray', alpha=0.6, linewidth=2,
                    length_includes_head=True)

            # Add displacement magnitude text
            disp_magnitude = np.sqrt((corner[0] - orig[0])**2 + (corner[1] - orig[1])**2)
            mid_point = [(orig[0] + corner[0])/2, (orig[1] + corner[1])/2]
            ax.annotate(f'{disp_magnitude:.1f} px',
                       xy=mid_point,
                       fontsize=8,
                       color='red',
                       ha='center')

    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f'ROI Polygon Detail - Frame {frame_idx}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Auto-scale with padding
    ax.relim()
    ax.autoscale_view(scalex=True, scaley=True)
    ax.margins(0.15)

    return fig, ax


# Example usage
if __name__ == "__main__":
    print("Load your movement dataset with:")
    print("  ds = xr.open_dataset('your_file.nc')")
    print("  # or")
    print("  ds = load_poses.from_sleap_file('your_sleap_file.h5')")
    print("")
    print("Then use:")
    print("  fig, axes = plot_roi_animation_grid(ds)")
    print("  fig, ax = plot_roi_trajectory_overlay(ds)")
    print("  fig, ax = plot_roi_single_frame(ds, frame_idx=100)")