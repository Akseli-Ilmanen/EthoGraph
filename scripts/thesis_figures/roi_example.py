"""Simple example for ROI polygon visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr

# If using in Jupyter notebook, use this instead of %matplotlib inline:
# plt.ion()  # Turn on interactive mode

# Import the ROI functions
from roi_polygon_movement import (
    plot_roi_animation_grid,
    plot_roi_trajectory_overlay,
    plot_roi_single_frame
)

# Load your dataset (replace with your actual data)
# ds2 = xr.open_dataset('your_dataset.nc')

# For testing with synthetic data:
def create_test_dataset():
    """Create a test dataset with movement data."""
    n_times = 100
    n_individuals = 1
    n_keypoints = 2

    # Create coordinates
    time = np.arange(n_times)
    individuals = ['individual1']
    keypoints = ['stickTip', 'stickStripeProx']
    space = ['x', 'y']

    # Generate synthetic positions (random walk)
    np.random.seed(42)
    positions = np.cumsum(np.random.randn(n_times, n_individuals, n_keypoints, 2) * 2, axis=0) + 50

    # Generate displacements
    displacements = np.random.randn(n_times, n_individuals, n_keypoints, 2) * 5

    # Create dataset
    ds = xr.Dataset({
        'position': (['time', 'individuals', 'keypoints', 'space'], positions),
        'displacement': (['time', 'individuals', 'keypoints', 'space'], displacements)
    }, coords={
        'time': time,
        'individuals': individuals,
        'keypoints': keypoints,
        'space': space
    })

    return ds

# Create or load dataset
# ds2 = create_test_dataset()  # For testing
# ds2 = xr.open_dataset('your_data.nc')  # For real data

# Example usage:
if __name__ == "__main__":
    # Create test data
    ds2 = create_test_dataset()

    # Plot grid of frames
    fig1, axes1 = plot_roi_animation_grid(
        ds2,
        start_frame=0,
        n_frames=50,
        keypoint_names=['stickTip', 'stickStripeProx'],
        n_cols=10
    )
    plt.show()

    # Plot trajectory overlay
    fig2, ax2 = plot_roi_trajectory_overlay(
        ds2,
        start_frame=0,
        n_frames=50,
        skip_frames=2
    )
    plt.show()

    # Plot single frame detail
    fig3, ax3 = plot_roi_single_frame(
        ds2,
        frame_idx=25,
        keypoint_names=['stickTip', 'stickStripeProx']
    )
    plt.show()