"""Features related to movements/kinematics."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mpl_cm
from typing import Dict, Tuple, Any, List
from pathlib import Path
import xarray as xr
from scipy.signal import find_peaks, peak_prominences


from typing import Union
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
from itertools import groupby


class Position3DCalibration:
    """ Convert 3D positions from DLC to real-world coordinates."""
    def __init__(
        self,
        conv_factor: float = -2.60976 + 0.0937,
        offset_x: float = 2.7,
        offset_y: float = -6.0415 + 13.3,
        offset_z: float = -45.7488,
        rot_x: float = 15,
        rot_y: float = -3.6802 + 9,
        rot_z: float = -7.47301 + 1.206338,
    ):
        self.conv_factor = conv_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z  
        
        
    def _rotate_x(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        return data @ rot_mat
    
    def _rotate_y(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        return data @ rot_mat
    
    def _rotate_z(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return data @ rot_mat
    
    def transform(self, ds):

        pos_data = ds.position.values.copy()
        

        space_dim = ds.position.dims.index('space')
        # Move space dimension to last position for easier indexing
        pos_data = np.moveaxis(pos_data, space_dim, -1)
        
        # Verify we have 3D coordinates
        if pos_data.shape[-1] != 3:
            raise ValueError(f"Expected 3 coordinates (x,y,z), got {pos_data.shape[-1]}")
        
        # Invert x-axis 
        pos_data[..., 0] = -pos_data[..., 0]
   
        # pos_data[..., 2] = -pos_data[..., 2] # z
        
        # Convert units to cm
        pos_data *= self.conv_factor
        

        pos_data[..., 0] -= self.offset_x
        pos_data[..., 1] -= self.offset_y
        pos_data[..., 2] -= self.offset_z
        
        # Apply rotations - reshape to (N, 3) for matrix multiplication
        original_shape = pos_data.shape
        pos_data_flat = pos_data.reshape(-1, 3)
        
        pos_data_flat = self._rotate_x(pos_data_flat, self.rot_x)
        pos_data_flat = self._rotate_y(pos_data_flat, self.rot_y)
        pos_data_flat = self._rotate_z(pos_data_flat, self.rot_z)
        
  
        pos_data = pos_data_flat.reshape(original_shape)
        
        # Move space dimension back to original position
        pos_data = np.moveaxis(pos_data, -1, space_dim)
        

        ds.position.values = pos_data
        
        return ds



def compute_distance_to_constant(
    data: xr.Dataset,
    reference_point: Union[np.ndarray, list, tuple],
    keypoint: str = None,
    individual: str = None,
    metric: str = "euclidean",
    **kwargs
) -> xr.DataArray:
    """
    Compute distance from keypoint(s) to a constant reference point.
    
    Parameters
    ----------
    data : xr.Dataset
        Dataset containing position data with dims: time, individuals, keypoints, space
        Space dimension must contain either ['x', 'y'] or ['x', 'y', 'z']
    reference_point : array-like
        Constant reference point [x, y] for 2D or [x, y, z] for 3D
    keypoint : str, optional
        Specific keypoint to compute distance for
    individual : str, optional
        Specific individual to compute distance for
    metric : str, default "euclidean"
        Distance metric (see scipy.spatial.distance.cdist)
    **kwargs
        Additional arguments for cdist
    
    Returns
    -------
    xr.DataArray
        Distances with preserved dimensions
    
    Raises
    ------
    ValueError
        If space coordinates and reference_point dimensions don't match
    """
    if keypoint:
        data = data.sel(keypoints=keypoint)
    if individual:
        data = data.sel(individuals=individual)
    
    if isinstance(reference_point, (list, tuple)):
        reference_point = np.array(reference_point)
    
    space_coords = list(data.coords['space'].values)
    ref_len = len(reference_point)
    
    if space_coords == ['x', 'y'] and ref_len == 2:
        space_selection = ['x', 'y']
    elif space_coords == ['x', 'y', 'z'] and ref_len == 3:
        space_selection = ['x', 'y', 'z']
    else:
        raise ValueError(
            f"Dimension mismatch: space coords {space_coords} "
            f"incompatible with reference_point length {ref_len}. "
            f"Expected either ['x', 'y'] with len=2 or ['x', 'y', 'z'] with len=3"
        )
    
    distances = xr.apply_ufunc(
        lambda pos: cdist(
            pos.reshape(-1, ref_len),
            reference_point.reshape(1, -1),
            metric=metric,
            **kwargs
        ).squeeze(),
        data.sel(space=space_selection),
        input_core_dims=[["space"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64]
    )
    
    distances.attrs["type"] = "features"
    return distances


def calculate_movement_angles(xy_pos):
   """
   Calculate movement angles between consecutive points in the x-y plane.
   
   Parameters
   ----------
   xy_pos : array-like of shape (N, D)
       N points with D dimensions. Uses first two columns as x, y coordinates.
       D must be at least 2.
   
   Returns
   -------
   angles : ndarray of shape (N-1,)
       Angles in degrees from positive y-axis, measured counterclockwise.
   """
   xy_pos = np.asarray(xy_pos)
   vectors = np.diff(xy_pos[:, :2], axis=0)  # Use only x-y plane
   return np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1]))



def get_angle_rgb(xy_pos, smooth_func=None, smoothing_params=None):
    """
    Convert movement angles to RGB colors using a colormap, with optional smoothing.

    Parameters
    ----------
    xy_pos : array-like of shape (N, D)
        N points with D dimensions. Uses first two columns as x, y coordinates.
    smooth_func : callable, optional
        Function to smooth xy_pos before angle calculation. Should accept and return array-like of shape (N, D).
    smooth_params: dict, optional
        Parameters to pass to the smoothing function.

    Returns
    -------
    rgb_matrix : ndarray of shape (N, 3)
        RGB values corresponding to each input point.
    angles : ndarray of shape (N,)
        Angles in degrees (0-360°).
    """
    xy_pos = np.asarray(xy_pos)
    
    if np.all(np.isnan(xy_pos)):
        nan_rgb = np.full((xy_pos.shape[0], 3), np.nan)
        nan_angles = np.full(xy_pos.shape[0], np.nan)
        return nan_rgb, nan_angles

    if smooth_func is not None:
        xy_pos = smooth_func(xy_pos, **smoothing_params)


    cmap = mpl_cm.get_cmap('hsv', 256)
    cm = cmap(np.linspace(0, 1, 256))[:, :3]  # Get RGB, exclude alpha


    curr_angles = calculate_movement_angles(xy_pos)

    # Add 0 at the end (forward scheme)
    curr_angles = np.append(curr_angles, 0)

    # Replace NaN with 0 and clip to [-180, 180]
    curr_angles = np.nan_to_num(curr_angles, 0)
    curr_angles = np.clip(curr_angles, -180, 180)

    # Map angles to colormap indices
    col_lines = np.linspace(0, len(cm) - 1, 360)

    # Shift angles from [-180, 180] to [0, 359] for indexing
    angles = np.ceil(curr_angles + 180).astype(int) - 1
    angles = np.clip(angles, 0, 359)

    # Get RGB values from colormap
    cmap_indices = np.round(col_lines[angles]).astype(int)
    rgb_matrix = cm[cmap_indices]

    return rgb_matrix, angles


def extract_speed_statistics(
    labels: Union[List[int], np.ndarray],
    features: Union[List[Any], np.ndarray],
    skip_background: bool = True
) -> Dict[int, Dict[str, List[Any]]]:
    """
    Extract statistical values for each label segment.
    
    Args:
        labels: Sequence of integer labels
        features: Feature values aligned with labels
        skip_background: Whether to skip label 0
    
    Returns:
        Dictionary: {label: {'starts': [...], 'ends': [...], 'peak_heights': [...]}}
    """
    if len(labels) != len(features):
        raise ValueError("Labels and features must have same length")
    
    labels = np.asarray(labels)
    features = np.asarray(features)
    
    # Filter NaN values first
    valid_mask = ~np.isnan(labels)
    labels = labels[valid_mask].astype(int)
    features = features[valid_mask]
    

    segment_stats = {}
    position = 0
    
    for label, group in groupby(labels):
        segment_length = sum(1 for _ in group)
        end_position = position + segment_length - 1
        
        if not (skip_background and label == 0):
            if label not in segment_stats:
                segment_stats[label] = {
                    'max_starts': [], 
                    'max_ends': []
                }
            

            start_value = features[position]
            end_value = features[end_position]
            
            segment_stats[label]['max_starts'].append(start_value)
            segment_stats[label]['max_ends'].append(end_value)

        
        position += segment_length
    
    return segment_stats



def compute_speed_statistics(
        boundary_stats: Dict[int, Dict[str, List[Any]]],
        n_mad: float = 1.0,
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Compute statistical boundaries (mean ± factor * std) for start and end values of each label.

        Args:
            boundary_stats: Output from extract_boundary_values
                            {label: {'max_starts': [...], 'max_ends': [...]} }
            factor: Scaling factor for the standard deviation (default: 2)

        Returns:
            Dictionary with upper/lower limits for each label's start/end values
            {label: {'max_starts': 150, 
                    'max_ends': 120}
        """
        final_stats = {}
        hist_stats = {}
        
        for label, boundaries in boundary_stats.items():
            label = int(label)  # Ensure label is int for dictionary keys
            final_stats[label] = {}
            hist_stats[label] = {}

            for boundary_type in ['max_starts', 'max_ends']:
                values = boundaries[boundary_type]

                if isinstance(values, list) and any(isinstance(v, (np.ndarray, list)) for v in values):
                    values = np.concatenate([np.asarray(v, dtype=np.float64).flatten() for v in values])
                else:
                    values = np.array(values, dtype=np.float64)
                if values.ndim > 1:
                    values = values.flatten()
                values = values[~np.isnan(values)]
                
                if len(values) == 0:
                    continue
                
                median = np.median(values)
                mad = _compute_mad(values)

                if boundary_type in ['max_starts', 'max_ends']:
                    final_stats[label][boundary_type] = median + n_mad * mad


                hist_stats[label][boundary_type] = values
            
        
        return final_stats, hist_stats

def _compute_mad(values: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation with consistency factor.
    
    Args:
        values: Array of values
    
    Returns:
        Scaled MAD value (comparable to standard deviation)
    """
    median = np.median(values)
    
    mad = np.median(np.abs(values - median))
    
    consistency_factor = 1.4826
    
    return consistency_factor * mad

def stats_histograms(speed_stats: Dict[int, Dict[str, float]], hist_stats: Dict[int, Dict[str, np.ndarray]], save_path: str, num_bins: int = 50):
    """
    Plot histograms of boundary statistics for each label.

    Args:
        speed_stats: {label: {stat_name: float}}
        hist_stats: {label: {stat_name: np.ndarray}}
        save_path: Path to save the figure
        num_bins: Number of bins for histogram (default: 50)
    """
    import matplotlib.pyplot as plt

    stat_names = ['max_starts', 'max_ends']
    labels = sorted(hist_stats.keys())
    n_labels = len(labels)
    n_stats = len(stat_names)

    fig, axes = plt.subplots(n_labels, n_stats, figsize=(4 * n_stats, 3 * n_labels), squeeze=False)

    for i, label in enumerate(labels):
        stats = hist_stats[label]
        cutoffs = speed_stats[label]
        for j, stat_name in enumerate(stat_names):
            ax = axes[i][j]
            value = stats.get(stat_name)
            cutoff = cutoffs.get(stat_name)
            if value is not None:
                ax.hist([value], bins=num_bins, alpha=0.7)
            ax.set_title(f'Label {label} - {stat_name}')
            ax.axvline(x=cutoff, color='r', linestyle='--', label='Cutoff')
            ax.set_xlabel(stat_name)
            ax.set_ylabel('Frequency')
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def create_event_aligned_heatmap(
    xy_positions: np.ndarray,
    event_indices: np.ndarray,
    fs: float,
    window: float = 1.0,
    sort_by_duration: np.ndarray = None,
    smooth_func=None,
    smoothing_params: dict = None,
    ax: plt.Axes = None,
    cmap: str = 'hsv',
    show_event_markers: bool = True,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Create event-aligned heatmap of movement angles.

    Parameters
    ----------
    xy_positions : np.ndarray
        Position data with shape (time, 2) or (time, >=2). Uses first two columns as x, y.
    event_indices : np.ndarray
        Frame indices to align events to (e.g., motif starts).
    fs : float
        Sampling rate in Hz.
    window : float
        Time window in seconds on each side of the event (default 1.0s).
    sort_by_duration : np.ndarray, optional
        If provided, sorts rows by these durations and draws end markers.
    smooth_func : callable, optional
        Function to smooth positions before angle calculation.
    smoothing_params : dict, optional
        Parameters for the smoothing function.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str
        Colormap name (default 'hsv' for cyclic angles).
    show_event_markers : bool
        Draw vertical lines at event onset (t=0) and optionally at durations.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure.
    ax : plt.Axes
        The axes with the heatmap.
    aligned_data : np.ndarray
        The aligned angle data matrix (n_events, time_points).
    """
    rgb_matrix, angles = get_angle_rgb(xy_positions, smooth_func, smoothing_params)

    window_frames = int(window * fs)
    time_axis = np.linspace(-window, window, window_frames * 2 + 1)
    n_events = len(event_indices)

    aligned_data = np.full((n_events, len(time_axis)), np.nan)

    for i, event_idx in enumerate(event_indices):
        start_frame = event_idx - window_frames
        end_frame = event_idx + window_frames

        data_start = max(0, start_frame)
        data_end = min(len(angles), end_frame + 1)

        if data_start >= data_end:
            continue

        extracted = angles[data_start:data_end]

        target_start = max(0, data_start - start_frame)
        target_end = target_start + len(extracted)

        aligned_data[i, target_start:target_end] = extracted

    if sort_by_duration is not None:
        sort_order = np.argsort(sort_by_duration)
        aligned_data = aligned_data[sort_order]
        sort_by_duration = sort_by_duration[sort_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    masked_data = np.ma.masked_invalid(aligned_data)
    im = ax.imshow(
        masked_data,
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], 0, n_events],
        origin='lower',
        cmap=cmap,
        vmin=0,
        vmax=359,
    )
    ax.set_facecolor('white')

    if show_event_markers:
        ax.axvline(0, color='black', linewidth=1, linestyle='-')

        if sort_by_duration is not None:
            for i, duration in enumerate(sort_by_duration):
                ax.plot([duration, duration], [i, i + 1], color='black', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Events')
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, label='Angle (°)')

    return fig, ax, aligned_data


def create_heatmap_from_segments(
    df: 'pd.DataFrame',
    angle_rgb_dict: Dict[Tuple[str, int], np.ndarray],
    fs: float,
    label_filter: int = None,
    window: float = 1.0,
    align_to: str = 'start',
    sort_by: str = 'duration',
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Create heatmap from segment DataFrame using pre-computed angle_rgb.

    Parameters
    ----------
    df : pd.DataFrame
        Segment DataFrame from ds_to_df with session and trial columns.
    angle_rgb_dict : Dict[Tuple[str, int], np.ndarray]
        Mapping of (session, trial) -> angle_rgb array (time, 3).
        Use build_angle_rgb_dict() to create this.
    fs : float
        Sampling rate in Hz.
    label_filter : int, optional
        Only include segments with this label.
    window : float
        Time window around event in seconds.
    align_to : str
        Align to 'start' or 'stop' of segments.
    sort_by : str
        Sort events by 'duration', 'trial', or None.

    Returns
    -------
    fig, ax, aligned_rgb
    """
    if label_filter is not None:
        df = df[df['label'] == label_filter]

    if df.empty:
        raise ValueError(f"No segments found for label {label_filter}")

    event_indices = []
    durations = []
    all_rgb = []
    offset = 0

    # Group by unique (session, trial) pairs
    for (session, trial), trial_df in df.groupby(['session', 'trial']):
        rgb_data = angle_rgb_dict.get((session, trial))


        for _, row in trial_df.iterrows():
            if align_to == 'start':
                event_idx = int(row['start'] * fs) + offset
            else:
                event_idx = int(row['stop'] * fs) + offset

            event_indices.append(event_idx)
            durations.append(row['duration'])

        all_rgb.append(rgb_data)
        offset += len(rgb_data)

    if not all_rgb:
        raise ValueError("No RGB data found for trials in DataFrame")

    combined_rgb = np.concatenate(all_rgb, axis=0)
    event_indices = np.array(event_indices)
    durations = np.array(durations)

    sort_durations = durations if sort_by == 'duration' else None

    return create_rgb_heatmap(
        combined_rgb,
        event_indices,
        fs,
        window=window,
        sort_by_duration=sort_durations,
    )


def create_rgb_heatmap(
    rgb_data: np.ndarray,
    event_indices: np.ndarray,
    fs: float,
    window: float = 1.0,
    sort_by_duration: np.ndarray = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Create event-aligned heatmap from pre-computed RGB data.

    Parameters
    ----------
    rgb_data : np.ndarray
        RGB values with shape (time, 3). Values in [0, 1] or [0, 255].
    event_indices : np.ndarray
        Indices of events to align to.
    fs : float
        Sampling rate in Hz.
    window : float
        Time window on each side of event in seconds.
    sort_by_duration : np.ndarray, optional
        If provided, sort rows by duration and draw end markers.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig, ax, aligned_rgb
    """
    window_frames = int(window * fs)
    n_frames = 2 * window_frames + 1
    n_events = len(event_indices)

    aligned_rgb = np.full((n_events, n_frames, 3), np.nan)

    for i, event_idx in enumerate(event_indices):
        start_idx = event_idx - window_frames
        end_idx = event_idx + window_frames + 1

        src_start = max(0, start_idx)
        src_end = min(len(rgb_data), end_idx)

        if src_start >= src_end:
            continue

        tgt_start = src_start - start_idx
        tgt_end = tgt_start + (src_end - src_start)

        aligned_rgb[i, tgt_start:tgt_end, :] = rgb_data[src_start:src_end]

    # Normalize to [0, 1] if needed
    if np.nanmax(aligned_rgb) > 1.0:
        aligned_rgb = aligned_rgb / 255.0

    if sort_by_duration is not None:
        order = np.argsort(sort_by_duration)
        aligned_rgb = aligned_rgb[order]
        sort_by_duration = sort_by_duration[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Replace NaN with white for display
    display_rgb = np.nan_to_num(aligned_rgb, nan=1.0)

    ax.imshow(
        display_rgb,
        aspect='auto',
        extent=[-window, window, 0, n_events],
        origin='lower',
        interpolation='nearest',
    )

    ax.axvline(0, color='black', linewidth=1, linestyle='-')

    if sort_by_duration is not None:
        for i, dur in enumerate(sort_by_duration):
            ax.plot([dur, dur], [i, i + 1], color='black', linewidth=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trials')
    ax.set_yticks([])

    return fig, ax, aligned_rgb