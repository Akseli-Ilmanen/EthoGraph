from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ethograph.features.preprocessing import interpolate_nans

def create_heatmap_from_segments(
    df: pd.DataFrame,
    data_dict: Dict[Tuple[str, int], np.ndarray],
    fs: float,
    label_filter: int = None,
    window: float = 1.0,
    align_to: str = 'start',
    sort_by: str = 'duration',
    cmap: str = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Create heatmap from segment DataFrame using pre-computed data.

    Parameters
    ----------
    df : pd.DataFrame
        Segment DataFrame from ds_to_df with session and trial columns.
    data_dict : Dict[Tuple[str, int], np.ndarray]
        Mapping of (session, trial) -> data array.
        Shape (time, 3) for RGB or (time,) / (time, 1) for grayscale.
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
    cmap : str, optional
        Colormap for grayscale data. Ignored for RGB.

    Returns
    -------
    fig, ax, aligned_data
    """
    if label_filter is not None:
        df = df[df['label'] == label_filter]

    if df.empty:
        raise ValueError(f"No segments found for label {label_filter}")

    event_indices = []
    durations = []
    all_data = []
    offset = 0

    for (session, trial), trial_df in df.groupby(['session', 'trial']):
        data = data_dict.get((session, trial))

        for _, row in trial_df.iterrows():
            event_idx = int(row[align_to] * fs) + offset
            event_indices.append(event_idx)
            durations.append(row['duration'])

        all_data.append(data)
        offset += len(data)

    if not all_data:
        raise ValueError("No data found for trials in DataFrame")

    combined_data = np.concatenate(all_data, axis=0)
    event_indices = np.array(event_indices)
    durations = np.array(durations)

    sort_durations = durations if sort_by == 'duration' else None
    
    if window is None:
        window = np.max(durations) * 1.3

    return create_aligned_heatmap(
        combined_data,
        event_indices,
        fs,
        window=window,
        sort_by_duration=sort_durations,
        cmap=cmap,
    )


def create_aligned_heatmap(
    data: np.ndarray,
    event_indices: np.ndarray,
    fs: float,
    window: float = 1.0,
    sort_by_duration: np.ndarray = None,
    ax: plt.Axes = None,
    cmap: str = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Create event-aligned heatmap from data.

    Parameters
    ----------
    data : np.ndarray
        Data with shape (time, 3) for RGB or (time,) / (time, 1) for grayscale.
        RGB values in [0, 1] or [0, 255].
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
    cmap : str, optional
        Colormap for grayscale data (e.g., 'viridis', 'gray').
        Ignored for RGB data.

    Returns
    -------
    fig, ax, aligned_data
    """
    is_rgb = data.ndim == 2 and data.shape[1] == 3
    
    print("Creating heatmap...")
    print(f"Data shape: {data.shape}, Number of events: {len(event_indices)}")
    
    if data.ndim == 1:
        data = data[:, np.newaxis]

    
    if is_rgb:
        aligned_data = _align_data_to_events(data, event_indices, fs, window, normalize=False)
    else:
        aligned_data = _align_data_to_events(data, event_indices, fs, window, normalize=True)    
    
    
    if sort_by_duration is not None:
        aligned_data, sort_by_duration = _sort_by_duration(aligned_data, sort_by_duration)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    _plot_heatmap(ax, aligned_data, window, is_rgb, cmap)
    _add_event_markers(ax, sort_by_duration)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Syllables (n={aligned_data.shape[0]})')
    ax.set_yticks([])
    
    return fig, ax, aligned_data


def _align_data_to_events(
    data: np.ndarray,
    event_indices: np.ndarray,
    fs: float,
    window: float,
    normalize: bool = False,
) -> np.ndarray:
    """Align data to event indices with specified time window."""
    window_frames = int(window * fs)
    n_frames = 2 * window_frames + 1
    n_events = len(event_indices)
    n_channels = data.shape[1]
    
    aligned_data = np.full((n_events, n_frames, n_channels), np.nan)
    
    for i, event_idx in enumerate(event_indices):
        start_idx = event_idx - window_frames
        end_idx = event_idx + window_frames + 1
        
        src_start = max(0, start_idx)
        src_end = min(len(data), end_idx)
        
        if src_start >= src_end:
            continue
        
        tgt_start = src_start - start_idx
        tgt_end = tgt_start + (src_end - src_start)
        
        aligned_data[i, tgt_start:tgt_end, :] = data[src_start:src_end]
        
        if normalize:
            event_data = aligned_data[i, :, 0]

            
            
            event_data = interpolate_nans(event_data)
            event_data = np.clip(event_data, np.nanpercentile(event_data, 0), np.nanpercentile(event_data, 80))
            aligned_data[i, :, 0] = (event_data - np.mean(event_data)) / np.std(event_data)  

    
    return aligned_data


def _sort_by_duration(
    aligned_data: np.ndarray,
    durations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort aligned data by duration."""
    order = np.argsort(durations)
    return aligned_data[order], durations[order]


def _plot_heatmap(
    ax: plt.Axes,
    aligned_data: np.ndarray,
    window: float,
    is_rgb: bool,
    cmap: str,
) -> None:
    """Plot the aligned data as a heatmap."""
    n_events = aligned_data.shape[0]
    
    if is_rgb:
        if np.nanmax(aligned_data) > 1.0:
            aligned_data = aligned_data / 255.0
        display_data = np.nan_to_num(aligned_data, nan=1.0)
        ax.imshow(
            display_data,
            aspect='auto',
            extent=[-window, window, 0, n_events],
            origin='lower',
            interpolation='nearest',
        )
    else:
        display_data = aligned_data.squeeze()
        ax.imshow(
            display_data,
            aspect='auto',
            extent=[-window, window, 0, n_events],
            origin='lower',
            cmap=cmap or 'viridis',
            interpolation='nearest',
        )


def _add_event_markers(ax: plt.Axes, durations: np.ndarray = None) -> None:
    """Add event onset and duration markers to heatmap."""
    ax.axvline(0, color='red', linewidth=3, linestyle='-')
    
    if durations is not None:
        for i, dur in enumerate(durations):
            ax.plot([dur, dur], [i, i + 1], color='red', linewidth=3)