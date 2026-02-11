from itertools import groupby
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import patches




def load_mapping(mapping_file):
    """Load class name to index mapping"""
    class_to_idx = {}
    idx_to_class = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                idx = int(parts[0])
                class_name = parts[1]
                class_to_idx[class_name] = idx
                idx_to_class[idx] = class_name
    return class_to_idx, idx_to_class


def load_label_mapping(mapping_file: Union[str, Path] = "mapping.txt") -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load label mapping from a text file and return mapping dictionary with colors.

    Args:
        mapping_file: Path to the mapping file (default: "mapping.txt")

    Returns:
        Dictionary mapping labels to {'name': str, 'color': np.ndarray}

    Example mapping.txt file:
        0 background
        1 beakTip_pullOutStick
        2 beakTip_pushStick
        3 beakTip_peck
        4 beakTip_grasp
        5 beakTip_release
        6 beakTip_tap
        7 beakTip_touch
        8 beakTip_move
        9 beakTip_idle

    Example usage:
        >>> mapping = load_label_mapping("mapping.txt")
        >>> print(mapping[1]['name'])  # 'beakTip_pullOutStick'
        >>> print(mapping[1]['color'])  # [1.0, 0.4, 0.698]
    """
    mapping_file = Path(mapping_file)
   
    # RGB colours, where adjacent colours are maximally different
    label_colors = [
        [1, 1, 1],           # background class
        [255, 102, 178],     
        [102, 158, 255],     
        [153, 51, 255],      
        [255, 51, 51],  
        [102, 255, 102],    
        [255, 153, 102],    
        [0, 153, 0],       
        [0, 0, 128],        
        [255, 255, 0],       
        [0, 204, 204],      
        [128, 128, 0],       
        [255, 0, 255],       
        [255, 165, 0],       
        [0, 128, 255],              
        [  7,   7, 215],
        [128, 0, 255],
        [255,  215,  0],
        [ 73, 113, 233],
        [255, 128, 0],
        [138,  34,  34],
        [103, 176,  29],
        [  0, 230, 230],
        [220,  20,  60],
        [  3, 243,   3],
        [147,  24, 147],
        [188,  82, 223],
        [178, 111,  44],
        [ 16, 166, 166],
        [ 71, 197, 238],
        [255, 149, 114],
        [ 16,  89, 162],
        [ 26, 195,  68],
        [254, 216, 103],
        [  0, 237, 118],
        [177, 177,  36],
        [ 73, 243, 200],
    ]

    
    label_mappings = {}
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels = int(parts[0])
                name = parts[1]
                
                color = np.array(label_colors[labels]) / 255.0
                
                label_mappings[labels] = {
                    'name': name,
                    'color': color
                }
    
    return label_mappings



def labels_to_rgb(
    labels: Union[str, np.ndarray], 
    label_mapping: Dict[int, Dict[str, np.ndarray]]
) -> np.ndarray:
    """
    Convert label sequence to RGB colors based on label mapping.
    
    Args:
        labels: String of digits or array of integers representing label IDs
        label_mapping: Dictionary from load_label_mapping()
        
    Returns:
        Array of shape (N, 3) with RGB values [0, 1] for each frame
        
    Example:
        >>> mapping = load_label_mapping("mapping.txt")
        >>> labels = "000000111111000002222223333333300000444444444"
        >>> rgb_array = labels_to_rgb(labels, mapping)
        >>> print(rgb_array.shape)  # (45, 3)
    """
    # Convert string to integer array if needed
    if isinstance(labels, str):
        labels = np.array([int(c) for c in labels])
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Create RGB array
    n_frames = len(labels)
    rgb_array = np.zeros((n_frames, 3), dtype=np.float32)
    
    # Vectorized assignment for each unique label
    for labels in np.unique(labels):
        if labels in label_mapping:
            mask = labels == labels
            rgb_array[mask] = label_mapping[labels]['color']
        else:
            # Default to white for unmapped labels
            mask = labels == labels
            rgb_array[mask] = [1.0, 1.0, 1.0]
    
    return rgb_array


def get_segments(col, bg_class=0):
    """
    Example: [0,1,1,1,0,2,2] → [(1,1,4), (2,5,7)]
    """
    padded = np.concatenate([[bg_class - 1], col, [bg_class - 1]])
    change_indices = np.nonzero(padded[:-1] != padded[1:])[0]
    
    segments = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]
        label = int(col[start])
        if label != bg_class:
            segments.append((label, start, end))
    return segments


def get_labels_start_end_indices(col, bg_class=0):
    """Returns indices for array slicing (exclusive end).
    
    Example: [0,1,1,1,0,2,2] → labels=[1,2], starts=[1,5], ends=[4,7]
    """
    segments = get_segments(col, bg_class)
    labels = [s[0] for s in segments]
    starts = [s[1] for s in segments]
    ends = [s[2] for s in segments]
    return labels, starts, ends


def get_labels_start_end_times(col, time_coord, individual, bg_class=0):
    """Returns time intervals for storage (inclusive end).
    
    Returns:
        List of dicts with onset_s, offset_s (both inclusive), labels, individual.
        Example at 10Hz: [0,1,1,1,0,2,2] → 
            [{'onset_s': 0.1, 'offset_s': 0.3, 'labels': 1, 'individual': 'crow_A'},
             {'onset_s': 0.5, 'offset_s': 0.6, 'labels': 2, 'individual': 'crow_A'}]
    """
    segments = get_segments(col, bg_class)
    return [{
        "onset_s": float(time_coord[start]),
        "offset_s": float(time_coord[end - 1]),
        "labels": label,
        "individual": individual,
    } for label, start, end in segments]
    
    



def find_blocks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return starts, ends


def stitch_gaps(labels: np.ndarray, max_gap_len: int) -> np.ndarray:
    stitched = labels.copy()
    zero_starts, zero_ends = find_blocks(labels == 0)
    
    for start, end in zip(zero_starts, zero_ends):
        gap_len = end - start
        
        if gap_len > max_gap_len:
            continue
        
        left_label = labels[start - 1] if start > 0 else 0
        right_label = labels[end + 1] if end < len(labels) - 1 else 0
        
        # Toss exception - HARD CODED
        if left_label == 3:
            continue
        
        if left_label != 0 and left_label == right_label:
            stitched[start:end + 1] = left_label
    
    return stitched


def purge_small_blocks(
    labels: np.ndarray,
    min_length: int,
    label_thresholds: Dict[Union[int, str], int] = None
) -> np.ndarray:
    """
    Remove label blocks shorter than their threshold (set to 0).

    Args:
        labels: Array of integer labels
        min_length: Default minimum length for all labels
        label_thresholds: Optional dict mapping labels to custom threshold.
                         Keys can be int or str (for JSON compatibility).

    Returns:
        Labels array with short blocks set to 0

    Example:
        # All labels use threshold 3, except label 3 uses threshold 6
        purge_small_blocks(labels, min_length=3, label_thresholds={3: 6})
    """
    if isinstance(labels, (str, bytes)):
        labels = np.array([int(c) for c in str(labels)])
    else:
        labels = np.asarray(labels)

    if len(labels) == 0:
        return labels.copy()

    if label_thresholds is None:
        label_thresholds = {}
    else:
        label_thresholds = {int(k): v for k, v in label_thresholds.items()}

    output = labels.copy()

    padded = np.concatenate([[-1], labels, [-1]])
    change_mask = padded[:-1] != padded[1:]
    change_indices = np.nonzero(change_mask)[0]

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        if start_idx >= len(labels):
            continue

        label_val = int(labels[start_idx])
        if label_val == 0:
            continue

        threshold = label_thresholds.get(label_val, min_length)
        run_length = end_idx - start_idx

        if run_length < threshold:
            output[start_idx:end_idx] = 0

    return output


def fix_endings(labels, changepoints):
    """
    Args:
        labels (array-like): Sequence of integer labels.
        changepoints (array-like): Indices where changepoints occur (can be list, numpy array, etc.) 
            Can be binary (0/1) or list of indices.
    Returns:
        np.ndarray: Modified labels array.
    Example:
    % Changepoint binary:
    %               [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    % Labels in:    [0, 0, 2, 2, 3, 3, 0, 0, 0, 4, 4]
    % Labels out:   [0, 0, 2, 2, 3, 3, 3, 0, 0, 4, 4]
    """
    labels_out = np.array(labels).reshape(-1)
    

    changepoints_arr = np.array(changepoints)
    if changepoints_arr.dtype == bool or (
        changepoints_arr.dtype == int and set(np.unique(changepoints_arr)).issubset({0, 1})
    ):
        changepoints_idxs = set(np.where(changepoints_arr)[0])
    else:
        changepoints_idxs = set(changepoints)
    
    # Find segment endings: where current is non-zero and next is zero
    is_nonzero = labels_out != 0
    is_zero_next = np.concatenate([labels_out[1:] == 0, [False]])
    segment_ends = np.where(is_nonzero & is_zero_next)[0]
    
    # Extend labels at segment ends if there's a changepoint
    for seg_end in segment_ends:
        if (seg_end + 1) in changepoints_idxs:
            if labels_out[seg_end] != 0 and labels_out[seg_end + 1] == 0:
                labels_out[seg_end + 1] = labels_out[seg_end]

    return labels_out




    
def plot_label_segments(ax, time_data, labels, label_mappings, is_main=True, fraction=0.2):
    """Plot label segments for a given data array.
    
    Args:
        ax: Matplotlib axis to plot on
        labels: Label/prediction data array
        label_mappings: Dict mapping labels to color info
        fps: Frames per second for time conversion (optional)
        is_main: If True, plot full-height rectangles; if False, plot small rectangles at top
    """

    
    current_label = 0
    segment_start = None
    
    for i, label in enumerate(labels):
        if label != 0:
            if label != current_label:
                if current_label != 0 and segment_start is not None:
                    draw_label_rectangle(
                        ax,
                        time_data[segment_start],
                        time_data[i - 1],
                        current_label,
                        label_mappings,
                        is_main,
                        fraction=fraction
                    )
                
                current_label = label
                segment_start = i
        else:
            if current_label != 0 and segment_start is not None:
                draw_label_rectangle(
                    ax,
                    time_data[segment_start],
                    time_data[i - 1],
                    current_label,
                    label_mappings,
                    is_main,
                    fraction=fraction
                )
                current_label = 0
                segment_start = None
    
    if current_label != 0 and segment_start is not None:
        draw_label_rectangle(
            ax,
            time_data[segment_start],
            time_data[-1],
            current_label,
            label_mappings,
            is_main,
            fraction=fraction
        )

def draw_label_rectangle(ax, start_time, end_time, labels, label_mappings, is_main=True, fraction=None):
    """Draw label rectangle using matplotlib.
    
    Args:
        ax: Matplotlib axis to plot on
        start_time: Start time of the label
        end_time: End time of the label
        labels: ID of the label for color mapping
        label_mappings: Dict mapping labels to color info
        is_main: If True, draw full-height rectangle; if False, draw small rectangle at top
    """
    if labels not in label_mappings:
        return
    
    color = label_mappings[labels]["color"]
    
    if is_main:
        ax.axvspan(
            start_time, end_time,
            alpha=0.7,
            color=color,
            zorder=-10
        )
    else:
        y_min, y_max = ax.get_ylim()
        height = (y_max - y_min) * fraction
        
        rect = plt.Rectangle(
            (start_time, y_max - height),
            end_time - start_time,
            height,
            color=color,
            alpha=0.8,
            zorder=10
        )
        ax.add_patch(rect)
        
        
        
def plot_label_segments_multirow(
    ax: plt.Axes,
    time_data: np.ndarray,
    labels: np.ndarray,
    label_mappings: Dict[int, Dict[str, str]],
    row_index: int = 0,
    row_spacing: float = 0.8,
    rect_height: float = 0.7,
    alpha: float = 0.7
) -> None:
    """Plot label segments at a specific row position.
    
    Args:
        ax: Matplotlib axis to plot on
        time_data: Time array for x-axis positioning
        labels: Label/prediction data array
        label_mappings: Dict mapping labels to color info
        row_index: Row number (0-based) for vertical positioning
        row_spacing: Vertical spacing between rows
        rect_height: Height of each rectangle
        alpha: Transparency of rectangles
    """
    y_base = row_index * row_spacing
    
    current_label = 0
    segment_start = None
    
    for i, label in enumerate(labels):
        # Ensure label is a scalar integer
        label = int(label) if hasattr(label, 'item') else int(label)
        
        if label != 0:
            if label != current_label:
                if current_label != 0 and segment_start is not None:
                    _draw_rectangle(
                        ax, time_data[segment_start], time_data[i - 1],
                        y_base, rect_height, current_label,
                        label_mappings, alpha
                    )
                
                current_label = label
                segment_start = i
        else:
            if current_label != 0 and segment_start is not None:
                _draw_rectangle(
                    ax, time_data[segment_start], time_data[i - 1],
                    y_base, rect_height, current_label,
                    label_mappings, alpha
                )
                current_label = 0
                segment_start = None
    
    if current_label != 0 and segment_start is not None:
        _draw_rectangle(
            ax, time_data[segment_start], time_data[-1],
            y_base, rect_height, current_label,
            label_mappings, alpha
        )


def _draw_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    y_base: float,
    height: float,
    labels: int,
    label_mappings: Dict[int, Dict[str, str]],
    alpha: float
) -> None:
    """Draw a single label rectangle."""
    # Ensure labels is a scalar integer
    labels = int(labels) if hasattr(labels, 'item') else int(labels)
    
    if labels not in label_mappings:
        return
    
    color = label_mappings[labels]["color"]
    
    rect = patches.Rectangle(
        (start_time, y_base),
        end_time - start_time,
        height,
        color=color,
        alpha=alpha,
        zorder=-10
    )
    ax.add_patch(rect)
    
    
    