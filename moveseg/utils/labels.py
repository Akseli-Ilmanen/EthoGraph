from matplotlib import patches
import numpy as np
from typing import Union, Dict
from pathlib import Path
import numpy as np
import os
from moveseg.utils.paths import get_project_root
from itertools import groupby
from itertools import groupby
from typing import Dict, List, Tuple, Any, Union
from scipy.signal import find_peaks, peak_prominences
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr




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


def load_motif_mapping(mapping_file: Union[str, Path] = "mapping.txt") -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load motif mapping from a text file and return mapping dictionary with colors.

    Args:
        mapping_file: Path to the mapping file (default: "mapping.txt")

    Returns:
        Dictionary mapping motif_id to {'name': str, 'color': np.ndarray}

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
        >>> mapping = load_motif_mapping("mapping.txt")
        >>> print(mapping[1]['name'])  # 'beakTip_pullOutStick'
        >>> print(mapping[1]['color'])  # [1.0, 0.4, 0.698]
    """
    mapping_file = Path(mapping_file)
   
    # RGB colours, where adjacent colours are maximally different
    motif_colors = [
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

    
    motif_mappings = {}
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                motif_id = int(parts[0])
                name = parts[1]
                
                color = np.array(motif_colors[motif_id]) / 255.0
                
                motif_mappings[motif_id] = {
                    'name': name,
                    'color': color
                }
    
    return motif_mappings



def labels_to_rgb(
    labels: Union[str, np.ndarray], 
    motif_mapping: Dict[int, Dict[str, np.ndarray]]
) -> np.ndarray:
    """
    Convert label sequence to RGB colors based on motif mapping.
    
    Args:
        labels: String of digits or array of integers representing motif IDs
        motif_mapping: Dictionary from load_motif_mapping()
        
    Returns:
        Array of shape (N, 3) with RGB values [0, 1] for each frame
        
    Example:
        >>> mapping = load_motif_mapping("mapping.txt")
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
    for motif_id in np.unique(labels):
        if motif_id in motif_mapping:
            mask = labels == motif_id
            rgb_array[mask] = motif_mapping[motif_id]['color']
        else:
            # Default to white for unmapped labels
            mask = labels == motif_id
            rgb_array[mask] = [1.0, 1.0, 1.0]
    
    return rgb_array


def get_labels_start_end_time(frame_wise_labels, bg_class=[0.0]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i) 
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
        
    return labels, starts, ends


def get_segment_indices(arr: List[int]) -> List[Tuple[int, int, int]]:
    """
    Alternative using groupby - more Pythonic for lists.
    """
    segments = []
    start = 0
    
    for value, group in groupby(arr):
        length = len(list(group))
        if value != 0:  # Skip zeros if desired
            segments.append((value, start, start + length))
        start += length
    
    return segments



def count_continuous_segments(ds: xr.Dataset) -> Dict[int, int]:
    """Count continuous segments for each unique label (excluding 0).
    
    Args:
        ds: Dataset with 'labels' DataArray containing sequential labels
        
    Returns:
        Dictionary mapping label ID to number of continuous segments
    """
    labels = ds.labels.values
    
    # Find where labels change (including transitions from/to 0)
    diff = np.diff(labels, prepend=labels[0]-1)
    change_indices = np.where(diff != 0)[0]
    
    # Get segment labels and count
    segment_counts = {}
    for i in range(len(change_indices)):
        end_idx = change_indices[i+1] if i+1 < len(change_indices) else len(labels)
        label = labels[change_indices[i]]
        
        if label == 0:
            continue

        segment_counts[int(label)] = segment_counts.get(int(label), 0) + 1

    return segment_counts



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


# Purge and remove equivalent except for toss exception
def purge_small_motifs(labels: np.ndarray, min_motif_len: int) -> np.ndarray:
    purged = labels.copy()
    
    for label in np.unique(labels):
        if label == 0:
            continue
        
        starts, ends = find_blocks(labels == label)
        
        for start, end in zip(starts, ends):
            block_len = end - start + 1
            
            # Toss exception - HARD CODED
            threshold = 6 if label == 3 else min_motif_len
            
            if block_len < threshold:
                purged[start:end + 1] = 0
    
    return purged

def remove_small_blocks(input_vec, min_motif_len):
    """
    Remove blocks shorter than min_motif_len from input_vec (set to 0).
    """
    
    if isinstance(input_vec, (str, bytes)):
        input_vec = np.array([int(c) for c in str(input_vec)])
    else:
        input_vec = np.array(input_vec)
    output_vec = input_vec.copy()
    i = 0
    while i < len(input_vec):
        if input_vec[i] != 0:
            val = input_vec[i]
            j = i
            while j < len(input_vec) and input_vec[j] == val:
                j += 1
            run_length = j - i
            if run_length < min_motif_len:
                output_vec[i:j] = 0
            i = j
        else:
            i += 1
    return output_vec


def fix_endings(labels, changepoints):
    """
    Args:
        labels (array-like): Sequence of integer labels.
        changepoints (array-like): Indices where changepoints occur (can be list, numpy array, etc.) 
            Can be binary (0/1) or list of indices.
    Returns:
        np.ndarray: Modified labels array.
    """
    labels_out = np.array(labels).reshape(-1)
    # Convert changepoints to indices if binary
    changepoints_arr = np.array(changepoints)
    if changepoints_arr.dtype == bool or (
        changepoints_arr.dtype == int and set(np.unique(changepoints_arr)).issubset({0, 1})
    ):
        changepoints_idxs = np.where(changepoints_arr)[0]
    else:
        changepoints_idxs = np.array(changepoints)
        
        
    change_positions = np.where(np.diff(labels_out) != 0)[0]
    segment_ends = change_positions
    for seg_end in segment_ends:
        if (seg_end + 1) in changepoints_idxs:
            if labels_out[seg_end] != 0 and labels_out[seg_end + 1] == 0:
                labels_out[seg_end + 1] = labels_out[seg_end]
    return labels_out


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


def create_classification_probabilities_pdf(label_dt, output_path: Union[str, Path],
                                           confidence_threshold: float = 0.95,
                                           segment_confidence_threshold: float = 0.85):
    """
    Create a PDF with classification probabilities plots for all trials in a label datatree.

    Args:
        label_dt: xarray DataTree containing labels and labels_confidence
        output_path: Path where to save the PDF file
        confidence_threshold: Threshold below which to highlight low confidence regions
        segment_confidence_threshold: Threshold for mean confidence within each label segment
    """
    output_path = Path(output_path)

    trial_nums = label_dt.trials
    N = len(trial_nums)

    mapping_path = get_project_root() / "configs" / "mapping.txt"
    motif_mappings = load_motif_mapping(mapping_path)
    class_colors = [motif_mappings[i]['color'] for i in range(len(motif_mappings))]
    num_classes = len(motif_mappings)
    
    n_cols = 3
    n_rows = (N + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, trial_num in enumerate(trial_nums):
        trial_ds = label_dt.sel(trials=trial_num)

        if 'labels_confidence' not in trial_ds:
            continue

        labels_confidence = trial_ds.labels_confidence.values.squeeze()
        labels = trial_ds.labels.values.squeeze()

        conf_per_class = np.zeros((num_classes, len(labels)))
        valid_mask = (labels >= 0) & (labels < num_classes)
        conf_per_class[labels[valid_mask].astype(int), np.where(valid_mask)[0]] = labels_confidence[valid_mask]

        remaining = 1.0 - labels_confidence[valid_mask]
        for t_idx, (label, rem) in enumerate(zip(labels[valid_mask], remaining)):
            label = int(label)
            conf_per_class[:, np.where(valid_mask)[0][t_idx]] += rem / (num_classes - 1)
            conf_per_class[label, np.where(valid_mask)[0][t_idx]] -= rem / (num_classes - 1)
                        
                        
                        
        ax = axes[idx]
        for i in range(1, len(conf_per_class)):
            if np.any(conf_per_class[i, :] > 0):  # Only plot if there's data
                ax.plot(conf_per_class[i, :], alpha=0.7, color=class_colors[i])


        max_probs = np.max(conf_per_class, axis=0)

        mask_low = max_probs < confidence_threshold
        ax.scatter(np.where(mask_low)[0], max_probs[mask_low],
                  color='black', s=10, label='Low Confidence', alpha=0.5)

        first = np.argmax(labels != 0)
        last = len(labels) - 1 - np.argmax((labels != 0)[::-1])
    
        confidence = np.mean(labels_confidence[first:last+1])


        
        segment_boundaries = np.concatenate(([0], np.where(np.diff(labels) != 0)[0] + 1, [len(labels)]))
        has_low_segment = False
        
        
        for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
            segment_label = labels[start]
            if segment_label > 0:
                segment_conf = np.mean(max_probs[start:end])
                if segment_conf < segment_confidence_threshold:
                    has_low_segment = True
                    ax.axvspan(start, end, color='red', alpha=0.3, zorder=5)
                    
            

        trial_name = f"trial-{trial_num}"
        ax.set_title(f'{trial_name}\nMean confidence: {confidence:.3f}', fontsize=10)

        label_dt.sel(trials=trial_num).attrs['mean_model_confidence'] = float(confidence)
        
        
        if confidence < confidence_threshold or has_low_segment:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
            ax.set_title(f'{trial_name}\nConfidence: {confidence:.3f}',
                        fontsize=10, color='red', weight='bold')
            label_dt.sel(trials=trial_num).attrs['model_confidence'] = 'low'
        else:
            label_dt.sel(trials=trial_num).attrs['model_confidence'] = 'high'
            
        
            
        
        
    
            
            

        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return label_dt

    
def plot_motif_segments(ax, time_data, labels, motif_mappings, is_main=True, fraction=0.2):
    """Plot motif segments for a given data array.
    
    Args:
        ax: Matplotlib axis to plot on
        labels: Label/prediction data array
        motif_mappings: Dict mapping motif_id to color info
        fps: Frames per second for time conversion (optional)
        is_main: If True, plot full-height rectangles; if False, plot small rectangles at top
    """

    
    current_motif_id = 0
    segment_start = None
    
    for i, label in enumerate(labels):
        if label != 0:
            if label != current_motif_id:
                if current_motif_id != 0 and segment_start is not None:
                    draw_motif_rectangle(
                        ax,
                        time_data[segment_start],
                        time_data[i - 1],
                        current_motif_id,
                        motif_mappings,
                        is_main,
                        fraction=fraction
                    )
                
                current_motif_id = label
                segment_start = i
        else:
            if current_motif_id != 0 and segment_start is not None:
                draw_motif_rectangle(
                    ax,
                    time_data[segment_start],
                    time_data[i - 1],
                    current_motif_id,
                    motif_mappings,
                    is_main,
                    fraction=fraction
                )
                current_motif_id = 0
                segment_start = None
    
    if current_motif_id != 0 and segment_start is not None:
        draw_motif_rectangle(
            ax,
            time_data[segment_start],
            time_data[-1],
            current_motif_id,
            motif_mappings,
            is_main,
            fraction=fraction
        )

def draw_motif_rectangle(ax, start_time, end_time, motif_id, motif_mappings, is_main=True, fraction=None):
    """Draw motif rectangle using matplotlib.
    
    Args:
        ax: Matplotlib axis to plot on
        start_time: Start time of the motif
        end_time: End time of the motif
        motif_id: ID of the motif for color mapping
        motif_mappings: Dict mapping motif_id to color info
        is_main: If True, draw full-height rectangle; if False, draw small rectangle at top
    """
    if motif_id not in motif_mappings:
        return
    
    color = motif_mappings[motif_id]["color"]
    
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
        
        
        
def plot_motif_segments_multirow(
    ax: plt.Axes,
    time_data: np.ndarray,
    labels: np.ndarray,
    motif_mappings: Dict[int, Dict[str, str]],
    row_index: int = 0,
    row_spacing: float = 0.8,
    rect_height: float = 0.7,
    alpha: float = 0.7
) -> None:
    """Plot motif segments at a specific row position.
    
    Args:
        ax: Matplotlib axis to plot on
        time_data: Time array for x-axis positioning
        labels: Label/prediction data array
        motif_mappings: Dict mapping motif_id to color info
        row_index: Row number (0-based) for vertical positioning
        row_spacing: Vertical spacing between rows
        rect_height: Height of each rectangle
        alpha: Transparency of rectangles
    """
    y_base = row_index * row_spacing
    
    current_motif_id = 0
    segment_start = None
    
    for i, label in enumerate(labels):
        # Ensure label is a scalar integer
        label = int(label) if hasattr(label, 'item') else int(label)
        
        if label != 0:
            if label != current_motif_id:
                if current_motif_id != 0 and segment_start is not None:
                    _draw_rectangle(
                        ax, time_data[segment_start], time_data[i - 1],
                        y_base, rect_height, current_motif_id,
                        motif_mappings, alpha
                    )
                
                current_motif_id = label
                segment_start = i
        else:
            if current_motif_id != 0 and segment_start is not None:
                _draw_rectangle(
                    ax, time_data[segment_start], time_data[i - 1],
                    y_base, rect_height, current_motif_id,
                    motif_mappings, alpha
                )
                current_motif_id = 0
                segment_start = None
    
    if current_motif_id != 0 and segment_start is not None:
        _draw_rectangle(
            ax, time_data[segment_start], time_data[-1],
            y_base, rect_height, current_motif_id,
            motif_mappings, alpha
        )


def _draw_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    y_base: float,
    height: float,
    motif_id: int,
    motif_mappings: Dict[int, Dict[str, str]],
    alpha: float
) -> None:
    """Draw a single motif rectangle."""
    # Ensure motif_id is a scalar integer
    motif_id = int(motif_id) if hasattr(motif_id, 'item') else int(motif_id)
    
    if motif_id not in motif_mappings:
        return
    
    color = motif_mappings[motif_id]["color"]
    
    rect = patches.Rectangle(
        (start_time, y_base),
        end_time - start_time,
        height,
        color=color,
        alpha=alpha,
        zorder=-10
    )
    ax.add_patch(rect)
    
    
    