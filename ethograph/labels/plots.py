from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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