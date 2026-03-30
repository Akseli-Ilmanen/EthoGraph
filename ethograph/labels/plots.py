from __future__ import annotations

from typing import Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_label_segments(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_mappings: Dict[int, Dict],
    individual: Optional[str] = None,
    is_main: bool = True,
    fraction: float = 0.2,
) -> None:
    """Plot label segments from an intervals DataFrame.

    Args:
        ax: Matplotlib axis to plot on
        df: Intervals DataFrame with columns onset_s, offset_s, labels, individual
        label_mappings: Dict mapping label IDs to color info
        individual: If given, only plot segments for this individual
        is_main: If True, plot full-height rectangles; if False, plot small rectangles at top
        fraction: Height fraction for non-main rectangles

    Example::

        import ethograph as eto
        from ethograph.labels.intervals import load_label_mapping

        dt = eto.open("data.nc")
        label_mappings = load_label_mapping("mapping.txt")

        fig, ax = plt.subplots()
        # df is an intervals DataFrame with onset_s, offset_s, labels, individual
        plot_label_segments(ax, df, label_mappings)
        plt.show()
    """
    if individual is not None:
        df = df[df["individual"] == individual]

    for _, row in df.iterrows():
        draw_label_rectangle(
            ax, row["onset_s"], row["offset_s"], int(row["labels"]),
            label_mappings, is_main, fraction=fraction,
        )


def draw_label_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    labels: int,
    label_mappings: Dict[int, Dict],
    is_main: bool = True,
    fraction: Optional[float] = None,
) -> None:
    """Draw a label rectangle on a matplotlib axis.

    Args:
        ax: Matplotlib axis to plot on
        start_time: Start time of the label
        end_time: End time of the label
        labels: Label class ID for color mapping
        label_mappings: Dict mapping label IDs to color info
        is_main: If True, draw full-height rectangle; if False, draw small rectangle at top
        fraction: Height fraction for non-main rectangles

    Example::

        fig, ax = plt.subplots()
        ax.plot(time, signal)
        draw_label_rectangle(ax, 1.2, 3.5, label_id=1, label_mappings=label_mappings)
    """
    if labels not in label_mappings:
        return

    color = label_mappings[labels]["color"]

    if is_main:
        ax.axvspan(start_time, end_time, alpha=0.7, color=color, zorder=-10)
    else:
        y_min, y_max = ax.get_ylim()
        height = (y_max - y_min) * fraction
        rect = patches.Rectangle(
            (start_time, y_max - height),
            end_time - start_time,
            height,
            color=color,
            alpha=0.8,
            zorder=10,
        )
        ax.add_patch(rect)


def plot_label_segments_multirow(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_mappings: Dict[int, Dict[str, str]],
    row_index: int = 0,
    row_spacing: float = 0.8,
    rect_height: float = 0.7,
    alpha: float = 0.7,
    individual: Optional[str] = None,
) -> None:
    """Plot label segments at a specific row position.

    Useful for comparing ground truth vs. predictions on the same axis by
    placing each on a different row.

    Args:
        ax: Matplotlib axis to plot on
        df: Intervals DataFrame with columns onset_s, offset_s, labels, individual
        label_mappings: Dict mapping label IDs to color info
        row_index: Row number (0-based) for vertical positioning
        row_spacing: Vertical spacing between rows
        rect_height: Height of each rectangle
        alpha: Transparency of rectangles
        individual: If given, only plot segments for this individual

    Example::

        import ethograph as eto
        from ethograph.labels.intervals import load_label_mapping

        dt = eto.open("data.nc")
        pred_dt = eto.open("predictions.nc")
        label_mappings = load_label_mapping("mapping.txt")

        fig, ax = plt.subplots()
        ax.set_yticks([0, 0.8])
        ax.set_yticklabels(["ground truth", "predictions"])

        # gt_df, pred_df are intervals DataFrames with onset_s, offset_s, labels, individual
        gt_df = ...
        pred_df = ...

        plot_label_segments_multirow(ax, gt_df, label_mappings, row_index=0)
        plot_label_segments_multirow(ax, pred_df, label_mappings, row_index=1)
        plt.show()
    """
    if individual is not None:
        df = df[df["individual"] == individual]

    y_base = row_index * row_spacing
    for _, row in df.iterrows():
        _draw_rectangle(
            ax, row["onset_s"], row["offset_s"],
            y_base, rect_height, int(row["labels"]), label_mappings, alpha,
        )


def _draw_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    y_base: float,
    height: float,
    labels: int,
    label_mappings: Dict[int, Dict[str, str]],
    alpha: float,
) -> None:
    if labels not in label_mappings:
        return

    color = label_mappings[labels]["color"]
    rect = patches.Rectangle(
        (start_time, y_base),
        end_time - start_time,
        height,
        color=color,
        alpha=alpha,
        zorder=-10,
    )
    ax.add_patch(rect)
