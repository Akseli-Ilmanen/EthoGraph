"""Core label primitives shared across dense and interval representations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np


def get_segments(col, bg_class=0):
    """Example: [0,1,1,1,0,2,2] → [(1,1,4), (2,5,7)]"""
    padded = np.concatenate([[-1], col, [-1]])
    change_indices = np.nonzero(padded[:-1] != padded[1:])[0]

    segments = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]
        label = int(col[start])
        if label != bg_class:
            segments.append((label, start, end))
    return segments


def find_blocks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return starts, ends


def load_mapping(mapping_file):
    """Load class name to index mapping."""
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


def load_label_mapping(
    mapping_file: Union[str, Path] = "mapping.txt",
) -> Dict[int, Dict]:
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    label_colors = [
        [1, 1, 1],
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
        [7, 7, 215],
        [128, 0, 255],
        [255, 215, 0],
        [73, 113, 233],
        [255, 128, 0],
        [138, 34, 34],
        [188, 82, 223],
        [103, 176, 29],
        [220, 20, 60],
        [3, 243, 3],
        [147, 24, 147],
        [178, 111, 44],
        [16, 166, 166],
        [71, 197, 238],
        [255, 149, 114],
        [16, 89, 162],
        [26, 195, 68],
        [254, 216, 103],
        [0, 237, 118],
        [177, 177, 36],
        [73, 243, 200],
    ]

    GAP_COLOR = np.array([128, 128, 128]) / 255.0

    label_mappings = {}
    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0].startswith("("):
                nums = parts[0].strip("()").split(",")
                label_id = (int(nums[0]), int(nums[1]))
                order = int(parts[-1])
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": GAP_COLOR,
                    "order": order,
                }
            else:
                label_id = int(parts[0])
                order = int(parts[-1]) if len(parts) >= 3 else label_id
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": np.array(label_colors[label_id]) / 255.0,
                    "order": order,
                }

    return label_mappings

