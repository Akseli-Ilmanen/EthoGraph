"""Quick test for TSV store with string label conversion."""
import tempfile
import os
import numpy as np
import pandas as pd
from ethograph.labels.tsv_store import (
    load_labels_tsv, save_labels_tsv, get_trial_from_tsv, set_trial_in_tsv,
    load_labels_meta, save_labels_meta, set_trial_meta_attr, get_trial_meta,
)
from ethograph.labels.intervals import empty_intervals

# Simulate label_mappings: {0: {"name": "background", ...}, 1: {"name": "Head bob", ...}}
mappings = {
    0: {"name": "background", "color": (0, 0, 0), "order": 0},
    1: {"name": "Head bob", "color": (1, 0, 0), "order": 1},
    2: {"name": "Wing flap", "color": (0, 1, 0), "order": 2},
    3: {"name": "Song", "color": (0, 0, 1), "order": 3},
}

# Internal format (integer labels)
all_df = pd.DataFrame({
    "trial": [1, 1, 1, 2, 2],
    "onset_s": [0.41, 0.51, 0.77, 0.10, 0.50],
    "offset_s": [0.505, 0.62, 0.885, 0.35, 0.80],
    "labels": np.array([1, 2, 3, 1, 2], dtype=np.int32),
    "individual": ["Poppy"] * 5,
})

# Save with mapping (should write string labels)
tmp = tempfile.mktemp(suffix="_labels.tsv")
save_labels_tsv(tmp, all_df, mappings)

# Check file content
content = open(tmp).read()
print("Saved TSV:")
print(content)
assert "Head bob" in content, "Should contain string labels"
assert "labels" not in content.split("\n")[0], "Should NOT have 'labels' column header"
assert "label" in content.split("\n")[0], "Should have 'label' column header"

# Load back with mapping (should restore integer labels)
loaded = load_labels_tsv(tmp, mappings)
print("Loaded DataFrame:")
print(loaded)
assert "labels" in loaded.columns, "Should have integer 'labels' column"
assert loaded["labels"].dtype == np.int32
assert list(loaded["labels"]) == [1, 2, 3, 1, 2]

# Per-trial access
trial1 = get_trial_from_tsv(loaded, 1)
print("\nTrial 1:")
print(trial1)
assert len(trial1) == 3
assert list(trial1["labels"]) == [1, 2, 3]

trial2 = get_trial_from_tsv(loaded, 2)
assert len(trial2) == 2

# Modify trial
new_trial1 = trial1.copy()
new_trial1.loc[0, "labels"] = 3  # Change first label
loaded = set_trial_in_tsv(loaded, 1, new_trial1)
check = get_trial_from_tsv(loaded, 1)
assert check.loc[0, "labels"] == 3

# Metadata round-trip
meta = {}
set_trial_meta_attr(meta, 1, "human_verified", 1)
set_trial_meta_attr(meta, 2, "human_verified", 0)
set_trial_meta_attr(meta, 1, "model_confidence", "high")

meta_tmp = tempfile.mktemp(suffix="_meta.tsv")
save_labels_meta(meta_tmp, meta)
loaded_meta = load_labels_meta(meta_tmp)
assert get_trial_meta(loaded_meta, 1)["human_verified"] == 1
assert get_trial_meta(loaded_meta, 1)["model_confidence"] == "high"
assert get_trial_meta(loaded_meta, 2)["human_verified"] == 0

os.unlink(tmp)
os.unlink(meta_tmp)
print("\nAll TSV store tests passed!")
