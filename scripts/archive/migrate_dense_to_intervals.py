"""Batch migrate .nc files to current ethograph format.

Usage:
    python migrate_dense_to_intervals.py /path/to/file1.nc /path/to/file2.nc
    python migrate_dense_to_intervals.py /path/to/folder/  # all .nc files in folder

Migrations performed:
    Labels:
        1. Dense format (labels with time dim) -> interval format
        2. Old interval format (label_id column) -> renames to labels
        3. Current interval format -> skipped

    Media attrs:
        4. Key indirection (cameras: [cam1, cam2] + cam1: file.mp4) -> direct paths (cameras: [file.mp4])
        5. tracking -> pose rename
        6. Removes stale individual key attrs (cam1, cam2, mic1, dlc1, etc.)
"""

import gc
import re
import sys
from pathlib import Path

import numpy as np

from ethograph import TrialTree


def detect_label_format(dt: TrialTree) -> str:
    """Detect label format across all trials.

    Returns:
        'dense' - old format with labels as time-series array
        'old_interval' - interval format with legacy 'label_id' column
        'interval' - current interval format with 'labels' column
        'empty' - no label data found
    """
    for node in dt.children.values():
        ds = node.ds
        if ds is None:
            continue

        if "labels" in ds.data_vars:
            da = ds["labels"]
            has_time_dim = any("time" in str(d).lower() for d in da.dims)
            has_segment_dim = "segment" in da.dims
            if has_time_dim:
                return "dense"
            if has_segment_dim:
                return "interval"

        if "label_id" in ds.data_vars and "segment" in ds.dims:
            return "old_interval"

        if "onset_s" in ds.data_vars and "segment" in ds.dims:
            return "interval"

    return "empty"


def needs_media_migration(ds) -> bool:
    """Check if dataset uses old key-indirection media attrs.

    Old format: cameras: ['cam1', 'cam2'] + cam1: 'file.mp4'
    New format: cameras: ['file.mp4', 'file2.mp4']

    Also detects 'tracking' attr (should be renamed to 'pose').
    """
    if "tracking" in ds.attrs:
        return True

    for attr_name in ("cameras", "mics"):
        keys = ds.attrs.get(attr_name)
        if keys is None:
            continue
        keys = np.atleast_1d(keys)
        if len(keys) > 0 and str(keys[0]) in ds.attrs:
            return True

    return False


def migrate_media_attrs(ds) -> tuple[dict, list[str]]:
    """Migrate media attrs from key-indirection to direct file paths.

    Returns (new_attrs, changes) where changes is a list of descriptions.
    """
    attrs = dict(ds.attrs)
    changes = []

    for old_name, new_name in [("cameras", "cameras"), ("mics", "mics"), ("tracking", "pose")]:
        keys = attrs.pop(old_name, None)
        if keys is None:
            continue

        keys = list(np.atleast_1d(keys))

        resolved = []
        stale_keys = []
        for key in keys:
            key_str = str(key)
            if key_str in attrs:
                resolved.append(attrs.pop(key_str))
                stale_keys.append(key_str)
            else:
                resolved.append(key_str)

        if not resolved:
            continue

        attrs[new_name] = resolved

        if old_name != new_name:
            changes.append(f"{old_name}->{new_name}")
        if stale_keys:
            changes.append(f"resolved {len(stale_keys)} keys ({', '.join(stale_keys)})")

    stale_pattern = re.compile(r"^(cam|mic|dlc)\d+$")
    stale_remaining = [k for k in attrs if stale_pattern.match(k)]
    for k in stale_remaining:
        attrs.pop(k)
    if stale_remaining:
        changes.append(f"removed {', '.join(stale_remaining)}")

    return attrs, changes


def migrate_file(path: Path) -> str:
    """Migrate a single .nc file. Returns status string."""
    dt = TrialTree.open(str(path))
    dt.load()
    dt.close()
    gc.collect()

    label_fmt = detect_label_format(dt)
    parts = []
    changed = False

    # --- Label migration ---
    if label_fmt in ("dense", "old_interval"):
        n_trials = len(dt.trials)
        label_dt = dt.get_label_dt()
        dt = dt.overwrite_with_labels(label_dt)
        changed = True
        if label_fmt == "dense":
            parts.append(f"labels: dense->interval ({n_trials} trials)")
        else:
            parts.append(f"labels: label_id->labels ({n_trials} trials)")

    # --- Media attrs migration ---
    sample_ds = dt.itrial(0)
    if needs_media_migration(sample_ds):
        for trial_num in dt.trials:
            trial_key = TrialTree.trial_key(trial_num)
            trial_ds = dt[trial_key].ds.copy()
            new_attrs, _ = migrate_media_attrs(trial_ds)
            trial_ds.attrs = new_attrs
            dt[trial_key] = TrialTree(trial_ds)
        changed = True
        _, change_details = migrate_media_attrs(sample_ds)
        parts.append(f"attrs: {', '.join(change_details)}")

    if not changed:
        return "skipped (already current format)"

    dt.save(path)
    return "; ".join(parts)


def collect_nc_files(args: list[str]) -> list[Path]:
    """Resolve arguments to a list of .nc file paths."""
    files = []
    for arg in args:
        cleaned = arg.strip().strip(",").strip("'\"")
        if cleaned.startswith(("r'", 'r"')):
            cleaned = cleaned[2:].strip("'\"")
        p = Path(cleaned)
        if p.is_dir():
            files.extend(sorted(p.glob("*.nc")))
        elif p.is_file() and p.suffix == ".nc":
            files.append(p)
        else:
            print(f"  WARNING: skipping {arg} (not a .nc file or directory)")
    return files


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = collect_nc_files(sys.argv[1:])
    if not files:
        print("No .nc files found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to process.\n")

    for path in files:
        print(f"  {path.name} ... ", end="", flush=True)
        try:
            status = migrate_file(path)
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

# e.g. use
# (ethograph) PS D:\Akseli\Code\ethograph> python scripts\migrate_dense_to_intervals.py       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250306_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250309_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250503_02\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250514_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250504_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250505_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250307_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250308_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250506_02\behav\Trial_data.nc",
# >>       
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250507_02\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250507_03\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250508_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250508_02\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250509_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250512_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250513_01\behav\Trial_data.nc",      
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250515_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250516_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250519_01\behav\Trial_data.nc",
# >>
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250521_01\behav\Trial_data.nc",
# >>       "D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250522_01\behav\Trial_data.nc"
# Found 21 file(s) to process.