"""Batch migrate .nc files from dense label format to interval-based format.

Usage:
    python migrate_dense_to_intervals.py /path/to/file1.nc /path/to/file2.nc
    python migrate_dense_to_intervals.py /path/to/folder/  # all .nc files in folder

Handles three cases:
    1. Dense format (labels with time dim) -> converts to interval format
    2. Old interval format (label_id column) -> renames to labels
    3. Current interval format (labels with segment dim) -> skipped

The script uses TrialTree.get_label_dt() which auto-detects and converts,
then overwrites the labels back and saves.
"""

import sys
from pathlib import Path

from ethograph import TrialTree


def detect_format(dt: TrialTree) -> str:
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

        # Check for dense: 'labels' variable with a time dimension
        if "labels" in ds.data_vars:
            da = ds["labels"]
            has_time_dim = any("time" in str(d).lower() for d in da.dims)
            has_segment_dim = "segment" in da.dims
            if has_time_dim:
                return "dense"
            if has_segment_dim:
                return "interval"

        # Check for old interval format: label_id with segment dimension
        if "label_id" in ds.data_vars and "segment" in ds.dims:
            return "old_interval"

        # Check for interval format without labels var (onset_s present)
        if "onset_s" in ds.data_vars and "segment" in ds.dims:
            return "interval"

    return "empty"


def migrate_file(path: Path) -> str:
    """Migrate a single .nc file. Returns status string."""
    dt = TrialTree.open(str(path))
    fmt = detect_format(dt)

    if fmt == "interval":
        return "skipped (already interval format)"

    if fmt == "empty":
        return "skipped (no label data)"

    n_trials = len(dt.trials)

    # get_label_dt() handles both dense->interval conversion
    # and label_id->labels rename
    label_dt = dt.get_label_dt()
    dt = dt.overwrite_with_labels(label_dt)
    dt.save(path)

    if fmt == "dense":
        return f"migrated dense->interval ({n_trials} trials)"
    return f"renamed label_id->labels ({n_trials} trials)"


def collect_nc_files(args: list[str]) -> list[Path]:
    """Resolve arguments to a list of .nc file paths."""
    files = []
    for arg in args:
        p = Path(arg)
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
