"""Batch migrate .nc files from dense label format to interval-based format.

Usage:
    python migrate_dense_to_intervals.py /path/to/file1.nc /path/to/file2.nc
    python migrate_dense_to_intervals.py /path/to/folder/  # all .nc files in folder

The script opens each file, converts dense labels to interval format via
TrialTree.get_label_dt() (which auto-detects and converts), then overwrites
the labels back and saves. Files already in interval format are skipped.
"""

import sys
from pathlib import Path

from ethograph import TrialTree


def is_dense_format(dt: TrialTree) -> bool:
    """Check if any trial in the tree uses the old dense label format."""
    for node in dt.children.values():
        ds = node.ds
        if ds is None:
            continue
        if "labels" in ds.data_vars:
            da = ds["labels"]
            has_time_dim = any("time" in str(d).lower() for d in da.dims)
            if has_time_dim:
                return True
    return False


def migrate_file(path: Path) -> str:
    """Migrate a single .nc file. Returns status string."""
    dt = TrialTree.open(str(path))

    if not is_dense_format(dt):
        return "skipped (already interval format)"

    n_trials = len(dt.trials)
    label_dt = dt.get_label_dt()
    dt = dt.overwrite_with_labels(label_dt)
    dt.save(path)
    return f"migrated ({n_trials} trials)"


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
