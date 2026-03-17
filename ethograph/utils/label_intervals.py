"""Interval-based label representation for EthoGraph.

Labels are stored as a pandas DataFrame with columns:
    onset_s   (float64) - start time in seconds
    offset_s  (float64) - end time in seconds
    labels  (int)     - label class ID (nonzero)
    individual (str)    - individual identifier
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from ethograph.utils.labels import get_labels_start_end_times

if TYPE_CHECKING:
    from ethograph.utils.trialtree import TrialTree

INTERVAL_COLUMNS = ["onset_s", "offset_s", "labels", "individual"]

INTERVAL_DTYPES = {
    "onset_s": np.float64,
    "offset_s": np.float64,
    "labels": np.int32,
    "individual": object,
}


def empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=INTERVAL_DTYPES[col]) for col in INTERVAL_COLUMNS}
    )


def dense_to_intervals(
    dense_array: np.ndarray,
    time_coord: np.ndarray,
    individuals: list[str],
) -> pd.DataFrame:
    """Convert dense (time, individuals) label array to intervals DataFrame.

    Handles both 1-D (single individual) and 2-D arrays.
    """
    dense_array = np.asarray(dense_array)
    time_coord = np.asarray(time_coord)

    if dense_array.ndim == 1:
        dense_array = dense_array[:, np.newaxis]

    if dense_array.shape[1] != len(individuals):
        raise ValueError(
            f"dense_array has {dense_array.shape[1]} columns but "
            f"{len(individuals)} individuals given"
        )

    rows = []
    for ind_idx, ind_name in enumerate(individuals):
        col = dense_array[:, ind_idx]
        rows.extend(get_labels_start_end_times(col, time_coord, str(ind_name)))

    return _rows_to_df(rows)



def intervals_to_dense(
    df: pd.DataFrame,
    sample_rate: float,
    duration: float,
    individuals: list[str],
    n_samples: int | None = None,
) -> np.ndarray:
    """Convert intervals DataFrame to dense (n_samples, n_individuals) array.

    Args:
        n_samples: If given, overrides the duration-based calculation.
    """
    if n_samples is None:
        n_samples = int(round(duration * sample_rate)) + 1
    dense = np.zeros((n_samples, len(individuals)), dtype=np.int8)

    ind_to_idx = {name: i for i, name in enumerate(individuals)}

    for _, row in df.iterrows():
        ind_idx = ind_to_idx.get(row["individual"])
        if ind_idx is None:
            continue
        start_idx = int(round(row["onset_s"] * sample_rate))
        end_idx = int(round(row["offset_s"] * sample_rate))
        start_idx = max(0, start_idx)
        end_idx = min(n_samples - 1, end_idx)
        dense[start_idx : end_idx + 1, ind_idx] = int(row["labels"])

    return dense


def intervals_to_xr(df: pd.DataFrame) -> xr.Dataset:
    """Convert intervals DataFrame to xarray Dataset with 'segment' dimension."""
    if df.empty:
        return xr.Dataset(
            {
                "onset_s": ("segment", np.array([], dtype=np.float64)),
                "offset_s": ("segment", np.array([], dtype=np.float64)),
                "labels": ("segment", np.array([], dtype=np.int32)),
                "individual": ("segment", np.array([], dtype="<U1")),
            }
        )
    df_reset = df.reset_index(drop=True)
    ind_values = np.array(df_reset["individual"].tolist(), dtype="U")
    return xr.Dataset(
        {
            "onset_s": ("segment", df_reset["onset_s"].values.astype(np.float64)),
            "offset_s": ("segment", df_reset["offset_s"].values.astype(np.float64)),
            "labels": ("segment", df_reset["labels"].values.astype(np.int32)),
            "individual": ("segment", ind_values),
        }
    )


def xr_to_intervals(ds: xr.Dataset) -> pd.DataFrame:
    """Convert xarray Dataset with 'segment' dimension back to intervals DataFrame."""
    if "onset_s" not in ds.data_vars:
        return empty_intervals()
    df = pd.DataFrame(
        {
            "onset_s": ds["onset_s"].values.astype(np.float64),
            "offset_s": ds["offset_s"].values.astype(np.float64),
            "labels": ds["labels"].values.astype(np.int32),
            "individual": ds["individual"].values.astype(str),
        }
    )
    return df


def add_interval(
    df: pd.DataFrame,
    onset_s: float,
    offset_s: float,
    labels: int,
    individual: str,
) -> pd.DataFrame:
    """Add an interval, resolving overlaps for the same individual.

    Any existing intervals that overlap [onset_s, offset_s] for the same
    individual are split, trimmed, or deleted as needed.
    """
    if onset_s > offset_s:
        onset_s, offset_s = offset_s, onset_s

    mask_same_ind = df["individual"] == individual
    other = df[~mask_same_ind]
    same = df[mask_same_ind].copy()

    kept: list[dict] = []
    for _, row in same.iterrows():
        ro, rf = row["onset_s"], row["offset_s"]
        rid = row["labels"]

        if rf <= onset_s or ro >= offset_s:
            kept.append(row.to_dict())
            continue

        eps = 1e-3
        if ro < onset_s:
            kept.append(
                {"onset_s": ro, "offset_s": onset_s - eps, "labels": rid, "individual": individual}
            )
        if rf > offset_s:
            kept.append(
                {"onset_s": offset_s + eps, "offset_s": rf, "labels": rid, "individual": individual}
            )

    kept.append(
        {"onset_s": onset_s, "offset_s": offset_s, "labels": labels, "individual": individual}
    )

    new_same = _rows_to_df(kept)
    result = pd.concat([other, new_same], ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def delete_interval(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Drop interval by DataFrame index."""
    return df.drop(index=idx).reset_index(drop=True)


def find_interval_at(df: pd.DataFrame, time_s: float, individual: str) -> int | None:
    """Return DataFrame index of interval containing time_s for given individual."""
    mask = (
        (df["individual"] == individual)
        & (df["onset_s"] <= time_s)
        & (df["offset_s"] >= time_s)
        & (df["labels"] != 0)
    )
    matches = df.index[mask]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_interval_bounds(df: pd.DataFrame, idx: int) -> tuple[float, float, int]:
    """Return (onset_s, offset_s, labels) for interval at index."""
    row = df.loc[idx]
    return float(row["onset_s"]), float(row["offset_s"]), int(row["labels"])


def purge_short_intervals(
    df: pd.DataFrame,
    min_duration_s: float,
    label_thresholds_s: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Drop intervals shorter than threshold (in seconds)."""
    if label_thresholds_s is None:
        label_thresholds_s = {}

    durations = df["offset_s"] - df["onset_s"]
    thresholds = df["labels"].map(
        lambda lid: label_thresholds_s.get(lid, min_duration_s)
    )
    keep = durations >= thresholds
    return df[keep].reset_index(drop=True)


def stitch_intervals(
    df: pd.DataFrame,
    max_gap_s: float,
    individual: str | None = None,
) -> pd.DataFrame:
    """Merge adjacent same-label intervals where gap <= threshold."""
    if df.empty:
        return df.copy()

    if individual is not None:
        mask = df["individual"] == individual
        other = df[~mask]
        target = df[mask].copy()
    else:
        other = empty_intervals()
        target = df.copy()

    target.sort_values(["individual", "onset_s"], inplace=True)
    target.reset_index(drop=True, inplace=True)

    merged: list[dict] = []
    i = 0
    while i < len(target):
        row = target.iloc[i]
        current = row.to_dict()
        j = i + 1
        while j < len(target):
            nxt = target.iloc[j]
            if (
                nxt["individual"] == current["individual"]
                and nxt["labels"] == current["labels"]
                and (nxt["onset_s"] - current["offset_s"]) < max_gap_s
            ):
                current["offset_s"] = nxt["offset_s"]
                j += 1
            else:
                break
        merged.append(current)
        i = j

    result = pd.concat([other, _rows_to_df(merged)], ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def snap_boundaries(
    df: pd.DataFrame,
    cp_times: np.ndarray,
    max_expansion_s: float,
    max_shrink_s: float,
) -> pd.DataFrame:
    """Snap interval onset/offset to nearest changepoint times.

    For each boundary, finds the nearest CP time and snaps if the move is
    within the expansion/shrink limits. Falls back to the original boundary
    if snapping would create an invalid range (onset >= offset).

    Post-processes to resolve overlaps between adjacent intervals of different
    labels by trimming the earlier interval's offset.
    """
    if df.empty or len(cp_times) == 0:
        return df.copy()

    cp_times = np.sort(cp_times)
    rows = []

    for _, row in df.iterrows():
        onset = row["onset_s"]
        offset = row["offset_s"]

        snap_onset = _snap_onset(onset, cp_times, max_expansion_s, max_shrink_s)
        snap_offset = _snap_offset(offset, cp_times, max_expansion_s, max_shrink_s)

        if snap_onset >= snap_offset:
            snap_onset = onset
            snap_offset = offset

        rows.append({
            "onset_s": snap_onset,
            "offset_s": snap_offset,
            "labels": row["labels"],
            "individual": row["individual"],
        })

    result = _rows_to_df(rows)
    result.sort_values(["individual", "onset_s"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = _resolve_overlaps(result)
    return result


def _snap_onset(
    boundary: float, cp_times: np.ndarray, max_expansion_s: float, max_shrink_s: float
) -> float:
    """Snap an onset boundary. Expanding = moving earlier, shrinking = moving later."""
    nearest_idx = np.argmin(np.abs(cp_times - boundary))
    cp_val = float(cp_times[nearest_idx])
    expansion = boundary - cp_val  # positive when cp_val < boundary (expanding left)
    shrink = cp_val - boundary     # positive when cp_val > boundary (shrinking from left)
    if expansion > max_expansion_s or shrink > max_shrink_s:
        return boundary
    return cp_val


def _snap_offset(
    boundary: float, cp_times: np.ndarray, max_expansion_s: float, max_shrink_s: float
) -> float:
    """Snap an offset boundary. Expanding = moving later, shrinking = moving earlier."""
    nearest_idx = np.argmin(np.abs(cp_times - boundary))
    cp_val = float(cp_times[nearest_idx])
    expansion = cp_val - boundary  # positive when cp_val > boundary (expanding right)
    shrink = boundary - cp_val     # positive when cp_val < boundary (shrinking from right)
    if expansion > max_expansion_s or shrink > max_shrink_s:
        return boundary
    return cp_val


def _resolve_overlaps(df: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    """Trim overlaps between adjacent intervals of different labels per individual.

    Also ensures a minimum gap of *eps* between consecutive intervals so
    that boundaries remain visually distinguishable and pynapple does not
    merge them.
    """
    if df.empty:
        return df

    groups = []
    for ind, group in df.groupby("individual", sort=False):
        group = group.sort_values("onset_s").reset_index(drop=True)
        for i in range(len(group) - 1):
            if group.at[i, "offset_s"] > group.at[i + 1, "onset_s"]:
                if group.at[i, "labels"] != group.at[i + 1, "labels"]:
                    group.at[i, "offset_s"] = group.at[i + 1, "onset_s"] - eps
            elif group.at[i, "offset_s"] == group.at[i + 1, "onset_s"]:
                group.at[i, "offset_s"] = group.at[i + 1, "onset_s"] - eps
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result





CROWSETTA_SEQ_FORMATS = [
    "aud-seq",
    "simple-seq",
    "generic-seq",
    "notmat",
    "textgrid",
    "timit",
    "yarden",
]


def crowsetta_to_intervals(
    file_path: str | Path,
    format_name: str,
    name_to_id: dict[str, int],
    individual: str = "ind0",
) -> pd.DataFrame:
    """Convert a crowsetta annotation file to an intervals DataFrame.

    Args:
        file_path: Path to the annotation file.
        format_name: Crowsetta format name (e.g. 'aud-seq', 'simple-seq').
        name_to_id: Mapping from string label names to integer IDs.
        individual: Individual name to assign to all intervals.

    Returns:
        Intervals DataFrame with columns (onset_s, offset_s, labels, individual).
    """
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    seq = annot.seq

    rows: list[dict] = []
    for segment in seq.segments:
        label_str = str(segment.label)
        label_id = name_to_id.get(label_str)
        if label_id is None or label_id == 0:
            continue
        rows.append({
            "onset_s": float(segment.onset_s),
            "offset_s": float(segment.offset_s),
            "labels": label_id,
            "individual": individual,
        })

    return _rows_to_df(rows)


def build_mapping_from_labels(string_labels: list[str]) -> dict[str, int]:
    """Build a name->id mapping from a list of unique string labels.

    Sorts labels alphabetically; 0 is reserved for 'background'.
    """
    unique = sorted(set(string_labels))
    mapping = {"background": 0}
    for i, name in enumerate(unique, start=1):
        mapping[name] = i
    return mapping


def extract_crowsetta_labels(
    file_path: str | Path,
    format_name: str,
) -> list[str]:
    """Extract unique string labels from a crowsetta annotation file."""
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    return list({str(seg.label) for seg in annot.seq.segments})


def write_mapping_file(
    output_path: str | Path,
    name_to_id: dict[str, int],
) -> None:
    """Write a mapping file in '<id> <name>' format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{idx} {name}" for name, idx in sorted(name_to_id.items(), key=lambda x: x[1])]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_crowsetta_mapping(
    file_path: str | Path,
    format_name: str,
    mapping_path: str | Path,
    configs_dir: str | Path,
) -> tuple[dict[str, int], str | None, str | None]:
    """Check existing mapping against crowsetta labels; create new if needed.

    Returns:
        (name_to_id, new_mapping_path_or_None, warning_message_or_None)
        - name_to_id: the mapping to use for conversion
        - new_mapping_path: path to newly created mapping file, or None if existing was used
        - warning: message if partial overlap was detected
    """
    file_labels = extract_crowsetta_labels(file_path, format_name)

    mapping_path = Path(mapping_path)
    configs_dir = Path(configs_dir)

    existing_names: set[str] = set()
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        existing_names.add(parts[1])
        except (OSError, UnicodeDecodeError):
            pass

    non_bg_labels = [l for l in file_labels if l.lower() not in ("background", "sil")]
    if not non_bg_labels:
        non_bg_labels = file_labels

    overlap = existing_names & set(non_bg_labels)

    if overlap == set(non_bg_labels) and len(overlap) > 0:
        from ethograph.utils.labels import load_mapping
        class_to_idx, _ = load_mapping(str(mapping_path))
        return class_to_idx, None, None

    warning = None
    if overlap and overlap != set(non_bg_labels):
        missing = set(non_bg_labels) - overlap
        warning = (
            f"Mapping file contains {len(overlap)} of {len(non_bg_labels)} labels. "
            f"Missing: {sorted(missing)}"
        )

    name_to_id = build_mapping_from_labels(non_bg_labels)
    new_path = configs_dir / f"mapping_{format_name.replace('-', '_')}.txt"
    write_mapping_file(new_path, name_to_id)

    return name_to_id, str(new_path), warning


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_intervals()
    df = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    for col, dtype in INTERVAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


# ---------------------------------------------------------------------------
# NWB interval label converter
# ---------------------------------------------------------------------------

class NWBLabelConverter:
    name = "interval_labels"

    def __init__(self, include_sources: set[str] | None = None):
        self._epochs: list[dict] | None = None
        self._label_map: dict[str, int] = {}

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    def _load(self, nwb) -> None:
        self._epochs = self._extract_behavioral_epochs(nwb)
        self._label_map = build_mapping_from_labels(
            sorted({e["label_name"] for e in self._epochs})
        )

    def from_nwb(self, nwb, trials_df: pd.DataFrame) -> TrialTree:
        from ethograph.utils.nwb import _assign_individual
        
        if self._epochs is None:
            self._load(nwb)
            
            

        ds_list = []
        for _, row in trials_df.iterrows():
            t_start, t_stop = row["start_time"], row["stop_time"]
            rows = [
                {
                    "onset_s": max(0.0, ep["onset_s"] - t_start),
                    "offset_s": min(t_stop - t_start, ep["offset_s"] - t_start),
                    "labels": self._label_map.get(ep["label_name"], 0),
                    "individual": ep["individual"],
                }
                for ep in self._epochs
                if ep["offset_s"] > t_start
                and ep["onset_s"] < t_stop
                and self._label_map.get(ep["label_name"], 0) != 0
            ]
            ds = intervals_to_xr(_rows_to_df(rows)) if rows else xr.Dataset()
            ds = _assign_individual(ds, nwb)
            ds_list.append(ds.assign_attrs(trial=row["trial"]))

        import ethograph as eto  # local import to avoid circular dependency
        return eto.from_datasets(ds_list)

    def _extract_behavioral_epochs(self, nwb) -> list[dict]:
        import pynwb
        individual = _get_nwb_individual(nwb)
        epochs: list[dict] = []

        if nwb.epochs is not None and len(nwb.epochs) > 0:
            self._collect_time_intervals(nwb.epochs, "epochs", individual, epochs)

        for mod_key, mod in nwb.processing.items():
            for iface_key, iface in mod.data_interfaces.items():
                if isinstance(iface, pynwb.epoch.TimeIntervals):
                    self._collect_time_intervals(iface, f"{mod_key}/{iface_key}", individual, epochs)
                elif isinstance(iface, pynwb.behavior.BehavioralEpochs):
                    for series_name, series in iface.interval_series.items():
                        self._collect_interval_series(series, f"{mod_key}/{iface_key}/{series_name}", individual, epochs)

        return epochs

    @staticmethod
    def _collect_time_intervals(table, source: str, individual: str, out: list[dict]) -> None:
        df = table.to_dataframe()
        label_col = "label" if "label" in df.columns else None
        for _, row in df.iterrows():
            out.append({
                "onset_s": float(row["start_time"]),
                "offset_s": float(row["stop_time"]),
                "label_name": str(row[label_col]) if label_col else source.split("/")[-1],
                "individual": individual,
                "source": source,
            })

    @staticmethod
    def _collect_interval_series(series, source: str, individual: str, out: list[dict]) -> None:
        data = np.asarray(series.data[:])
        timestamps = np.asarray(series.timestamps[:])
        label_name = getattr(series, "name", source.split("/")[-1])
        starts = np.where(data > 0)[0]
        for i in starts:
            j = next((k for k in range(i + 1, len(data)) if data[k] < 0), None)
            if j is not None:
                out.append({
                    "onset_s": float(timestamps[i]),
                    "offset_s": float(timestamps[j]),
                    "label_name": label_name,
                    "individual": individual,
                    "source": source,
                })

def _get_nwb_individual(nwb) -> str:
    subject = getattr(nwb, "subject", None)
    if subject:
        sid = getattr(subject, "subject_id", None)
        if sid:
            return str(sid)
    return "individual_0"



