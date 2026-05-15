"""Data loading utilities for the ethograph GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load
from movement.kinematics import (
    compute_acceleration,
    compute_pairwise_distances,
    compute_speed,
    compute_velocity,
)

import ethograph as eto
from ethograph.gui.notify import notify_dialog
from ethograph.io.catalog import (
    DataCatalog,
    PynappleLoader,
    XarrayLoader,
    catalog_from_pynapple,
    catalog_from_xarray,
)
from ethograph.io.metadata_table import (
    empty_metadata_df,
    load_metadata_df,
    load_metadata_tsv,
    metadata_tsv_path,
    trials_ep_from_metadata_df,
    validate_metadata_timing,
)
from ethograph.io.nwb_alignment import (
    EmpytAlignment,
    TableAlignment,
    discover_nwb,
    make_nwb_alignment,
)
from ethograph.io.time_model import SourceCollection
from ethograph.io.trialtree import TrialTree
from ethograph.io.validation import validate_datatree
from ethograph.labels.converters import (
    PynappleLabelConverter,
    resolve_labels_tsv,
)
from ethograph.labels.tsv_store import init_empty_labels

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Everything ``load_dataset`` returns — no dt.attrs transport."""

    dt: Any = None  # TrialTree (or None for NWB-only path)
    trial_ids: list[int | str] = field(default_factory=lambda: [1])
    nwb_alignment: Any = field(default_factory=EmpytAlignment)
    metadata_df: pd.DataFrame = field(default_factory=lambda: empty_metadata_df([1]))
    metadata_path: str | None = None
    all_labels_df: pd.DataFrame = field(default_factory=lambda: init_empty_labels([]))
    labels_file_path: str | None = None  # Path to the source labels TSV
    catalog: DataCatalog = None
    data_loader: Any = None
    source_collection: SourceCollection = field(default_factory=SourceCollection)
    nwb_local: str | None = None
    nwb_video_folder: str | None = None
    pynapple_data: dict | None = None


def _detect_audio_rate(audio_path: str) -> float:
    """Detect sample rate from an audio file via audioio."""
    from audioio import AudioLoader

    with AudioLoader(audio_path) as loader:
        return float(loader.rate)


def _resolve_trials_ep(data: dict, trials_ep, *, metadata_path: str | Path | None = None):
    """Resolve trials IntervalSet from available sources.

    Priority:
    1. User-supplied metadata file (``metadata_path``) with timing columns.
    2. ``trials_ep`` detected from the data source.

    Returns None when no trial information is available.  The caller
    (or app_state) is responsible for a final synthetic-single-trial
    fallback if needed.
    """
    if metadata_path is not None:
        path = Path(metadata_path)
        if not path.exists():
            raise ValueError(f"Metadata file not found: {path}")
        df = load_metadata_tsv(path)
        if "start_time" in df.columns and "stop_time" in df.columns:
            validate_metadata_timing(df, path)
            return trials_ep_from_metadata_df(df)

    if trials_ep is not None and len(trials_ep) > 0:
        return trials_ep

    return None


def synthesize_single_trial(data: dict):
    """Create a single-trial IntervalSet spanning all loaded time-series.

    Raises ValueError if no time-series data is found.
    """
    import pynapple as nap

    starts: list[float] = []
    ends: list[float] = []
    for obj in data.values():
        if isinstance(obj, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)) and len(obj) > 0:
            starts.append(float(obj.t[0]))
            ends.append(float(obj.t[-1]))
    if not starts:
        raise ValueError("No time-series data found — cannot determine session extent")
    return nap.IntervalSet(start=min(starts), end=max(ends))


def _is_pynapple_path_folder(file_path: str) -> bool:
    """Check if path is a pynapple folder or .npz file."""
    p = Path(file_path)
    return (
        p.suffix in {".npz", ".nwb"}
        or (p.is_dir() and any(p.glob("**/*.npz")))
        or (p.is_dir() and any(p.glob("**/*.nwb")))
    )


def _resolve_alignment(source_path: str | Path):
    """Resolve alignment source with priority order.

    For source ``.nwb`` files: use the source NWB directly. Trials and
    ImageSeries (media paths, stream rates) are read from the source.
    Falls back to a sidecar metadata TSV for timing if the source has
    no trials table.

    For other source paths (``.nc``, ``.npz``, pynapple folders):
    1. Sidecar ``.ethograph/alignment.nwb``.
    2. Sidecar metadata TSV with ``start_time`` and ``stop_time``.
    """
    source = Path(source_path)

    def _tsv_timing_alignment(path: Path):
        try:
            if not path.exists() or not path.is_file():
                return None
            df = load_metadata_tsv(path)
            required = {"start_time", "stop_time"}
            if not required.issubset(df.columns):
                return None
            return TableAlignment(df)
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to read timing TSV %s: %s", path, e)
            return None

    if source.suffix.lower() == ".nwb" and source.exists():
        source_alignment = make_nwb_alignment(source)
        if not source_alignment.trials_df.empty:
            return source_alignment
        sidecar_tsv = metadata_tsv_path(source)
        tsv_alignment = _tsv_timing_alignment(sidecar_tsv)
        if tsv_alignment is not None:
            return tsv_alignment
        return source_alignment

    sidecar_nwb = discover_nwb(source)
    if sidecar_nwb is not None and Path(sidecar_nwb).exists():
        return make_nwb_alignment(sidecar_nwb)

    sidecar_tsv = metadata_tsv_path(source)
    tsv_alignment = _tsv_timing_alignment(sidecar_tsv)
    if tsv_alignment is not None:
        return tsv_alignment

    return EmpytAlignment()


# ---------------------------------------------------------------------------
# Pynapple loading (.npz, folders)
# ---------------------------------------------------------------------------


def _load_pynapple_dataset(
    file_path: str,
    metadata_path: str | None = None,
) -> LoadResult:
    """Load a pynapple .npz file or folder."""
    from ethograph.io.pynapple import load_nap_data

    data, trials_ep = load_nap_data(file_path)
    trials_ep = _resolve_trials_ep(data, trials_ep, metadata_path=metadata_path)

    catalog = catalog_from_pynapple(data, source_path=file_path)
    loader = PynappleLoader(data, catalog)

    parent = Path(file_path).parent if not Path(file_path).is_dir() else Path(file_path)
    sidecar = parent / ".ethograph" / "alignment.nwb"
    nwb_path = str(sidecar) if sidecar.exists() else None

    trial_ids = list(range(1, len(trials_ep) + 1)) if trials_ep is not None else [1]

    sio = make_nwb_alignment(nwb_path)

    converter = PynappleLabelConverter(data)
    all_labels_df = converter.resolve_labels(
        source_path=file_path,
        trial_ids=trial_ids,
    )

    resolved_metadata_df, resolved_metadata_path = load_metadata_df(
        source_path=file_path,
        metadata_path=metadata_path,
        nwb_alignment=sio,
        trial_ids=trial_ids,
    )

    # Determine which labels file path was used
    from ethograph.labels.tsv_store import labels_tsv_path

    tsv_path = labels_tsv_path(Path(file_path))
    labels_file_path = str(tsv_path) if tsv_path.exists() else None

    return LoadResult(
        dt=None,
        trial_ids=trial_ids,
        nwb_alignment=sio,
        metadata_df=resolved_metadata_df,
        metadata_path=resolved_metadata_path,
        all_labels_df=all_labels_df,
        labels_file_path=labels_file_path,
        catalog=catalog,
        data_loader=loader,
        source_collection=_build_source_collection_pynapple(data, trials_ep),
        pynapple_data=data,
    )


def _load_trialtree(
    file_path: str,
    metadata_path: str | None = None,
) -> LoadResult:
    """Load a TrialTree or xarray.Dataset from a .nc file."""
    dt = eto.open(file_path)

    # Plain Dataset .nc files (e.g. Movement datasets) have no trial children.
    # Wrap them as a single-trial TrialTree so the GUI can work with them directly.
    if not dt.children or not any(node.ds is not None and "trial" in node.ds.attrs for node in dt.children.values()):
        ds = xr.open_dataset(file_path, engine="netcdf4")
        dt = _wizard_ds_to_continuous_dt(ds)
        dt._source_path = file_path

    sio = _resolve_alignment(file_path)
    resolved_metadata_df, resolved_metadata_path = load_metadata_df(
        source_path=file_path,
        metadata_path=metadata_path,
        nwb_alignment=sio,
        trial_ids=dt.trials,
    )

    catalog = catalog_from_xarray(dt.itrial(0), dt, nwb_alignment=sio)

    # TODO: ugly, rewreite with better validation, notify code.
    errors = validate_datatree(dt)
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        suffix_msg = "\n\nSee documentation: XXX"
        msg = "Validation failed:\n" + error_msg + suffix_msg
        notify_dialog(msg, "error", "Validation Error")
        raise ValueError(msg)

    # Determine which labels file path was used
    from ethograph.labels.tsv_store import labels_tsv_path

    tsv_path = labels_tsv_path(Path(file_path))
    labels_file_path = str(tsv_path) if tsv_path.exists() else None

    all_labels_df = resolve_labels_tsv(file_path, dt.trials)

    return LoadResult(
        dt=dt,
        trial_ids=dt.trials,
        nwb_alignment=sio,
        metadata_df=resolved_metadata_df,
        metadata_path=resolved_metadata_path,
        all_labels_df=all_labels_df,
        labels_file_path=labels_file_path,
        catalog=catalog,
        data_loader=XarrayLoader(dt.itrial(0), catalog),
        source_collection=_build_source_collection_xarray(dt, nwb_alignment=sio),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_dataset(
    file_path: str,
    progress_callback: Callable[[str], None] | None = None,
    metadata_path: str | None = None,
) -> LoadResult:
    """Load dataset from file path.

    Supports ``.nc`` (NetCDF), ``.nwb``, ``.npz``, pynapple folders.

    Parameters
    ----------
    metadata_path
        Optional path to a TSV/CSV/Excel file with ``trial``, ``start_time``,
        ``stop_time`` columns.  When provided, trial boundaries are read from
        this file instead of the data source.

    Returns a :class:`LoadResult` with dt, labels, catalog, and metadata.
    """
    if _is_pynapple_path_folder(file_path):
        return _load_pynapple_dataset(file_path, metadata_path=metadata_path)

    if file_path.endswith(".nc"):
        return _load_trialtree(file_path, metadata_path=metadata_path)

    raise ValueError(
        f"Unsupported file type: {Path(file_path).suffix!r}. "
        "Expected .nc, .nwb, .npz, a pynapple folder, or a DANDI project directory."
    )


# ---------------------------------------------------------------------------
# SourceCollection builders
# ---------------------------------------------------------------------------


def _build_source_collection_pynapple(
    data: dict,
    trials_ep=None,
) -> SourceCollection:
    """Build SourceCollection from pynapple objects."""
    import pynapple as nap

    from ethograph.io.time_sources import PynappleSource

    sc = SourceCollection()
    for key, obj in data.items():
        if isinstance(obj, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
            sc.add(PynappleSource(key, obj, trials_ep))
    if trials_ep is not None and len(trials_ep) > 0:
        sc.set_trials(
            ids=list(range(1, len(trials_ep) + 1)),
            starts=[float(s) for s in trials_ep.start],
            stops=[float(e) for e in trials_ep.end],
        )
    return sc


def _build_source_collection_xarray(dt: TrialTree, nwb_alignment=None) -> SourceCollection:
    """Build SourceCollection from a TrialTree's xarray datasets."""
    from ethograph.io.time_sources import XarrayTrialSource
    from ethograph.utils.xr_utils import get_time_coord

    sc = SourceCollection()
    ds = dt.itrial(0)
    for var_name in ds.data_vars:
        da = ds[var_name]
        tc = get_time_coord(da)
        if tc is not None:
            sc.add(XarrayTrialSource(var_name, ds, tc.name))

    sio = nwb_alignment if nwb_alignment is not None else EmpytAlignment()
    try:
        trial_ids = dt.trials
        starts, stops = [], []
        for tid in trial_ids:
            start = sio.start_time(tid)
            stop = sio.stop_time(tid)
            if stop is None:
                break
            starts.append(start)
            stops.append(stop)
        else:
            if starts:
                sc.set_trials(ids=trial_ids, starts=starts, stops=stops)
    except (AttributeError, ValueError, KeyError):
        pass

    return sc


# ---------------------------------------------------------------------------
# Wizard helpers
# ---------------------------------------------------------------------------


def _wizard_ds_to_continuous_dt(ds: xr.Dataset) -> TrialTree:
    """Wrap a single xr.Dataset as a one-trial continuous TrialTree."""

    duration = eto.get_ds_duration(ds)
    if duration is not None:
        epochs = pd.DataFrame(
            {
                "trial": [1],
                "start_time": [0.0],
                "stop_time": [duration],
            }
        )
    else:
        epochs = pd.DataFrame(
            {
                "trial": [1],
                "start_time": [0.0],
            }
        )

    return TrialTree.from_continuous(ds, epochs)


def _wizard_single_media_helper(
    dt,
    video_path=None,
    pose_path=None,
    audio_path=None,
    video_offset: float | None = None,
    audio_offset: float | None = None,
    nwb_dir: Path | None = None,
):
    """Create a minimal NWB alignment file for a single-trial wizard.

    Parameters
    ----------
    nwb_dir:
        Directory in which to write ``.ethograph/alignment.nwb``.
        Should be the directory of the output ``.nc`` file so that
        ``discover_nwb`` finds the sidecar on next load.
        Falls back to the media-file parent when not supplied.
    """
    from ethograph.io.nwb_alignment import align_media_per_trial

    row: dict = {"trial": 1, "start_time": 0.0}

    if video_path is not None:
        row["video_cam-1"] = Path(video_path).name
        if video_offset is not None and video_offset != 0.0:
            row["video_cam-1_start"] = float(video_offset)

    if pose_path is not None:
        row["pose_cam-1"] = Path(pose_path).name

    if audio_path is not None:
        row["audio_mic-1"] = Path(audio_path).name
        if audio_offset is not None and audio_offset != 0.0:
            row["audio_mic-1_start"] = float(audio_offset)

    trial_table = pd.DataFrame([row])

    if isinstance(dt, xr.DataTree):
        fps = dt.itrial(0).attrs.get("fps")
    elif isinstance(dt, xr.Dataset):
        fps = dt.attrs.get("fps")
    else:
        fps = None

    stream_rates: dict[str, float] = {}
    if fps is not None:
        if video_path:
            stream_rates["video"] = float(fps)
        if pose_path:
            stream_rates["pose"] = float(fps)
    if audio_path:
        stream_rates["audio"] = _detect_audio_rate(audio_path)

    ref_path = video_path or pose_path or audio_path
    media_root = Path(ref_path).parent if ref_path else None
    output_dir = nwb_dir if nwb_dir is not None else (media_root or Path.cwd())

    nwb_path = output_dir / ".ethograph" / "alignment.nwb"
    align_media_per_trial(
        trial_table,
        stream_rates=stream_rates,
        output_path=nwb_path,
        media_root=media_root,
        pose_fps=fps,
    )

    return dt


def wizard_single_from_pose(
    video_path,
    fps,
    pose_path,
    source_software,
    video_offset: float | None = None,
    output_nc_path: str | Path | None = None,
):
    """Create a minimal TrialTree from pose or bounding-box data."""
    try:
        ds = load.load_dataset(
            pose_path,
            fps=fps,
            source_software=source_software,
        )
    except (OSError, ValueError, KeyError):
        notify_dialog(
            f"Failed to load data from {pose_path}. Please check the file and try again.",
            "error",
            "Load Error",
        )
        raise

    ds["velocity"] = compute_velocity(ds.position)
    ds["speed"] = compute_speed(ds.position)
    ds["acceleration"] = compute_acceleration(ds.position)

    if "keypoints" in ds.coords and len(ds.keypoints) > 1:
        compute_pairwise_distances(ds.position, dim="keypoints", pairs="all")

    if len(ds.individuals) > 1:
        compute_pairwise_distances(ds.position, dim="individuals", pairs="all")

    nwb_dir = Path(output_nc_path).parent if output_nc_path else None
    _wizard_single_media_helper(
        ds, video_path=video_path, pose_path=pose_path, video_offset=video_offset, nwb_dir=nwb_dir
    )
    return _wizard_ds_to_continuous_dt(ds)


def wizard_single_from_ds(
    video_path,
    ds: xr.Dataset,
    video_offset: float | None = None,
    output_nc_path: str | Path | None = None,
):
    nwb_dir = Path(output_nc_path).parent if output_nc_path else None
    _wizard_single_media_helper(ds, video_path=video_path, video_offset=video_offset, nwb_dir=nwb_dir)
    return ds


def wizard_single_from_npy_file(
    video_path,
    fps,
    npy_path,
    data_sr,
    individuals=None,
    video_motion: bool = False,
    video_offset: float | None = None,
    output_nc_path: str | Path | None = None,
):
    if individuals is None:
        individuals = [
            "individual 1",
            "individual 2",
            "individual 3",
            "individual 4",
        ]

    data = np.load(npy_path)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_variables = data.shape

    if n_samples < n_variables:
        data = data.T
        n_samples, n_variables = data.shape

    time_coords = np.arange(n_samples) / data_sr

    ds = xr.Dataset(
        data_vars={"data": (["time", "variable"], data)},
        coords={"time": time_coords, "individuals": individuals},
    )
    ds.attrs["fps"] = fps

    if video_motion and video_path is not None:
        from ethograph.features.movement import extract_video_motion

        ds["video_motion"] = extract_video_motion(video_path, fps=ds.attrs["fps"], time_coord_name="time_video")

    nwb_dir = Path(output_nc_path).parent if output_nc_path else None
    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(dt, video_path=video_path, video_offset=video_offset, nwb_dir=nwb_dir)
    return dt


def wizard_single_from_ephys(
    video_path: str | None = None,
    fps: int = 30,
    audio_path: str | None = None,
    individuals: list[str] | None = None,
    video_motion: bool = False,
    video_offset: float | None = None,
    audio_offset: float | None = None,
    output_nc_path: str | Path | None = None,
):
    if individuals is None:
        individuals = [
            "individual 1",
            "individual 2",
            "individual 3",
            "individual 4",
        ]

    ds = xr.Dataset(coords={"individuals": individuals})
    ds.attrs["fps"] = fps

    if video_motion and video_path is not None:
        from ethograph.features.movement import extract_video_motion

        ds["video_motion"] = extract_video_motion(video_path, fps=ds.attrs["fps"], time_coord_name="time_video")

    nwb_dir = Path(output_nc_path).parent if output_nc_path else None
    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(
        dt,
        video_path=video_path,
        audio_path=audio_path,
        video_offset=video_offset,
        audio_offset=audio_offset,
        nwb_dir=nwb_dir,
    )
    return dt


def wizard_single_from_video(
    video_path: str,
    fps: int | None = None,
    individuals: list[str] | None = None,
    scale_width: int = 160,
):
    """Create a TrialTree from a video file with motion-energy feature."""
    from ethograph.features.movement import extract_video_motion
    from ethograph.gui.wizard_single import get_video_fps

    if fps is None:
        fps = get_video_fps(video_path)
        if fps is None:
            raise ValueError(f"Cannot determine FPS from {video_path}")

    if individuals is None:
        individuals = [
            "individual 1",
            "individual 2",
            "individual 3",
            "individual 4",
        ]

    motion = extract_video_motion(video_path, fps=fps, verbose=False, scale_width=scale_width)

    ds = xr.Dataset(
        {"video_motion": motion},
        coords={"individuals": individuals},
    )
    ds.attrs["fps"] = fps

    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(dt, video_path=video_path)
    return dt


def wizard_single_from_audio(
    video_path,
    fps,
    audio_path,
    individuals=None,
    video_motion: bool = False,
    audio_sr: int = 44100,
    video_offset: float | None = None,
    output_nc_path: str | Path | None = None,
):
    if individuals is None:
        individuals = [
            "individual 1",
            "individual 2",
            "individual 3",
            "individual 4",
        ]

    ds = xr.Dataset(coords={"individuals": individuals})
    ds.attrs["fps"] = fps

    if video_motion and video_path is not None:
        from ethograph.features.movement import extract_video_motion

        ds["video_motion"] = extract_video_motion(video_path, fps=ds.attrs["fps"], time_coord_name="time_video")

    nwb_dir = Path(output_nc_path).parent if output_nc_path else None
    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(
        dt,
        video_path=video_path,
        audio_path=audio_path,
        video_offset=video_offset,
        nwb_dir=nwb_dir,
    )
    return dt
