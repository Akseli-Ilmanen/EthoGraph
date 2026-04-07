"""Data loading utilities for the ethograph GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

import ethograph as eto
from ethograph.io.catalog import (
    DataCatalog,
    NWBLoader,
    PynappleLoader,
    XarrayLoader,
    catalog_from_nwb,
    catalog_from_pynapple,
    catalog_from_xarray,
)
from ethograph.io.time_model import SourceCollection
from ethograph.io.validation import validate_datatree
from movement.io import load
from movement.kinematics import (
    compute_acceleration,
    compute_pairwise_distances,
    compute_speed,
    compute_velocity,
)

from ethograph.gui.notify import notify_dialog
from ethograph.io.metadata_table import load_metadata_df, load_metadata_tsv, metadata_tsv_path
from ethograph.io.trialtree import TrialTree
from ethograph.io.nwb_alignment import EmpytAlignment, TableAlignment, discover_nwb, make_nwb_alignment
from ethograph.labels.tsv_store import (
    init_empty_labels,
    labels_tsv_path,
    load_labels_tsv,
    save_labels_tsv,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Everything ``load_dataset`` returns — no dt.attrs transport."""

    dt: Any  # TrialTree (or None for future NWB-only path)
    trial_ids: list[int | str]
    nwb_alignment: Any = None  # NWBAlignment 
    metadata_df: pd.DataFrame = None
    metadata_path: str | None = None
    all_labels_df: pd.DataFrame = None
    catalog: DataCatalog = None
    data_loader: Any = None
    source_collection: Any = None
    nwb_local: str | None = None
    nwb_pose_keys: list[str] | None = None
    nwb_ephys_series: str | None = None
    nwb_ephys_path: str | None = None
    nwb_video_folder: str | None = None


def _detect_audio_rate(audio_path: str) -> float:
    """Detect sample rate from an audio file via audioio."""
    from audioio import AudioLoader
    with AudioLoader(audio_path) as loader:
        return float(loader.rate)


def _is_nwb_file(file_path: str) -> bool:
    """Check if path is a standalone .nwb file."""
    return Path(file_path).suffix == ".nwb"


def _is_pynapple_path(file_path: str) -> bool:
    """Check if path is a pynapple folder or .npz file."""
    p = Path(file_path)
    return p.is_dir() or p.suffix == ".npz"


def _is_nwb_project_dir(file_path: str) -> bool:
    """Check if path is an NWB project directory (created by the NWB wizard)."""
    p = Path(file_path)
    return p.is_dir() and (p / ".ethograph" / "nwb_metadata").exists()


def _resolve_alignment(source_path: str | Path):
    """Resolve alignment source with priority order.

    For source ``.nwb`` files:
    1. Source NWB trials table.
    2. Sidecar ``.ethograph/alignment.nwb``.
    3. Sidecar metadata TSV with ``start_time`` and ``stop_time``.

    For other source paths:
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

        sidecar_nwb = discover_nwb(source)
        if sidecar_nwb is not None and Path(sidecar_nwb).exists():
            sidecar_alignment = make_nwb_alignment(sidecar_nwb)
            if not sidecar_alignment.trials_df.empty:
                return sidecar_alignment

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
# Lightweight TrialTree from trial info (no temp NWB needed)
# ---------------------------------------------------------------------------


def _minimal_trialtree(
    trial_ids: list[int],
    durations: list[float] | None = None,
) -> TrialTree:
    """Build a minimal TrialTree with lightweight trial nodes.

    Parameters
    ----------
    trial_ids
        List of trial identifiers.
    durations
        Per-trial durations in seconds.  Used only for the ``time``
        coordinate on each dummy dataset.  If *None*, a placeholder
        ``[0.0, 1.0]`` coordinate is used (actual data comes from the
        DataLoader, not from these datasets).
    """
    datasets = []
    for i, tid in enumerate(trial_ids):
        dur = durations[i] if durations else 1.0
        ds = xr.Dataset(
            coords={
                "time": np.array([0.0, dur]),
                "individuals": ["individual_0"],
            }
        )
        ds.attrs["trial"] = tid
        datasets.append(ds)

    return TrialTree.from_datasets(datasets, validate=False)


# ---------------------------------------------------------------------------
# NWB direct loading
# ---------------------------------------------------------------------------


def _load_nwb_dataset(file_path: str) -> LoadResult:
    """Load a standalone .nwb file via direct HDF5 slicing.

    Uses ``nwb_backend`` for catalog + combo detection and ``NWBLoader``
    for time-sliced data access (no pynapple intermediate).
    """
    from ethograph.io.nwb_backend import read_trial_intervals

    catalog, combo_cat = catalog_from_nwb(file_path)
    trial_intervals = read_trial_intervals(file_path)

    loader = NWBLoader(file_path, catalog, combo_catalog=combo_cat)
    if trial_intervals:
        loader.set_trial_intervals(trial_intervals)
        trial_ids = list(range(1, len(trial_intervals) + 1))
    else:
        trial_ids = [1]

    tsv_path = labels_tsv_path(Path(file_path))
    if tsv_path.exists():
        all_labels_df = load_labels_tsv(tsv_path)
        logger.info("Loaded labels from %s", tsv_path.name)
    else:
        all_labels_df = init_empty_labels(trial_ids)

    sio = _resolve_alignment(file_path)
    metadata_df, metadata_path = load_metadata_df(
        source_path=file_path,
        nwb_alignment=sio,
        trial_ids=trial_ids,
    )

    return LoadResult(
        dt=None,
        trial_ids=trial_ids,
        nwb_alignment=sio,
        metadata_df=metadata_df,
        metadata_path=metadata_path,
        all_labels_df=all_labels_df,
        catalog=catalog,
        data_loader=loader,
        source_collection=_build_source_collection_nwb(file_path, combo_cat, trial_intervals),
        nwb_local=file_path,
    )


# ---------------------------------------------------------------------------
# Pynapple loading (.npz, folders)
# ---------------------------------------------------------------------------


def _load_pynapple_dataset(file_path: str) -> LoadResult:
    """Load a pynapple .npz file or folder."""
    from ethograph.io.pynapple import load_nap_data

    data, trials_ep = load_nap_data(file_path)
    catalog = catalog_from_pynapple(data, trials_ep)
    loader = PynappleLoader(data, trials_ep, catalog)
    nwb_path = file_path if _is_nwb_file(file_path) else None
    trial_ids = list(range(1, len(trials_ep) + 1)) if trials_ep is not None and len(trials_ep) > 0 else [1]

    
    return LoadResult(
        dt=None,
        trial_ids=trial_ids,
        nwb_alignment=make_nwb_alignment(nwb_path),
        metadata_df=load_metadata_df(
            source_path=file_path,
            nwb_alignment=make_nwb_alignment(nwb_path),
            trial_ids=trial_ids,
        )[0],
        metadata_path=load_metadata_df(
            source_path=file_path,
            nwb_alignment=make_nwb_alignment(nwb_path),
            trial_ids=trial_ids,
        )[1],
        all_labels_df=init_empty_labels(trial_ids),
        catalog=catalog,
        data_loader=loader,
        source_collection=_build_source_collection_pynapple(data, trials_ep),
    )


def _load_nwb_project(project_dir: str) -> LoadResult:
    """Load an NWB project directory created by the NWB import wizard."""
    import json

    from ethograph.io.pynapple import load_nap_data

    project_path = Path(project_dir)
    config_path = project_path / ".ethograph" / "nwb_metadata"
    alignment_path = project_path / ".ethograph" / "alignment.nwb"

    with open(config_path) as f:
        config = json.load(f)

    nwb_source = config.get("nwb_local")
    if not nwb_source:
        dandiset_id = config.get("nwb_source_dandiset")
        session_eid = config.get("nwb_source_session")
        if dandiset_id and session_eid:
            raise NotImplementedError(
                f"DANDI streaming re-open not yet supported. "
                f"Download the NWB file locally first. "
                f"(dandiset={dandiset_id}, session={session_eid})"
            )
        raise ValueError("No NWB source path found in nwb_metadata")

    data, trials_ep = load_nap_data(nwb_source)
    catalog = catalog_from_pynapple(data, trials_ep)
    loader = PynappleLoader(data, trials_ep, catalog)
    nwb = str(alignment_path) if alignment_path.exists() else nwb_source
    trial_ids = list(range(1, len(trials_ep) + 1)) if trials_ep is not None and len(trials_ep) > 0 else [1]
    
    labels_path = project_path / "labels.tsv"
    if labels_path.exists():
        all_labels_df = load_labels_tsv(labels_path)
        logger.info("Loaded labels from %s", labels_path)
    else:
        all_labels_df = init_empty_labels(trial_ids)

    sio = make_nwb_alignment(nwb)
    metadata_df, metadata_path = load_metadata_df(
        source_path=nwb_source,
        nwb_alignment=sio,
        trial_ids=trial_ids,
    )

    return LoadResult(
        dt=None,
        trial_ids=trial_ids,
        nwb_alignment=sio,
        metadata_df=metadata_df,
        metadata_path=metadata_path,
        all_labels_df=all_labels_df,
        catalog=catalog,
        data_loader=loader,
        source_collection=_build_source_collection_pynapple(data, trials_ep),
        nwb_local=nwb_source,
        nwb_pose_keys=config.get("nwb_pose_keys"),
        nwb_ephys_series=config.get("nwb_ephys_series"),
        nwb_ephys_path=config.get("nwb_ephys_path"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_dataset(
    file_path: str,
    require_fps: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    max_trials: int | None = None,
    dandiset_id: str | None = None,
    import_labels: bool = True,
) -> LoadResult:
    """Load dataset from file path.

    Supports ``.nc`` (NetCDF), ``.nwb``, ``.npz``, pynapple folders,
    and NWB project directories (with ``.ethograph/nwb_metadata``).

    Returns a :class:`LoadResult` with dt, labels, catalog, and metadata.
    """
    if _is_nwb_project_dir(file_path):
        return _load_nwb_project(file_path)

    if _is_nwb_file(file_path):
        return _load_nwb_dataset(file_path)

    if _is_pynapple_path(file_path):
        return _load_pynapple_dataset(file_path)

    # --- xarray (.nc) path ---
    dt = eto.open(file_path)

    # Plain Dataset .nc files (e.g. Movement datasets) have no trial children.
    # Wrap them as a single-trial TrialTree so the GUI can work with them directly.
    if not dt.children or not any(
        node.ds is not None and "trial" in node.ds.attrs
        for node in dt.children.values()
    ):
        ds = xr.open_dataset(file_path, engine="netcdf4")
        dt = eto.dataset_to_basic_trialtree(ds)
        dt._source_path = file_path

    sio = _resolve_alignment(file_path)
    metadata_df, metadata_path = load_metadata_df(
        source_path=file_path,
        nwb_alignment=sio,
        trial_ids=dt.trials,
    )

    catalog = catalog_from_xarray(dt.itrial(0), dt, nwb_alignment=sio)

    errors = validate_datatree(dt, require_fps=require_fps)
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        suffix_msg = "\n\nSee documentation: XXX"
        msg = "Validation failed:\n" + error_msg + suffix_msg
        notify_dialog(msg, "error", "Validation Error")
        raise ValueError(msg)

    nc_path = Path(file_path)
    tsv_path = labels_tsv_path(nc_path)

    if tsv_path.exists():
        all_labels_df = load_labels_tsv(tsv_path)
        logger.info("Loaded labels from %s", tsv_path.name)
    else:
        all_labels_df = init_empty_labels(dt.trials)

    data_loader = None
    nwb_local = dt.attrs.get("nwb_local")
    if nwb_local and Path(nwb_local).exists():
        try:
            from ethograph.io.pynapple import load_nap_data

            data, trials_ep = load_nap_data(nwb_local)
            nap_catalog = catalog_from_pynapple(data, trials_ep)
            data_loader = PynappleLoader(data, trials_ep, nap_catalog)
            for f in nap_catalog.features:
                if f not in catalog.features:
                    catalog.features.append(f)
            for c in nap_catalog.colors:
                if c not in catalog.colors:
                    catalog.colors.append(c)
            for cp in nap_catalog.changepoints:
                if cp not in catalog.changepoints:
                    catalog.changepoints.append(cp)
            for name, spec in nap_catalog.combos.items():
                if name not in catalog.combos:
                    catalog.combos[name] = spec
        except Exception as e:
            logger.warning("Failed to load pynapple store from %s: %s", nwb_local, e)

    return LoadResult(
        dt=dt,
        trial_ids=dt.trials,
        nwb_alignment=sio,
        metadata_df=metadata_df,
        metadata_path=metadata_path,
        all_labels_df=all_labels_df,
        catalog=catalog,
        data_loader=data_loader,
        source_collection=_build_source_collection_xarray(dt, nwb_alignment=sio),
        nwb_local=nwb_local,
    )


# ---------------------------------------------------------------------------
# SourceCollection builders
# ---------------------------------------------------------------------------


def _build_source_collection_pynapple(
    data: dict, trials_ep=None,
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


def _build_source_collection_nwb(
    file_path: str,
    combo_catalog,
    trial_intervals: list[tuple[float, float]] | None = None,
) -> SourceCollection:
    """Build SourceCollection from NWB combo catalog."""
    from ethograph.io.time_sources import NWBTimeSource

    sc = SourceCollection()
    for entry in combo_catalog.features:
        sc.add(NWBTimeSource(
            name=entry.display_name,
            source_path=file_path,
            entry=entry,
            combo_catalog=combo_catalog,
        ))
    if trial_intervals:
        sc.set_trials(
            ids=list(range(1, len(trial_intervals) + 1)),
            starts=[t[0] for t in trial_intervals],
            stops=[t[1] for t in trial_intervals],
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
# Wizard helpers (unchanged)
# ---------------------------------------------------------------------------


def _wizard_single_media_helper(
    dt,
    video_path=None,
    pose_path=None,
    audio_path=None,
    video_offset: float | None = None,
    audio_offset: float | None = None,
):
    """Create a minimal NWB alignment file for a single-trial wizard."""
    from ethograph.utils.nwb import build_nwb_from_trial_table

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
    fps = dt.itrial(0).attrs.get("fps", 30)

    stream_rates: dict[str, float] = {}
    if video_path:
        stream_rates["video"] = float(fps)
    if pose_path:
        stream_rates["pose"] = float(fps)
    if audio_path:
        stream_rates["audio"] = _detect_audio_rate(audio_path)

    ref_path = video_path or pose_path or audio_path
    if ref_path:
        output_dir = Path(ref_path).parent
    else:
        output_dir = Path.cwd()

    nwb_path = output_dir / ".ethograph" / "alignment.nwb"
    build_nwb_from_trial_table(
        trial_table, stream_rates=stream_rates, output_path=nwb_path
    )

    return dt


def wizard_single_from_pose(
    video_path,
    fps,
    pose_path,
    source_software,
    video_offset: float | None = None,
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

    dt = eto.dataset_to_basic_trialtree(ds, video_motion=False)
    _wizard_single_media_helper(
        dt, video_path=video_path, pose_path=pose_path, video_offset=video_offset
    )
    return dt


def wizard_single_from_ds(
    video_path, ds: xr.Dataset, video_offset: float | None = None
):
    dt = eto.dataset_to_basic_trialtree(ds)
    _wizard_single_media_helper(dt, video_path=video_path, video_offset=video_offset)
    return dt


def wizard_single_from_npy_file(
    video_path,
    fps,
    npy_path,
    data_sr,
    individuals=None,
    video_motion: bool = False,
    video_offset: float | None = None,
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

    dt = eto.dataset_to_basic_trialtree(
        ds, video_path=video_path, video_motion=video_motion
    )
    _wizard_single_media_helper(dt, video_path=video_path, video_offset=video_offset)
    return dt


def wizard_single_from_ephys(
    video_path: str | None = None,
    fps: int = 30,
    audio_path: str | None = None,
    individuals: list[str] | None = None,
    video_motion: bool = False,
    video_offset: float | None = None,
    audio_offset: float | None = None,
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

    dt = eto.dataset_to_basic_trialtree(
        ds, video_path=video_path, video_motion=video_motion
    )
    _wizard_single_media_helper(
        dt,
        video_path=video_path,
        audio_path=audio_path,
        video_offset=video_offset,
        audio_offset=audio_offset,
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

    motion = extract_video_motion(
        video_path, fps=fps, verbose=False, scale_width=scale_width
    )

    ds = xr.Dataset(
        {"video_motion": motion},
        coords={"individuals": individuals},
    )
    ds.attrs["fps"] = fps

    dt = eto.dataset_to_basic_trialtree(ds, video_motion=False)
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

    dt = eto.dataset_to_basic_trialtree(
        ds, video_path=video_path, video_motion=video_motion
    )
    _wizard_single_media_helper(
        dt,
        video_path=video_path,
        audio_path=audio_path,
        video_offset=video_offset,
    )
    return dt
