"""Data loading utilities for the ethograph GUI."""

import logging
import tempfile
from pathlib import Path
from typing import Callable

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
from ethograph.io.trialtree import TrialTree, _discover_nwb
from ethograph.labels.tsv_store import (
    init_empty_labels,
    labels_tsv_path,
    load_labels_tsv,
    save_labels_tsv,
)

logger = logging.getLogger(__name__)

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
    return p.is_dir() and (p / ".ethograph" / "project.json").exists()


# ---------------------------------------------------------------------------
# Lightweight TrialTree from trial intervals (replaces nap_to_metadata_trialtree)
# ---------------------------------------------------------------------------


def _trialtree_from_trials_ep(trials_ep, data: dict | None = None) -> TrialTree:
    """Build a minimal TrialTree from trial intervals.

    Labels are interval-based (onset_s / offset_s), so only trial
    boundaries matter — no need to discover the densest feature.
    """
    import pynapple as nap
    from ethograph.utils.nwb import build_nwb_from_trial_table

    if trials_ep is None or len(trials_ep) == 0:
        # Single-trial: infer range from data if available
        feature_objs = {}
        if data:
            feature_objs = {
                k: v
                for k, v in data.items()
                if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
            }
        if feature_objs:
            t_min = min(
                obj.t[0] for obj in feature_objs.values() if len(obj) > 0
            )
            t_max = max(
                obj.t[-1] for obj in feature_objs.values() if len(obj) > 0
            )
            duration = t_max - t_min
        else:
            raise ValueError(
                "Cannot build TrialTree: no trial intervals and no data to infer duration from"
            )

        ds = xr.Dataset(
            coords={"time": np.array([0.0, duration]), "individuals": ["individual_0"]}
        )
        ds.attrs["trial"] = 1
        return TrialTree.from_datasets([ds], validate=False)

    datasets = []
    for i in range(len(trials_ep)):
        start = float(trials_ep.start[i])
        end = float(trials_ep.end[i])
        duration = end - start
        ds = xr.Dataset(
            coords={
                "time": np.array([0.0, duration]),
                "individuals": ["individual_0"],
            }
        )
        ds.attrs["trial"] = i + 1
        datasets.append(ds)

    dt = TrialTree.from_datasets(datasets, validate=False)

    # Create NWB alignment file for trial timing
    trial_df = pd.DataFrame(
        {
            "trial": list(range(1, len(trials_ep) + 1)),
            "start_time": list(trials_ep.start),
            "stop_time": list(trials_ep.end),
        }
    )
    nwb_dir = Path(tempfile.gettempdir()) / ".ethograph"
    nwb_path = nwb_dir / "alignment.nwb"
    build_nwb_from_trial_table(trial_df, output_path=nwb_path)
    dt.nwb_path = str(nwb_path)

    return dt


# ---------------------------------------------------------------------------
# NWB direct loading
# ---------------------------------------------------------------------------


def _load_nwb_dataset(file_path: str) -> tuple:
    """Load a standalone .nwb file via direct HDF5 slicing.

    Uses ``nwb_backend`` for catalog + combo detection and ``NWBLoader``
    for time-sliced data access (no pynapple intermediate).

    Returns ``(dt, all_labels_df, catalog)``.
    """
    from ethograph.io.nwb_backend import read_trial_intervals

    catalog, combo_cat = catalog_from_nwb(file_path)
    trial_intervals = read_trial_intervals(file_path)

    # Build NWBLoader
    loader = NWBLoader(file_path, catalog, combo_catalog=combo_cat)
    if trial_intervals:
        loader.set_trial_intervals(trial_intervals)

    # Build lightweight TrialTree from trial intervals
    if trial_intervals:
        import pynapple as nap

        starts = [t[0] for t in trial_intervals]
        stops = [t[1] for t in trial_intervals]
        trials_ep = nap.IntervalSet(start=starts, end=stops)
        dt = _trialtree_from_trials_ep(trials_ep)
    else:
        dt = _trialtree_from_trials_ep(None)

    dt.attrs["data_loader"] = loader
    dt.attrs["nwb_source"] = file_path
    dt.attrs["source_collection"] = _build_source_collection_nwb(
        file_path, combo_cat, trial_intervals,
    )

    # Labels
    tsv_path = labels_tsv_path(Path(file_path))
    if tsv_path.exists():
        all_labels_df = load_labels_tsv(tsv_path)
        logger.info("Loaded labels from %s", tsv_path.name)
    else:
        all_labels_df = init_empty_labels(dt.trials)

    return dt, all_labels_df, catalog


# ---------------------------------------------------------------------------
# Pynapple loading (.npz, folders)
# ---------------------------------------------------------------------------


def _load_pynapple_dataset(file_path: str) -> tuple:
    """Load a pynapple .npz file or folder.

    Returns ``(dt, all_labels_df, catalog)``.
    """
    from ethograph.io.pynapple import load_nap_data

    data, trials_ep = load_nap_data(file_path)
    catalog = catalog_from_pynapple(data, trials_ep)
    loader = PynappleLoader(data, trials_ep, catalog)
    dt = _trialtree_from_trials_ep(trials_ep, data)
    dt.attrs["data_loader"] = loader
    dt.attrs["source_collection"] = _build_source_collection_pynapple(data, trials_ep)
    all_labels_df = init_empty_labels(dt.trials)
    return dt, all_labels_df, catalog


def _load_nwb_project(project_dir: str) -> tuple:
    """Load an NWB project directory created by the NWB import wizard.

    Returns ``(dt, all_labels_df, catalog)``.
    """
    import json

    from ethograph.io.pynapple import load_nap_data

    project_path = Path(project_dir)
    config_path = project_path / ".ethograph" / "project.json"
    alignment_path = project_path / ".ethograph" / "alignment.nwb"

    with open(config_path) as f:
        config = json.load(f)

    nwb_source = config.get("nwb_source")
    if not nwb_source:
        dandiset_id = config.get("nwb_source_dandiset")
        session_eid = config.get("nwb_source_session")
        if dandiset_id and session_eid:
            raise NotImplementedError(
                f"DANDI streaming re-open not yet supported. "
                f"Download the NWB file locally first. "
                f"(dandiset={dandiset_id}, session={session_eid})"
            )
        raise ValueError("No NWB source path found in project.json")

    data, trials_ep = load_nap_data(nwb_source)
    catalog = catalog_from_pynapple(data, trials_ep)
    loader = PynappleLoader(data, trials_ep, catalog)
    dt = _trialtree_from_trials_ep(trials_ep, data)
    dt.attrs["data_loader"] = loader
    dt.attrs["source_collection"] = _build_source_collection_pynapple(data, trials_ep)

    # Apply project config
    if config.get("nwb_pose_keys"):
        dt.attrs["nwb_pose_keys"] = config["nwb_pose_keys"]
    dt.attrs["nwb_source"] = nwb_source
    for key in (
        "nwb_ephys_series",
        "nwb_ephys_path",
        "nwb_ephys_dandiset_id",
        "nwb_ephys_asset_id",
        "nwb_raw_asset_id",
    ):
        if config.get(key):
            dt.attrs[key] = config[key]

    if alignment_path.exists():
        dt.nwb_path = str(alignment_path)

    # Load labels
    labels_path = project_path / "labels.tsv"
    if labels_path.exists():
        all_labels_df = load_labels_tsv(labels_path)
        logger.info("Loaded labels from %s", labels_path)
    else:
        all_labels_df = init_empty_labels(dt.trials)

    return dt, all_labels_df, catalog


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
) -> tuple:
    """Load dataset from file path.

    Supports ``.nc`` (NetCDF), ``.nwb``, ``.npz``, pynapple folders,
    and NWB project directories (with ``.ethograph/project.json``).

    Returns
    -------
    tuple
        ``(dt, all_labels_df, catalog)`` where *catalog* is a
        :class:`~ethograph.io.catalog.DataCatalog`.  A ``DataLoader``
        is stored on ``dt.attrs["data_loader"]`` for NWB / pynapple
        backends.  For xarray backends the caller creates an
        :class:`~ethograph.io.catalog.XarrayLoader` after setting the
        dataset.

    Raises
    ------
    ValueError
        On validation or format errors (popup shown before raising).
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
        nwb = _discover_nwb(file_path)
        if nwb is not None:
            dt._nwb_path = str(nwb)

    catalog = catalog_from_xarray(dt.itrial(0), dt)

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

    # If a .nc has nwb_source, attach a PynappleLoader for lazy feature access
    nwb_source = dt.attrs.get("nwb_source")
    if nwb_source and Path(nwb_source).exists():
        try:
            from ethograph.io.pynapple import load_nap_data

            data, trials_ep = load_nap_data(nwb_source)
            nap_catalog = catalog_from_pynapple(data, trials_ep)
            loader = PynappleLoader(data, trials_ep, nap_catalog)
            dt.attrs["data_loader"] = loader
            # Merge pynapple features into the catalog
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
            logger.warning("Failed to load pynapple store from %s: %s", nwb_source, e)

    # Build SourceCollection from xarray
    dt.attrs["source_collection"] = _build_source_collection_xarray(dt)

    return dt, all_labels_df, catalog


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


def _build_source_collection_xarray(dt: TrialTree) -> SourceCollection:
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

    # Populate trial intervals from session_io only when real timing exists.
    try:
        session = dt.session_io
        trial_ids = dt.trials
        starts, stops = [], []
        for tid in trial_ids:
            start = session.start_time(tid)
            stop = session.stop_time(tid)
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
    dt.nwb_path = str(nwb_path)

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
