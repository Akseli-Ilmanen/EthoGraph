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
from ethograph.io.metadata_table import empty_metadata_df, load_metadata_df, load_metadata_tsv, metadata_tsv_path
from ethograph.io.trialtree import TrialTree
from ethograph.io.nwb_alignment import EmpytAlignment, TableAlignment, discover_nwb, make_nwb_alignment
from ethograph.labels.converters import (
    NWBLabelConverter,
    PynappleLabelConverter,
    resolve_labels_tsv,
)
from ethograph.labels.tsv_store import init_empty_labels

logger = logging.getLogger(__name__)


def _default_trial_ids() -> list[int]:
    return [1]


@dataclass
class LoadResult:
    """Everything ``load_dataset`` returns — no dt.attrs transport."""

    dt: Any = None  # TrialTree (or None for NWB-only path)
    trial_ids: list[int | str] = field(default_factory=_default_trial_ids)
    nwb_alignment: Any = field(default_factory=EmpytAlignment)
    metadata_df: pd.DataFrame = field(default_factory=lambda: empty_metadata_df([1]))
    metadata_path: str | None = None
    all_labels_df: pd.DataFrame = field(default_factory=lambda: init_empty_labels([]))
    catalog: DataCatalog = None
    data_loader: Any = None
    source_collection: SourceCollection = field(default_factory=SourceCollection)
    nwb_local: str | None = None
    nwb_video_folder: str | None = None


def _detect_audio_rate(audio_path: str) -> float:
    """Detect sample rate from an audio file via audioio."""
    from audioio import AudioLoader
    with AudioLoader(audio_path) as loader:
        return float(loader.rate)


def _ensure_trials_ep(data: dict, trials_ep):
    """Guarantee a valid trials IntervalSet.

    When no explicit trials are found, synthesize a single trial spanning
    the full time range of all loaded time-series objects.
    """
    if trials_ep is not None and len(trials_ep) > 0:
        return trials_ep

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


def _is_nwb_file(file_path: str) -> bool:
    """Check if path is a standalone .nwb file."""
    return Path(file_path).suffix == ".nwb"


def _is_pynapple_path_folder(file_path: str) -> bool:
    """Check if path is a pynapple folder or .npz file."""
    p = Path(file_path)
    return p.suffix == ".npz" or (p.is_dir() and any(p.glob("**/*.npz")))
    


def _read_dandi_provenance(ethograph_dir: Path) -> dict | None:
    """Read DANDI provenance from .ethograph/provenance.yaml."""
    from ethograph.utils.nwb import read_provenance

    prov = read_provenance(ethograph_dir)
    if prov and prov.get("nwb_dandiset_id") and prov.get("nwb_asset_id"):
        return prov
    return None


def _is_remote_nwb(file_path: str) -> bool:
    """Check if path is a remote NWB project with DANDI provenance."""
    p = Path(file_path)
    ethograph_dir = p / ".ethograph"
    if not (p.is_dir() and ethograph_dir.exists()):
        return False
    return _read_dandi_provenance(ethograph_dir) is not None


def _bootstrap_alignment_nwb(source_nwb: Path) -> Path | None:
    """Create ``.ethograph/alignment.nwb`` from a source NWB's acquisition ImageSeries.

    When loading a standalone ``.nwb`` that has ImageSeries with
    ``external_file`` but no sidecar alignment.nwb yet, this bootstraps
    one so the GUI can resolve media paths and stream rates.
    """
    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.image import ImageSeries

    with NWBHDF5IO(str(source_nwb), "r") as io:
        nwb = io.read()

        series_info = []
        for name, obj in nwb.acquisition.items():
            if not isinstance(obj, ImageSeries):
                continue
            if obj.external_file is None or len(obj.external_file) == 0:
                continue
            info = {
                "name": name,
                "description": obj.description or name,
                "external_file": list(obj.external_file),
                "starting_frame": (
                    np.array(obj.starting_frame, dtype=np.int32)
                    if obj.starting_frame is not None
                    else np.zeros(1, dtype=np.int32)
                ),
            }
            if obj.timestamps is not None:
                info["timestamps"] = np.array(obj.timestamps)
            elif obj.rate is not None:
                info["rate"] = float(obj.rate)
                info["starting_time"] = float(obj.starting_time or 0.0)
            series_info.append(info)

        if not series_info:
            return None

        trials_rows = []
        if nwb.trials is not None and len(nwb.trials) > 0:
            trials_df = nwb.trials.to_dataframe()
            for _, row in trials_df.iterrows():
                trials_rows.append((float(row["start_time"]), float(row["stop_time"])))

    output_dir = source_nwb.parent / ".ethograph"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alignment.nwb"

    nwbfile = NWBFile(
        session_description="Alignment (bootstrapped from source NWB by ethograph).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    for start, stop in trials_rows:
        nwbfile.add_trial(start_time=start, stop_time=stop)

    for info in series_info:
        kwargs = {
            "name": info["name"],
            "description": info["description"],
            "external_file": info["external_file"],
            "format": "external",
            "starting_frame": info["starting_frame"],
        }
        if "timestamps" in info:
            kwargs["timestamps"] = info["timestamps"]
        elif "rate" in info:
            kwargs["rate"] = info["rate"]
            kwargs["starting_time"] = info["starting_time"]
        nwbfile.add_acquisition(ImageSeries(**kwargs))

    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwbfile)

    logger.info(
        "Bootstrapped alignment.nwb from %s (%d ImageSeries)",
        source_nwb.name,
        len(series_info),
    )
    return output_path


def _resolve_alignment(source_path: str | Path):
    """Resolve alignment source with priority order.

    For source ``.nwb`` files:
    1. Source NWB trials table.
    2. Sidecar ``.ethograph/alignment.nwb``.
    3. Bootstrap alignment.nwb from source NWB ImageSeries (if any have external files).
    4. Sidecar metadata TSV with ``start_time`` and ``stop_time``.

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
        else:
            bootstrapped = _bootstrap_alignment_nwb(source)
            if bootstrapped is not None:
                return make_nwb_alignment(bootstrapped)

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
# NWB direct loading
# ---------------------------------------------------------------------------


def _load_nwb_dataset(file_path: str) -> LoadResult:
    """Load a standalone .nwb file via direct HDF5 slicing.

    Uses ``catalog`` for combo detection and ``NWBLoader``
    for time-sliced data access (no pynapple intermediate).
    """
    from ethograph.io.catalog import read_trial_intervals

    catalog, combo_cat = catalog_from_nwb(file_path)
    trial_intervals = read_trial_intervals(file_path)

    loader = NWBLoader(file_path, catalog, combo_catalog=combo_cat)
    if trial_intervals:
        loader.set_trial_intervals(trial_intervals)
        trial_ids = list(range(1, len(trial_intervals) + 1))
    else:
        trial_ids = [1]

    sio = _resolve_alignment(file_path)

    converter = NWBLabelConverter(nwb_path=file_path)
    all_labels_df = converter.resolve_labels(
        source_path=file_path,
        trial_ids=trial_ids,
        trials_df=sio.trials_df if hasattr(sio, "trials_df") else None,
    )
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
    trials_ep = _ensure_trials_ep(data, trials_ep)
    catalog = catalog_from_pynapple(data, trials_ep)
    loader = PynappleLoader(data, trials_ep, catalog)

    parent = Path(file_path).parent if not Path(file_path).is_dir() else Path(file_path)
    sidecar = parent / ".ethograph" / "alignment.nwb"
    nwb_path = str(sidecar) if sidecar.exists() else None
        
        
    trial_ids = list(range(1, len(trials_ep) + 1))

    sio = make_nwb_alignment(nwb_path)

    converter = PynappleLabelConverter(data, trials_ep)
    all_labels_df = converter.resolve_labels(
        source_path=file_path,
        trial_ids=trial_ids,
    )

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
        source_collection=_build_source_collection_pynapple(data, trials_ep),
    )



def _load_trialtree(file_path: str) -> LoadResult:
    """Load a TrialTree or xarray.Dataset from a .nc file."""
    dt = eto.open(file_path)


    # Plain Dataset .nc files (e.g. Movement datasets) have no trial children.
    # Wrap them as a single-trial TrialTree so the GUI can work with them directly.
    if not dt.children or not any(
        node.ds is not None and "trial" in node.ds.attrs
        for node in dt.children.values()
    ):
        ds = xr.open_dataset(file_path, engine="netcdf4")
        dt = _wizard_ds_to_continuous_dt(ds)
        dt._source_path = file_path



    sio = _resolve_alignment(file_path)
    metadata_df, metadata_path = load_metadata_df(
        source_path=file_path,
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

    all_labels_df = resolve_labels_tsv(file_path, dt.trials)

    return LoadResult(
        dt=dt,
        trial_ids=dt.trials,
        nwb_alignment=sio,
        metadata_df=metadata_df,
        metadata_path=metadata_path,
        all_labels_df=all_labels_df,
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
) -> LoadResult:
    """Load dataset from file path.

    Supports ``.nc`` (NetCDF), ``.nwb``, ``.npz``, pynapple folders,
    and remote NWB projects (DANDI provenance in ``.ethograph/alignment.nwb``).

    Returns a :class:`LoadResult` with dt, labels, catalog, and metadata.
    """
    if _is_remote_nwb(file_path):
        return _load_remote_nwb(file_path)

    if _is_nwb_file(file_path):
        return _load_nwb_dataset(file_path)

    if _is_pynapple_path_folder(file_path):
        return _load_pynapple_dataset(file_path)

    if file_path.endswith(".nc"):
        return _load_trialtree(file_path)

    raise ValueError(
        f"Unsupported file type: {Path(file_path).suffix!r}. "
        "Expected .nc, .nwb, .npz, a pynapple folder, or a DANDI project directory."
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
# Wizard helpers
# ---------------------------------------------------------------------------


def _wizard_ds_to_continuous_dt(ds: xr.Dataset) -> TrialTree:
    """Wrap a single xr.Dataset as a one-trial continuous TrialTree."""
    
    
    
    duration = eto.get_ds_duration(ds)
    if duration is not None:
        epochs = pd.DataFrame({
            "trial": [1],
            "start_time": [0.0],
            "stop_time": [duration],
        })
    else:
        epochs = pd.DataFrame({
            "trial": [1],
            "start_time": [0.0],
        })
        
    
    return TrialTree.from_continuous(ds, epochs)


def _wizard_single_media_helper(
    dt,
    video_path=None,
    pose_path=None,
    audio_path=None,
    video_offset: float | None = None,
    audio_offset: float | None = None,
):
    """Create a minimal NWB alignment file for a single-trial wizard."""
    from ethograph.io.nwb_alignment import alignment_media_per_trial

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
    
    if isinstance(dt.itrial(0), xr.Dataset):
        fps = dt.itrial(0).attrs.get("fps")
    elif isinstance(dt, xr.Dataset):
        fps = dt.attrs.get("fps")
    else:
        fps = None

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
    alignment_media_per_trial(
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

    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(
        dt, video_path=video_path, pose_path=pose_path, video_offset=video_offset
    )
    return dt


def wizard_single_from_ds(
    video_path, ds: xr.Dataset, video_offset: float | None = None
):
    _wizard_single_media_helper(ds, video_path=video_path, video_offset=video_offset)
    return ds


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

    if video_motion and video_path is not None:
        from ethograph.features.movement import extract_video_motion
        ds["video_motion"] = extract_video_motion(
            video_path, fps=ds.attrs["fps"], time_coord_name="time_video"
        )

    dt = _wizard_ds_to_continuous_dt(ds)
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

    if video_motion and video_path is not None:
        from ethograph.features.movement import extract_video_motion
        ds["video_motion"] = extract_video_motion(
            video_path, fps=ds.attrs["fps"], time_coord_name="time_video"
        )

    dt = _wizard_ds_to_continuous_dt(ds)
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
        ds["video_motion"] = extract_video_motion(
            video_path, fps=ds.attrs["fps"], time_coord_name="time_video"
        )

    dt = _wizard_ds_to_continuous_dt(ds)
    _wizard_single_media_helper(
        dt,
        video_path=video_path,
        audio_path=audio_path,
        video_offset=video_offset,
    )
    return dt
