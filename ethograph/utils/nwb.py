"""NWB → TrialTree bridge (read direction) for the NWB import wizard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import subprocess
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import xarray as xr

try:
    import h5py
    import pynwb
    from pynwb import NWBFile
except ImportError:
    h5py = None
    pynwb = None
    NWBFile = None

try:
    import remfile
except ImportError:
    remfile = None

try:
    from dandi.dandiapi import DandiAPIClient
except ImportError:
    DandiAPIClient = None

try:
    from movement.io import load_poses
except ImportError:
    load_poses = None


def _require_nwb():
    if pynwb is None:
        raise ImportError(
            "h5py and pynwb are required for NWB support. "
            "Install them with: uv pip install \"ethograph[nwb]\""
        )


def _require_dandi():
    if DandiAPIClient is None:
        raise ImportError(
            "dandi is required for DANDI support. "
            "Install with: uv pip install \"ethograph[nwb]\""
        )

try:
    import lindi as _lindi
    _LINDI_AVAILABLE = True
except Exception:
    _lindi = None
    _LINDI_AVAILABLE = False

import ethograph as eto
from ethograph import TrialTree, get_time_coord


# ---------------------------------------------------------------------------
# DANDI URL parsing
# ---------------------------------------------------------------------------

_DANDI_HOSTS = frozenset({
    "api.dandiarchive.org",
    "dandiarchive.org",
    "lindi.neurosift.org",
    "neurosift.app",
})
_UUID_RE = re.compile(
    r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    re.IGNORECASE,
)
_DANDISET_RE = re.compile(r"/dandisets/(\d+)/")


def parse_dandi_url(url: str) -> dict | None:
    if not url:
        return None
    url = url.strip()
    parsed = urlparse(url)
    if not any(host in parsed.netloc for host in _DANDI_HOSTS):
        return None

    query_params = parse_qs(parsed.query)
    dandiset_id = (query_params.get("dandisetId") or [None])[0]

    embedded = (query_params.get("url") or [None])[0]
    if embedded:
        m = _UUID_RE.search(embedded)
        if m:
            asset_id = m.group(1)
            if not dandiset_id:
                dm = _DANDISET_RE.search(embedded)
                if dm:
                    dandiset_id = dm.group(1)
            return {"dandiset_id": dandiset_id, "asset_id": asset_id, "streaming_url": embedded}

    m = _UUID_RE.search(url)
    if m:
        asset_id = m.group(1)
        if not dandiset_id:
            dm = _DANDISET_RE.search(url)
            if dm:
                dandiset_id = dm.group(1)
        return {"dandiset_id": dandiset_id, "asset_id": asset_id, "streaming_url": url}

    return None


# ---------------------------------------------------------------------------
# NWB file openers
# ---------------------------------------------------------------------------

def open_nwb_local(path: str) -> tuple:
    """Open a local NWB file. Returns (nwb, io, h5_file, None)."""
    _require_nwb()
    h5_file = h5py.File(path, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
    return io.read(), io, h5_file, None


def open_nwb_dandi(dandiset_id: str, asset_id: str) -> tuple:
    """Open a DANDI NWB file, trying lindi index first for speed.

    Lindi provides a pre-built JSON index on neurosift.org, making metadata
    access nearly instant compared to streaming via remfile. Falls back to
    remfile if lindi is unavailable for this asset.

    Returns (nwb, io, h5_file, rf) where rf=None when lindi is used.
    """
    _require_nwb()
    _require_dandi()
    if _LINDI_AVAILABLE:
        lindi_url = (
            f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}"
            f"/assets/{asset_id}/nwb.lindi.json"
        )
        try:
            lindi_file = _lindi.LindiH5pyFile.from_lindi_file(lindi_url)
            io = pynwb.NWBHDF5IO(file=lindi_file, mode="r", load_namespaces=True)
            return io.read(), io, lindi_file, None
        except Exception:
            pass

    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id).get_asset(asset_id)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    rf = remfile.File(url)
    h5_file = h5py.File(rf, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
    return io.read(), io, h5_file, rf


def find_video_assets(
    dandiset_id: str,
    nwb: Any,
    asset_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    _require_dandi()
    video_extensions = frozenset({".mp4", ".avi", ".mov", ".mkv"})

    for item in getattr(nwb, "acquisition", {}).values():
        external_files = getattr(item, "external_file", None)
        if external_files is None:
            continue
        files = external_files[:] if hasattr(external_files, "__getitem__") else [external_files]
        videos = [
            (Path(str(f)).stem, str(f))
            for f in files
            if Path(str(f)).suffix.lower() in video_extensions
        ]
        if videos:
            return videos

    subject = getattr(nwb, "subject", None)
    identifier = getattr(nwb, "identifier", None)
    search_terms = [
        t
        for t in [
            getattr(nwb, "session_id", None),
            identifier[:8] if identifier else None,
            getattr(subject, "subject_id", None) if subject else None,
            asset_id,
        ]
        if t
    ]

    if not search_terms:
        return []

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id)
        video_assets = []

        for asset in dandiset.get_assets():
            if Path(asset.path).suffix.lower() not in video_extensions:
                continue
            if not any(term in asset.path for term in search_terms):
                continue

            video_assets.append((Path(asset.path).stem, f"https://api.dandiarchive.org/api/assets/{asset.identifier}/download/"))

            if progress_callback:
                progress_callback(f"Found video: {Path(asset.path).name}")

        return video_assets




def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Video accessor
# ---------------------------------------------------------------------------

def download_clip(
    source: str,
    t_start: float,
    t_stop: float,
    output_path: Path,
) -> Path | None:
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try copying slice of file, e.g. mp4 (session length) -> mp4 (trial length).
    cmd = [
        "ffmpeg", "-ss", str(t_start), "-i", source,
        "-t", str(t_stop - t_start),
        "-c", "copy",
        str(output_path), "-y"
    ]

    result = subprocess.run(cmd, capture_output=True)

    # Convert to new format (e.g. .mkv to .mp4) to trial length slice
    if result.returncode != 0:
        subprocess.run([
            "ffmpeg", "-ss", str(t_start), "-i", source,
            "-t", str(t_stop - t_start),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(output_path), "-y"
        ])
    return output_path


# ---------------------------------------------------------------------------
# NWB metadata probing
# ---------------------------------------------------------------------------

def probe_behavioral_series(nwb: NWBFile) -> list[dict]:
    """List all behavioral time-series interfaces available for import."""
    SKIP_MODULES = {"ecephys", "ophys", "ogen"}
    results = []
    for mod_name, mod in nwb.processing.items():
        if mod_name in SKIP_MODULES:
            continue
        for iface_name, iface in mod.data_interfaces.items():
            if hasattr(iface, "pose_estimation_series"):
                continue
            if not hasattr(iface, "data"):
                continue
            if not _has_valid_timing(iface):
                continue
            try:
                n = iface.data.shape[0] if hasattr(iface.data, "shape") else len(iface.data)
            except Exception:
                n = 0
            results.append({
                "source": f"{mod_name}/{iface_name}",
                "module": mod_name,
                "interface": iface_name,
                "n_samples": n,
            })
    return results


def probe_electrical_series(nwb: NWBFile) -> list[dict]:
    """List all ElectricalSeries in nwb.acquisition."""
    results = []
    for name, obj in nwb.acquisition.items():
        if not isinstance(obj, pynwb.ecephys.ElectricalSeries):
            continue
        n_samples = obj.data.shape[0] if hasattr(obj.data, "shape") else len(obj.data)
        n_channels = obj.data.shape[1] if hasattr(obj.data, "shape") and obj.data.ndim > 1 else 1
        rate = float(obj.rate) if obj.rate else None
        results.append({
            "name": name,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "rate": rate,
        })
    return results


def probe_label_sources(nwb: NWBFile) -> list[dict]:
    """List all potential interval label sources in the NWB file."""
    results = []

    if nwb.epochs is not None:
        try:
            n = len(nwb.epochs)
        except Exception:
            n = 0
        results.append({"source": "epochs", "description": f"nwb.epochs ({n} rows)"})

    for mod_name, mod in nwb.processing.items():
        for iface_name, iface in mod.data_interfaces.items():
            if isinstance(iface, pynwb.epoch.TimeIntervals):
                try:
                    n = len(iface)
                except Exception:
                    n = 0
                results.append({
                    "source": f"{mod_name}/{iface_name}",
                    "description": f"TimeIntervals: {mod_name}/{iface_name} ({n} rows)",
                })
            elif isinstance(iface, pynwb.behavior.BehavioralEpochs):
                for series_name in iface.interval_series:
                    results.append({
                        "source": f"{mod_name}/{iface_name}/{series_name}",
                        "description": f"IntervalSeries: {mod_name}/{iface_name}/{series_name}",
                    })
    return results


# ---------------------------------------------------------------------------
# Behavioral series converter
# ---------------------------------------------------------------------------

class BehaviorSpatialTimeSeriesConverter:
    name = "spatial_series"
    SKIP_MODULES = {"ecephys", "ophys", "ogen"}

    def __init__(self, include_sources: set[str] | None = None):
        self._include_sources = include_sources
        self._data: dict[str, xr.DataArray] = {}

    def _iter_ifaces(self, nwb: NWBFile):
        for mod_name, mod in nwb.processing.items():
            if mod_name in self.SKIP_MODULES:
                continue
            for name, iface in mod.data_interfaces.items():
                if self._include_sources is not None and f"{mod_name}/{name}" not in self._include_sources:
                    continue
                if hasattr(iface, "pose_estimation_series"):
                    continue
                if not hasattr(iface, "data"):
                    continue
                if not _has_valid_timing(iface):
                    continue
                yield name, iface

    def load(self, nwb: NWBFile) -> None:
        for name, iface in self._iter_ifaces(nwb):
            timestamps = _get_absolute_timestamps(iface)
            data = np.asarray(iface.data[:])

            if data.ndim == 1:
                self._data[name] = xr.DataArray(data, dims=("time",), coords={"time": timestamps})
            elif self._is_spatial(iface, data):
                space = ["x", "y", "z"][:data.shape[-1]]
                self._data[name] = xr.DataArray(data, dims=("time", "space"), coords={"time": timestamps, "space": space})
            else:
                channels = [f"ch_{i}" for i in range(data.reshape(len(data), -1).shape[1])]
                self._data[name] = xr.DataArray(data.reshape(len(data), -1), dims=("time", "channel"), coords={"time": timestamps, "channel": channels})

    def from_nwb(self, nwb: NWBFile, trial_idx: int, t_start: float, t_stop: float) -> dict[str, xr.DataArray]:
        if not self._data:
            self.load(nwb)
        return {
            name: da.sel(time=slice(t_start, t_stop)).assign_coords(time=da.sel(time=slice(t_start, t_stop)).time - t_start)
            for name, da in self._data.items()
        }

    @staticmethod
    def _is_spatial(iface: Any, data: np.ndarray) -> bool:
        return hasattr(iface, "reference_frame") or (data.ndim == 2 and data.shape[-1] in (2, 3))


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def _get_pose_timestamps(pose_estimation: Any) -> np.ndarray:
    """Extract session-absolute timestamps from a PoseEstimation interface.

    movement's ``load_poses.from_nwb_file`` discards the original NWB
    timestamps and reconstructs time as ``arange(n) / fps`` starting at 0.
    This helper retrieves the real timestamps so they can be reassigned.
    """
    series = next(iter(pose_estimation.pose_estimation_series.values()))
    return _get_absolute_timestamps(series)


def _estimate_fps(pose_estimation: Any, n_frames: int = 100) -> float:
    series = next(iter(pose_estimation.pose_estimation_series.values()))
    return series.rate or 1 / np.diff(series.timestamps[:n_frames]).mean()


def _get_keypoints(pose_estimation: Any) -> set[str]:
    return set(pose_estimation.pose_estimation_series.keys())


def _find_pose_module_key(nwb: NWBFile, camera_name: str) -> str:
    for mod_name, mod in nwb.processing.items():
        if camera_name in mod.data_interfaces:
            return mod_name
    return "pose_estimation"


def load_nwb_session(
    nwb_file: NWBFile,
    pose_containers: dict[str, Any] | None = None,
    cameras_with_pose: list[str] | None = None,
    trial_indices: list[int] | None = None,
    include_pose: bool = True,
    behavioral_sources: set[str] | None = None,
) -> tuple[TrialTree, pd.DataFrame]:
    _require_nwb()
    trials_df = read_trials_table(nwb_file)
    if trial_indices is not None:
        trials_df = trials_df.iloc[trial_indices].reset_index(drop=True)

    assert not include_pose or pose_containers is not None, "pose_containers dict must be provided to include pose data"


    if include_pose:
        fps_per_cam = {k: _estimate_fps(v) for k, v in pose_containers.items()}
        kps_per_cam = {k: _get_keypoints(v) for k, v in pose_containers.items()}
        shared_keypoints = set.intersection(*kps_per_cam.values())
        same_fps = len(set(fps_per_cam.values())) == 1
        

        pose_datasets = {
            cam_name: load_poses.from_nwb_file(
                nwb_file,
                processing_module_key=_find_pose_module_key(nwb_file, cam_name),
                pose_estimation_key=cam_name,
            )
            for cam_name in pose_containers
        }

        # movement discards NWB timestamps and rebuilds time from 0.
        # Restore session-absolute timestamps so trial slicing works.
        for cam_name, pose_est in pose_containers.items():
            abs_ts = _get_pose_timestamps(pose_est)
            ds = pose_datasets[cam_name] # in case single item
            pose_datasets[cam_name] = ds.assign_coords(time=abs_ts)


            
            # Single camera view
            if len(pose_containers) == 1:
                pose_ds = ds
                pose_keys = [str(k) for k in pose_containers]
            
            # Same keypoints/fps -> can use Movement multiview convention
            elif len(pose_containers) > 1 and bool(shared_keypoints) and same_fps:
                pose_ds = xr.concat(
                    [d.sel(keypoints=list(shared_keypoints)) for d in pose_datasets.values()],
                    dim=xr.DataArray(list(pose_datasets.keys()), dims="view"),
                )
                pose_keys = [str(v) for v in ds["position"].coords["view"].values]
                

            elif same_fps:
                pose_ds = xr.merge([
                    d.rename({"position": f"position_{cam}", "confidence": f"confidence_{cam}"})
                    for cam, d in pose_datasets.items()
                ])
                pose_keys = [k for k in cameras_with_pose if f"position_{k}" in pose_ds.data_vars]

            else:
                pose_ds = xr.merge([
                    d.rename({
                        "position": f"position_{cam}",
                        "confidence": f"confidence_{cam}",
                        "time": f"time_{int(round(fps_per_cam[cam]))}Hz",
                    })
                    for cam, d in pose_datasets.items()
                ])
                pose_keys = [k for k in cameras_with_pose if f"position_{k}" in pose_ds.data_vars]



    behavior_converter = BehaviorSpatialTimeSeriesConverter(include_sources=behavioral_sources)
    behavior_converter.load(nwb_file)

    ds_list = []
    for _, row in trials_df.iterrows():
        t_start, t_stop = float(row["start_time"]), float(row["stop_time"])
        behavior_trial = behavior_converter.from_nwb(nwb_file, int(row["trial"]), t_start, t_stop)

        if include_pose:
            pose_slices = {
                get_time_coord(pose_ds[var]).name: slice(t_start, t_stop)
                for var in pose_ds.data_vars
                if "position" in str(var)
            }
            pose_trial = pose_ds.sel(pose_slices)
            pose_trial = pose_trial.assign_coords({
                dim: pose_trial[dim].values - t_start
                for dim in pose_slices
            })

            if pose_slices:
                pose_time_dim = next(iter(pose_slices))
                time_vals = pose_trial[pose_time_dim].values
                pose_hz = int(round(1.0 / float(np.diff(time_vals[:2]).item()))) if len(time_vals) >= 2 else None
                aligned_behavior: dict[str, xr.DataArray] = {}
                for var, da in behavior_trial.items():
                    time_dim = next((d for d in da.dims if "time" in d), None)
                    if time_dim is None:
                        aligned_behavior[var] = da
                        continue
                    hz = int(time_dim.replace("time_", "").replace("Hz", "")) if "Hz" in time_dim else None
                    aligned_behavior[var] = da.rename({time_dim: pose_time_dim}) if hz is not None and hz == pose_hz else da
                ds_trial = xr.merge([pose_trial, xr.Dataset(aligned_behavior)])
            else:
                ds_trial = xr.Dataset(behavior_trial)
        else:
            ds_trial = xr.Dataset(behavior_trial)

        ds_trial = ds_trial.assign_attrs(
            trial=int(row["trial"]),
            start_time=t_start,
            stop_time=t_stop,
            **{col: _coerce_attr(row[col]) for col in trials_df.columns if col not in ("trial", "start_time", "stop_time")},
        )
        ds_trial = _assign_individual(ds_trial, nwb_file)
        
        for var in ds_trial.data_vars:
            ds_trial[var].attrs["type"] = "features"
        
        ds_list.append(ds_trial)
        
    dt = eto.from_datasets(ds_list)
    
    if include_pose:
        dt.attrs["nwb_pose_keys"] = pose_keys

    return dt, trials_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_timeseries_timing(iface: Any) -> tuple[float, float]:
    """Extract (rate_hz, starting_time_s) from any NWB TimeSeries.

    Handles both NWB timing schemes:
    - ``rate`` + ``starting_time``: returns them directly.
    - ``timestamps``: derives rate from median inter-sample interval,
      starting_time from ``timestamps[0]``.

    Raises ``ValueError`` if neither scheme is available.
    """
    if getattr(iface, "rate", None) is not None and iface.rate:
        t0 = float(iface.starting_time) if getattr(iface, "starting_time", None) is not None else 0.0
        return float(iface.rate), t0
    ts = getattr(iface, "timestamps", None)
    if ts is not None and len(ts) >= 2:
        ts_arr = np.asarray(ts[:min(len(ts), 10_000)], dtype=np.float64)
        diffs = np.diff(ts_arr)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            rate = 1.0 / float(np.median(diffs))
            return rate, float(ts_arr[0])
    raise ValueError(
        f"TimeSeries '{getattr(iface, 'name', '?')}' has neither rate nor timestamps."
    )


def _has_valid_timing(iface: Any) -> bool:
    """Return True if the interface has either an explicit timestamps array or a rate."""
    if getattr(iface, "timestamps", None) is not None:
        return True
    return getattr(iface, "rate", None) is not None


def _get_absolute_timestamps(iface: Any) -> np.ndarray:
    """Return timestamps in absolute session time for any NWB TimeSeries.

    NWB supports two timing schemes:
    - ``timestamps``: explicit array already in absolute session time.
    - ``rate`` + ``starting_time``: regularly sampled; absolute times are
      ``starting_time + arange(n) / rate``.  ``starting_time`` defaults to 0
      when absent (i.e. recording starts at session time 0).
    """
    if getattr(iface, "timestamps", None) is not None:
        return np.asarray(iface.timestamps[:], dtype=np.float64)
    n = iface.data.shape[0] if hasattr(iface.data, "shape") else len(iface.data)
    t0 = float(iface.starting_time) if getattr(iface, "starting_time", None) is not None else 0.0
    return t0 + np.arange(n, dtype=np.float64) / float(iface.rate)


def _get_individual_coord(nwb: NWBFile) -> list[str]:
    subject = getattr(nwb, "subject", None)
    sid = getattr(subject, "subject_id", None) if subject else None
    return [str(sid) if sid else "individual_0"]

def _assign_individual(ds: xr.Dataset, nwb: NWBFile) -> xr.Dataset:
    return ds.assign_coords(individuals=_get_individual_coord(nwb))


def read_trials_table(nwb: NWBFile) -> pd.DataFrame:
    if nwb.trials is None or len(nwb.trials) == 0:
        duration = _get_max_duration(nwb)
        return pd.DataFrame([{"trial": 1, "start_time": 0.0, "stop_time": duration}])

    df = nwb.trials.to_dataframe()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_resolve)

    if "trial" not in df.columns:
        df = df.reset_index(drop=True)
        df["trial"] = df.index + 1
    return df

def _resolve(val):
    if hasattr(val, 'data'):  # h5py / NWB lazy wrapper
        val = val.data
    if hasattr(val, '__array__'):
        val = val.item() if val.ndim == 0 else val.tolist()
    return val

def _coerce_attr(val: Any) -> Any:
    if isinstance(val, (np.bool_, bool)):
        return int(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val



def _get_max_duration(nwb: NWBFile) -> float:
    durations = []
    for ts in nwb.acquisition.values():
        if isinstance(ts, pynwb.TimeSeries):
            dur = _ts_duration(ts)
            if dur is not None:
                durations.append(dur)
    return max(durations) if durations else 1.0


def _ts_duration(ts: Any) -> float | None:
    n = ts.data.shape[0] if hasattr(ts.data, "shape") else len(ts.data)
    if ts.timestamps is not None and len(ts.timestamps) > 0:
        return float(ts.timestamps[-1])
    if ts.rate and ts.rate > 0:
        start = float(ts.starting_time) if ts.starting_time else 0.0
        return start + n / float(ts.rate)
    return None


# ---------------------------------------------------------------------------
# NWB session creation helpers
# ---------------------------------------------------------------------------


_KNOWN_STREAMS = ("video", "pose", "audio", "ephys")


def _parse_stream_devices(columns: list[str]) -> dict[str, list[str]]:
    """Detect ``{stream}_{device}`` columns → ``{stream: [device, ...]}``."""
    result: dict[str, list[str]] = {}
    for col in columns:
        for stream in _KNOWN_STREAMS:
            prefix = f"{stream}_"
            if col.startswith(prefix) and not col.endswith("_start"):
                device = col[len(prefix):]
                if device:
                    result.setdefault(stream, []).append(device)
    return result


def sync_acquisition_for_streams(
    nwbfile: NWBFile,
    stream_rates: dict[str, float],
) -> None:
    """Create ImageSeries acquisition items for ALL external media streams.

    Reads the trials table to discover ``{stream}_{device}`` columns.
    For each stream+device pair, creates an ``ImageSeries`` in
    ``nwbfile.acquisition`` with ``external_file``, ``starting_frame``,
    and ``rate`` (or ``timestamps`` if offsets are present).

    Parameters
    ----------
    nwbfile
        NWB file with a populated trials table.
    stream_rates
        Mapping of stream name to sampling rate, e.g.
        ``{"video": 30.0, "audio": 44100.0, "pose": 30.0}``.
    """
    from pynwb.image import ImageSeries

    df = nwbfile.trials.to_dataframe()
    stream_devices = _parse_stream_devices(list(df.columns))

    for stream, devices in stream_devices.items():
        rate = stream_rates.get(stream)
        if rate is None or rate <= 0:
            continue

        for device in devices:
            col = f"{stream}_{device}"
            if col not in df.columns:
                continue

            valid = df[df[col] != ""]
            if valid.empty:
                continue

            external_files = valid[col].tolist()

            start_col = f"{col}_start"
            if start_col in df.columns:
                starts = valid[start_col].values.astype(float)
            else:
                starts = valid["start_time"].values.astype(float)

            timestamps_parts: list[np.ndarray] = []
            starting_frames: list[int] = []
            frame_count = 0

            # Check if we have real trial durations
            has_real_durations = (
                "stop_time" in valid.columns
                and valid["stop_time"].notna().all()
                and (valid["stop_time"].astype(float) - valid["start_time"].astype(float) > 0).all()
            )

            for i, (_, row) in enumerate(valid.iterrows()):
                file_start = float(starts[i])
                if has_real_durations:
                    duration = float(row["stop_time"]) - float(row["start_time"])
                    n_samples = max(1, int(duration * rate))
                else:
                    n_samples = 1
                ts = file_start + np.arange(n_samples) / rate
                timestamps_parts.append(ts)
                starting_frames.append(frame_count)
                frame_count += n_samples

            if device not in [d.name for d in nwbfile.devices.values()]:
                nwbfile.create_device(
                    name=device, description=f"{stream} device {device}"
                )

            acq_name = f"{stream}_{device}"
            if acq_name in nwbfile.acquisition:
                del nwbfile.acquisition[acq_name]

            if has_real_durations:
                nwbfile.add_acquisition(
                    ImageSeries(
                        name=acq_name,
                        description=f"{stream} from {device}",
                        external_file=external_files,
                        format="external",
                        starting_frame=np.array(starting_frames, dtype=np.int32),
                        timestamps=np.concatenate(timestamps_parts),
                    )
                )
            else:
                nwbfile.add_acquisition(
                    ImageSeries(
                        name=acq_name,
                        description=f"{stream} from {device}",
                        external_file=external_files,
                        format="external",
                        starting_frame=np.array(starting_frames, dtype=np.int32),
                        rate=rate,
                    )
                )


def build_nwb_session(
    media_by_trial: dict[int, dict[str, dict[str, Path]]],
    cam_labels: list[str],
    stream_names: list[str],
    stream_rates: dict[str, float] | None = None,
    output_path: Path | None = None,
) -> NWBFile:
    """Create an NWB file with trials table and acquisition items.

    Parameters
    ----------
    media_by_trial
        ``{trial_id: {stream: {device: Path}}}`` nested dict.
    cam_labels
        Camera device names.
    stream_names
        Stream names to include (e.g. ``["video", "pose"]``).
    stream_rates
        Rate per stream. Streams not listed are skipped.
    output_path
        If given, writes the NWB file to this path.
    """
    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    all_trials = sorted(media_by_trial.keys())

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for media alignment (ethograph generated).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Original trial number")
    for cam in cam_labels:
        for stream in stream_names:
            nwbfile.add_trial_column(
                name=f"{stream}_{cam}",
                description=f"{stream} filename for {cam}",
            )

    for trial in all_trials:
        row: dict[str, Any] = {"trial": trial, "start_time": 0.0, "stop_time": 1.0}
        for cam in cam_labels:
            for stream in stream_names:
                path = media_by_trial[trial].get(stream, {}).get(cam)
                row[f"{stream}_{cam}"] = path.name if path else ""
        nwbfile.add_trial(**row)

    if stream_rates:
        sync_acquisition_for_streams(nwbfile, stream_rates)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(output_path), "w") as io:
            io.write(nwbfile)

    return nwbfile


def build_nwb_from_trial_table(
    trial_table: pd.DataFrame,
    stream_rates: dict[str, float] | None = None,
    output_path: Path | None = None,
    session_description: str = "NWB file for media alignment (ethograph generated).",
) -> NWBFile:
    """Create an NWB file from a pandas DataFrame trial table.

    The DataFrame must have a ``trial`` column. Media columns are detected
    by the ``{stream}_{device}`` naming convention (e.g. ``video_cam-1``,
    ``audio_mic-1``, ``pose_cam-1``).  An ``ImageSeries`` is created in
    acquisition for each stream+device pair.

    Parameters
    ----------
    trial_table
        DataFrame with ``trial``, ``start_time``, ``stop_time`` and
        media columns like ``video_cam-1``, ``audio_mic-1``.
    stream_rates
        Sampling rate per stream, e.g.
        ``{"video": 30.0, "audio": 44100.0, "pose": 30.0}``.
        Streams not listed are skipped (no ImageSeries created).
    output_path
        Write path. Creates parent directories.
    session_description
        NWB session description string.
    """
    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    nwbfile = pynwb.NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    # Detect media columns (everything except trial/start_time/stop_time)
    reserved = {"trial", "start_time", "stop_time"}
    media_cols = [c for c in trial_table.columns if c not in reserved]

    has_trial_col = "trial" in trial_table.columns
    if has_trial_col:
        nwbfile.add_trial_column(name="trial", description="Trial number")

    has_stop_time = "stop_time" in trial_table.columns

    for col in media_cols:
        nwbfile.add_trial_column(name=col, description=f"{col} filename")

    for _, row in trial_table.iterrows():
        start = float(row.get("start_time", 0.0))
        if has_stop_time and pd.notna(row.get("stop_time")):
            stop = float(row["stop_time"])
        else:
            stop = start + 1.0  # NWB requires stop > start
        trial_row: dict[str, Any] = {"start_time": start, "stop_time": stop}
        if has_trial_col:
            trial_row["trial"] = row["trial"]
        for col in media_cols:
            trial_row[col] = str(row[col]) if pd.notna(row[col]) else ""
        nwbfile.add_trial(**trial_row)

    # Create ImageSeries for all detected streams
    if stream_rates:
        sync_acquisition_for_streams(nwbfile, stream_rates)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(output_path), "w") as io:
            io.write(nwbfile)

    return nwbfile


def create_alignment(
    trial_table: pd.DataFrame,
    stream_rates: dict[str, float],
    output_path: str | Path,
) -> Path:
    """Create an alignment.nwb from a trial table.

    This is the primary user-facing function for creating alignment files.

    Parameters
    ----------
    trial_table
        DataFrame with ``trial`` column and ``{stream}_{device}`` filename
        columns.  ``start_time`` / ``stop_time`` are optional — omit for
        aligned-to-trial data.

        Example::

            trial | video_cam-1    | audio_mic-1   | pose_cam-1
            1     | cam1_t1.mp4    | mic1_t1.wav   | cam1_t1.h5
            2     | cam1_t2.mp4    | mic1_t2.wav   | cam1_t2.h5

    stream_rates
        Sampling rate per stream.  Must include every stream that has
        columns in the table.  Example:
        ``{"video": 30.0, "audio": 48000.0, "pose": 30.0}``
    output_path
        Where to write the ``.nwb`` file.

    Returns
    -------
    Path to the created NWB file.

    Examples
    --------
    >>> import pandas as pd, ethograph as eto
    >>> table = pd.DataFrame({
    ...     "trial": [1, 2, 3],
    ...     "video_cam-1": ["t1.mp4", "t2.mp4", "t3.mp4"],
    ...     "pose_cam-1": ["t1.h5", "t2.h5", "t3.h5"],
    ... })
    >>> eto.create_alignment(table, {"video": 30.0, "pose": 30.0}, "out/.ethograph/alignment.nwb")
    """
    output = Path(output_path)
    build_nwb_from_trial_table(trial_table, stream_rates=stream_rates, output_path=output)
    return output


def create_alignment_from_streams(
    trials: pd.DataFrame,
    streams: list[dict],
    output_path: str | Path,
) -> Path:
    """Create an alignment.nwb for unaligned / complex scenarios.

    The trials table contains only timing (no filenames).  All file
    references go into ImageSeries acquisition items.

    Parameters
    ----------
    trials
        DataFrame with ``trial``, ``start_time``, ``stop_time``.
    streams
        List of stream dicts, each with::

            {
                "name": "video_cam-1",       # acquisition item name
                "files": ["t1.mp4", ...],    # one per trial (full paths)
                "rate": 30.0,                # sampling rate
            }

        For session-wide files (one file spanning all trials)::

            {
                "name": "audio_mic-1",
                "files": ["session.wav"],
                "rate": 44100.0,
                "starting_time": 0.0,        # when file starts in session time
            }

        For streams with explicit timestamps (irregular)::

            {
                "name": "ephys_probe-1",
                "files": ["session.dat"],
                "timestamps": np.array([0.0, 0.001, ...]),
            }

    output_path
        Where to write the ``.nwb`` file.

    Returns
    -------
    Path to the created NWB file.

    Examples
    --------
    Per-trial video + pose, session-wide audio::

        >>> trials = pd.DataFrame({
        ...     "trial": [1, 2, 3],
        ...     "start_time": [0.0, 10.5, 22.3],
        ...     "stop_time": [8.2, 19.1, 30.0],
        ... })
        >>> streams = [
        ...     {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4", "t3.mp4"], "rate": 30.0},
        ...     {"name": "pose_cam-1", "files": ["t1.h5", "t2.h5", "t3.h5"], "rate": 30.0},
        ...     {"name": "audio_mic-1", "files": ["session.wav"], "rate": 48000.0, "starting_time": 0.0},
        ... ]
        >>> eto.create_alignment_from_streams(trials, streams, ".ethograph/alignment.nwb")
    """
    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for media alignment (ethograph generated).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    # Trials table: only timing, no filenames
    nwbfile.add_trial_column(name="trial", description="Trial number")
    for _, row in trials.iterrows():
        nwbfile.add_trial(
            trial=row["trial"],
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
        )

    trial_starts = trials["start_time"].values.astype(float)
    trial_stops = trials["stop_time"].values.astype(float)
    n_trials = len(trials)

    for spec in streams:
        name = spec["name"]
        files = spec["files"]
        rate = spec.get("rate")
        explicit_ts = spec.get("timestamps")
        starting_time = spec.get("starting_time", None)

        # Parse stream_device for device creation
        parts = name.split("_", 1)
        device_name = parts[1] if len(parts) > 1 else parts[0]
        if device_name not in [d.name for d in nwbfile.devices.values()]:
            nwbfile.create_device(name=device_name, description=f"Device {device_name}")

        if explicit_ts is not None:
            # Irregular timestamps provided directly
            nwbfile.add_acquisition(
                ImageSeries(
                    name=name,
                    description=name,
                    external_file=files,
                    format="external",
                    starting_frame=np.array([0] * len(files), dtype=np.int32),
                    timestamps=np.asarray(explicit_ts, dtype=np.float64),
                )
            )
        elif len(files) == 1 and n_trials > 1:
            # Session-wide: one file spanning all trials
            t0 = starting_time if starting_time is not None else float(trial_starts[0])
            t1 = float(trial_stops[-1])
            n_samples = max(1, int((t1 - t0) * rate)) if rate else 1
            timestamps = t0 + np.arange(n_samples) / rate if rate else np.array([t0])
            nwbfile.add_acquisition(
                ImageSeries(
                    name=name,
                    description=name,
                    external_file=files,
                    format="external",
                    starting_frame=np.array([0], dtype=np.int32),
                    timestamps=timestamps,
                )
            )
        else:
            # Per-trial: one file per trial
            timestamps_parts = []
            starting_frames = []
            frame_count = 0
            for i in range(min(len(files), n_trials)):
                t0 = float(trial_starts[i])
                dur = float(trial_stops[i]) - t0
                n_samples = max(1, int(dur * rate)) if rate else 1
                ts = t0 + np.arange(n_samples) / rate if rate else np.array([t0])
                timestamps_parts.append(ts)
                starting_frames.append(frame_count)
                frame_count += n_samples
            nwbfile.add_acquisition(
                ImageSeries(
                    name=name,
                    description=name,
                    external_file=files[:n_trials],
                    format="external",
                    starting_frame=np.array(starting_frames, dtype=np.int32),
                    timestamps=np.concatenate(timestamps_parts),
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output), "w") as io:
        io.write(nwbfile)

    return output


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

