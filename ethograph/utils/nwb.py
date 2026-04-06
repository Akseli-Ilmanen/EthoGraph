"""NWB session creation, alignment helpers, and timing utilities.

Also re-exports names that moved to ``utils.dandi`` and ``io.nwb_import``
for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pynwb
    from pynwb import NWBFile
except ImportError:
    pynwb = None
    NWBFile = None


# ---------------------------------------------------------------------------
# Timing helpers (used by nwb_alignment, plots_ephystrace, and creation code)
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


# ---------------------------------------------------------------------------
# Stream column detection (shared with nwb_alignment)
# ---------------------------------------------------------------------------

_KNOWN_STREAMS = ("video", "pose", "audio", "ephys")


def _parse_stream_devices(columns: list[str]) -> dict[str, list[str]]:
    """Detect ``{stream}_{device}`` columns -> ``{stream: [device, ...]}``."""
    result: dict[str, list[str]] = {}
    for col in columns:
        for stream in _KNOWN_STREAMS:
            prefix = f"{stream}_"
            if col.startswith(prefix) and not col.endswith("_start"):
                device = col[len(prefix):]
                if device:
                    result.setdefault(stream, []).append(device)
    return result


# ---------------------------------------------------------------------------
# NWB session creation helpers
# ---------------------------------------------------------------------------


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
        columns.  ``start_time`` / ``stop_time`` are optional -- omit for
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


