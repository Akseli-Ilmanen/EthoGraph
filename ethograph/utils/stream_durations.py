"""Pure functions for probing media file durations.

No Qt, no TrialTree, no GUI dependencies — just path → float.
Used by the wizard timeline and any future TrialTree-level alignment helpers.
"""

from __future__ import annotations

from pathlib import Path


def get_video_duration(path: str) -> float | None:
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if stream.frames and stream.average_rate:
                return stream.frames / float(stream.average_rate)
    except Exception:
        pass
    return None


def get_audio_duration(path: str) -> float | None:
    try:
        import soundfile as sf

        return sf.info(path).duration
    except Exception:
        pass
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return None


def _count_csv_headers(path: str) -> int:
    """Count header rows in a pose CSV by finding where numeric data starts."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                float(parts[0])
                float(parts[1])
                return i
            except (ValueError, IndexError):
                continue
    return 1


def get_pose_duration(path: str, fps: float) -> float | None:
    """Estimate pose file duration from frame count and fps."""
    try:
        suffix = Path(path).suffix.lower()
        n_frames = None

        if suffix == ".csv":
            n_headers = _count_csv_headers(path)
            with open(path, "r") as fh:
                n_frames = sum(1 for _ in fh) - n_headers

        elif suffix in (".h5", ".hdf5", ".slp"):
            import h5py

            with h5py.File(path, "r") as f:
                if suffix == ".slp":
                    n_frames = f["instances"].shape[0]
                else:
                    for key in f.keys():
                        data = f[key]
                        if hasattr(data, "shape") and len(data.shape) >= 2:
                            n_frames = data.shape[0]
                            break

        if n_frames is not None and n_frames > 0:
            return n_frames / fps
    except Exception as e:
        print(f"Could not estimate duration for pose file {path}: {e}")
    return None


def get_ephys_duration(path: str) -> float | None:
    try:
        from ethograph.gui.plots_ephystrace import GenericEphysLoader

        loader = GenericEphysLoader(path)
        return len(loader) / loader.rate
    except Exception:
        pass
    return None


def get_kilosort_duration(folder: str) -> float | None:
    """Estimate recording duration from a kilosort output folder.

    Uses ``spike_times.npy`` (in samples) and ``params.py`` (sample_rate).
    Falls back to the raw ``.dat`` file referenced in params if available.
    """
    import numpy as np

    folder_path = Path(folder)
    spike_times_file = folder_path / "spike_times.npy"
    params_file = folder_path / "params.py"

    if not spike_times_file.exists() or not params_file.exists():
        return None

    try:
        namespace: dict = {}
        exec(params_file.read_text(), namespace)
        sample_rate = float(namespace.get("sample_rate", 0))
        if sample_rate <= 0:
            return None
    except Exception:
        return None

    # Try raw dat file first (most accurate)
    dat_path_str = namespace.get("dat_path", "")
    if dat_path_str:
        dat_path = Path(dat_path_str)
        if not dat_path.is_absolute():
            dat_path = folder_path / dat_path
        dur = get_ephys_duration(str(dat_path)) if dat_path.is_file() else None
        if dur is not None:
            return dur

    # Fall back to max spike time
    try:
        spike_times = np.load(str(spike_times_file)).ravel()
        if len(spike_times) == 0:
            return None
        max_sample = float(spike_times.max())
        return max_sample / sample_rate
    except Exception:
        return None


def probe_duration(path: str, stream: str, fps: float | None = None) -> float | None:
    """Dispatch to the appropriate duration probe based on stream type.

    Parameters
    ----------
    path:
        Path to the media file.
    stream:
        One of ``"video"``, ``"audio"``, ``"pose"``, ``"ephys"``.
    fps:
        Required for ``"pose"`` stream; ignored for all others.
    """
    if stream == "video":
        return get_video_duration(path)
    if stream == "audio":
        return get_audio_duration(path)
    if stream == "pose":
        if fps is None:
            raise ValueError(f"fps required for pose duration: {path}")
        return get_pose_duration(path, fps)
    if stream == "ephys":
        return get_ephys_duration(path)
    return None
