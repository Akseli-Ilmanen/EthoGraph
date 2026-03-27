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
