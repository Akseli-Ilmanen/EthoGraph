"""Wrapper around pynapple's detect_oscillatory_events for GUI integration."""

import numpy as np
import pynapple as nap


def detect_oscillatory_events_np(
    data: np.ndarray,
    sr: float,
    freq_band: tuple = (6.0, 10.0),
    thresh_band: tuple = (1.0, 10.0),
    duration_band: tuple = (0.3, 1.5),
    min_inter_duration: float = 0.02,
    wsize: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect oscillatory events in a 1-D signal using pynapple.

    Bandpass-filters the signal, thresholds the normalized squared envelope,
    and merges short gaps to identify rhythmic events (e.g., theta bursts,
    ripples, spindles).

    Args:
        data: 1-D numpy array of the signal.
        sr: Sampling rate in Hz.
        freq_band: (low_hz, high_hz) for bandpass filtering.
        thresh_band: (min_thresh, max_thresh) for normalized squared signal.
        duration_band: (min_duration_s, max_duration_s) for event duration.
        min_inter_duration: Minimum gap between events before merging (seconds).
        wsize: Window size for the FIR bandpass filter.

    Returns:
        (onsets, offsets) as numpy arrays of times in seconds.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    timestamps = np.arange(len(data)) / sr
    tsd = nap.Tsd(t=timestamps, d=data)
    epoch = nap.IntervalSet(start=timestamps[0], end=timestamps[-1])

    events = nap.process.filtering.detect_oscillatory_events(
        data=tsd,
        epoch=epoch,
        freq_band=freq_band,
        thresh_band=thresh_band,
        duration_band=duration_band,
        min_inter_duration=min_inter_duration,
        fs=sr,
        wsize=wsize,
    )

    if events is None or len(events) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    onsets = np.asarray(events["start"], dtype=np.float64)
    offsets = np.asarray(events["end"], dtype=np.float64)
    return onsets, offsets
