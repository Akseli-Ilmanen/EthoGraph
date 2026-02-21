"""Spectral changepoint detection using vocalseg dynamic threshold segmentation.

Uses vocalseg library for detecting vocal onset/offset candidates in audio files.
Reference: https://github.com/timsainb/vocalization-segmentation
"""

import audioio as aio
import numpy as np
import vocalpy as voc
from scipy.signal import stft

if not hasattr(np, "product"):
    np.product = np.prod

from vocalseg.continuity_filtering import continuity_segmentation
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation

from ethograph.features.energy import _to_sound, env_ava, env_meansquared


def _prepare_audio(
    audio_path: str | None = None,
    signal: np.ndarray | None = None,
    sr: float | None = None,
    channel_idx: int = 0,
) -> tuple[np.ndarray, float]:
    """Load audio and return (data_1d, sample_rate).

    Uses audioio for file loading instead of vocalpy.
    """
    if audio_path is not None:
        data, file_sr = aio.load_audio(audio_path)
        sr = float(file_sr)
        if data.ndim > 1:
            ch = min(channel_idx, data.shape[1] - 1)
            data = data[:, ch]
        else:
            data = data.ravel()
        return np.asarray(data, dtype=np.float64), sr

    if signal is None:
        raise ValueError("Either audio_path or signal must be provided")
    if sr is None:
        raise ValueError("sr required when using signal array")

    return np.asarray(signal, dtype=np.float64).ravel(), float(sr)


def _compute_spect_range(
    data_1d: np.ndarray, samplerate: int, **kwargs
) -> tuple[float, float]:
    nperseg = kwargs.get("nperseg", 1024)
    noverlap = kwargs.get("noverlap", 512)
    min_freq = kwargs.get("min_freq", 30e3)
    max_freq = kwargs.get("max_freq", 110e3)

    scaled = (data_1d * 2**15).astype(np.int16)
    f, _, spect = stft(scaled, samplerate, nperseg=nperseg, noverlap=noverlap)
    i1 = np.searchsorted(f, min_freq)
    i2 = np.searchsorted(f, max_freq)
    spect = np.log(np.abs(spect[i1:i2]) + 1e-9)
    return float(np.min(spect)), float(np.max(spect))


def get_audio_changepoints(
    method: str = "meansquared",
    audio_path: str | None = None,
    signal: np.ndarray | None = None,
    sr: float | None = None,
    channel_idx: int = 0,
    **kwargs,
) -> tuple:

    data_1d, sr = _prepare_audio(audio_path, signal, sr, channel_idx)
    sound = _to_sound(data_1d, sr)

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if method == "meansquared":
        segments = voc.segment.meansquared(sound, **kwargs)
        env_time, envelope = env_meansquared(data_1d, sr, **kwargs)

    elif method == "ava":
        if "spect_min_val" not in kwargs or "spect_max_val" not in kwargs:
            smin, smax = _compute_spect_range(data_1d, sr, **kwargs)
            kwargs.setdefault("spect_min_val", smin)
            kwargs.setdefault("spect_max_val", smax)

        segments = voc.segment.ava(sound, **kwargs)
        env_time, envelope = env_ava(data_1d, sr, **kwargs)

    elif method == "vocalseg":
        n_fft = kwargs.get("n_fft")
        if n_fft is not None:
            min_n_fft = int(np.ceil(0.005 * sr))
            if n_fft < min_n_fft:
                kwargs["n_fft"] = min_n_fft

        results = dynamic_threshold_segmentation(vocalization=signal, rate=sr, **kwargs)
        hop_length_ms = kwargs.get("hop_length_ms", 1)
        fft_rate = sr / int(hop_length_ms / 1000 * sr)
        envelope = results["vocal_envelope"]
        env_time = np.arange(len(envelope)) / fft_rate

        onsets = results["onsets"]
        offsets = results["offsets"]
        if len(onsets) == 0:
            return (np.array([]), np.array([])), env_time, envelope
        return (onsets, offsets), env_time, envelope

    elif method == "continuity":
        n_fft = kwargs.get("n_fft")
        if n_fft is not None:
            min_n_fft = int(np.ceil(0.005 * sr))
            if n_fft < min_n_fft:
                kwargs["n_fft"] = min_n_fft

        results = continuity_segmentation(vocalization=signal, rate=sr, **kwargs)
        hop_length_ms = kwargs.get("hop_length_ms", 1)
        fft_rate = sr / int(hop_length_ms / 1000 * sr)
        envelope = results["vocal_envelope"]
        env_time = np.arange(len(envelope)) / fft_rate

        onsets = results["onsets"]
        offsets = results["offsets"]
        if len(onsets) == 0:
            return (np.array([]), np.array([])), env_time, envelope
        return (onsets, offsets), env_time, envelope

    else:
        raise ValueError(f"Unknown method: {method!r}")

    onsets = segments.start_inds.astype(float) / sr
    offsets = segments.stop_inds.astype(float) / sr

    if segments.start_inds.size == 0:
        return (np.array([]), np.array([])), env_time, envelope

    return (onsets, offsets), env_time, envelope
