"""Spectral changepoint detection using vocalseg dynamic threshold segmentation.

Uses vocalseg library for detecting vocal onset/offset candidates in audio files.
Reference: https://github.com/timsainb/vocalization-segmentation
"""

import audioio as aio
import noisereduce as nr
import numpy as np
import vocalpy as voc
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation



def _run_dynamic_threshold(
    signal: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    n_fft = kwargs.get("n_fft")
    if n_fft is not None:
        min_n_fft = int(np.ceil(0.005 * sample_rate))
        if n_fft < min_n_fft:
            kwargs["n_fft"] = min_n_fft

    results = dynamic_threshold_segmentation(
        vocalization=signal,
        rate=sample_rate,
        **kwargs,
    )

    if results is None:
        return np.array([]), np.array([])
    return np.array(results['onsets']), np.array(results['offsets'])



def vocalseg_from_array(
    signal: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect onset and offset times in a 1D signal using vocalseg.dynamic_threshold_segmentation()."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    return _run_dynamic_threshold(signal, sample_rate, **kwargs)

def _prepare_sound(
    audio_path: str | None = None,
    signal: np.ndarray | None = None,
    sample_rate: float | None = None,
    channel_idx: int = 0,
) -> voc.Sound:
    if audio_path is not None:
        sound = voc.Sound.read(audio_path)
        if sound.data.ndim > 1 and sound.data.shape[0] > 1:
            ch = min(channel_idx, sound.data.shape[0] - 1)
            sound = voc.Sound(data=sound.data[ch:ch + 1], samplerate=sound.samplerate)
        return sound

    if signal is None:
        raise ValueError("Either audio_path or signal must be provided")
    if sample_rate is None:
        raise ValueError("sample_rate required when using signal array")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    return voc.Sound(data=signal.reshape(1, -1), samplerate=int(sample_rate))


def _unpack_segment_result(result):
    """Handle both forked vocalpy (Segments, envelope) and upstream (Segments)."""
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _compute_meansquared_envelope(sound: voc.Sound, **kwargs) -> tuple:
    freq_cutoffs = kwargs.get("freq_cutoffs", (500, 10000))
    smooth_win = kwargs.get("smooth_win", 2)
    envelope = np.squeeze(
        voc.signal.audio.meansquared(sound, freq_cutoffs, smooth_win), axis=0
    )
    env_time = np.arange(len(envelope)) / sound.samplerate
    return env_time, envelope


def _compute_ava_envelope(data_1d: np.ndarray, samplerate: int, **kwargs) -> tuple:
    EPSILON = 1e-9
    nperseg = kwargs.get("nperseg", 1024)
    noverlap = kwargs.get("noverlap", 512)
    min_freq = kwargs.get("min_freq", 30e3)
    max_freq = kwargs.get("max_freq", 110e3)
    spect_min_val = kwargs.get("spect_min_val")
    spect_max_val = kwargs.get("spect_max_val")
    use_softmax_amp = kwargs.get("use_softmax_amp", True)
    temperature = kwargs.get("temperature", 0.5)
    smoothing_timescale = kwargs.get("smoothing_timescale", 0.007)

    scaled = (data_1d * 2**15).astype(np.int16)
    f, t, spect = stft(scaled, samplerate, nperseg=nperseg, noverlap=noverlap)
    i1 = np.searchsorted(f, min_freq)
    i2 = np.searchsorted(f, max_freq)
    spect = spect[i1:i2]
    spect = np.log(np.abs(spect) + EPSILON)

    if spect_min_val is None:
        spect_min_val = np.min(spect)
    if spect_max_val is None:
        spect_max_val = np.max(spect)

    spect -= spect_min_val
    denom = spect_max_val - spect_min_val
    if denom > 0:
        spect /= denom
    spect = np.clip(spect, 0.0, 1.0)

    dt = t[1] - t[0]
    if use_softmax_amp:
        temp = np.exp(spect / temperature)
        temp /= np.sum(temp, axis=0) + EPSILON
        amps = np.sum(np.multiply(spect, temp), axis=0)
    else:
        amps = np.sum(spect, axis=0)
    amps = gaussian_filter(amps, smoothing_timescale / dt)
    return t, amps


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


def vocalpy_segment(
    method: str = "meansquared",
    audio_path: str | None = None,
    signal: np.ndarray | None = None,
    sample_rate: float | None = None,
    channel_idx: int = 0,
    **kwargs,
) -> tuple:
    """Segment audio using vocalpy.

    Works with both upstream vocalpy (returns Segments) and forked version
    (returns (Segments, envelope) tuple).

    Returns:
        ((onsets, offsets), time_array, envelope) tuple.
        Returns empty arrays when no segments are found.
    """
    empty_result = (np.array([]), np.array([])), np.array([]), np.array([])
    sound = _prepare_sound(audio_path, signal, sample_rate, channel_idx)
    sr = sound.samplerate
    data_1d = np.squeeze(sound.data, axis=0)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if method == "meansquared":
        result = voc.segment.meansquared(sound, **kwargs)
        segments, envelope = _unpack_segment_result(result)
        if segments is None:
            if envelope is None:
                envelope_fallback = _compute_meansquared_envelope(sound, **kwargs)
                return (np.array([]), np.array([])), *envelope_fallback
            env_time = np.arange(len(envelope)) / sr
            return (np.array([]), np.array([])), env_time, envelope
        if envelope is None:
            env_time, envelope = _compute_meansquared_envelope(sound, **kwargs)
        else:
            env_time = np.arange(len(envelope)) / sr

    elif method == "ava":
        if "spect_min_val" not in kwargs or "spect_max_val" not in kwargs:
            smin, smax = _compute_spect_range(data_1d, sr, **kwargs)
            kwargs.setdefault("spect_min_val", smin)
            kwargs.setdefault("spect_max_val", smax)
        result = voc.segment.ava(sound, **kwargs)
        segments, envelope = _unpack_segment_result(result)
        if envelope is None:
            env_time, envelope = _compute_ava_envelope(data_1d, sr, **kwargs)
        else:
            nperseg = kwargs.get("nperseg", 1024)
            noverlap = kwargs.get("noverlap", nperseg // 2)
            hop = nperseg - noverlap
            env_time = (nperseg / 2 + np.arange(len(envelope)) * hop) / sr

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'meansquared' or 'ava'.")

    if segments.start_inds.size == 0:
        return (np.array([]), np.array([])), env_time, envelope

    onsets = segments.start_inds.astype(float) / sr
    offsets = segments.stop_inds.astype(float) / sr
    return (onsets, offsets), env_time, envelope



def apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_fraction: float,
    stationary: bool = True,
    prop_decrease: float = 1.0,
) -> np.ndarray:
    """Apply noise reduction to audio using noisereduce library.

    Reference: https://github.com/timsainb/noisereduce (Sainburg, 2019)

    Args:
        audio: Audio samples (1D array)
        sr: Sample rate
        n_fft: FFT window size
        hop_fraction: Hop length as fraction of n_fft
        stationary: If True, use stationary noise reduction (faster)
        prop_decrease: Proportion to reduce noise by (0.0-1.0), default 1.0

    Returns:
        Noise-reduced audio samples
    """
    
    hop_length = int(n_fft * hop_fraction)
    return nr.reduce_noise(
        y=audio, sr=sr, stationary=stationary,
        n_fft=n_fft, hop_length=hop_length, prop_decrease=prop_decrease
    )
