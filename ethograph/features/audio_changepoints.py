"""Spectral changepoint detection using vocalseg dynamic threshold segmentation.

Uses vocalseg library for detecting vocal onset/offset candidates in audio files.
Reference: https://github.com/timsainb/vocalization-segmentation
"""

from typing import Tuple

import audioio as aio
import noisereduce as nr
import numpy as np
import vocalpy as voc
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation



def _run_dynamic_threshold(
    signal: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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


def vocalseg_from_path(
    audio_path: str,
    channel_idx: int = 0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect onset and offset times using vocalseg.dynamic_threshold_segmentation().

    All segmentation parameters are forwarded via **kwargs to let vocalseg
    handle its own defaults.
    """

    audio, sr = aio.load_audio(audio_path)

    if audio.ndim > 1:
        ch = min(channel_idx, audio.shape[1] - 1)
        audio = audio[:, ch]

    onsets, offsets = _run_dynamic_threshold(audio, sr, **kwargs)


    return onsets, offsets


def vocalseg_from_array(
    signal: np.ndarray,
    sample_rate: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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


def vocalpy_segment(
    method: str = "meansquared",
    audio_path: str | None = None,
    signal: np.ndarray | None = None,
    sample_rate: float | None = None,
    channel_idx: int = 0,
    **kwargs,
) -> tuple:
    """Segment audio using vocalpy.

    Returns:
        ((onsets, offsets), time_array, envelope) tuple.
    """
    sound = _prepare_sound(audio_path, signal, sample_rate, channel_idx)
    sr = sound.samplerate

    if method == "meansquared":
        segments, envelope = voc.segment.meansquared(sound, scale=False, **kwargs)
    elif method == "ava":
        segments, envelope = voc.segment.ava(sound, scale=False, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'meansquared' or 'ava'.")
    
    if segments is None:
        return None,  None, None

    onsets = segments.start_inds.astype(float) / sr
    offsets = segments.stop_inds.astype(float) / sr
    time_array = np.arange(len(envelope)) / sound.samplerate

    return (onsets, offsets), time_array, envelope



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
