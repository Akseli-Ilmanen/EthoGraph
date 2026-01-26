"""Audio changepoint detection using vocalseg dynamic threshold segmentation.

Uses vocalseg library for detecting vocal onset/offset candidates in audio files.
Reference: https://github.com/timsainb/vocalization-segmentation
"""

import numpy as np
from typing import Tuple, Optional
import hashlib


_audio_changepoint_cache: dict[str, dict] = {}


def get_file_hash(audio_path: str) -> str:
    """Get a simple hash for cache key based on path and modification time."""
    import os
    try:
        stat = os.stat(audio_path)
        return hashlib.md5(f"{audio_path}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()
    except OSError:
        return hashlib.md5(audio_path.encode()).hexdigest()


def detect_audio_changepoints(
    audio_path: str,
    hop_length_ms: float = 5.0,
    min_level_db: float = -70.0,
    min_syllable_length_s: float = 0.02,
    silence_threshold: float = 0.1,
    ref_level_db: int = 20,
    spectral_range: Optional[Tuple[float, float]] = None,
    n_fft: int = 1024,
    use_cache: bool = True,
    channel_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect onset and offset times in audio using vocalseg.
    Args:
        audio_path: Path to audio file
        hop_length_ms: Hop length in milliseconds for spectrogram
        min_level_db: Minimum dB level threshold for segmentation
        min_syllable_length_s: Minimum syllable duration in seconds
        silence_threshold: Envelope threshold for offset detection (0-1).
            Lower values detect offsets later (captures more tail).
        ref_level_db: Reference level dB of audio (default: 20)
        spectral_range: Spectral range (min_hz, max_hz) to care about, or None for full range
        n_fft: FFT window size
        use_cache: Whether to use cached results
        channel_idx: Audio channel to use (for multi-channel files)

    Returns:
        Tuple of (onset_times, offset_times) in seconds
    """
    from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
    import audioio as aio

    cache_key = get_file_hash(audio_path)
    spectral_range_str = f"{spectral_range}" if spectral_range else "None"
    param_key = f"{hop_length_ms}:{min_level_db}:{min_syllable_length_s}:{silence_threshold}:{ref_level_db}:{spectral_range_str}:{n_fft}:{channel_idx}"
    full_key = f"{cache_key}:{param_key}"

    if use_cache and full_key in _audio_changepoint_cache:
        cached = _audio_changepoint_cache[full_key]
        return cached['onsets'], cached['offsets']

    audio, sr = aio.load_audio(audio_path)

    if audio.ndim > 1:
        ch = min(channel_idx, audio.shape[1] - 1)
        audio = audio[:, ch]

    results = dynamic_threshold_segmentation(
        vocalization=audio,
        rate=sr,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        min_level_db=min_level_db,
        min_syllable_length_s=min_syllable_length_s,
        silence_threshold=silence_threshold,
        ref_level_db=ref_level_db,
        spectral_range=spectral_range,
    )

    # Shift times left by half hop to align with spectrogram center-of-frame convention
    hop_correction = 0
    onset_times = np.array(results['onsets']) - hop_correction
    offset_times = np.array(results['offsets']) - hop_correction

    if use_cache:
        _audio_changepoint_cache[full_key] = {
            'onsets': onset_times,
            'offsets': offset_times,
        }

    return onset_times, offset_times


def clear_audio_changepoint_cache():
    """Clear the audio changepoint cache."""
    _audio_changepoint_cache.clear()


def get_all_audio_changepoints(
    audio_path: str,
    **kwargs
) -> np.ndarray:
    """Get combined onset and offset times as single sorted array.

    Useful for snap-to-changepoint functionality.

    Args:
        audio_path: Path to audio file
        **kwargs: Arguments passed to detect_audio_changepoints

    Returns:
        Sorted array of all changepoint times (onsets and offsets combined)
    """
    onsets, offsets = detect_audio_changepoints(audio_path, **kwargs)
    all_changepoints = np.concatenate([onsets, offsets])
    return np.unique(np.sort(all_changepoints))


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
    try:
        import noisereduce as nr
        hop_length = int(n_fft * hop_fraction)
        return nr.reduce_noise(
            y=audio, sr=sr, stationary=stationary,
            n_fft=n_fft, hop_length=hop_length, prop_decrease=prop_decrease
        )
    except ImportError:
        print("noisereduce not installed. Install with: pip install noisereduce")
        return audio
