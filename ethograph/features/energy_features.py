from pathlib import Path

import audioio as aio
import numpy as np
import vocalpy as voc

from ethograph.features.filter import envelope
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
from ethograph.utils.audio import mp4_to_wav



def compute_meansquared_envelope(
    audio_data_1d: np.ndarray,
    sample_rate: float,
    freq_cutoffs: tuple[float, float] = (500, 10000),
    smooth_win: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute meansquared energy envelope from a 1-D audio array.

    Returns (env_time, envelope) where env_time is seconds and
    envelope is the meansquared energy at the original sample rate.
    """
    sound = voc.Sound(
        data=np.asarray(audio_data_1d, dtype=np.float64).reshape(1, -1),
        samplerate=int(sample_rate),
    )
    env = np.squeeze(
        voc.signal.audio.meansquared(sound, freq_cutoffs, smooth_win),
        axis=0,
    )
    env_time = np.arange(len(env)) / sample_rate
    return env_time, env


def get_envelope(audio_path, audio_sr, env_rate):
    "Create audio envelope amplitude."


    if isinstance(audio_path, Path):
        audio_path = str(audio_path)

    try:
        audio_data, _ = aio.load_audio(audio_path)
    except Exception:
        wav_path = mp4_to_wav(audio_path, audio_sr)
        audio_data, _ = aio.load_audio(wav_path)
        print(f"Could not load audio from {Path(audio_path).name}, therefore created file: {Path(wav_path).name}")



    # Handle both mono and multi-channel audio
    if audio_data.ndim == 1:
        # Mono audio - add channel dimension
        audio_data = audio_data[:, np.newaxis]

    # Get envelope for each channel
    num_channels = audio_data.shape[1]
    env_list = []

    for ch in range(num_channels):
        # Nyquist frequency is env_rate/2, so cutoff should be less than that
        ch_env = envelope(audio_data[:, ch], rate=audio_sr, cutoff=env_rate/4, env_rate=env_rate)
        env_list.append(np.squeeze(ch_env))

    # Stack to create (T, Ch) shape
    env = np.stack(env_list, axis=1)

    return env, wav_path if 'wav_path' in locals() else None




def band_envelope(
    data: np.ndarray, fs: float, band: tuple[float, float], env_rate: float = 1000.0
) -> tuple[np.ndarray, float]:
    """Band-specific power envelope for extracellular data."""
    
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)
    envelope = np.abs(hilbert(filtered, axis=0))
    
    down = int(fs // env_rate)
    if down > 1:
        envelope = resample_poly(envelope, 1, down, axis=0)
    
    return envelope, fs / down