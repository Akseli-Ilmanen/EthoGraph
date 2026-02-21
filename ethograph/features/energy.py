import numpy as np
import vocalpy as voc
from scipy.signal import butter, sosfiltfilt, stft


def _sosfilter(data, rate, cutoff, mode, order=4, axis=0):
    if mode == 'lp' and cutoff > rate / 2:
        return data
    if mode == 'bp' and cutoff[1] > rate / 2:
        mode, cutoff = 'hp', cutoff[0]
    sos = butter(order, cutoff, mode, fs=rate, output='sos')
    return sosfiltfilt(sos, data, axis=axis)


def _downsample(data, rate, new_rate):
    if new_rate >= rate:
        return data, rate
    step = int(round(rate / new_rate))
    return data[::step], rate / step


def _validate_envelope_params(cutoff: float, env_rate: float, margin: float = 0.8) -> None:
    nyquist = env_rate / 2
    if cutoff >= nyquist:
        raise ValueError(
            f"Cutoff ({cutoff} Hz) must be below Nyquist ({nyquist} Hz) for env_rate={env_rate} Hz"
        )
    if cutoff > nyquist * margin:
        import warnings
        warnings.warn(
            f"Cutoff ({cutoff} Hz) is close to Nyquist ({nyquist} Hz). "
            f"Recommend cutoff < {nyquist * margin:.0f} Hz for env_rate={env_rate} Hz",
            stacklevel=3,
        )

def lowpass_envelope(
    data: np.ndarray, rate: float, cutoff: float = 500.0, env_rate: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute amplitude envelope via lowpass filtering.

    Applies full-wave rectification, lowpass filters the result,
    and downsamples to ``env_rate``.

    Parameters
    ----------
    data : np.ndarray
        Single-channel audio signal.
    rate : float
        Sampling rate of ``data`` in Hz.
    cutoff : float
        Lowpass filter cutoff in Hz. Must be below ``env_rate / 2``.
    env_rate : float
        Target sampling rate for the output envelope in Hz.

    Returns
    -------
    env_time : np.ndarray
        Time axis for the envelope in seconds.
    envelope : np.ndarray
        Amplitude envelope at ``env_rate``.
    """
    _validate_envelope_params(cutoff, env_rate)
    filtered = _sosfilter(np.abs(data), rate, cutoff, mode='lp')
    envelope, actual_rate = _downsample(filtered, rate, env_rate)
    env_time = np.arange(len(envelope)) / actual_rate
    return env_time, envelope


def highpass_envelope(
    data: np.ndarray, rate: float, cutoff: float = 300.0, env_rate: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute amplitude envelope of highpass-filtered signal.

    Highpass filters the signal at ``cutoff``, applies full-wave
    rectification, lowpass filters at 500 Hz, and downsamples
    to ``env_rate``.

    Parameters
    ----------
    data : np.ndarray
        Single-channel audio signal.
    rate : float
        Sampling rate of ``data`` in Hz.
    cutoff : float
        Highpass filter cutoff in Hz.
    env_rate : float
        Target sampling rate for the output envelope in Hz.

    Returns
    -------
    env_time : np.ndarray
        Time axis for the envelope in seconds.
    envelope : np.ndarray
        Amplitude envelope at ``env_rate``.
    """
    _validate_envelope_params(cutoff, env_rate)
    filtered = _sosfilter(data, rate, cutoff, mode='hp')
    envelope = _sosfilter(np.abs(filtered), rate, cutoff=500.0, mode='lp')
    envelope, actual_rate = _downsample(envelope, rate, env_rate)
    env_time = np.arange(len(envelope)) / actual_rate
    return env_time, envelope


def bandpass_envelope(
    data: np.ndarray,
    rate: float,
    band: tuple = (300.0, 6000.0),
    env_rate: float = 1000.0,
    cutoff: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute amplitude envelope of bandpass-filtered signal.

    Bandpass filters the signal to ``band``, applies full-wave
    rectification, lowpass filters at ``cutoff``, and downsamples
    to ``env_rate``.

    Parameters
    ----------
    data : np.ndarray
        Single-channel audio signal.
    rate : float
        Sampling rate of ``data`` in Hz.
    band : tuple
        Bandpass frequency range as ``(low, high)`` in Hz.
    env_rate : float
        Target sampling rate for the output envelope in Hz.
    cutoff : float
        Lowpass cutoff for smoothing the rectified signal in Hz.
        Must be below ``env_rate / 2``.

    Returns
    -------
    env_time : np.ndarray
        Time axis for the envelope in seconds.
    envelope : np.ndarray
        Amplitude envelope at ``env_rate``.
    """
    _validate_envelope_params(cutoff, env_rate)
    filtered = _sosfilter(data, rate, band, mode='bp')
    envelope = _sosfilter(np.abs(filtered), rate, cutoff=cutoff, mode='lp')
    envelope, actual_rate = _downsample(envelope, rate, env_rate)
    env_time = np.arange(len(envelope)) / actual_rate
    return env_time, envelope


def _to_sound(audio_data_1d: np.ndarray, sample_rate: float) -> voc.Sound:
    return voc.Sound(
        data=np.asarray(audio_data_1d, dtype=np.float64).reshape(1, -1),
        samplerate=int(sample_rate),
    )


def env_meansquared(
    data: np.ndarray,
    rate: float,
    freq_cutoffs: tuple | None = None,
    smooth_win: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute meansquared energy envelope from a 1-D audio array.

    Returns (env_time, envelope) where env_time is seconds and
    envelope is the meansquared energy at the original sample rate.
    """
    sound = _to_sound(data, rate)
    kwargs = {}
    if freq_cutoffs is not None:
        kwargs["freq_cutoffs"] = freq_cutoffs
    if smooth_win != 2:
        kwargs["smooth_win"] = smooth_win
    envelope = np.squeeze(
        voc.signal.energy.meansquared(sound, **kwargs),
        axis=0,
    )
    env_time = np.arange(len(envelope)) / rate
    return env_time, envelope


def env_ava(
    data: np.ndarray,
    rate: float,
    nperseg: int = 1024,
    noverlap: int = 512,
    min_freq: float = 30000.0,
    max_freq: float = 110000.0,
    smoothing_timescale: float = 0.007,
    use_softmax_amp: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AVA amplitude envelope from a 1-D audio array.

    Returns (env_time, envelope).
    """
    energy_kwargs = {
        "nperseg": nperseg,
        "noverlap": noverlap,
        "min_freq": min_freq,
        "max_freq": max_freq,
        "smoothing_timescale": smoothing_timescale,
        "use_softmax_amp": use_softmax_amp,
    }
    _VALID_KEYS = {
        "nperseg", "noverlap", "min_freq", "max_freq",
        "spect_min_val", "spect_max_val", "use_softmax_amp",
        "temperature", "smoothing_timescale", "scale", "scale_val",
        "scale_dtype", "epsilon",
    }
    energy_kwargs.update({k: v for k, v in kwargs.items() if k in _VALID_KEYS})

    sound = _to_sound(data, rate)

    needs_spect_bounds = (
        energy_kwargs.get("spect_min_val") is None
        or energy_kwargs.get("spect_max_val") is None
    )
    if needs_spect_bounds:
        spect_min, spect_max = _ava_spect_bounds(
            data, int(rate), energy_kwargs,
        )
        energy_kwargs.setdefault("spect_min_val", spect_min)
        energy_kwargs.setdefault("spect_max_val", spect_max)

    amps, dt = voc.signal.energy.ava(sound, **energy_kwargs)
    t = np.arange(len(amps)) * dt
    return t, amps


def _ava_spect_bounds(
    data_1d: np.ndarray, samplerate: int, kwargs: dict,
) -> tuple[float, float]:
    scale = kwargs.get("scale", True)
    data = np.asarray(data_1d, dtype=np.float64)
    if scale:
        scale_val = kwargs.get("scale_val", 2**15)
        scale_dtype = kwargs.get("scale_dtype", np.int16)
        data = (data * scale_val).astype(scale_dtype)
    nperseg = kwargs.get("nperseg", 1024)
    noverlap = kwargs.get("noverlap", 512)
    min_freq = kwargs.get("min_freq", 30e3)
    max_freq = kwargs.get("max_freq", 110e3)
    epsilon = kwargs.get("epsilon", 1e-9)

    f, _, spect = stft(data, samplerate, nperseg=nperseg, noverlap=noverlap)
    i1 = np.searchsorted(f, min_freq)
    i2 = np.searchsorted(f, max_freq)
    log_spect = np.log(np.abs(spect[i1:i2]) + epsilon)
    return float(log_spect.min()), float(log_spect.max())


def get_lowpass_envelope(audio_path: str, audio_sr: int | None, fps: float):
    from pathlib import Path

    import audioio as aio
    from scipy.interpolate import interp1d

    from ethograph.utils.audio import mp4_to_wav

    gen_wav_path = None
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".mp4":
        gen_wav_path = mp4_to_wav(audio_path)
        audio_path = gen_wav_path

    data, sr = aio.load_audio(audio_path)
    if audio_sr is not None:
        sr = audio_sr

    if data.ndim > 1:
        data = data[:, 0]

    env_time, envelope = lowpass_envelope(data, sr)
    n_video_frames = int(len(data) / sr * fps)

    video_time = np.arange(n_video_frames) / fps
    interp_fn = interp1d(env_time, envelope, kind='linear', fill_value='extrapolate')
    envelope = interp_fn(video_time)

    return envelope, gen_wav_path
