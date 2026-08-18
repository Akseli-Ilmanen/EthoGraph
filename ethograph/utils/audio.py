from __future__ import annotations


def get_audio_sr(audio_path: str) -> int | None:
    """Read sample rate from audio file using audioio, rounded to 3 decimals.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    int or None
        Sample rate, or ``None`` if the file cannot be read.
    """
    try:
        import audioio as aio
    except ImportError as e:
        raise ImportError('audioio is required. Install it with: uv pip install "ethograph[audio]"') from e
    from ethograph.io.audio_extract import resolve_audio_path

    try:
        _, audio_sr = aio.load_audio(resolve_audio_path(audio_path))
        return round(audio_sr, 3)
    except Exception:
        return None
