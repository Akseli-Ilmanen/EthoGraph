from __future__ import annotations

import subprocess
from pathlib import Path


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
    try:
        _, audio_sr = aio.load_audio(audio_path)
        return round(audio_sr, 3)
    except Exception:
        return None


def mp4_to_wav(mp4_path: str | Path, audio_sr: int) -> Path:
    """Convert MP4 to WAV using ffmpeg.

    Parameters
    ----------
    mp4_path : str or Path
        Input MP4 file.
    audio_sr : int
        Target sample rate.

    Returns
    -------
    Path
        Path to the created WAV file.
    """
    mp4_path = Path(mp4_path)
    wav_path = mp4_path.with_suffix(".wav")

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(mp4_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(audio_sr),
            "-ac",
            "1",
            "-y",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )

    if not wav_path.exists():
        raise RuntimeError(f"Failed to create WAV file: {wav_path}")

    return str(wav_path)
