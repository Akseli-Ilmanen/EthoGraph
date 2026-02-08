import subprocess
from pathlib import Path


def mp4_to_wav(mp4_path: str | Path, audio_sr: int) -> Path:
    """Convert MP4 to WAV using ffmpeg.
    
    Args:
        mp4_path: Input MP4 file
        audio_sr: Sample rate
        
    Returns:
        Path to created WAV file
    """
    mp4_path = Path(mp4_path)
    wav_path = mp4_path.with_suffix('.wav')

    
    subprocess.run([
        'ffmpeg', '-i', str(mp4_path),
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(audio_sr), '-ac', '1',
        '-y', str(wav_path)
    ], check=True, capture_output=True)
    
    if not wav_path.exists():
        raise RuntimeError(f"Failed to create WAV file: {wav_path}")
    
    return str(wav_path)