import audioio as aio
from thunderhopper.filters import envelope
import numpy as np
from ethograph.utils.audio import mp4_to_wav
from pathlib import Path


def get_synced_envelope(audio_path, sr, fps):
    "Create audio envelope amplitude at same sampling rate as video fps."
    

    if isinstance(audio_path, Path):
        audio_path = str(audio_path)
        
    try: 
        audio_data, _ = aio.load_audio(audio_path) 
    except Exception:
        wav_path = mp4_to_wav(audio_path, sr)
        audio_data, _ = aio.load_audio(wav_path)
        print(f"Could not load audio from {Path(audio_path).name}, therefore created file: {Path(wav_path).name}")
    
    
    
    # Only get first channel if stereo
    if audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = audio_data[:, 0]
        
        
    # Nyquist frequency is fps/2, so cutoff should be less than that
    env = envelope(audio_data, rate=sr, cutoff=fps/4, env_rate=fps) 
    env = np.squeeze(env)
    
    
    return env, wav_path if 'wav_path' in locals() else None


