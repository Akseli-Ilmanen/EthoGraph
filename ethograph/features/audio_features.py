from pathlib import Path

import audioio as aio
import numpy as np
from ethograph.features.filter import envelope
from ethograph.utils.audio import mp4_to_wav


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


