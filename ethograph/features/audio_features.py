import audioio as aio
from thunderhopper.filters import envelope
import numpy as np
from ethograph.utils.audio import mp4_to_wav
from pathlib import Path


def get_synced_envelope(audio_path, audio_sr, fps):
    "Create audio envelope amplitude at same sampling rate as video fps."
    

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
        # Nyquist frequency is fps/2, so cutoff should be less than that
        ch_env = envelope(audio_data[:, ch], rate=audio_sr, cutoff=fps/4, env_rate=fps)
        env_list.append(np.squeeze(ch_env))
    
    # Stack to create (T, Ch) shape
    env = np.stack(env_list, axis=1)
    
    return env, wav_path if 'wav_path' in locals() else None


