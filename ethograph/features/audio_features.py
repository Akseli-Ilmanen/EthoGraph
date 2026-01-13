import audioio as aio
from thunderhopper.filters import envelope
import numpy as np


def get_synced_envelope(audio_path, sr, fps):
    "Create audio envelope amplitude at same sampling rate as video fps."
    audio_data, _ = aio.load_audio(audio_path) 
    
    # Only get first channel if stereo
    if audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = audio_data[:, 0]
        
        
    # Nyquist frequency is fps/2, so cutoff should be less than that
    env = envelope(audio_data, rate=sr, cutoff=fps/4, env_rate=fps) 
    env = np.squeeze(env)
    
    return env
