"""Source-agnostic data providers for audio pipelines."""

from __future__ import annotations

from .modality import FileSource, ModalitySource
from .plots_spectrogram import SharedAudioCache


def build_audio_source(app_state) -> FileSource | None:
    """Build a FileSource for audio from the current app_state."""
    audio_path = getattr(app_state, 'audio_path', None)
    if not audio_path:
        return None
    _, channel_idx = app_state.get_audio_source()
    loader = SharedAudioCache.get_loader(audio_path)
    if loader is None:
        return None
    return FileSource("audio", loader, channel=channel_idx)
