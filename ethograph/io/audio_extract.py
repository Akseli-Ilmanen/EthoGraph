"""Sidecar WAV extraction for video containers that carry an audio track.

Audio in ethograph is read through ``audioio.AudioLoader`` — random access,
sample-exact, one window at a time (waveform, spectrogram, playback clock, CP
detection).  Its seekable backends are libsndfile/wavefile, which decode no
video container at all: AAC-in-MP4 is not a libsndfile format, and the
sequential fallbacks (audioread → ffmpeg) cannot answer "give me samples
s0:s1" without re-decoding from the start.  AAC also carries encoder priming
samples, so even a sequential decode is not sample-aligned with the video
timeline for free.

So a container's track is decoded **once** into a cached PCM WAV and every
audio consumer opens that file instead.  Same lifecycle as the video proxies
(:mod:`ethograph.io.video_proxy`): a deterministic key from source identity, a
central cache under ``~/.ethograph/cache/audio_tracks``, generated on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ethograph.io.validation import VIDEO_EXTENSIONS
from ethograph.utils.paths import cache_dir, media_cache_key

logger = logging.getLogger(__name__)

#: Bump when the decode recipe changes so stale extracts are regenerated.
_EXTRACT_RECIPE_VERSION = 1


def audio_cache_dir() -> Path:
    """Central directory holding every extracted audio track.

    One shared location, like the proxy cache: the source video may live on
    read-only or network media, and a flat folder keyed by source identity
    never collides.
    """
    return cache_dir("audio_tracks")


def is_video_container(path: str | Path) -> bool:
    """Whether *path* names a video container rather than an audio file."""
    return Path(str(path)).suffix.lower() in VIDEO_EXTENSIONS


def has_embedded_audio(path: str | Path) -> bool:
    """True when the video container holds at least one audio stream."""
    try:
        import av

        with av.open(str(path)) as container:
            return any(s.type == "audio" for s in container.streams)
    except Exception:  # noqa: BLE001 - unreadable container = no usable audio
        return False


def extracted_audio_path(video_path: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Return the deterministic extract path for *video_path*."""
    cache_dir = Path(cache_dir) if cache_dir is not None else audio_cache_dir()
    return cache_dir / f"{media_cache_key(video_path, _EXTRACT_RECIPE_VERSION)}.wav"


def extract_audio_wav(video_path: str | Path, out_path: str | Path) -> Path:
    """Decode the first audio track of *video_path* into the WAV *out_path*.

    Written through a ``.tmp`` sibling so an interrupted decode never leaves a
    truncated file that the cache would then treat as valid.
    """
    import av
    import numpy as np
    import soundfile as sf

    video_path, out_path = Path(video_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".wav.tmp")

    subtypes = {np.dtype(np.int16): "PCM_16", np.dtype(np.int32): "PCM_32"}
    writer = None
    try:
        with av.open(str(video_path)) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise RuntimeError(f"No audio track in {video_path.name}.")
            stream = streams[0]
            rate = int(stream.rate)
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if not frame.format.is_planar:
                    # Packed formats decode to (1, samples*channels) interleaved.
                    arr = arr.reshape(-1, len(frame.layout.channels)).T
                block = arr.T  # (samples, channels)
                if block.dtype not in subtypes and block.dtype != np.float32:
                    block = block.astype(np.float32)
                if writer is None:
                    writer = sf.SoundFile(
                        str(tmp_path),
                        mode="w",
                        samplerate=rate,
                        channels=block.shape[1],
                        # The writer opens the ``.tmp`` sibling, so the format
                        # cannot be inferred from the extension.
                        format="WAV",
                        subtype=subtypes.get(block.dtype, "FLOAT"),
                    )
                writer.write(block)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Audio track in {video_path.name} holds no samples.")
    tmp_path.replace(out_path)
    return out_path


def ensure_extracted_audio(video_path: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Return the cached WAV for *video_path*'s audio track, decoding it if missing."""
    out_path = extracted_audio_path(video_path, cache_dir)
    if out_path.exists():
        return out_path
    logger.info("Extracting audio track from %s → %s", Path(video_path).name, out_path)
    return extract_audio_wav(video_path, out_path)


def resolve_audio_path(audio_path: str | Path, cache_dir: str | Path | None = None) -> str:
    """Return a path ``audioio`` can open for the audio at *audio_path*.

    Audio files pass through untouched; a video container is decoded (once)
    into the extract cache and the WAV is returned.  Every reader of an audio
    file goes through here, so an alignment that points a mic stream straight
    at an ``.mp4`` works exactly like one pointing at a ``.wav``.

    Raises ``RuntimeError`` when the container carries no decodable track —
    callers already treat an unreadable audio source that way.
    """
    if not is_video_container(audio_path):
        return str(audio_path)
    return str(ensure_extracted_audio(audio_path, cache_dir))


def cache_size(cache_dir: str | Path | None = None) -> int:
    """Total bytes of extracted audio (and any temp files) in the cache."""
    cache_dir = Path(cache_dir) if cache_dir is not None else audio_cache_dir()
    if not cache_dir.exists():
        return 0
    return sum(f.stat().st_size for f in cache_dir.glob("*") if f.is_file())


def clear_cache(cache_dir: str | Path | None = None, keep: set[str] | None = None) -> int:
    """Delete extracted audio files; return bytes freed.

    Files whose absolute path is in *keep* are preserved (e.g. the extract of
    the session currently loaded — deleting that one breaks playback).
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else audio_cache_dir()
    if not cache_dir.exists():
        return 0
    keep = {str(Path(p)) for p in (keep or set())}
    freed = 0
    for f in cache_dir.glob("*"):
        if not f.is_file() or str(f) in keep:
            continue
        size = f.stat().st_size
        f.unlink()
        freed += size
    return freed
