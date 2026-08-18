"""Embedded-container audio (AAC in MP4) resolving to a cached WAV."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ethograph.io import audio_extract

av = pytest.importorskip("av")
sf = pytest.importorskip("soundfile")


def _write_aac_mp4(path: Path, rate: int = 24000, duration: float = 1.0) -> Path:
    """Write a synthetic stereo AAC-in-MP4 tone."""
    t = np.arange(int(rate * duration)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.ascontiguousarray(np.stack([tone, tone]))

    with av.open(str(path), "w") as container:
        stream = container.add_stream("aac", rate=rate)
        stream.layout = "stereo"
        for pos in range(0, stereo.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(stereo[:, pos : pos + 1024]), format="fltp", layout="stereo"
            )
            frame.sample_rate = rate
            frame.pts = pos
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def test_audio_file_passes_through(tmp_path):
    wav = tmp_path / "mic.wav"
    sf.write(wav, np.zeros(100, dtype=np.float32), 16000)
    assert audio_extract.resolve_audio_path(wav) == str(wav)


def test_mp4_resolves_to_readable_wav(tmp_path):
    mp4 = _write_aac_mp4(tmp_path / "clip.mp4")
    resolved = Path(audio_extract.resolve_audio_path(mp4, cache_dir=tmp_path / "cache"))

    assert resolved.suffix == ".wav"
    data, rate = sf.read(resolved)
    assert rate == 24000
    assert data.shape[1] == 2
    # AAC carries priming/padding samples, so the extract is only approximately
    # the source duration — never assert an exact sample count.
    assert 0.99 < len(data) / rate < 1.05


def test_extract_is_cached_not_redecoded(tmp_path, monkeypatch):
    mp4 = _write_aac_mp4(tmp_path / "clip.mp4")
    cache = tmp_path / "cache"
    first = audio_extract.resolve_audio_path(mp4, cache_dir=cache)

    def _fail(*args, **kwargs):
        raise AssertionError("cached extract was decoded again")

    monkeypatch.setattr(audio_extract, "extract_audio_wav", _fail)
    assert audio_extract.resolve_audio_path(mp4, cache_dir=cache) == first


def test_cache_key_follows_source_identity(tmp_path):
    mp4 = _write_aac_mp4(tmp_path / "clip.mp4")
    before = audio_extract.extracted_audio_path(mp4, tmp_path)
    _write_aac_mp4(mp4, duration=2.0)  # re-recorded source, same name
    assert audio_extract.extracted_audio_path(mp4, tmp_path) != before


def test_container_without_audio_raises(tmp_path):
    silent = tmp_path / "video.mp4"
    with av.open(str(silent), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    assert not audio_extract.has_embedded_audio(silent)
    with pytest.raises(RuntimeError, match="No audio track"):
        audio_extract.resolve_audio_path(silent, cache_dir=tmp_path / "cache")


def test_clear_cache_frees_files(tmp_path):
    mp4 = _write_aac_mp4(tmp_path / "clip.mp4")
    cache = tmp_path / "cache"
    resolved = audio_extract.resolve_audio_path(mp4, cache_dir=cache)

    assert audio_extract.cache_size(cache) > 0
    assert audio_extract.clear_cache(cache, keep={resolved}) == 0
    assert audio_extract.clear_cache(cache) > 0
    assert not Path(resolved).exists()
