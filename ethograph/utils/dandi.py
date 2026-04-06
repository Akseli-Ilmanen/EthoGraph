"""DANDI archive access: URL parsing, remote NWB opening, video discovery."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

try:
    import h5py
    import pynwb
except ImportError:
    h5py = None
    pynwb = None

try:
    import remfile
except ImportError:
    remfile = None

try:
    from dandi.dandiapi import DandiAPIClient
except ImportError:
    DandiAPIClient = None

try:
    import lindi as _lindi
    _LINDI_AVAILABLE = True
except Exception:
    _lindi = None
    _LINDI_AVAILABLE = False


def _require_nwb():
    if pynwb is None:
        raise ImportError(
            "h5py and pynwb are required for NWB support. "
            'Install them with: uv pip install "ethograph[nwb]"'
        )


def _require_dandi():
    if DandiAPIClient is None:
        raise ImportError(
            "dandi is required for DANDI support. "
            'Install with: uv pip install "ethograph[nwb]"'
        )


# ---------------------------------------------------------------------------
# DANDI URL parsing
# ---------------------------------------------------------------------------

_DANDI_HOSTS = frozenset({
    "api.dandiarchive.org",
    "dandiarchive.org",
    "lindi.neurosift.org",
    "neurosift.app",
})
_UUID_RE = re.compile(
    r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    re.IGNORECASE,
)
_DANDISET_RE = re.compile(r"/dandisets/(\d+)/")


def parse_dandi_url(url: str) -> dict | None:
    if not url:
        return None
    url = url.strip()
    parsed = urlparse(url)
    if not any(host in parsed.netloc for host in _DANDI_HOSTS):
        return None

    query_params = parse_qs(parsed.query)
    dandiset_id = (query_params.get("dandisetId") or [None])[0]

    embedded = (query_params.get("url") or [None])[0]
    if embedded:
        m = _UUID_RE.search(embedded)
        if m:
            asset_id = m.group(1)
            if not dandiset_id:
                dm = _DANDISET_RE.search(embedded)
                if dm:
                    dandiset_id = dm.group(1)
            return {"dandiset_id": dandiset_id, "asset_id": asset_id, "streaming_url": embedded}

    m = _UUID_RE.search(url)
    if m:
        asset_id = m.group(1)
        if not dandiset_id:
            dm = _DANDISET_RE.search(url)
            if dm:
                dandiset_id = dm.group(1)
        return {"dandiset_id": dandiset_id, "asset_id": asset_id, "streaming_url": url}

    return None


# ---------------------------------------------------------------------------
# NWB file openers
# ---------------------------------------------------------------------------

def open_nwb_local(path: str) -> tuple:
    """Open a local NWB file. Returns (nwb, io, h5_file, None)."""
    _require_nwb()
    h5_file = h5py.File(path, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
    return io.read(), io, h5_file, None


def open_nwb_dandi(dandiset_id: str, asset_id: str) -> tuple:
    """Open a DANDI NWB file, trying lindi index first for speed.

    Lindi provides a pre-built JSON index on neurosift.org, making metadata
    access nearly instant compared to streaming via remfile. Falls back to
    remfile if lindi is unavailable for this asset.

    Returns (nwb, io, h5_file, rf) where rf=None when lindi is used.
    """
    _require_nwb()
    _require_dandi()
    if _LINDI_AVAILABLE:
        lindi_url = (
            f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}"
            f"/assets/{asset_id}/nwb.lindi.json"
        )
        try:
            lindi_file = _lindi.LindiH5pyFile.from_lindi_file(lindi_url)
            io = pynwb.NWBHDF5IO(file=lindi_file, mode="r", load_namespaces=True)
            return io.read(), io, lindi_file, None
        except Exception:
            pass

    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id).get_asset(asset_id)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    rf = remfile.File(url)
    h5_file = h5py.File(rf, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
    return io.read(), io, h5_file, rf



# TODO: Remove, once nwb_video uses public AP:
# https://github.com/catalystneuro/nwb-video-widgets/issues/33

# ---------------------------------------------------------------------------
# DANDI video asset discovery
# ---------------------------------------------------------------------------

def find_video_assets(
    dandiset_id: str,
    nwb: Any,
    asset_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    _require_dandi()
    video_extensions = frozenset({".mp4", ".avi", ".mov", ".mkv"})

    for item in getattr(nwb, "acquisition", {}).values():
        external_files = getattr(item, "external_file", None)
        if external_files is None:
            continue
        files = external_files[:] if hasattr(external_files, "__getitem__") else [external_files]
        videos = [
            (Path(str(f)).stem, str(f))
            for f in files
            if Path(str(f)).suffix.lower() in video_extensions
        ]
        if videos:
            return videos

    subject = getattr(nwb, "subject", None)
    identifier = getattr(nwb, "identifier", None)
    search_terms = [
        t
        for t in [
            getattr(nwb, "session_id", None),
            identifier[:8] if identifier else None,
            getattr(subject, "subject_id", None) if subject else None,
            asset_id,
        ]
        if t
    ]

    if not search_terms:
        return []

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id)
        video_assets = []

        for asset in dandiset.get_assets():
            if Path(asset.path).suffix.lower() not in video_extensions:
                continue
            if not any(term in asset.path for term in search_terms):
                continue

            video_assets.append((Path(asset.path).stem, f"https://api.dandiarchive.org/api/assets/{asset.identifier}/download/"))

            if progress_callback:
                progress_callback(f"Found video: {Path(asset.path).name}")

        return video_assets


# ---------------------------------------------------------------------------
# Video clip download (ffmpeg)
# ---------------------------------------------------------------------------

def download_clip(
    source: str,
    t_start: float,
    t_stop: float,
    output_path: Path,
) -> Path | None:
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try copying slice of file, e.g. mp4 (session length) -> mp4 (trial length).
    cmd = [
        "ffmpeg", "-ss", str(t_start), "-i", source,
        "-t", str(t_stop - t_start),
        "-c", "copy",
        str(output_path), "-y"
    ]

    result = subprocess.run(cmd, capture_output=True)

    # Convert to new format (e.g. .mkv to .mp4) to trial length slice
    if result.returncode != 0:
        subprocess.run([
            "ffmpeg", "-ss", str(t_start), "-i", source,
            "-t", str(t_stop - t_start),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(output_path), "-y"
        ])
    return output_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
