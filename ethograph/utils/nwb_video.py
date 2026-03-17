"""Shared utilities for NWB video widgets. - Adapted from https://github.com/catalystneuro/nwb-video-widgets"""

from __future__ import annotations

import struct
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Optional

import av
import numpy as np
from pynwb import NWBFile
from pynwb.image import ImageSeries

from ethograph.utils.nwb import open_nwb_dandi


if TYPE_CHECKING:
    from dandi.dandiapi import RemoteAsset


# Codecs natively supported by all major browsers via HTML5 <video>
BROWSER_COMPATIBLE_CODECS = {"h264", "H264", "avc1", "vp8", "vp9", "VP8", "VP9", "vp09", "av01", "AV01"}

_HEADER_READ_SIZE = 32 * 1024  # 32 KB is enough for codec detection



def _detect_avi_codec(data: bytes) -> str | None:
    """Extract the video codec FourCC from AVI (RIFF) header bytes.

    Walks the RIFF chunk structure to find the ``strh`` chunk with
    ``fccType == b'vids'`` and returns the ``fccHandler`` field.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        return None

    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        if len(data) < pos + 8:
            break
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]

        if chunk_id == b"LIST":
            pos += 12  # enter LIST, skip list type
            continue

        if chunk_id == b"strh" and chunk_size >= 8:
            fcc_type = data[pos + 8 : pos + 12]
            fcc_handler = data[pos + 12 : pos + 16]
            if fcc_type == b"vids":
                codec = fcc_handler.decode("ascii", errors="replace").strip("\x00")
                return codec if codec else None

        pos += 8 + chunk_size + (chunk_size % 2)

    return None


def _find_mp4_box(data: bytes, start: int, end: int, target: bytes) -> tuple[int, int] | None:
    """Find an ISO BMFF box by type within a byte range.

    Returns ``(payload_start, payload_end)`` or ``None``.
    """
    pos = start
    while pos + 8 <= end:
        box_size = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]

        if box_size == 1 and pos + 16 <= end:
            box_size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_start = pos + 16
        elif box_size < 8:
            break
        else:
            payload_start = pos + 8

        if box_type == target:
            return payload_start, min(pos + box_size, end)

        pos += box_size

    return None


def _detect_mp4_codec(data: bytes) -> str | None:
    """Extract the video codec FourCC from MP4/MOV header bytes.

    Navigates ``moov > trak > mdia > minf > stbl > stsd`` and reads the
    codec identifier from the first sample entry.  If the ``moov`` box
    is not found at a top-level box boundary (e.g. when parsing the tail
    of a file), falls back to scanning for the ``moov`` signature.
    """
    end = len(data)
    inner_path = [b"trak", b"mdia", b"minf", b"stbl", b"stsd"]

    # Try structured traversal first
    moov = _find_mp4_box(data, 0, end, b"moov")

    # Fallback: scan for moov signature (useful for tail-of-file reads)
    if moov is None:
        search_pos = 0
        while True:
            found = data.find(b"moov", search_pos)
            if found == -1 or found < 4:
                return None
            box_size = struct.unpack_from(">I", data, found - 4)[0]
            if 16 < box_size <= end - (found - 4):
                moov = (found + 4, found - 4 + box_size)
                break
            search_pos = found + 4

    start, end = moov
    for box_type in inner_path:
        result = _find_mp4_box(data, start, end, box_type)
        if result is None:
            return None
        start, end = result

    # stsd FullBox: version(1) + flags(3) + entry_count(4) = 8 bytes
    # then SampleEntry: size(4) + codec_fourcc(4)
    entry_offset = start + 8
    if entry_offset + 8 > end:
        return None

    codec_fourcc = data[entry_offset + 4 : entry_offset + 8]
    return codec_fourcc.decode("ascii", errors="replace").strip("\x00") or None


def detect_video_codec(video_path: Path) -> str | None:
    """Detect the video codec of a file by reading its header bytes.

    Supports AVI (RIFF) and MP4/MOV (ISO BMFF) containers. Returns the
    codec identifier string (e.g. ``"avc1"``, ``"MJPG"``, ``"mp4v"``)
    or ``None`` if the format is not recognized.

    For MP4 files where the ``moov`` box is at the end of the file (common
    when not encoded with ``faststart``), the tail of the file is also read.

    Parameters
    ----------
    video_path : Path
        Path to a video file.

    Returns
    -------
    str or None
        Codec identifier, or None if unrecognized.
    """
    file_size = video_path.stat().st_size
    with open(video_path, "rb") as f:
        data = f.read(_HEADER_READ_SIZE)

    if len(data) < 12:
        return None

    # AVI: RIFF....AVI
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return _detect_avi_codec(data)

    # MP4/MOV: ftyp box at start
    if data[4:8] == b"ftyp" or data[4:8] == b"moov":
        codec = _detect_mp4_codec(data)
        if codec is not None:
            return codec

        # moov may be at the end of the file (no faststart)
        if file_size > _HEADER_READ_SIZE:
            tail_size = min(file_size, _HEADER_READ_SIZE * 8)  # up to 256KB
            with open(video_path, "rb") as f:
                f.seek(file_size - tail_size)
                tail_data = f.read(tail_size)
            return _detect_mp4_codec(tail_data)

    return None


def validate_video_codec(video_path: Path) -> None:
    """Raise ``ValueError`` if the video uses a non-browser-compatible codec.

    Parameters
    ----------
    video_path : Path
        Path to a video file.

    Raises
    ------
    ValueError
        If the detected codec is not in ``BROWSER_COMPATIBLE_CODECS``.
    """
    codec = detect_video_codec(video_path)
    if codec is None:
        return  # unrecognized format, don't block

    if codec not in BROWSER_COMPATIBLE_CODECS:
        stem = video_path.stem
        raise ValueError(
            f"Video '{video_path.name}' uses the '{codec}' codec which cannot be played in the browser. "
            f"Re-encode with: ffmpeg -i {video_path.name} -c:v libx264 -crf 18 -pix_fmt yuv420p {stem}_h264.mp4"
        )


def discover_video_series(nwbfile: NWBFile) -> dict[str, ImageSeries]:
    """Discover all ImageSeries with external video files in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to search for video series

    Returns
    -------
    dict[str, ImageSeries]
        Mapping of series names to ImageSeries objects that have external_file
    """
    video_series = {}
    for name, obj in nwbfile.acquisition.items():
        if isinstance(obj, ImageSeries) and obj.external_file is not None:
            video_series[name] = obj
    return video_series


def get_video_timestamps(nwbfile: NWBFile) -> dict[str, list[float]]:
    """Extract video timestamps from all ImageSeries in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing video ImageSeries in acquisition

    Returns
    -------
    dict[str, list[float]]
        Mapping of video names to timestamp arrays
    """
    video_series = discover_video_series(nwbfile)
    timestamps = {}

    for name, series in video_series.items():
        if series.timestamps is not None:
            timestamps[name] = [float(t) for t in series.timestamps[:]]
        elif series.starting_time is not None:
            timestamps[name] = [float(series.starting_time)]
        else:
            timestamps[name] = [0.0]

    return timestamps


def get_video_info(nwbfile: NWBFile) -> dict[str, dict]:
    """Extract video time range information from all ImageSeries in an NWB file.

    Uses indexed access (timestamps[0], timestamps[-1]) instead of loading
    the full timestamps array, which is important for DANDI streaming where
    each slice triggers HTTP range requests.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing video ImageSeries in acquisition

    Returns
    -------
    dict[str, dict]
        Mapping of video names to info dictionaries with keys:
        - start: float, start time in seconds
        - end: float, end time in seconds
    """
    video_series = discover_video_series(nwbfile)
    info = {}

    for name, series in video_series.items():
        if series.timestamps is not None and len(series.timestamps) > 0:
            start = float(series.timestamps[0])
            end = float(series.timestamps[-1])
        elif series.starting_time is not None:
            start = float(series.starting_time)
            # Without timestamps, we can't determine end time accurately
            # Use starting_time as both start and end
            end = start
        else:
            start = 0.0
            end = 0.0

        info[name] = {
            "start": start,
            "end": end,
        }

    return info



def discover_pose_estimation_cameras(nwbfile: NWBFile) -> dict:
    """Discover all PoseEstimation containers in an NWB file.

    Searches all objects in the file regardless of where they are stored,
    so PoseEstimation data in any processing module (e.g. 'pose_estimation',
    'behavior') is found.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to search for pose estimation data

    Returns
    -------
    dict
        Mapping of camera names to PoseEstimation objects.
    """
    cameras = {}
    for obj in nwbfile.objects.values():
        if obj.neurodata_type == "PoseEstimation":
            assert obj.name not in cameras, f"Duplicate PoseEstimation name found: {obj.name}"
            cameras[obj.name] = obj
    return cameras


def get_camera_to_video_mapping(nwbfile: NWBFile) -> dict[str, str]:
    """Auto-map pose estimation camera names to video series names.

    Uses the naming convention: camera name prefixed with "Video"
    - 'LeftCamera' -> 'VideoLeftCamera'
    - 'BodyCamera' -> 'VideoBodyCamera'

    Only returns mappings where both the camera and corresponding video exist.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing pose estimation and video data

    Returns
    -------
    dict[str, str]
        Mapping from camera names to video series names
    """
    cameras = discover_pose_estimation_cameras(nwbfile)
    video_series = discover_video_series(nwbfile)

    mapping = {}
    for camera_name in cameras:
        video_name = f"Video{camera_name}"
        if video_name in video_series:
            mapping[camera_name] = video_name

    return mapping


def get_pose_estimation_info(nwbfile: NWBFile) -> dict[str, dict]:
    """Extract pose estimation info for all cameras in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing pose estimation in processing['pose_estimation']

    Returns
    -------
    dict[str, dict]
        Mapping of camera names to info dictionaries with keys:
        - start: float, start time in seconds
        - end: float, end time in seconds
        - keypoints: list[str], names of keypoints
    """
    cameras = discover_pose_estimation_cameras(nwbfile)
    info = {}

    for camera_name, pose_estimation in cameras.items():
        # Get keypoint names (remove PoseEstimationSeries suffix)
        keypoint_names = [
            name.replace("PoseEstimationSeries", "") for name in pose_estimation.pose_estimation_series.keys()
        ]

        # Get start/end times from the first pose estimation series using indexed
        # access to avoid loading the full timestamps array into memory. This is
        # important for DANDI streaming where each slice triggers HTTP range requests.
        first_series = next(iter(pose_estimation.pose_estimation_series.values()), None)
        if first_series is not None and first_series.timestamps is not None:
            start = float(first_series.timestamps[0])
            end = float(first_series.timestamps[-1])
        else:
            start = 0.0
            end = 0.0

        info[camera_name] = {
            "start": start,
            "end": end,
            "keypoints": keypoint_names,
        }

    return info


def probe_dandi_video_metadata(url: str) -> dict:
    """Probe video metadata from a remote URL using HTTP HEAD and PyAV.

    Returns a dict with available keys: size_bytes, width, height, fps,
    duration_s, frame_count.  If the HTTP HEAD fails (e.g. S3 redirects),
    the size is simply omitted.
    """
    metadata: dict = {}

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                metadata["size_bytes"] = int(content_length)
    except (urllib.error.URLError, OSError, TimeoutError):
        pass

    container = av.open(url)
    stream = container.streams.video[0]
    metadata["width"] = stream.codec_context.width
    metadata["height"] = stream.codec_context.height
    if stream.average_rate:
        metadata["fps"] = float(stream.average_rate)
    if stream.duration is not None and stream.time_base is not None:
        metadata["duration_s"] = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        metadata["duration_s"] = container.duration / 1_000_000
    if stream.frames:
        metadata["frame_count"] = stream.frames
    container.close()

    return metadata


def stream_video_in_browser(url: str, title: str = "DANDI Video") -> None:
    """Open a DANDI video URL in the default browser for streaming playback."""
    html = (
        "<!DOCTYPE html>\n"
        f"<html><head><title>{title}</title></head>"
        '<body style="margin:0;background:#000">\n'
        '<video controls autoplay style="width:100%;height:100vh">\n'
        f'<source src="{url}" type="video/mp4">\n'
        "</video>\n"
        "</body></html>"
    )
    path = Path.home() / ".ethograph" / "dandi_video.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    webbrowser.open(path.as_uri())


class NWBDANDIPoseEstimationWidget():
    """Video player with pose estimation overlay for DANDI-hosted NWB files.

    Overlays DeepLabCut keypoints on streaming video with support for
    camera selection via a settings panel.

    This widget discovers PoseEstimation containers anywhere in the NWB file
    and resolves video paths to S3 URLs via the DANDI API. An interactive
    settings panel allows users to select which camera to display.

    Supports two common NWB patterns:
    1. Single file: both videos and pose estimation in same NWB file
    2. Split files: videos in raw NWB file, pose estimation in processed file

    Parameters
    ----------
    asset : RemoteAsset
        DANDI asset object for the processed NWB file containing pose estimation.
        The dandiset_id and asset path are extracted from this object.
    nwbfile : pynwb.NWBFile, optional
        Pre-loaded NWB file containing pose estimation. If not provided, the widget
        will load the NWB file via streaming from `asset`.
    video_asset : RemoteAsset, optional
        DANDI asset object for the raw NWB file containing videos. If not provided,
        videos are assumed to be accessible relative to `asset`.
    video_nwbfile : pynwb.NWBFile, optional
        Pre-loaded NWB file containing video ImageSeries. If not provided but
        `video_asset` is provided, the widget will extract video URLs from `video_asset`.
        If neither is provided, videos are assumed to be in `nwbfile`.
    keypoint_colors : str or dict, default 'tab10'
        Either a matplotlib colormap name (e.g., 'tab10', 'Set1', 'Paired') for
        automatic color assignment, or a dict mapping keypoint names to hex colors
        (e.g., {'LeftPaw': '#FF0000', 'RightPaw': '#00FF00'}).
    default_camera : str, optional
        Camera to display initially. Falls back to first available if not found.

    Example
    -------
    Single file (videos + pose in same file):

    >>> from dandi.dandiapi import DandiAPIClient
    >>> client = DandiAPIClient()
    >>> dandiset = client.get_dandiset("000409", "draft")
    >>> asset = dandiset.get_asset_by_path("sub-.../sub-..._combined.nwb")
    >>> widget = NWBDANDIPoseEstimationWidget(asset=asset)
    >>> display(widget)

    Split files (videos in raw, pose in processed):

    >>> raw_asset = dandiset.get_asset_by_path("sub-.../sub-..._desc-raw.nwb")
    >>> processed_asset = dandiset.get_asset_by_path("sub-.../sub-..._desc-processed.nwb")
    >>> widget = NWBDANDIPoseEstimationWidget(
    ...     asset=processed_asset,
    ...     video_asset=raw_asset,
    ... )
    >>> display(widget)

    With pre-loaded NWB files (avoids re-loading):

    >>> widget = NWBDANDIPoseEstimationWidget(
    ...     asset=processed_asset,
    ...     nwbfile=nwbfile_processed,
    ...     video_asset=raw_asset,
    ...     video_nwbfile=nwbfile_raw,
    ... )

    Raises
    ------
    ValueError
        If no cameras have both pose data and video.
    """

    def __init__(
        self,
        processed_asset: RemoteAsset,
        nwbfile: Optional[NWBFile] = None,
        raw_asset: Optional[RemoteAsset] = None,
        video_nwbfile: Optional[NWBFile] = None,
    ):
        # Load NWB file if not provided (for pose estimation)
        if nwbfile is None:
            nwbfile = self._load_nwbfile_from_dandi(processed_asset)

        # Determine video source
        # Priority: video_nwbfile > video_asset > nwbfile
        if video_nwbfile is not None:
            video_source_nwbfile = video_nwbfile
        elif raw_asset is not None:
            video_source_nwbfile = self._load_nwbfile_from_dandi(raw_asset)
        else:
            video_source_nwbfile = nwbfile

        # Determine which asset to use for video URLs
        video_source_asset = raw_asset if raw_asset is not None else processed_asset

        # Compute video URLs from DANDI

        video_urls = self._get_video_urls_from_dandi(video_source_nwbfile, video_source_asset)

        # Get all PoseEstimation containers (location-agnostic)
        pose_containers = discover_pose_estimation_cameras(nwbfile)
        if not pose_containers:
            raise ValueError("NWB file does not contain any PoseEstimation objects")
        available_cameras = list(pose_containers.keys())
        

        video_info = self._get_video_info(video_source_nwbfile, video_urls)
        
        self.nwbfile = nwbfile
        self.pose_containers = pose_containers
        self.available_cameras = available_cameras
        self.video_info = video_info
        




    @staticmethod
    def _load_nwbfile_from_dandi(asset: RemoteAsset) -> NWBFile:
        """Load an NWB file from DANDI, trying lindi index first for speed."""
        nwb, _io, _h5, _rf = open_nwb_dandi(asset.dandiset_id, asset.identifier)
        return nwb

    @staticmethod
    def _get_video_info(nwbfile: NWBFile, video_urls: dict[str, str]) -> dict[str, dict]:
        """Get metadata for all video series.

        Enriches existing video_urls dict with start/end times from the NWB file.
        Each entry will have: url, start, end.
        """
        video_series = discover_video_series(nwbfile)
        video_info = {}

        for name, series in video_series.items():
            info = {"url": video_urls.get(name, "")}

            if series.timestamps is not None and len(series.timestamps) > 0:
                info["start"] = float(series.timestamps[0])
                info["end"] = float(series.timestamps[-1])
            elif series.starting_time is not None and series.rate is not None:
                n_frames = series.data.shape[0] if hasattr(series.data, "shape") else 0
                if n_frames > 0:
                    info["start"] = float(series.starting_time)
                    info["end"] = float(series.starting_time + (n_frames - 1) / series.rate)
                else:
                    info["start"] = 0.0
                    info["end"] = 0.0
            else:
                info["start"] = 0.0
                info["end"] = 0.0

            video_info[name] = info

        return video_info

    @staticmethod
    def _get_video_urls_from_dandi(
        nwbfile: NWBFile,
        asset: RemoteAsset,
    ) -> dict[str, str]:
        """Extract video S3 URLs from NWB file using DANDI API."""
        dandiset = asset.client.get_dandiset(asset.dandiset_id, asset.version_id)

        # Use PurePosixPath because DANDI paths always use forward slashes
        nwb_parent = PurePosixPath(asset.path).parent
        video_series = discover_video_series(nwbfile)
        video_urls = {}

        for name, series in video_series.items():
            relative_path = series.external_file[0].lstrip("./")
            full_path = str(nwb_parent / relative_path)

            video_asset = dandiset.get_asset_by_path(full_path)
            if video_asset is not None:
                video_urls[name] = video_asset.get_content_url(follow_redirects=1, strip_query=True)

        return video_urls

    @staticmethod
    def _load_camera_pose_data(pose_containers: dict, camera_name: str) -> dict:
        """Load pose data for a single camera.

        Returns a dict with:
        - keypoint_metadata: {name: {color, label}}
        - pose_coordinates: {name: [[x, y], ...]} as JSON-serializable lists
        - timestamps: [t0, t1, ...] as JSON-serializable list
        """
        camera_pose = pose_containers[camera_name]

        keypoint_names = list(camera_pose.pose_estimation_series.keys())
        n_kp = len(keypoint_names)

        metadata = {}
        coordinates = {}
        timestamps = None

        for index, (series_name, series) in enumerate(camera_pose.pose_estimation_series.items()):
            short_name = series_name.replace("PoseEstimationSeries", "")

            # Bulk C-level conversion via tolist(), then replace sparse NaN rows with None.
            data = series.data[:]
            nan_mask = np.isnan(data).any(axis=1)
            coords_list = data.tolist()
            for nan_index in np.flatnonzero(nan_mask):
                coords_list[nan_index] = None
            coordinates[short_name] = coords_list

            if timestamps is None:
                timestamps = series.get_timestamps()[:].tolist()
                fps = 1 / np.median(np.diff(timestamps[:5]))


        return {
            "keypoint_metadata": metadata,
            "pose_coordinates": coordinates,
            "timestamps": timestamps,
            "fps": fps,
        }
