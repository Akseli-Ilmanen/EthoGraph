"""S3D video features: extract them, then merge them into a session.

Two ways in, one output format — a sidecar ``{video stem}_s3d.nc`` holding a
``(time_s3d, s3d_dims)`` DataArray on the *video's* clock:

* **A folder of videos**, before any session exists — :func:`extract_videos`
  (``eto.segment.extract_videos([folder], out_dir)``).
* **A config's sessions**, whose alignment names each trial's video —
  :func:`extract_video_features` (``Project.video_features()``). Sidecars land
  under a per-session subfolder (:func:`session_video_features_dir`, keyed by
  a hash of the session's resolved source path) so that two sessions whose
  video files share a name — e.g. every session has its own ``cam-1.mp4`` —
  never overwrite each other.

:func:`merge_video_features` then samples a sidecar onto each trial's own time
axis and writes a session copy carrying ``s3d (time, s3d_dims)`` — an ordinary
feature, plottable in the GUI and selectable by name in ``features.columns``.
Merging is xarray-only; pynapple/NWB sessions merge by hand.
"""

from __future__ import annotations

import hashlib
import logging
import re
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import xarray as xr

from ethograph.io.validation import VIDEO_EXTENSIONS
from ethograph.segment.config import SegmentConfig, VideoFeaturesConfig
from ethograph.segment.sessions import Session, filter_trials, open_session
from ethograph.utils.logging import log_to_file
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

VIDEO_FEATURES_DIR = "video_features"
S3D_VAR = "s3d"
S3D_TIME_DIM = "time_s3d"


def sidecar_path(video: Path, out_dir: Path) -> Path:
    """Where *video*'s features live: ``{out_dir}/{stem}_s3d.nc``."""
    return Path(out_dir) / f"{Path(video).stem}_s3d.nc"


def _compile_patterns(include: Iterable[str] | None) -> list[re.Pattern[str]] | None:
    """Compile *include* into regexes, naming the offender if one is malformed."""
    if include is None:
        return None
    patterns = []
    for raw in include:
        try:
            patterns.append(re.compile(str(raw)))
        except re.error as exc:
            raise ValueError(f"include pattern {raw!r} is not a valid regular expression: {exc}") from exc
    if not patterns:
        raise ValueError("include is empty — pass None to keep every video, not an empty list.")
    return patterns


def iter_video_files(paths: Iterable[str | Path], include: Iterable[str] | None = None) -> Iterator[Path]:
    """Expand folders, globs and plain paths into video files, deduplicated.

    A folder is searched recursively; a path with no video extension is an
    error rather than a silent skip.

    *include* keeps only the videos whose **full path** matches one of the
    given regular expressions (``re.search``, so a plain substring like
    ``"cam-1"`` works). Matching the whole path means it catches a camera in
    either place — ``cam-1/trial003.mp4`` and ``trial003_cam-1.mp4`` alike.
    A filter that keeps nothing is an error, not an empty run.
    """
    patterns = _compile_patterns(include)
    seen: dict[Path, None] = {}
    for raw in paths:
        entry = Path(raw).expanduser()
        if entry.is_dir():
            found = sorted(p for p in entry.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
            if not found:
                raise FileNotFoundError(f"No video files under {entry} (looked for {sorted(VIDEO_EXTENSIONS)})")
            for p in found:
                seen.setdefault(p.resolve(), None)
            continue
        matches = [Path(p) for p in sorted(glob(str(entry)))] if any(c in str(entry) for c in "*?[") else [entry]
        if not matches:
            raise FileNotFoundError(f"Nothing matches {entry}")
        for p in matches:
            if not p.exists():
                raise FileNotFoundError(f"No such file or folder: {p}")
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"{p} is not a video ({sorted(VIDEO_EXTENSIONS)})")
            seen.setdefault(p.resolve(), None)
    if patterns is None:
        return iter(seen)
    kept = [p for p in seen if any(rx.search(str(p)) for rx in patterns)]
    if not kept:
        raise FileNotFoundError(
            f"include={[rx.pattern for rx in patterns]} matched none of the {len(seen)} videos found, e.g. "
            f"{[str(p) for p in list(seen)[:3]]}"
        )
    logger.info("include kept %d of %d videos", len(kept), len(seen))
    return iter(kept)


def extract_videos(
    videos: Iterable[str | Path],
    out_dir: str | Path,
    cfg: VideoFeaturesConfig | None = None,
    overwrite: bool = False,
    include: Iterable[str] | None = None,
) -> list[Path]:
    """Run S3D over each video, writing one sidecar per video into *out_dir*.

    Videos whose sidecar already exists are skipped unless *overwrite*.
    *include* narrows the videos found (see :func:`iter_video_files`) — the
    usual reason being that two cameras see nearly the same thing, so only
    one of them is worth an hour of S3D.
    """
    from ethograph.video_features import extract_s3d, plan_s3d
    from ethograph.video_features.frames import probe_video

    cfg = cfg if cfg is not None else VideoFeaturesConfig()
    s3d_cfg = cfg.s3d_config()
    out_dir = Path(out_dir)
    logger.info("Extracting S3D video features into %s (overwrite=%s)", out_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for video in iter_video_files(videos, include):
        target = sidecar_path(video, out_dir)
        if target.exists() and not overwrite:
            logger.info("%s exists — skipping (use overwrite)", target.name)
            continue
        plan = plan_s3d(probe_video(str(video)).fps, s3d_cfg)
        logger.info("%s: %s, mode=%s", video.name, plan.describe(), s3d_cfg.mode)
        da = extract_s3d(video, s3d_cfg)
        da.to_netcdf(target)
        written.append(target)
        logger.info("  → %s %s", target, tuple(da.shape))
    return written


# ---------------------------------------------------------------------------
# The config's sessions
# ---------------------------------------------------------------------------


def session_video_features_dir(session: Session, config: SegmentConfig) -> Path:
    """Where *session*'s S3D sidecars live: ``video_features_dir/{hash}``.

    Namespaced by a hash of the session's resolved source path (the same
    hashing scheme as :func:`ethograph.labels.onset_model.session_id`, minus
    the stem prefix) so that sessions whose video files happen to share a
    name don't overwrite each other's sidecars.
    """
    digest = hashlib.sha1(str(session.source.resolve()).encode("utf-8")).hexdigest()[:8]
    return config.video_features_dir / digest


def session_videos(session: Session, config: SegmentConfig) -> dict[int | str, Path]:
    """Trial → video path, for every trial passing the filter; missing videos are an error."""
    out: dict[int | str, Path] = {}
    missing = []
    for trial in filter_trials(session, config.trials):
        path = session.media_path(trial, "video", config.video_features.camera)
        if path is None:
            missing.append(trial)
        else:
            out[trial] = path
    if missing:
        raise FileNotFoundError(
            f"{session.source}: no video found for trials {missing} — "
            f"check the alignment's file names and the session's video_dir"
        )
    return out


def extract_video_features(config: SegmentConfig, overwrite: bool = False) -> list[Path]:
    """Run S3D over every video the config's sessions use.

    Each session's videos are extracted into their own subfolder
    (:func:`session_video_features_dir`) — sessions never share an output
    folder, so identically named video files across sessions can't collide.
    """
    with log_to_file(config.video_features_dir / "extract.log"):
        written: list[Path] = []
        for spec in config.sessions:
            session = open_session(spec, config)
            videos: dict[Path, None] = {}
            for path in session_videos(session, config).values():
                videos.setdefault(path.resolve(), None)
            out_dir = session_video_features_dir(session, config)
            written += extract_videos(videos, out_dir, config.video_features, overwrite=overwrite)
        return written


def merge_video_features(
    session: Session,
    config: SegmentConfig,
    features_dir: Path | None = None,
    out_path: Path | None = None,
    in_place: bool = False,
) -> Path:
    """Write a copy of *session* carrying ``s3d`` on every trial's time axis.

    Each trial's video offset (``stream_offset_for_trial``) maps the trial
    clock onto the video's; the sidecar is sampled onto the trial's own time
    axis by nearest neighbour. Returns the path written — a sibling
    ``{stem}_s3d.nc`` unless *out_path* or *in_place* says otherwise, because
    the pipeline never overwrites a session file unasked.
    """
    dt = session.result.dt
    if dt is None:
        raise ValueError(
            f"{session.source}: merging needs an xarray (.nc) session; "
            "pynapple/NWB sessions carry the sidecar in by hand."
        )
    features_dir = Path(features_dir) if features_dir is not None else session_video_features_dir(session, config)
    alignment = session.result.nwb_alignment
    videos = session_videos(session, config)

    for trial, video in videos.items():
        sidecar = sidecar_path(video, features_dir)
        if not sidecar.is_file():
            raise FileNotFoundError(f"{sidecar} missing — extract the video features first")
        # update_trial mutates the tree in place and returns None.
        dt.update_trial(trial, lambda ds, s=sidecar, t=trial: _with_s3d(ds, s, alignment, t))

    if in_place:
        target = session.source
    elif out_path is not None:
        target = Path(out_path)
    else:
        target = session.source.with_name(f"{session.source.stem}_s3d.nc")
    dt.save(str(target))
    logger.info("%s: merged s3d into %d trials → %s", session.id, len(videos), target)
    return target


def _with_s3d(ds: xr.Dataset, sidecar: Path, alignment, trial) -> xr.Dataset:
    da = xr.load_dataarray(sidecar)
    offset = float(alignment.stream_offset_for_trial(trial, "video"))
    reference = next(iter(ds.data_vars.values()))
    time_coord = get_time_coord(reference)
    if time_coord is None:
        raise ValueError(f"Trial {trial} has no time coord to sample the video features onto.")
    time = np.asarray(time_coord.values)
    # The sidecar's clock is the video's own (frame 0 at 0). The stream offset
    # places that frame 0 at trial time `offset` (VideoSync: trial = video +
    # offset), so a trial time samples the video at `t - offset`.
    sampled = da.interp(**{S3D_TIME_DIM: time - offset}, method="nearest", kwargs={"fill_value": "extrapolate"})
    sampled = sampled.rename({S3D_TIME_DIM: time_coord.name}).assign_coords({time_coord.name: time})
    sampled.attrs = {**da.attrs, "description": "S3D video features sampled onto the trial clock"}
    return ds.assign({S3D_VAR: sampled})
