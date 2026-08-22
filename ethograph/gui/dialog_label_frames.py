"""Grid of video frames at label times (Tools ▸ Labels: Show frames as Grid/PDF…).

One window with two tabs. On *Setup* the user ticks label classes from
``mapping.txt`` (point and state events), narrows the trials through the
metadata table's condition columns, and picks which cameras matter.
*Generate* decodes, for every matching label instance, the video frame closest
to its time — one frame per point event, a start and an end frame per state
event — overlays the pose when a pose file exists for that (trial, camera),
and fills the *Frames* tab with a scrollable grid of the thumbnails (the
window carries minimise/maximise buttons, so the grid goes full screen in one
click and the tiles refit to the new width). Each tile is titled with the
label, trial, camera, time and the label's confidence; clicking it jumps the
main GUI to that trial with the
cursor on the label's time. A confidence threshold outlines every tile below
it in red — the review loop for model predictions, which carry the model's own
score while human labels are 1.0. **Histogram…** next to it shows where those
scores actually pile up: one histogram per label class (per individual too,
when more than one is labelled), the part below the threshold drawn in the
same red, and its threshold spin bound both ways to the grid's. The grid's
column count is adjustable and the whole grid exports to a paginated PDF.

The grid is also where a frame-by-frame review pass is *scoped*: tick the
tiles that look wrong (**Tick flagged** ticks everything the confidence
threshold outlines) and **Refine ticked frame-by-frame…** hands exactly those
boundaries to :mod:`ethograph.gui.dialog_refine`, whose queue then walks them
and nothing else — Enter to move a boundary onto the right frame, Backspace to
delete an event that should not be there.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QEventLoop, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QImage, QPageSize, QPainter, QPdfWriter, QPen, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.app_constants import MULTIDIM_COLORS
from ethograph.gui.dialog_refine import open_refine_dialog
from ethograph.gui.file_dialogs import browse_save_file
from ethograph.gui.notify import notify
from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.gui.pose_render import POSES_DATASET_SUFFIX, PoseRenderData, load_pose_from_file
from ethograph.gui.table_filter import CategoryFilterDialog
from ethograph.gui.video_manager import probe_video
from ethograph.io.metadata_table import allowed_trials_from_metadata, condition_columns
from ethograph.io.time_model import TimeRange
from ethograph.labels.intervals import EVENT_TYPE_POINT, HUMAN_CONFIDENCE

logger = logging.getLogger(__name__)

#: Longest side frames are decoded at — thumbnails, not the full video.
THUMB_MAX_SIDE = 512

#: How long the GUI gets to redraw before a panel is screenshotted. The
#: panels paint synchronously (pyqtgraph) — the jump itself returns with the
#: plots updated — so this only needs to cover deferred zero-timers (labels
#: redraw, marker sync) and short range-change debounces; longer after a
#: trial switch, whose refresh cascade schedules more of them.
PANEL_SETTLE_MS = 100
PANEL_TRIAL_SETTLE_MS = 300

#: Size the dialog grows to when the grid first appears — the setup form is
#: narrow, a grid of thumbnails is not.
_GRID_MIN_WIDTH = 1100
_GRID_MIN_HEIGHT = 780

#: Quiet period after a resize before the tiles are rescaled, so dragging the
#: window edge does not rescale every pixmap per pixel.
_REFLOW_DEBOUNCE_MS = 120

#: Outline drawn around a tile whose label falls below the confidence
#: threshold — in the grid (stylesheet) and in the PDF (pen colour).
LOW_CONFIDENCE_COLOR = "#d94040"
_LOW_CONFIDENCE_STYLE = f"QFrame#frameCell {{ border: 2px solid {LOW_CONFIDENCE_COLOR}; border-radius: 3px; }}"

#: Confidence-histogram popup: dark canvas (the label colours are picked for
#: one), plots per row and each plot's floor.
_HIST_BG = "#1a1d21"
_HIST_COLUMNS = 3
_HIST_MIN_SIZE = (300, 210)
_HIST_BINS = 20

#: A label class drawn in its own colour would hide the flagged part of its
#: histogram if that colour is itself reddish — those bars go neutral instead
#: (the plot title still carries the label's colour).
_HIST_NEUTRAL = "#8a9099"
_HIST_MIN_COLOR_DISTANCE = 90.0


# ----------------------------------------------------------------------
# Pure logic (Qt-free, unit-tested in tests/test_unit/test_label_frames.py)
# ----------------------------------------------------------------------


@dataclass
class FrameEntry:
    """One tile of the grid: a label boundary seen by one camera."""

    trial: object
    camera: str | None
    label_id: int
    name: str
    event_type: str
    boundary: str  # "point" | "start" | "end"
    t_rel: float
    onset_s: float
    offset_s: float
    individual: object = None
    individual_rec: object = None
    #: How sure the label is: 1.0 for a human label, the model's own score for
    #: a predicted one (see ``labels/onset_model.py``).
    confidence: float = HUMAN_CONFIDENCE
    color_hex: str = "#ffffff"
    image: np.ndarray | None = None
    frame_idx: int | None = None
    cropped: bool = False
    error: str | None = None
    #: (panel title, QImage) screenshots of ticked GUI panels around t_rel,
    #: shared between the cameras of the same label boundary.
    panels: list = field(default_factory=list)


def is_low_confidence(entry: "FrameEntry", threshold: float) -> bool:
    """Whether *entry* should be flagged. A threshold of 0 flags nothing."""
    return threshold > 0.0 and entry.confidence < threshold


def _mapping_color_hex(info: dict) -> str:
    color = info.get("color")
    if color is None:
        return "#ffffff"
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3]))


def build_frame_entries(
    labels_df: pd.DataFrame,
    mappings: dict,
    label_ids: list[int],
    cameras: list[str | None],
    allowed_trials: set[str] | None = None,
) -> list[FrameEntry]:
    """Expand matching label rows into grid entries.

    One entry per point event, a start + end entry per state event, times
    trial-relative — each repeated for every selected camera so a label's
    views sit next to each other in the grid.
    """
    if labels_df is None or labels_df.empty:
        return []
    rows = labels_df[labels_df["labels"].isin(label_ids)]
    if allowed_trials is not None:
        rows = rows[rows["trial"].astype(str).isin(allowed_trials)]
    rows = rows.sort_values(["trial", "onset_s"])

    entries: list[FrameEntry] = []
    for _, row in rows.iterrows():
        label_id = int(row["labels"])
        info = mappings.get(label_id, {})
        name = str(info.get("name", label_id))
        event_type = str(info.get("event_type", "state"))
        onset = float(row["onset_s"])
        offset = float(row["offset_s"])
        is_point = event_type == EVENT_TYPE_POINT or not math.isfinite(offset)
        boundaries = [("point", onset)] if is_point else [("start", onset), ("end", offset)]
        for boundary, t_rel in boundaries:
            for camera in cameras:
                entries.append(
                    FrameEntry(
                        trial=row["trial"],
                        camera=camera,
                        label_id=label_id,
                        name=name,
                        event_type=EVENT_TYPE_POINT if is_point else "state",
                        boundary=boundary,
                        t_rel=t_rel,
                        onset_s=onset,
                        offset_s=offset,
                        individual=row.get("individual"),
                        individual_rec=row.get("individual_rec"),
                        confidence=_row_confidence(row),
                        color_hex=_mapping_color_hex(info),
                    )
                )
    return entries


def _subject_key(value) -> str:
    """Subject columns compare as text; ``None`` and NaN are the same blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def seeds_from_entries(entries: list[FrameEntry]) -> list[dict]:
    """Refine seeds for *entries* — one per boundary, cameras deduplicated.

    A boundary two cameras saw is two tiles but one label, and the refine
    queue must stop at it once. Each seed is the label row plus the ``field``
    to edit, which is what :func:`ethograph.gui.dialog_refine.targets_from_seeds`
    consumes.
    """
    seeds: dict[tuple, dict] = {}
    for entry in entries:
        key = (
            str(entry.trial),
            entry.label_id,
            round(entry.onset_s, 6),
            _subject_key(entry.individual),
            _subject_key(entry.individual_rec),
            entry.boundary,
        )
        seeds.setdefault(
            key,
            {
                "trial": entry.trial,
                "labels": entry.label_id,
                "onset_s": entry.onset_s,
                "offset_s": entry.offset_s,
                "individual": entry.individual,
                "individual_rec": entry.individual_rec,
                "event_type": entry.event_type,
                "field": entry.boundary,
            },
        )
    return list(seeds.values())


def flagged_trials(entries: list[FrameEntry], threshold: float) -> set[str]:
    """Trials holding at least one entry below *threshold* (as strings).

    A trial the model got wrong once is worth reading end to end: its other
    events may score high and still be misplaced, so this is what widens a
    review from the flagged frames to the trials they sit in.
    """
    return {str(entry.trial) for entry in entries if is_low_confidence(entry, threshold)}


@dataclass
class ConfidenceGroup:
    """One histogram's worth of confidences: a label class, one animal's."""

    label_id: int
    name: str
    #: ``None`` when the labels name a single individual — then the label
    #: class alone is the split.
    individual: str | None
    color_hex: str
    values: list[float] = field(default_factory=list)


def confidence_groups(entries: list[FrameEntry]) -> list[ConfidenceGroup]:
    """Group the entries' confidences, one group per histogram.

    A label instance seen by two cameras is two tiles but one score, and a
    state event's start and end tiles share their row's — both collapse here,
    so a histogram counts events, not tiles. Groups split per individual as
    well as per label class whenever more than one individual is labelled: a
    model is rarely equally sure about every animal.
    """
    rows: dict[tuple, FrameEntry] = {}
    for entry in entries:
        key = (
            str(entry.trial),
            entry.label_id,
            round(entry.onset_s, 6),
            _subject_key(entry.individual),
            _subject_key(entry.individual_rec),
        )
        rows.setdefault(key, entry)

    per_individual = len({_subject_key(e.individual) for e in rows.values()}) > 1
    groups: dict[tuple[int, str], ConfidenceGroup] = {}
    for entry in rows.values():
        individual = _subject_key(entry.individual) if per_individual else None
        group = groups.get((entry.label_id, individual or ""))
        if group is None:
            group = ConfidenceGroup(entry.label_id, entry.name, individual, entry.color_hex)
            groups[(entry.label_id, individual or "")] = group
        group.values.append(entry.confidence)
    return [groups[key] for key in sorted(groups)]


def split_histogram(values, threshold: float, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin *values* into ``(edges, below threshold, the rest)``.

    Both halves share one set of edges over [0, 1], so the bin the threshold
    falls inside splits into a red part and a normal part instead of being
    coloured all one way. A threshold of 0 flags nothing, as in the grid.
    """
    data = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    low = data < threshold if threshold > 0.0 else np.zeros(data.shape, dtype=bool)
    below, _ = np.histogram(data[low], bins=edges)
    above, _ = np.histogram(data[~low], bins=edges)
    return edges, below, above


def _row_confidence(row) -> float:
    """A label row's confidence; fully confident when the column is absent."""
    value = row.get("confidence", HUMAN_CONFIDENCE)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return HUMAN_CONFIDENCE
    return float(value)


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    return tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))


def _draw_disc(image: np.ndarray, cx: int, cy: int, radius: int, rgb: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[y0:y1, x0:x1][disc] = rgb


def draw_pose_points(
    image: np.ndarray,
    pose: PoseRenderData,
    frame_idx: int,
    scale: float,
    color_by: str,
) -> None:
    """Draw the pose points of one video frame onto a decoded thumbnail.

    ``scale`` is the source→decoded pixel ratio (``VideoFrameSource.scale``);
    colour encodes ``color_by`` (keypoint/individual), matching the video
    overlay's one-axis colour rule.
    """
    frame_col = 1 if pose.data.shape[1] > 3 else 0
    frames = np.full(len(pose.data), -1, dtype=int)
    valid = pose.data_not_nan
    frames[valid] = np.round(pose.data[valid, frame_col]).astype(int)
    mask = frames == frame_idx
    if not mask.any():
        return

    ys = pose.data[mask, frame_col + 1] / scale
    xs = pose.data[mask, frame_col + 2] / scale
    if color_by in pose.properties.columns:
        cats = pose.properties.iloc[np.flatnonzero(mask)][color_by].astype(str).to_numpy()
        order = pose.properties[color_by].astype(str).unique()
    else:
        cats = np.array([""] * len(xs))
        order = [""]
    palette = {val: _hex_to_rgb(MULTIDIM_COLORS[i % len(MULTIDIM_COLORS)]) for i, val in enumerate(order)}

    h, w = image.shape[:2]
    radius = max(2, round(min(h, w) / 130))
    for x, y, cat in zip(xs, ys, cats):
        _draw_disc(image, int(round(x)), int(round(y)), radius, palette.get(cat, (255, 255, 255)))


def crop_thumbnail(image: np.ndarray, rect: tuple[int, int, int, int], scale: float) -> np.ndarray:
    """Cut a GUI camera crop — ``(x0, y0, x1, y1)`` source pixels, y down —
    out of a thumbnail decoded at ``1/scale``. A rect degenerate after
    clamping returns the image unchanged."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v / scale)) for v in rect)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return image
    return np.ascontiguousarray(image[y0:y1, x0:x1])


# ----------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------


def _pose_device(alignment, camera: str | None) -> str | None:
    """Which pose stream feeds *camera* — same pairing as the video overlay."""
    pose_keys = list(getattr(alignment, "pose_keys", None) or [])
    if camera in pose_keys:
        return camera
    cameras = list(getattr(alignment, "cameras", None) or [])
    if camera in cameras:
        idx = cameras.index(camera)
        if idx < len(pose_keys):
            return pose_keys[idx]
    return None


def _load_group_pose(
    alignment,
    trial,
    camera: str | None,
    pose_folder: str | None,
    source_software: str | None,
    fps: float,
) -> PoseRenderData | None:
    device = _pose_device(alignment, camera) or camera
    path = alignment.resolve_media_path(trial, "pose", device=device, fallback_folder=pose_folder)
    if not path or not Path(path).exists():
        return None
    if not source_software and Path(path).suffix.lower() != POSES_DATASET_SUFFIX:
        logger.info("Pose file %s skipped — source software unknown.", Path(path).name)
        return None
    try:
        return load_pose_from_file(path, source_software, fps)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed to load pose %s: %s", Path(path).name, exc)
        return None


def decode_entry_images(
    entries: list[FrameEntry],
    *,
    alignment,
    video_folder: str | None,
    pose_folder: str | None,
    source_software: str | None,
    pose_color_by: str,
    camera_crops: dict[str | None, tuple[int, int, int, int]] | None = None,
    current_trial=None,
    current_video_path: str | None = None,
    progress_cb=None,
) -> None:
    """Fill each entry's ``image`` (RGB thumbnail with pose overlay) in place.

    Entries are grouped per (trial, camera) so each video is opened once and
    visited in frame order. Media metadata (paths, rates, offsets, pose
    files) resolves sequentially — the alignment NWB is not thread-safe —
    then the groups decode concurrently in a small thread pool, one video
    per worker. A file that cannot be resolved or decoded marks its entries
    with ``error`` instead of raising — missing media is a runtime
    condition, and the grid shows the message on the tile.
    ``camera_crops`` maps a selected camera to its GUI display crop (source
    pixels); a cropped camera's thumbnails show only that region.
    ``progress_cb(done)`` returning False cancels the remaining work.
    """
    groups: dict[tuple[str, str | None], list[FrameEntry]] = {}
    for entry in entries:
        groups.setdefault((str(entry.trial), entry.camera), []).append(entry)

    # Phase 1 — sequential: resolve paths, rates, offsets and pose files.
    # The alignment NWB (h5py) is not safe to read from worker threads.
    jobs: list[tuple[list[FrameEntry], str, float, float, PoseRenderData | None, object, int]] = []
    for (_, camera), group in groups.items():
        trial = group[0].trial
        path = alignment.resolve_media_path(trial, "video", device=camera, fallback_folder=video_folder)
        if not path and current_video_path and str(trial) == str(current_trial):
            path = current_video_path
        if not path or not Path(path).exists():
            for entry in group:
                entry.error = "video not found"
            continue
        try:
            probe = probe_video(path)
            fps = alignment.get_stream_rate("video", camera) or probe.fps
            offset = alignment.stream_offset_for_trial(trial, "video", camera)
            pose = _load_group_pose(alignment, trial, camera, pose_folder, source_software, fps)
        except (OSError, ValueError) as exc:
            logger.warning("Frame extraction failed for %s: %s", path, exc)
            for entry in group:
                entry.error = str(exc)
            continue
        jobs.append((group, path, fps, offset, pose, (camera_crops or {}).get(camera), probe.nframes))

    def report() -> bool:
        if progress_cb is None:
            return True
        return progress_cb(sum(1 for e in entries if e.image is not None or e.error is not None))

    if not report() or not jobs:
        return

    # Phase 2 — parallel: pure PyAV decode + numpy pose/crop work, one video
    # per worker (each opens its own container, so no shared decode state).
    cancel = threading.Event()

    def run_job(job) -> None:
        group, path, fps, offset, pose, crop, nframes = job
        try:
            with VideoFrameSource(path, fps, nframes, max_side=THUMB_MAX_SIDE) as source:
                for entry in sorted(group, key=lambda e: e.t_rel):
                    if cancel.is_set():
                        return
                    frame = int(round((entry.t_rel - offset) * fps))
                    frame = min(max(frame, 0), max(nframes - 1, 0))
                    image = np.ascontiguousarray(source[frame])
                    if pose is not None:
                        draw_pose_points(image, pose, frame, source.scale, pose_color_by)
                    if crop is not None:
                        image = crop_thumbnail(image, crop, source.scale)
                        entry.cropped = True
                    entry.frame_idx = frame
                    entry.image = image
        except (OSError, ValueError) as exc:
            logger.warning("Frame extraction failed for %s: %s", path, exc)
            for entry in group:
                if entry.image is None:
                    entry.error = str(exc)

    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        while not all(f.done() for f in futures):
            if not report():
                cancel.set()
            # Keep the progress dialog painting while the workers decode.
            _settle(50)
    report()
    for future in futures:
        future.result()


# ----------------------------------------------------------------------
# GUI panel capture
# ----------------------------------------------------------------------


def _settle(ms: int) -> None:
    """Let the GUI redraw for *ms* — a local event loop, not a blocking sleep."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _parent_dock(widget: QWidget) -> QDockWidget | None:
    parent = widget.parent()
    while parent is not None and not isinstance(parent, QDockWidget):
        parent = parent.parent()
    return parent


def _dock_label(dock: QDockWidget | None, fallback: str) -> str:
    """A dock's display title (the slim title bar's label), else *fallback*."""
    if dock is None:
        return fallback
    bar = dock.titleBarWidget()
    label = getattr(bar, "_label", None)
    if label is not None and label.text():
        return label.text()
    return dock.windowTitle() or fallback


def open_gui_panels(meta) -> list[tuple[str, QWidget]]:
    """(title, widget) of every plot panel currently visible in the GUI."""
    panels: list[tuple[str, QWidget]] = []
    data_widget = getattr(meta, "data_widget", None)
    container = getattr(data_widget, "plot_container", None)
    if container is not None:
        for plot in list(getattr(container, "_dyn_panels", []) or []):
            dock = (getattr(container, "_dyn_docks", {}) or {}).get(plot)
            if dock is None or dock.isHidden():
                continue
            panels.append((_dock_label(dock, str(getattr(plot, "panel_type", "panel"))), plot))
        for name, dock in (getattr(container, "_panel_docks", {}) or {}).items():
            if not dock.isHidden():
                panels.append((_dock_label(dock, name), dock.widget()))
    for kind, plots in (
        ("Space", getattr(data_widget, "space_plots", None) or []),
        ("Radial", getattr(data_widget, "radial_plots", None) or []),
    ):
        for i, plot in enumerate(plots, start=1):
            if plot.isVisible():
                panels.append((_dock_label(_parent_dock(plot), f"{kind} plot {i}"), plot))
    return panels


def _panel_viewbox(widget: QWidget):
    """The pyqtgraph ViewBox behind a panel, or None (space/GL widgets)."""
    get_vb = getattr(widget, "getViewBox", None)
    if callable(get_vb):
        return get_vb()
    return getattr(widget, "vb", None)


def capture_panel_images(
    entries: list[FrameEntry],
    *,
    nav,
    panels: list[tuple[str, QWidget]],
    window_s: float,
    autoscale: bool = True,
    progress_cb=None,
) -> int:
    """Screenshot the ticked GUI panels around each entry's time, in place.

    Navigates the real GUI once per unique (trial, time) — the plots redraw
    with the time marker on the label time and the viewport spanning
    ``window_s`` around it — then grabs each panel widget (``QWidget.grab``,
    same capture the screen recorder uses, so pygfx canvases come out too).
    Cameras of the same boundary share the captures.

    ``autoscale`` fits each panel's y-range to the data visible in that
    capture's time window; with it off, the ranges the user has set now stay
    frozen for the duration. The panels' own view state is restored
    afterwards. Returns the number of positions visited;
    ``progress_cb(done)`` returning False stops early.
    """
    keyed: dict[tuple[str, float], list[FrameEntry]] = {}
    for entry in entries:
        keyed.setdefault((str(entry.trial), round(entry.t_rel, 6)), []).append(entry)

    view_boxes = [vb for vb in (_panel_viewbox(widget) for _, widget in panels) if vb is not None]
    saved_states = [vb.getState(copy=True) for vb in view_boxes]
    for vb in view_boxes:
        if autoscale:
            # Fit y to the data visible in the x window, not the whole trace.
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(y=True)
        else:
            vb.enableAutoRange(y=False)

    half = window_s / 2.0
    done = 0
    last_trial: str | None = None
    try:
        for (trial_key, _), group in sorted(keyed.items()):
            lead = group[0]
            nav.jump_to_label_instance(
                {
                    "trial": lead.trial,
                    "onset_s": lead.onset_s,
                    "offset_s": lead.offset_s,
                    "individual": lead.individual,
                    "individual_rec": lead.individual_rec,
                },
                seek_rel=lead.t_rel,
                play=False,
                view_rel=TimeRange(lead.t_rel - half, lead.t_rel + half),
            )
            _settle(PANEL_TRIAL_SETTLE_MS if trial_key != last_trial else PANEL_SETTLE_MS)
            last_trial = trial_key

            shots: list[tuple[str, QImage]] = []
            for title, widget in panels:
                try:
                    shots.append((title, widget.grab().toImage().convertToFormat(QImage.Format_RGB888)))
                except RuntimeError:
                    logger.warning("Panel %r was closed during capture — skipped.", title)
            for entry in group:
                entry.panels = shots
            done += 1
            if progress_cb is not None and not progress_cb(done):
                return done
        return done
    finally:
        for vb, state in zip(view_boxes, saved_states):
            vb.setState(state)


# ----------------------------------------------------------------------
# Tile text
# ----------------------------------------------------------------------


def _entry_title(entry: FrameEntry) -> str:
    title = f"{entry.name} ({entry.label_id})"
    if entry.boundary != "point":
        title += f" — {entry.boundary.upper()}"
    return title


def _entry_info(entry: FrameEntry) -> str:
    parts = [f"trial {entry.trial}"]
    if entry.camera:
        parts.append(str(entry.camera))
    individual = entry.individual
    if individual is not None and not (isinstance(individual, float) and math.isnan(individual)):
        parts.append(str(individual))
    parts.append(f"{entry.t_rel:.3f} s")
    parts.append(f"conf {entry.confidence:.2f}")
    if entry.cropped:
        parts.append("cropped")
    return "  ·  ".join(parts)


# ----------------------------------------------------------------------
# PDF export
# ----------------------------------------------------------------------


def _to_qimage(image: np.ndarray) -> QImage:
    h, w, _ = image.shape
    return QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def write_frames_pdf(
    path: str | Path,
    entries: list[FrameEntry],
    columns: int,
    confidence_threshold: float = 0.0,
) -> None:
    """Write the grid as a paginated PDF, *columns* tiles per row.

    Tiles below *confidence_threshold* get the same red outline they carry in
    the grid, so a printed review sheet flags what to check.
    """
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setResolution(150)
    painter = QPainter(writer)
    try:
        page_w, page_h = writer.width(), writer.height()
        margin, gap = 40, 18
        cell_w = (page_w - 2 * margin - (columns - 1) * gap) // columns
        painter.setFont(QFont("Helvetica", 7))
        line_h = painter.fontMetrics().height()
        text_h = 2 * line_h + 4

        def frame_height(entry: FrameEntry) -> int:
            if entry.image is not None:
                h, w = entry.image.shape[:2]
                return int(cell_w * h / w)
            return int(cell_w * 9 / 16)

        def cell_height(entry: FrameEntry) -> int:
            total = text_h + frame_height(entry)
            for _, qimg in entry.panels:
                total += line_h + int(cell_w * qimg.height() / max(1, qimg.width()))
            return total

        y = margin
        for start in range(0, len(entries), columns):
            row = entries[start : start + columns]
            row_h = max(cell_height(entry) for entry in row)
            if y + row_h > page_h - margin and y > margin:
                writer.newPage()
                y = margin
            for i, entry in enumerate(row):
                x = margin + i * (cell_w + gap)
                if is_low_confidence(entry, confidence_threshold):
                    painter.save()
                    painter.setPen(QPen(QColor(LOW_CONFIDENCE_COLOR), 2))
                    painter.drawRect(x - 5, y - 5, cell_w + 10, cell_height(entry) + 10)
                    painter.restore()
                painter.drawText(x, y + line_h, _entry_title(entry))
                painter.drawText(x, y + 2 * line_h, _entry_info(entry))
                cy = y + text_h
                h_img = frame_height(entry)
                if entry.image is not None:
                    painter.drawImage(QRect(x, cy, cell_w, h_img), _to_qimage(entry.image))
                else:
                    painter.drawRect(x, cy, cell_w, h_img)
                    painter.drawText(x + 4, cy + line_h, f"(no frame: {entry.error or 'unavailable'})")
                cy += h_img
                for title, qimg in entry.panels:
                    painter.drawText(x, cy + line_h - 2, title)
                    cy += line_h
                    h_panel = int(cell_w * qimg.height() / max(1, qimg.width()))
                    painter.drawImage(QRect(x, cy, cell_w, h_panel), qimg)
                    cy += h_panel
            y += row_h + gap
    finally:
        painter.end()


# ----------------------------------------------------------------------
# Grid dialog
# ----------------------------------------------------------------------


class _ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


def histogram_bar_color(color_hex: str) -> str:
    """The colour the unflagged bars take.

    The label's own, unless it sits so close to the flag red that the flagged
    part of the same histogram would not read as a separate colour.
    """
    if math.dist(_hex_to_rgb(color_hex), _hex_to_rgb(LOW_CONFIDENCE_COLOR)) < _HIST_MIN_COLOR_DISTANCE:
        return _HIST_NEUTRAL
    return color_hex


def _group_title(group: ConfidenceGroup, flagged: int) -> str:
    title = f"{group.name} ({group.label_id})"
    if group.individual:
        title += f" — {group.individual}"
    counts = f"n={len(group.values)}"
    if flagged:
        counts += f" · {flagged} flagged"
    return f"{title}   {counts}"


class ConfidenceHistogramsDialog(QDialog):
    """How the confidences of each label class are distributed.

    One histogram per label class — per (class, individual) when more than one
    individual is labelled — with the part below the threshold drawn in the
    same red the grid outlines flagged tiles in, so the threshold can be
    chosen by looking at where the model's scores actually pile up. The
    threshold spin is bound both ways to the grid's.
    """

    threshold_changed = Signal(float)

    def __init__(self, groups: list[ConfidenceGroup], threshold: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confidence histograms")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._groups = list(groups)
        self._plots: list[tuple[ConfidenceGroup, pg.PlotWidget]] = []

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Flag confidence below:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setSpecialValueText("off")
        self.threshold_spin.setValue(threshold)
        self.threshold_spin.setToolTip("Shared with the grid — moving it here recolours the tiles too.")
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        bar.addWidget(self.threshold_spin)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(_HIST_BINS)
        self.bins_spin.valueChanged.connect(self._redraw)
        bar.addWidget(self.bins_spin)
        bar.addStretch()
        layout.addLayout(bar)

        hint = QLabel(
            "Each event counts once, whatever it was seen by · red is what falls below the threshold · "
            "human labels score 1.0."
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)
        for i, group in enumerate(self._groups):
            plot = pg.PlotWidget()
            plot.setBackground(_HIST_BG)
            plot.setLabel("bottom", "confidence")
            plot.setLabel("left", "events")
            plot.setXRange(0.0, 1.0, padding=0.02)
            plot.setMinimumSize(*_HIST_MIN_SIZE)
            grid.addWidget(plot, i // _HIST_COLUMNS, i % _HIST_COLUMNS)
            self._plots.append((group, plot))
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self._redraw()
        columns = min(_HIST_COLUMNS, max(1, len(self._groups)))
        rows = math.ceil(len(self._groups) / _HIST_COLUMNS) if self._groups else 1
        self.resize(columns * (_HIST_MIN_SIZE[0] + 20) + 40, min(rows, 2) * (_HIST_MIN_SIZE[1] + 20) + 120)

    def set_threshold(self, value: float) -> None:
        """Follow the grid's threshold (a no-op when it already matches)."""
        self.threshold_spin.setValue(float(value))

    def _on_threshold(self, value: float) -> None:
        self.threshold_changed.emit(float(value))
        self._redraw()

    def _redraw(self) -> None:
        threshold = self.threshold_spin.value()
        bins = self.bins_spin.value()
        for group, plot in self._plots:
            edges, below, above = split_histogram(group.values, threshold, bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            width = (edges[1] - edges[0]) * 0.9
            plot.clear()
            plot.addItem(
                pg.BarGraphItem(
                    x=centers,
                    height=above,
                    y0=below,
                    width=width,
                    brush=pg.mkBrush(histogram_bar_color(group.color_hex)),
                    pen=pg.mkPen(None),
                )
            )
            plot.addItem(
                pg.BarGraphItem(
                    x=centers,
                    height=below,
                    width=width,
                    brush=pg.mkBrush(LOW_CONFIDENCE_COLOR),
                    pen=pg.mkPen(None),
                )
            )
            if threshold > 0.0:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=threshold,
                        angle=90,
                        pen=pg.mkPen(LOW_CONFIDENCE_COLOR, width=1, style=Qt.DashLine),
                    )
                )
            plot.setTitle(_group_title(group, int(below.sum())), color=group.color_hex, size="10pt")


class LabelFramesGridView(QWidget):
    """Scrollable grid of label frames — the *Frames* tab of the dialog.

    A tile click navigates the GUI there; a tile's tick box queues that
    boundary for frame-by-frame refinement, which is the review loop this grid
    exists for — scan the sheet, tick what is wrong, refine exactly those.
    """

    def __init__(self, meta, entries: list[FrameEntry], parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._entries = entries
        #: The refine dialog handed the ticked boundaries — kept alive here
        #: only when no top bar owns one (tests, embedded use).
        self._refine_dialog = None
        #: The confidence-histogram popup while it is open.
        self._hist_dialog: ConfidenceHistogramsDialog | None = None
        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.timeout.connect(self._relayout)

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 12)
        self.columns_spin.setValue(4)
        self.columns_spin.valueChanged.connect(self._relayout)
        bar.addWidget(self.columns_spin)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Flag confidence below:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setSpecialValueText("off")
        self.threshold_spin.setToolTip(
            "Outline every tile whose label scores below this in red.\n"
            "Human labels are 1.0; a predicted label carries the model's own score."
        )
        self.threshold_spin.valueChanged.connect(self._apply_threshold)
        bar.addWidget(self.threshold_spin)
        self.histogram_btn = QPushButton("Histogram…")
        self.histogram_btn.setAutoDefault(False)
        self.histogram_btn.setToolTip(
            "How the confidences are distributed, one histogram per label class\n"
            "(per individual too, when more than one is labelled). The part below\n"
            "the threshold is red, and the threshold can be set from there."
        )
        self.histogram_btn.clicked.connect(self._show_histograms)
        bar.addWidget(self.histogram_btn)
        self.tick_flagged_btn = QPushButton("Tick flagged")
        self.tick_flagged_btn.setAutoDefault(False)
        self.tick_flagged_btn.setToolTip("Tick every tile the threshold outlines in red")
        self.tick_flagged_btn.clicked.connect(self._tick_flagged)
        bar.addWidget(self.tick_flagged_btn)
        self.tick_flagged_trials_btn = QPushButton("Tick their whole trials")
        self.tick_flagged_trials_btn.setAutoDefault(False)
        self.tick_flagged_trials_btn.setToolTip(
            "Tick every event of every trial that holds a flagged one.\n"
            "A trial the model got one event wrong in is worth reviewing\n"
            "end to end: its other events may score high and still sit\n"
            "on the wrong frame."
        )
        self.tick_flagged_trials_btn.clicked.connect(self._tick_flagged_trials)
        bar.addWidget(self.tick_flagged_trials_btn)
        bar.addStretch()
        bar.addWidget(QLabel(f"{len(entries)} frames"))
        export_btn = QPushButton("Export PDF…")
        export_btn.setAutoDefault(False)
        export_btn.clicked.connect(self._export_pdf)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        select_bar = QHBoxLayout()
        self.clear_ticks_btn = QPushButton("Clear ticks")
        self.clear_ticks_btn.setAutoDefault(False)
        self.clear_ticks_btn.clicked.connect(self._clear_ticks)
        select_bar.addWidget(self.clear_ticks_btn)
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: grey; font-size: 10px;")
        select_bar.addWidget(self.selection_label)
        select_bar.addStretch()
        self.refine_btn = QPushButton("Refine ticked frame-by-frame…")
        self.refine_btn.setAutoDefault(False)
        self.refine_btn.setToolTip(
            "Hand the ticked boundaries to the frame-by-frame refinement dialog:\n"
            "its queue then walks exactly these — ←/→ nudge the video, Enter\n"
            "commits the frame on screen, Backspace deletes an event that\n"
            "should not exist at all."
        )
        self.refine_btn.clicked.connect(self._refine_ticked)
        select_bar.addWidget(self.refine_btn)
        layout.addLayout(select_bar)

        hint = QLabel(
            "Click a frame to jump the GUI to that trial and time · "
            "tick the frames that look wrong, then refine them one by one."
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        container = QWidget()
        self._grid = QGridLayout(container)
        self._grid.setSpacing(10)
        self._scroll.setWidget(container)
        layout.addWidget(self._scroll)

        self._cells = [self._make_cell(entry) for entry in entries]
        self._relayout()
        self._apply_threshold()
        self._sync_selection()

    def _make_cell(self, entry: FrameEntry) -> QFrame:
        cell = QFrame()
        cell.setObjectName("frameCell")
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(4)
        select_cb = QCheckBox()
        select_cb.setToolTip("Tick to queue this boundary for frame-by-frame refinement")
        select_cb.toggled.connect(self._sync_selection)
        head.addWidget(select_cb)
        title = QLabel(_entry_title(entry))
        title.setStyleSheet(f"font-weight: bold; color: {entry.color_hex};")
        head.addWidget(title, stretch=1)
        lay.addLayout(head)
        cell._select_cb = select_cb  # type: ignore[attr-defined]
        info = QLabel(_entry_info(entry))
        info.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(info)

        #: (label, unscaled pixmap) pairs — relayout rescales each to the
        #: current column width.
        pix_labels: list[tuple[QLabel, QPixmap]] = []

        image_label = _ClickableLabel()
        image_label.setCursor(Qt.PointingHandCursor)
        image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        if entry.image is not None:
            pixmap = QPixmap.fromImage(_to_qimage(entry.image))
            image_label.setPixmap(pixmap)
            pix_labels.append((image_label, pixmap))
        else:
            image_label.setText(f"(no frame:\n{entry.error or 'unavailable'})")
            image_label.setFrameShape(QFrame.StyledPanel)
            image_label.setMinimumSize(160, 90)
        image_label.clicked.connect(lambda e=entry: self._jump(e))
        lay.addWidget(image_label)

        for panel_title, qimage in entry.panels:
            caption = QLabel(panel_title)
            caption.setStyleSheet("color: grey; font-size: 9px;")
            lay.addWidget(caption)
            panel_label = QLabel()
            panel_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            pixmap = QPixmap.fromImage(qimage)
            panel_label.setPixmap(pixmap)
            pix_labels.append((panel_label, pixmap))
            lay.addWidget(panel_label)

        cell._pix_labels = pix_labels  # type: ignore[attr-defined]
        lay.addStretch()
        return cell

    def _apply_threshold(self):
        """Outline the tiles below the threshold; clear the rest."""
        threshold = self.threshold_spin.value()
        for cell, entry in zip(self._cells, self._entries):
            cell.setStyleSheet(_LOW_CONFIDENCE_STYLE if is_low_confidence(entry, threshold) else "")
        any_flagged = any(is_low_confidence(e, threshold) for e in self._entries)
        self.tick_flagged_btn.setEnabled(any_flagged)
        self.tick_flagged_trials_btn.setEnabled(any_flagged)
        if self._hist_dialog is not None:
            self._hist_dialog.set_threshold(threshold)

    def _show_histograms(self):
        """Open (or raise) the per-label confidence histograms."""
        if self._hist_dialog is not None:
            self._hist_dialog.show()
            self._hist_dialog.raise_()
            return
        groups = confidence_groups(self._entries)
        if not groups:
            notify("No labels to plot confidences for.", severity="warning")
            return
        self._hist_dialog = ConfidenceHistogramsDialog(groups, self.threshold_spin.value(), parent=self)
        self._hist_dialog.threshold_changed.connect(self.threshold_spin.setValue)
        self._hist_dialog.destroyed.connect(self._on_histograms_closed)
        self._hist_dialog.show()

    def _on_histograms_closed(self, *_args):
        self._hist_dialog = None

    # ------------------------------------------------------------------
    # Ticking tiles → frame-by-frame refinement
    # ------------------------------------------------------------------

    def _ticked_entries(self) -> list[FrameEntry]:
        return [entry for cell, entry in zip(self._cells, self._entries) if cell._select_cb.isChecked()]

    def _sync_selection(self, *_args):
        """Keep the tick count and the Refine button honest."""
        n = len(self._ticked_entries())
        self.selection_label.setText(f"{n} ticked" if n else "")
        self.refine_btn.setEnabled(bool(n))

    def _tick_flagged(self):
        """Tick every tile the confidence threshold flags."""
        threshold = self.threshold_spin.value()
        for cell, entry in zip(self._cells, self._entries):
            if is_low_confidence(entry, threshold):
                cell._select_cb.setChecked(True)

    def _tick_flagged_trials(self):
        """Tick every tile of every trial holding a flagged one."""
        trials = flagged_trials(self._entries, self.threshold_spin.value())
        for cell, entry in zip(self._cells, self._entries):
            if str(entry.trial) in trials:
                cell._select_cb.setChecked(True)

    def _clear_ticks(self):
        for cell in self._cells:
            cell._select_cb.setChecked(False)

    def _refine_ticked(self):
        """Hand the ticked boundaries to the frame-by-frame refine dialog."""
        entries = self._ticked_entries()
        if not entries:
            notify("Tick the frames that need refining first.", severity="warning")
            return
        seeds = seeds_from_entries(entries)
        self._refine_dialog = open_refine_dialog(self.meta, parent=self)
        if self._refine_dialog.start_from_seeds(seeds, from_grid=True):
            notify(f"Refining {len(seeds)} boundaries — ←/→ nudge the video, Enter commits, Backspace deletes.")

    def resizeEvent(self, event):
        """Refit the thumbnails whenever the width changes — maximizing the
        dialog grows every tile to the new column width."""
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self._reflow_timer.start(_REFLOW_DEBOUNCE_MS)

    def _relayout(self):
        columns = self.columns_spin.value()
        spacing = self._grid.spacing()
        viewport_w = max(self._scroll.viewport().width(), 400)
        thumb_w = max(100, (viewport_w - spacing * (columns + 1)) // columns)
        while self._grid.count():
            self._grid.takeAt(0)
        for i, cell in enumerate(self._cells):
            for label, pixmap in cell._pix_labels:
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaledToWidth(min(thumb_w, pixmap.width()), Qt.SmoothTransformation))
            self._grid.addWidget(cell, i // columns, i % columns, alignment=Qt.AlignTop)

    def _jump(self, entry: FrameEntry):
        nav = getattr(self.meta, "navigation_widget", None)
        if nav is None:
            return
        inst = {
            "trial": entry.trial,
            "labels": entry.label_id,
            "onset_s": entry.onset_s,
            "offset_s": entry.offset_s,
            "individual": entry.individual,
            "individual_rec": entry.individual_rec,
        }
        nav.jump_to_label_instance(inst, seek_rel=entry.t_rel, play=False)

    def _export_pdf(self):
        labels_path = self.app_state.labels_file_path()
        default = f"{labels_path.stem}_frames.pdf" if labels_path else "label_frames.pdf"
        path = browse_save_file(
            self,
            self.app_state,
            "Export label frames PDF",
            default,
            "PDF files (*.pdf)",
            preferred_dir=labels_path,
        )
        if not path:
            return
        write_frames_pdf(path, self._entries, self.columns_spin.value(), self.threshold_spin.value())
        notify(f"Wrote {Path(path).name}")


# ----------------------------------------------------------------------
# Dialog: Setup tab + Frames tab
# ----------------------------------------------------------------------


class LabelFramesDialog(QDialog):
    """One window, two tabs: *Setup* picks what to show, *Frames* is the grid.

    *label_ids* pre-ticks label classes and *trials* narrows the run to a set
    of trial ids on top of the metadata filters — how another dialog hands a
    batch of labels over for review (see the Predict dialog's "Review
    predictions" button).
    """

    def __init__(self, meta, parent=None, *, label_ids: list[int] | None = None, trials: set[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Label frames")
        # Minimise/maximise sit next to the close button, so the grid goes
        # full screen in one click.
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self.labels_widget = meta.labels_widget
        self._filters: dict[str, set[str]] = {}
        self._filter_buttons: dict[str, QPushButton] = {}
        self.grid_view: LabelFramesGridView | None = None
        self._restrict_trials = set(trials) if trials else None
        preselected = set(label_ids or ())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        setup_page = QWidget()
        layout = QVBoxLayout(setup_page)
        layout.setSpacing(8)

        labels_group = QGroupBox("Labels")
        labels_lay = QVBoxLayout(labels_group)
        labels_lay.addWidget(QLabel("Tick the label classes to show frames for:"))
        self.label_list = QListWidget()
        mappings = self._mappings()
        for label_id, info in sorted(mappings.items(), key=lambda x: x[0]):
            if not isinstance(label_id, int) or label_id == 0:
                continue
            name = info.get("name", str(label_id))
            event_type = info.get("event_type", "state")
            item = QListWidgetItem(f"{label_id} — {name}  ({event_type})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if label_id in preselected else Qt.Unchecked)
            self.label_list.addItem(item)
        labels_lay.addWidget(self.label_list)
        if self._restrict_trials is not None:
            restricted = QLabel(f"Restricted to {len(self._restrict_trials)} trials handed over for review.")
            restricted.setWordWrap(True)
            restricted.setStyleSheet("color: grey; font-size: 10px;")
            labels_lay.addWidget(restricted)
        layout.addWidget(labels_group)

        meta_group = QGroupBox("Trial metadata filters")
        meta_lay = QGridLayout(meta_group)
        mdf = getattr(self.app_state, "metadata_df", None)
        columns = condition_columns(mdf) if mdf is not None and not mdf.empty else []
        if not columns:
            meta_lay.addWidget(QLabel("No metadata columns available."), 0, 0)
        for i, column in enumerate(columns):
            meta_lay.addWidget(QLabel(column), i, 0)
            btn = QPushButton("All")
            btn.setAutoDefault(False)
            btn.clicked.connect(lambda _=False, c=column: self._edit_filter(c))
            meta_lay.addWidget(btn, i, 1)
            self._filter_buttons[column] = btn
        layout.addWidget(meta_group)

        self.camera_list: QListWidget | None = None
        cameras = list(getattr(getattr(self.app_state, "nwb_alignment", None), "cameras", None) or [])
        if cameras:
            cam_group = QGroupBox("Cameras")
            cam_lay = QVBoxLayout(cam_group)
            self.camera_list = QListWidget()
            for camera in cameras:
                cropped = self._gui_crop_for(camera) is not None
                item = QListWidgetItem(f"{camera}  (cropped)" if cropped else str(camera))
                item.setData(Qt.UserRole, camera)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.camera_list.addItem(item)
            self.camera_list.setMaximumHeight(90)
            cam_lay.addWidget(self.camera_list)
            layout.addWidget(cam_group)

        crop_hint = QLabel(
            "A camera cropped in the GUI keeps its crop here — crop a camera\n"
            "view first to zoom every frame onto the region of interest."
        )
        crop_hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(crop_hint)

        panel_group = QGroupBox("GUI panels under each frame")
        panel_lay = QVBoxLayout(panel_group)
        self.panel_list: QListWidget | None = None
        self.window_spin: QDoubleSpinBox | None = None
        self.skip_video_cb: QCheckBox | None = None
        self.axis_auto_cb: QCheckBox | None = None
        open_panels = open_gui_panels(self.meta)
        if open_panels:
            panel_lay.addWidget(QLabel("Tick open panels to screenshot below each frame:"))
            self.panel_list = QListWidget()
            for title, widget in open_panels:
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, widget)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.panel_list.addItem(item)
            self.panel_list.setMaximumHeight(90)
            panel_lay.addWidget(self.panel_list)
            window_row = QHBoxLayout()
            window_row.addWidget(QLabel("Time window:"))
            self.window_spin = QDoubleSpinBox()
            self.window_spin.setRange(0.01, 600.0)
            self.window_spin.setDecimals(2)
            self.window_spin.setValue(1.0)
            self.window_spin.setSuffix(" s")
            self.window_spin.setToolTip("Plot window shown around each label time — the marker sits on the label.")
            window_row.addWidget(self.window_spin, stretch=1)
            panel_lay.addLayout(window_row)
            self.axis_auto_cb = QCheckBox("Autoscale y per window")
            self.axis_auto_cb.setChecked(True)
            self.axis_auto_cb.setToolTip(
                "Fit each panel's y-range to the data visible in that capture's\n"
                "time window; unticked freezes the y-ranges as they are set now."
            )
            panel_lay.addWidget(self.axis_auto_cb)
            self.skip_video_cb = QCheckBox("Skip video loading during capture (faster)")
            self.skip_video_cb.setChecked(True)
            self.skip_video_cb.setToolTip(
                "Trial switches during capture skip loading each trial's video and\n"
                "pose — the screenshots only need the plot panels, and spawning a\n"
                "video decoder dominates the per-trial cost. Reloaded afterwards."
            )
            panel_lay.addWidget(self.skip_video_cb)
            panel_hint = QLabel("Capturing navigates the GUI through the selected labels' trials, then returns.")
            panel_hint.setStyleSheet("color: grey; font-size: 10px;")
            panel_lay.addWidget(panel_hint)
        else:
            panel_lay.addWidget(QLabel("No plot panels open in the GUI."))
        layout.addWidget(panel_group)

        generate_btn = QPushButton("Generate")
        generate_btn.setAutoDefault(False)
        generate_btn.clicked.connect(self._generate)
        layout.addWidget(generate_btn)

        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(setup_page)
        self.tabs.addTab(setup_scroll, "Setup")

        self._frames_placeholder = QLabel("Pick labels on the Setup tab and press Generate.")
        self._frames_placeholder.setAlignment(Qt.AlignCenter)
        self._frames_placeholder.setStyleSheet("color: grey;")
        self.tabs.addTab(self._frames_placeholder, "Frames")
        self.tabs.setTabEnabled(1, False)
        self.resize(460, 620)

    def _mappings(self) -> dict:
        return getattr(self.labels_widget, "_mappings", {}) or {}

    def _gui_crop_for(self, camera: str | None) -> tuple[int, int, int, int] | None:
        """The display crop the GUI holds for *camera* (source pixels).

        With no named cameras the entries carry ``camera=None`` — that maps
        to whatever camera the primary view currently shows.
        """
        vm = getattr(getattr(self.meta, "data_widget", None), "video_mgr", None)
        if vm is None:
            return None
        name = camera if camera is not None else getattr(vm.primary_view, "camera_name", None)
        return vm.camera_crop(name)

    def _camera_crops(self, cameras: list[str | None]) -> dict[str | None, tuple[int, int, int, int]]:
        crops = {}
        for camera in cameras:
            rect = self._gui_crop_for(camera)
            if rect is not None:
                crops[camera] = rect
        return crops

    def _edit_filter(self, column: str):
        mdf = getattr(self.app_state, "metadata_df", None)
        if mdf is None or column not in mdf.columns:
            return
        values = sorted(mdf[column].dropna().astype(str).unique())
        dialog = CategoryFilterDialog(0, values, self._filters.get(column, set()), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        allowed = dialog.get_allowed()
        self._filters[column] = allowed
        self._filter_buttons[column].setText("All" if not allowed else f"{len(allowed)} of {len(values)}")

    def _selected_label_ids(self) -> list[int]:
        ids = []
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(int(item.data(Qt.UserRole)))
        return ids

    def _checked_panels(self) -> list[tuple[str, QWidget]]:
        if self.panel_list is None:
            return []
        panels = []
        for i in range(self.panel_list.count()):
            item = self.panel_list.item(i)
            if item.checkState() == Qt.Checked:
                panels.append((item.text(), item.data(Qt.UserRole)))
        return panels

    def _current_position(self) -> tuple[object, float] | None:
        """Where the user currently is, to jump back after panel capture."""
        video = getattr(self.app_state, "video", None)
        if video is not None:
            hit = self.app_state.from_display(video.frame_to_time(int(self.app_state.current_frame)))
            if hit is not None:
                return hit
        trial = getattr(self.app_state, "trials_sel", None)
        return (trial, 0.0) if trial is not None else None

    def _restore_position(self, position: tuple[object, float] | None):
        nav = getattr(self.meta, "navigation_widget", None)
        if position is None or nav is None:
            return
        trial, t_rel = position
        nav.jump_to_label_instance(
            {"trial": trial, "onset_s": float(t_rel), "offset_s": float("nan")},
            play=False,
        )

    def _capture_panels(self, entries: list[FrameEntry]):
        """Screenshot the ticked panels for every entry, restoring the view."""
        panels = self._checked_panels()
        nav = getattr(self.meta, "navigation_widget", None)
        if not panels or nav is None:
            return
        n_positions = len({(str(e.trial), round(e.t_rel, 6)) for e in entries})
        progress = QProgressDialog("Capturing GUI panels…", "Cancel", 0, n_positions, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(done: int) -> bool:
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        position = self._current_position()
        data_widget = getattr(self.meta, "data_widget", None)
        skip_video = data_widget is not None and self.skip_video_cb is not None and self.skip_video_cb.isChecked()
        if skip_video:
            # Trial switches during capture skip video/pose loading — the
            # panels being screenshotted never show them, and spawning a
            # decode worker dominates the per-trial cost.
            data_widget.suppress_video_load = True
        try:
            capture_panel_images(
                entries,
                nav=nav,
                panels=panels,
                window_s=self.window_spin.value() if self.window_spin is not None else 4.0,
                autoscale=self.axis_auto_cb is None or self.axis_auto_cb.isChecked(),
                progress_cb=on_progress,
            )
        finally:
            progress.close()
            # Jump back while still suppressed (cheap), then reload the media
            # once for wherever we landed — a same-file reload reuses the
            # loaded PlotVideo, so this is cheap when the trial didn't change.
            self._restore_position(position)
            if skip_video:
                data_widget.suppress_video_load = False
                data_widget.update_video()
                data_widget._init_or_update_extra_cameras()
                data_widget.video_mgr.sync_proxies()
                data_widget.update_pose()

    def _selected_cameras(self) -> list[str | None]:
        if self.camera_list is None:
            return [None]
        cameras = []
        for i in range(self.camera_list.count()):
            item = self.camera_list.item(i)
            if item.checkState() == Qt.Checked:
                cameras.append(item.data(Qt.UserRole))
        return cameras

    def _generate(self):
        label_ids = self._selected_label_ids()
        if not label_ids:
            notify("Tick at least one label class.", severity="warning")
            return
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            notify("No labels loaded.", severity="warning")
            return
        cameras = self._selected_cameras()
        if not cameras:
            notify("Tick at least one camera.", severity="warning")
            return
        allowed = allowed_trials_from_metadata(getattr(self.app_state, "metadata_df", None), self._filters)
        if self._restrict_trials is not None:
            allowed = self._restrict_trials if allowed is None else allowed & self._restrict_trials
        entries = build_frame_entries(df, self._mappings(), label_ids, cameras, allowed)
        if not entries:
            notify("No label instances match the selected labels and metadata filters.", severity="warning")
            return

        progress = QProgressDialog("Extracting frames…", "Cancel", 0, len(entries), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(done: int) -> bool:
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        source_software = self.app_state.source_software or getattr(self.app_state.ds, "source_software", None)
        decode_entry_images(
            entries,
            alignment=self.app_state.nwb_alignment,
            video_folder=self.app_state.video_folder,
            pose_folder=self.app_state.pose_folder,
            source_software=source_software,
            pose_color_by=getattr(self.app_state, "pose_color_by", "keypoint") or "keypoint",
            camera_crops=self._camera_crops(cameras),
            current_trial=getattr(self.app_state, "trials_sel", None),
            current_video_path=getattr(self.app_state, "video_path", None),
            progress_cb=on_progress,
        )
        progress.close()
        if progress.wasCanceled():
            entries = [e for e in entries if e.image is not None or e.error]
            if not entries:
                return

        self._capture_panels(entries)
        self._show_grid(entries)

    def _show_grid(self, entries: list[FrameEntry]):
        """Put a freshly built grid on the *Frames* tab and go there."""
        old = self.tabs.widget(1)
        self.grid_view = LabelFramesGridView(self.meta, entries, parent=self)
        self.tabs.removeTab(1)
        self.tabs.insertTab(1, self.grid_view, f"Frames ({len(entries)})")
        self.tabs.setTabEnabled(1, True)
        if old is not self._frames_placeholder:
            old.deleteLater()
        self.tabs.setCurrentIndex(1)
        if self.width() < _GRID_MIN_WIDTH:
            self.resize(max(self.width(), _GRID_MIN_WIDTH), max(self.height(), _GRID_MIN_HEIGHT))
