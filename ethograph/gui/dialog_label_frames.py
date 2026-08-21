"""Grid of video frames at label times (Tools ▸ Labels: Show frames as PDF…).

A config dialog lets the user tick label classes from ``mapping.txt`` (point
and state events), narrow the trials through the metadata table's condition
columns, and pick which cameras matter. *Generate* decodes, for every matching
label instance, the video frame closest to its time — one frame per point
event, a start and an end frame per state event — overlays the pose when a
pose file exists for that (trial, camera), and opens a scrollable grid of the
thumbnails. Each tile is titled with the label, trial, camera and time;
clicking it jumps the main GUI to that trial with the cursor on the label's
time. The grid's column count is adjustable and the whole grid exports to a
paginated PDF.
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
from qtpy.QtCore import QEventLoop, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QFont, QImage, QPageSize, QPainter, QPdfWriter, QPixmap
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
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.app_constants import MULTIDIM_COLORS
from ethograph.gui.file_dialogs import browse_save_file
from ethograph.gui.notify import notify
from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.gui.pose_render import POSES_DATASET_SUFFIX, PoseRenderData, load_pose_from_file
from ethograph.gui.table_filter import CategoryFilterDialog
from ethograph.gui.video_manager import probe_video
from ethograph.io.metadata_table import condition_columns
from ethograph.io.time_model import TimeRange
from ethograph.labels.intervals import EVENT_TYPE_POINT

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
    color_hex: str = "#ffffff"
    image: np.ndarray | None = None
    frame_idx: int | None = None
    cropped: bool = False
    error: str | None = None
    #: (panel title, QImage) screenshots of ticked GUI panels around t_rel,
    #: shared between the cameras of the same label boundary.
    panels: list = field(default_factory=list)


def _mapping_color_hex(info: dict) -> str:
    color = info.get("color")
    if color is None:
        return "#ffffff"
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3]))


def allowed_trials_from_metadata(
    metadata_df: pd.DataFrame | None,
    filters: dict[str, set[str]],
) -> set[str] | None:
    """Trials (as strings) passing every column filter; ``None`` = no filtering."""
    if metadata_df is None or metadata_df.empty or not any(filters.values()):
        return None
    mask = pd.Series(True, index=metadata_df.index)
    for col, allowed in filters.items():
        if not allowed or col not in metadata_df.columns:
            continue
        mask &= metadata_df[col].astype(str).isin(allowed)
    return set(metadata_df.loc[mask, "trial"].astype(str))


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
                        color_hex=_mapping_color_hex(info),
                    )
                )
    return entries


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
    y_mode: str = "lock",
    progress_cb=None,
) -> int:
    """Screenshot the ticked GUI panels around each entry's time, in place.

    Navigates the real GUI once per unique (trial, time) — the plots redraw
    with the time marker on the label time and the viewport spanning
    ``window_s`` around it — then grabs each panel widget (``QWidget.grab``,
    same capture the screen recorder uses, so pygfx canvases come out too).
    Cameras of the same boundary share the captures.

    ``y_mode`` fixes each panel's y-axis behaviour for the duration:
    ``"lock"`` freezes the ranges the user has set now (tiles compare across
    trials on one scale), ``"autoscale"`` fits each window's visible data.
    The panels' own view state is restored afterwards. Returns the number of
    positions visited; ``progress_cb(done)`` returning False stops early.
    """
    keyed: dict[tuple[str, float], list[FrameEntry]] = {}
    for entry in entries:
        keyed.setdefault((str(entry.trial), round(entry.t_rel, 6)), []).append(entry)

    view_boxes = [vb for vb in (_panel_viewbox(widget) for _, widget in panels) if vb is not None]
    saved_states = [vb.getState(copy=True) for vb in view_boxes]
    for vb in view_boxes:
        if y_mode == "autoscale":
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
    if entry.cropped:
        parts.append("cropped")
    return "  ·  ".join(parts)


# ----------------------------------------------------------------------
# PDF export
# ----------------------------------------------------------------------


def _to_qimage(image: np.ndarray) -> QImage:
    h, w, _ = image.shape
    return QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def write_frames_pdf(path: str | Path, entries: list[FrameEntry], columns: int) -> None:
    """Write the grid as a paginated PDF, *columns* tiles per row."""
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


class LabelFramesGridDialog(QDialog):
    """Scrollable grid of label frames; a tile click navigates the GUI there."""

    def __init__(self, meta, entries: list[FrameEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label frames")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self._entries = entries

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 12)
        self.columns_spin.setValue(4)
        self.columns_spin.valueChanged.connect(self._relayout)
        bar.addWidget(self.columns_spin)
        bar.addStretch()
        bar.addWidget(QLabel(f"{len(entries)} frames"))
        export_btn = QPushButton("Export PDF…")
        export_btn.setAutoDefault(False)
        export_btn.clicked.connect(self._export_pdf)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        hint = QLabel("Click a frame to jump the GUI to that trial and time.")
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
        self.resize(1100, 780)
        self._relayout()

    def _make_cell(self, entry: FrameEntry) -> QWidget:
        cell = QWidget()
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        title = QLabel(_entry_title(entry))
        title.setStyleSheet(f"font-weight: bold; color: {entry.color_hex};")
        lay.addWidget(title)
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
        write_frames_pdf(path, self._entries, self.columns_spin.value())
        notify(f"Wrote {Path(path).name}")


# ----------------------------------------------------------------------
# Config dialog
# ----------------------------------------------------------------------


class LabelFramesConfigDialog(QDialog):
    """Pick labels, metadata filters and cameras, then generate the grid."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Show label frames")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self.labels_widget = meta.labels_widget
        self._filters: dict[str, set[str]] = {}
        self._filter_buttons: dict[str, QPushButton] = {}
        self._grid_dialog: LabelFramesGridDialog | None = None

        layout = QVBoxLayout(self)
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
            item.setCheckState(Qt.Unchecked)
            self.label_list.addItem(item)
        labels_lay.addWidget(self.label_list)
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
        self.axis_lock_rb: QRadioButton | None = None
        self.axis_auto_rb: QRadioButton | None = None
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
            axis_row = QHBoxLayout()
            axis_row.addWidget(QLabel("Y axes:"))
            self.axis_lock_rb = QRadioButton("Lock (as set now)")
            self.axis_lock_rb.setChecked(True)
            self.axis_lock_rb.setToolTip(
                "Freeze each panel's current y-range for every capture —\ntiles compare across trials on one scale."
            )
            axis_row.addWidget(self.axis_lock_rb)
            self.axis_auto_rb = QRadioButton("Autoscale per window")
            self.axis_auto_rb.setToolTip("Fit each panel's y-range to the data visible in that capture's time window.")
            axis_row.addWidget(self.axis_auto_rb)
            axis_row.addStretch()
            panel_lay.addLayout(axis_row)
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
        self.resize(420, 520)

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
                y_mode="autoscale" if self.axis_auto_rb is not None and self.axis_auto_rb.isChecked() else "lock",
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

        self._grid_dialog = LabelFramesGridDialog(self.meta, entries, parent=self.parent() or self)
        self._grid_dialog.show()
        self._grid_dialog.raise_()
