"""Temporal alignment timeline visualization for the NC wizard (Page 4)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QRegularExpression, QRectF, Qt
from qtpy.QtGui import QColor, QBrush, QFont, QPainterPath, QPen, QSyntaxHighlighter, QTextCharFormat
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPathItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.wizard_multi_codegen import generate_alignment_code
from ethograph.gui.dialog_function_params import _do_open_source
from ethograph.gui.plots_timeseriessource import compute_trial_alignment, TimeRange, TrialAlignment
from ethograph.gui.wizard_media_files import extract_file_row
from ethograph.gui.wizard_overview import ModalityConfig, WizardState
from ethograph.utils.stream_durations import get_audio_duration, get_ephys_duration, get_pose_duration, get_video_duration
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

# Colors per modality (matching dialog_media_files.py palette)
MODALITY_COLORS = {
    "video": "#50c8b4",
    "pose": "#e8737a",
    "audio": "#e8c75a",
    "ephys": "#b07ae8",
    "features": "#7ab0e8",
}


class PythonCodeHighlighter(QSyntaxHighlighter):
    """Simple Python syntax highlighter for the code preview panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor("#c586c0"))
        for kw in [
            "and", "as", "assert", "break", "class", "continue", "def", "del",
            "elif", "else", "except", "False", "finally", "for", "from", "if",
            "import", "in", "is", "lambda", "None", "nonlocal", "not", "or",
            "pass", "raise", "return", "True", "try", "while", "with", "yield",
        ]:
            self._rules.append((QRegularExpression(rf"\\b{kw}\\b"), keyword_fmt))

        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#b5cea8"))
        self._rules.append((QRegularExpression(r"\b-?\d+(\.\d+)?([eE][+-]?\d+)?\b"), number_fmt))

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#ce9178"))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), string_fmt))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), string_fmt))

        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6a9955"))
        self._rules.append((QRegularExpression(r"#.*$"), comment_fmt))

        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()"), func_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


# ---------------------------------------------------------------------------
# Notebook conversion helper
# ---------------------------------------------------------------------------

def _code_to_notebook(code: str) -> dict:
    """Convert Python code with section markers to Jupyter notebook format."""
    import re

    section_pattern = r'^# ─── \d+\. .+ ───$'
    lines = code.split('\n')

    cells = []
    current_cell_lines = []

    for line in lines:
        if re.match(section_pattern, line):
            if current_cell_lines:
                cell_code = '\n'.join(current_cell_lines).strip()
                if cell_code:
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": cell_code.split('\n')
                    })
            current_cell_lines = [line]
        else:
            current_cell_lines.append(line)

    if current_cell_lines:
        cell_code = '\n'.join(current_cell_lines).strip()
        if cell_code:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_code.split('\n')
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _make_rounded_bar(
    x0: float, x1: float, y0: float, y1: float,
    brush: QBrush, pen: QPen, radius: float = 0.08,
) -> QGraphicsPathItem:
    path = QPainterPath()
    rect = QRectF(x0, y0, x1 - x0, y1 - y0)
    path.addRoundedRect(rect, radius, radius)
    item = QGraphicsPathItem(path)
    item.setBrush(brush)
    item.setPen(pen)
    return item


def draw_session_timeline(
    plot: pg.PlotWidget,
    dt,
    items_out: list | None = None,
    extra_streams: list[str] | None = None,
) -> float:
    """Draw a timeline from NWB session data onto a pyqtgraph PlotWidget.

    Reads trial boundaries and media timing from ``dt.session_io``
    (NWB-backed or empty). For per-trial media, draws bars at each
    trial's time range. For session-wide media, uses acquisition
    ImageSeries timestamps/starting_time.

    Parameters
    ----------
    plot
        Target PlotWidget (cleared before drawing).
    dt
        TrialTree with a session_io (NWB-backed or empty).
    items_out
        If provided, appended with all created QGraphicsItems.
    extra_streams
        Additional stream names to draw (e.g. ``["features"]``).

    Returns
    -------
    float
        Total duration (max time) for x-axis scaling.
    """
    sio = dt.session_io
    sio.print_session()

    items: list = items_out if items_out is not None else []

    try:
        trials = dt.trials
    except ValueError:
        return 1.0

    if not trials:
        return 1.0

    # Discover streams and devices from session_io
    streams_to_draw: list[tuple[str, str, list[str]]] = []
    for stream_name in ["video", "pose", "audio", "ephys"] + (extra_streams or []):
        devices = sio.devices(stream_name)
        if devices:
            for dev in devices:
                streams_to_draw.append((f"{stream_name}: {dev}", stream_name, [dev]))
        elif stream_name in ("ephys",):
            # Check if any trial has media for this stream
            if sio.get_media(trials[0], stream_name) is not None:
                streams_to_draw.append((stream_name, stream_name, []))

    # Features: show if any trial has feature data variables
    has_features = False
    for trial_id in trials:
        try:
            ds = dt.trial(trial_id)
            if any(ds[v].attrs.get("type") == "features" for v in ds.data_vars):
                has_features = True
                break
        except (KeyError, ValueError):
            pass
    if has_features:
        streams_to_draw.append(("features", "features", []))

    if not streams_to_draw:
        streams_to_draw.append(("(no streams)", "features", []))

    n_rows = len(streams_to_draw)
    rows_rev = list(reversed(streams_to_draw))
    y_ticks = [(i + 0.5, rows_rev[i][0]) for i in range(n_rows)]
    plot.getAxis("left").setTicks([y_ticks])
    plot.setYRange(-0.2, n_rows + 0.2)

    max_t = 0.0
    artificial_gaps = False

    # --- Detect aligned-to-trial mode ---
    # Check app_state flag first, then infer from NWB data.
    # True only when ALL file modalities are aligned (not just some).
    is_aligned_mode = False
    app_state = getattr(dt, "_app_state", None)
    if app_state is not None:
        is_aligned_mode = getattr(app_state, "files_aligned_to_trials", False)
    else:
        # Infer: all trials same start_time and no real stop_time
        starts = [sio.start_time(t) for t in trials]
        stops = [sio.stop_time(t) for t in trials]
        all_same_start = len(set(starts)) <= 1
        no_real_stops = all(s is None for s in stops)
        is_aligned_mode = all_same_start and no_real_stops and len(trials) > 1

    # For aligned mode, estimate per-trial durations from media files
    trial_layout: dict[int | str, tuple[float, float]] = {}  # trial -> (x_start, x_end)

    if is_aligned_mode:
        artificial_gaps = True
        # Collect file durations per trial from the first video/pose/audio stream
        trial_durations: list[float] = []
        for trial_id in trials:
            dur = _estimate_trial_duration_from_nwb(dt, sio, trial_id)
            trial_durations.append(dur if dur and dur > 0 else 10.0)

        gap = 0.5  # artificial gap between trials in seconds
        cum = 0.0
        for i, trial_id in enumerate(trials):
            trial_layout[trial_id] = (cum, cum + trial_durations[i])
            cum += trial_durations[i] + gap
    else:
        for trial_id in trials:
            t0 = sio.start_time(trial_id)
            t1 = sio.stop_time(trial_id)
            if t1 is None:
                t1 = t0 + 10.0
            trial_layout[trial_id] = (t0, t1)

    # --- Trial boundary lines ---
    for trial_id in trials:
        t0, t1 = trial_layout.get(trial_id, (0.0, 1.0))

        line = pg.InfiniteLine(
            pos=t0, angle=90,
            pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
        )
        plot.addItem(line)
        items.append(line)

        mid = (t0 + t1) / 2
        lbl = pg.TextItem(str(trial_id), color="#aaaaaa", anchor=(0.5, 1.0))
        lbl.setPos(mid, n_rows + 0.1)
        plot.addItem(lbl)
        items.append(lbl)

        max_t = max(max_t, t1)

    # --- Stream bars ---
    nwb = None
    from ethograph.io.session_io import NWBSessionIO
    if isinstance(sio, NWBSessionIO):
        nwb = sio.nwb

    for row_idx, (label, stream_name, devices) in enumerate(rows_rev):
        color = pg.mkColor(MODALITY_COLORS.get(stream_name, "#888888"))
        color.setAlpha(160)
        bar_brush = pg.mkBrush(color)
        bar_pen = pg.mkPen(color.lighter(130), width=1)
        y_base = row_idx

        # Check if this stream has per-trial media entries
        has_per_trial_media = False
        if devices:
            has_per_trial_media = sio.get_media(trials[0], stream_name, devices[0]) is not None
        else:
            has_per_trial_media = sio.get_media(trials[0], stream_name) is not None

        if stream_name == "features":
            has_per_trial_media = True

        if has_per_trial_media:
            # Per-trial: draw one bar per trial using the layout
            for trial_id in trials:
                t_start, t_end = trial_layout.get(trial_id, (0.0, 1.0))
                bar = _make_rounded_bar(
                    t_start, t_end, y_base + 0.3, y_base + 0.7,
                    bar_brush, bar_pen,
                )
                plot.addItem(bar)
                items.append(bar)
                max_t = max(max_t, t_end)
        else:
            # Session-wide: use acquisition timestamps if available
            for dev in (devices or [None]):
                t_start = 0.0
                t_end = max_t if max_t > 0 else 60.0

                acq_name = f"{stream_name}_{dev}" if dev else stream_name
                if nwb and acq_name in nwb.acquisition:
                    acq = nwb.acquisition[acq_name]
                    from ethograph.utils.nwb import resolve_timeseries_timing
                    try:
                        rate, acq_t0 = resolve_timeseries_timing(acq)
                        t_start = acq_t0
                        ts = getattr(acq, "timestamps", None)
                        if ts is not None and len(ts) > 0:
                            t_end = float(ts[-1])
                        else:
                            n_samples = acq.data.shape[0] if hasattr(acq.data, "shape") else len(acq.data)
                            t_end = acq_t0 + n_samples / rate
                    except (ValueError, TypeError):
                        pass

                if t_end > t_start:
                    bar = _make_rounded_bar(
                        t_start, t_end, y_base + 0.3, y_base + 0.7,
                        bar_brush, bar_pen,
                    )
                    plot.addItem(bar)
                    items.append(bar)
                    max_t = max(max_t, t_end)

    # --- Note about artificial gaps ---
    if artificial_gaps:
        note = pg.TextItem(
            "Note: gaps between trials are artificial (files aligned to trials)",
            color="#888888", anchor=(0.0, 0.0),
        )
        note.setPos(0, -0.5)
        plot.addItem(note)
        items.append(note)

    total = max(max_t, 1.0)
    plot.setXRange(0, min(total, 120), padding=0.02)
    return total


def _estimate_trial_duration_from_nwb(dt, sio, trial_id) -> float | None:
    """Estimate a trial's duration from its media files or xarray data."""
    from ethograph.utils.stream_durations import (
        get_video_duration, get_audio_duration, get_pose_duration,
    )

    # Try video
    for cam in sio.cameras:
        filename = sio.get_media(trial_id, "video", cam)
        if filename:
            # Resolve path: check common folders
            for folder_attr in ("video_folder",):
                app_state = getattr(dt, "_app_state", None)
                folder = getattr(app_state, folder_attr, None) if app_state else None
                if folder:
                    import os
                    full = os.path.join(folder, filename)
                    if os.path.isfile(full):
                        dur = get_video_duration(full)
                        if dur:
                            return dur
            # Try filename as-is or relative to source path
            source = getattr(dt, "_source_path", None)
            if source:
                import os
                full = os.path.join(os.path.dirname(source), filename)
                if os.path.isfile(full):
                    dur = get_video_duration(full)
                    if dur:
                        return dur

    # Try pose
    fps = dt.get_video_fps() or 30.0
    for cam in sio.devices("pose"):
        filename = sio.get_media(trial_id, "pose", cam)
        if filename:
            source = getattr(dt, "_source_path", None)
            if source:
                import os
                full = os.path.join(os.path.dirname(source), filename)
                if os.path.isfile(full):
                    dur = get_pose_duration(full, fps)
                    if dur:
                        return dur

    # Try xarray feature timestamps
    try:
        from ethograph.utils.xr_utils import get_time_coord
        ds = dt.trial(trial_id)
        for var_name in ds.data_vars:
            tc = get_time_coord(ds[var_name])
            if tc is not None:
                vals = tc.values
                if len(vals) > 1:
                    return float(vals[-1]) - float(vals[0])
    except (KeyError, ValueError):
        pass

    return None


def _compute_file_durations(state: WizardState) -> dict[str, dict[str, float]]:
    durations: dict[str, dict[str, float]] = {}

    for name in ["video", "pose", "audio", "ephys"]:
        cfg: ModalityConfig = getattr(state, name)
        if not cfg.enabled:
            continue
        durs: dict[str, float] = {}

        if cfg.pattern and cfg.pattern.files:
            files = cfg.pattern.files
        elif cfg.single_file_path:
            files = [Path(cfg.single_file_path)]
        else:
            continue

        for f in files:
            fp = str(f)
            dur = None
            if name == "video":
                dur = get_video_duration(fp)
            elif name == "audio":
                dur = get_audio_duration(fp)
            elif name == "pose":
                dur = get_pose_duration(fp, cfg.fps)
            elif name == "ephys":
                dur = get_ephys_duration(fp)
            if dur is not None:
                durs[fp] = dur

        if durs:
            durations[name] = durs

    return durations



def _normalize_trial_key(value: object) -> object:
    """Normalize trial identifiers so numeric strings match integer trial IDs."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        return s
    return value


class TimelinePage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 4 — Temporal alignment</b>"))

        # Tab widget for visualization and code
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 24px;
                min-width: 120px;
            }
        """)

        # Tab 1: Timeline visualization
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        viz_layout.addWidget(QLabel(
            "Review how your files align in time. "
            "Colored bars show file durations; dotted lines mark trial boundaries."
        ))
        viz_layout.addSpacing(4)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1a1d21")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.getAxis("left").setTicks([])
        viz_layout.addWidget(self._plot, stretch=1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Pan:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider)
        viz_layout.addLayout(slider_row)

        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet("font-size: 11px; color: #555; padding: 2px 4px;")
        self._note_label.hide()
        viz_layout.addWidget(self._note_label)

        self._tabs.addTab(viz_tab, "1: Visualization")

        # Tab 2: Python code
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        code_layout.addWidget(QLabel(
            "Executable Python code that reproduces your alignment setup. "
            "Copy this code to customize or debug your workflow."
        ))
        code_layout.addSpacing(4)

        self._code_editor = QPlainTextEdit()
        self._code_editor.setReadOnly(True)
        self._code_editor.setFont(QFont("Consolas", 10))
        self._code_editor.setStyleSheet(
            "QPlainTextEdit { "
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3e3e3e; "
            "}"
        )
        self._code_highlighter = PythonCodeHighlighter(self._code_editor.document())
        code_layout.addWidget(self._code_editor, stretch=1)

        code_btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy code to clipboard")
        copy_btn.clicked.connect(self._on_copy_code)
        code_btn_row.addWidget(copy_btn)

        editor_btn = QPushButton("Open in code editor")
        editor_btn.clicked.connect(self._on_open_in_editor)
        code_btn_row.addWidget(editor_btn)
        code_btn_row.addStretch()
        code_layout.addLayout(code_btn_row)

        self._tabs.addTab(code_tab, "2: Python code")

        layout.addWidget(self._tabs, stretch=1)

        layout.addSpacing(8)

        # Output path (shared across tabs)
        self._out_widget = QWidget()
        _out_row = QHBoxLayout(self._out_widget)
        _out_row.setContentsMargins(0, 0, 0, 0)
        _out_row.addWidget(QLabel("Output path:"))
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output location for trials.nc...")
        self._output_edit.setReadOnly(True)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output)
        _out_row.addWidget(self._output_edit)
        _out_row.addWidget(out_browse)
        layout.addWidget(self._out_widget)

        self._total_duration = 1.0
        self._items: list = []
        self._state: WizardState | None = None

    def populate_from_state(self, state: WizardState):
        self._state = state
        self._clear()
        self._regenerate_code()

        durations = _compute_file_durations(state)
        state.file_durations = durations

        enabled_modalities = [
            name for name in ["video", "pose", "audio", "ephys"]
            if getattr(state, name).enabled
        ]

        # Build rows: each (label, modality_name, device_name_or_None)
        rows: list[tuple[str, str, str | None]] = []
        for name in enabled_modalities:
            devices = self._get_devices(name, state)
            if devices:
                for dev in devices:
                    rows.append((f"{name}: {dev}", name, dev))
            else:
                rows.append((name.capitalize(), name, None))

        n_rows = len(rows)
        rows_reversed = list(reversed(rows))
        y_ticks = [(i + 0.5, rows_reversed[i][0]) for i in range(n_rows)]
        self._plot.getAxis("left").setTicks([y_ticks])
        self._plot.setYRange(-0.2, n_rows + 0.2)

        max_time = 0.0

        # Build per-trial cumulative offsets for aligned mode
        trial_cum_offsets: dict[int, float] = {}
        trial_durs: list[float] = []
        if state.files_aligned_to_trials and state.trial_table is not None:
            n_trials = len(state.trial_table)

            if state.is_fully_aligned() and {
                "start_time", "stop_time",
            }.issubset(state.trial_table.columns):
                starts = pd.to_numeric(state.trial_table["start_time"], errors="coerce")
                stops = pd.to_numeric(state.trial_table["stop_time"], errors="coerce")
                table_durs = (stops - starts).to_numpy(dtype=float)
                trial_durs = [float(d) for d in table_durs if np.isfinite(d) and d > 0]

            if len(trial_durs) != n_trials:
                per_trial: list[list[float]] = [[] for _ in range(n_trials)]
                for mod_name in ["video", "audio", "pose", "ephys"]:
                    mod_values = list(durations.get(mod_name, {}).values())
                    if not mod_values:
                        continue
                    last = float(mod_values[-1])
                    for i in range(n_trials):
                        d = float(mod_values[i]) if i < len(mod_values) else last
                        if np.isfinite(d) and d > 0:
                            per_trial[i].append(d)

                trial_durs = [max(vals) if vals else 0.0 for vals in per_trial]

            cum = 0.0
            for i, d in enumerate(trial_durs):
                trial_cum_offsets[i] = cum
                cum += d

        trial_index_by_key: dict[object, int] = {}
        trial_start_by_index: dict[int, float] = {}
        if state.trial_table is not None and "trial" in state.trial_table.columns:
            for idx, tid in enumerate(state.trial_table["trial"].tolist()):
                trial_index_by_key[tid] = idx
                trial_index_by_key[str(tid)] = idx
                norm_tid = _normalize_trial_key(tid)
                if norm_tid is not None:
                    trial_index_by_key[norm_tid] = idx

            if "start_time" in state.trial_table.columns:
                starts = pd.to_numeric(state.trial_table["start_time"], errors="coerce")
                for idx, t0 in enumerate(starts.tolist()):
                    if np.isfinite(t0):
                        trial_start_by_index[idx] = float(t0)

        # Group files by device
        file_device_map: dict[str, dict[str, str]] = {}
        file_trial_index_map: dict[str, dict[str, int]] = {}
        for name in enabled_modalities:
            cfg: ModalityConfig = getattr(state, name)
            if cfg.pattern and cfg.pattern.files:
                dev_role = "mic" if name == "audio" else "camera"
                summary = cfg.pattern.summary()
                if dev_role in summary:
                    mapping: dict[str, str] = {}
                    trial_mapping: dict[str, int] = {}
                    for f in cfg.pattern.files:
                        row_data = extract_file_row(
                            f, cfg.pattern.segments, cfg.pattern.tokenize_mode,
                        )
                        fp = str(f)
                        mapping[fp] = row_data.get(dev_role, "")
                        trial_val = _normalize_trial_key(row_data.get("trial"))
                        if trial_val is not None:
                            idx = trial_index_by_key.get(trial_val)
                            if idx is None:
                                idx = trial_index_by_key.get(str(trial_val))
                            if idx is not None:
                                trial_mapping[fp] = idx
                    file_device_map[name] = mapping
                    if trial_mapping:
                        file_trial_index_map[name] = trial_mapping

        # Draw file bars
        for row_idx, (label, name, device) in enumerate(rows_reversed):
            cfg: ModalityConfig = getattr(state, name)
            color = pg.mkColor(MODALITY_COLORS.get(name, "#888888"))
            color.setAlpha(160)
            y_base = row_idx
            offset = cfg.constant_offset

            mod_durs = durations.get(name, {})
            if device is not None and name in file_device_map:
                dev_map = file_device_map[name]
                filtered = {fp: dur for fp, dur in mod_durs.items() if dev_map.get(fp) == device}
            else:
                filtered = mod_durs

            bar_pen = pg.mkPen(color.lighter(130), width=1)
            bar_brush = pg.mkBrush(color)

            if state.files_aligned_to_trials and trial_cum_offsets:
                cum = 0.0
                for i, (filepath, dur) in enumerate(filtered.items()):
                    x_start = offset + trial_cum_offsets.get(i, cum)
                    bar = _make_rounded_bar(
                        x_start, x_start + dur,
                        y_base + 0.3, y_base + 0.7,
                        bar_brush, bar_pen,
                    )
                    self._plot.addItem(bar)
                    self._items.append(bar)
                    end = x_start + dur
                    if end > max_time:
                        max_time = end
                    cum += dur
            else:
                aligned_cum = 0.0
                for filepath, dur in filtered.items():
                    x_start = offset
                    if cfg.file_mode == "aligned_to_trial":
                        trial_idx = file_trial_index_map.get(name, {}).get(filepath)
                        if trial_idx is not None:
                            if trial_start_by_index:
                                x_start = offset + trial_start_by_index.get(trial_idx, 0.0)
                            elif trial_cum_offsets:
                                x_start = offset + trial_cum_offsets.get(trial_idx, aligned_cum)
                            else:
                                x_start = offset + aligned_cum
                        else:
                            x_start = offset + aligned_cum

                    bar = _make_rounded_bar(
                        x_start, x_start + dur,
                        y_base + 0.3, y_base + 0.7,
                        bar_brush, bar_pen,
                    )
                    self._plot.addItem(bar)
                    self._items.append(bar)
                    end = x_start + dur
                    if end > max_time:
                        max_time = end
                    if cfg.file_mode == "aligned_to_trial":
                        aligned_cum += dur

        # Draw trial boundaries
        if state.trial_table is not None and "trial" in state.trial_table.columns:
            trial_ids = state.trial_table["trial"].tolist()
            if state.files_aligned_to_trials and trial_durs:
                cum = 0.0
                for i, tid in enumerate(trial_ids):
                    d = trial_durs[i] if i < len(trial_durs) else trial_durs[-1]
                    line = pg.InfiniteLine(
                        pos=cum, angle=90,
                        pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
                    )
                    self._plot.addItem(line)
                    self._items.append(line)

                    label = pg.TextItem(str(tid), color="#aaaaaa", anchor=(0.5, 1.0))
                    label.setPos(cum + d / 2, n_rows + 0.1)
                    self._plot.addItem(label)
                    self._items.append(label)
                    cum += d

            elif "start_time" in state.trial_table.columns:
                for _, row in state.trial_table.iterrows():
                    t0 = float(row["start_time"])
                    line = pg.InfiniteLine(
                        pos=t0, angle=90,
                        pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
                    )
                    self._plot.addItem(line)
                    self._items.append(line)

                    tid = row.get("trial", "")
                    label = pg.TextItem(str(tid), color="#aaaaaa", anchor=(0.5, 1.0))
                    label.setPos(t0, n_rows + 0.1)
                    self._plot.addItem(label)
                    self._items.append(label)

                    if "stop_time" in state.trial_table.columns:
                        t1 = float(row["stop_time"])
                        if t1 > max_time:
                            max_time = t1

        self._total_duration = max(max_time, 1.0)
        self._plot.setXRange(0, min(self._total_duration, 120), padding=0.02)

    def _get_devices(self, name: str, state: WizardState) -> list[str]:
        if name == "video" and state.camera_names:
            return state.camera_names
        if name == "audio" and state.mic_names:
            return state.mic_names
        cfg: ModalityConfig = getattr(state, name)
        if cfg.pattern:
            dev_role = "mic" if name == "audio" else "camera"
            summary = cfg.pattern.summary()
            if dev_role in summary:
                return summary[dev_role]
        return []

    def _clear(self):
        for item in self._items:
            self._plot.removeItem(item)
        self._items.clear()

    def _on_slider(self, value: int):
        frac = value / 1000.0
        window = min(self._total_duration, 120)
        center = frac * (self._total_duration - window) + window / 2
        self._plot.setXRange(center - window / 2, center + window / 2, padding=0)

    def collect_state(self, state: WizardState):
        state.output_path = self._output_edit.text()

    def _browse_output(self):
        result = QFileDialog.getSaveFileName(
            self, "Save dataset",
            "trials.nc",
            "NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self._output_edit.setText(path)
            if self._state:
                self._state.output_path = path
                self._regenerate_code()

    def _regenerate_code(self):
        """Generate and display Python code for current state."""
        if self._state is None:
            return
        code = generate_alignment_code(self._state)
        self._code_editor.setPlainText(code)

    def _on_copy_code(self):
        """Copy generated code to clipboard."""
        code = self._code_editor.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(code)

    def _on_open_in_editor(self):
        """Save code to .ethograph folder and open in user's code editor."""
        from datetime import datetime

        code = self._code_editor.toPlainText()
        if not code.strip():
            return

        notebook = _code_to_notebook(code)

        wizard_dir = Path.home() / ".ethograph" / "alignment_wizard"
        wizard_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = wizard_dir / f"ethograph_alignment_setup_{timestamp}.ipynb"

        output_file.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

        _do_open_source(str(output_file), self)

    # ------------------------------------------------------------------
    # Standalone (non-wizard) entry point
    # ------------------------------------------------------------------

    def configure_for_standalone(self):
        """Hide wizard-specific controls; call before populate_from_trialtree()."""
        self._out_widget.hide()

    def populate_from_trialtree(self, dt, app_state):
        """Populate timeline from a loaded TrialTree's NWB session data.

        Uses ``dt.session_io`` to read trial timing and acquisition
        metadata for the timeline visualization.
        """
        self._clear()
        self._state = None
        self._code_editor.setPlainText(
            "# Open via the New Dataset Wizard to generate alignment code."
        )

        self._total_duration = draw_session_timeline(
            self._plot, dt, items_out=self._items,
        )

    @staticmethod
    def _trial_has_modality(dt, trial_id, modality: str, device, ds) -> bool:
        if modality == "features":
            return ds is not None and bool(ds.data_vars)
        if modality == "ephys":
            return True
        if device is None:
            return False
        try:
            return bool(dt.get_media(trial_id, modality, device))
        except (KeyError, IndexError):
            return False

    @staticmethod
    def _has_pose_data(dt, cam: str) -> bool:
        for trial_id in (dt.trials or [])[:5]:
            try:
                if dt.get_media(trial_id, "pose", cam):
                    return True
            except (KeyError, IndexError):
                pass
        return False

    @staticmethod
    def _get_end_source(dt, trial_id, ds, alignment: TrialAlignment | None) -> str:
        session_io = getattr(dt, "session_io", None)
        if session_io is not None:
            try:
                if session_io.stop_time(trial_id) is not None:
                    return "session stop_time"
            except (KeyError, AttributeError):
                pass
        if ds is not None:
            for var_name in ds.data_vars:
                da = ds[var_name]
                if da.attrs.get("type", "") in ("features", "colors", ""):
                    tc = get_time_coord(da)
                    if tc is not None:
                        vals = getattr(tc, "values", tc)
                        if len(vals) > 0 and float(vals[-1]) > 0:
                            return "feature last timestamp"
        if alignment is not None and alignment.trial_range is not None:
            return "video/audio file length"
        return "unknown (10 s placeholder)"

    def _update_note(self, end_sources: list[tuple[str, str]]):
        src_count = Counter(src for _, src in end_sources)
        src_colors = {
            "session stop_time":       "#2a8a2a",
            "feature last timestamp":  "#2255cc",
            "video/audio file length": "#aa6600",
            "unknown (10 s placeholder)": "#cc4400",
        }
        parts = []
        for src, n in src_count.most_common():
            c = src_colors.get(src, "#555")
            parts.append(f"<span style='color:{c}'>■</span>&nbsp;{n} trial(s):&nbsp;<b>{src}</b>")

        needs_fallback = any(s != "session stop_time" for _, s in end_sources)
        priority = ""
        if needs_fallback:
            priority = (
                "<br><span style='color:#888; font-size:10px;'>"
                "Trial end priority: "
                "(1)&nbsp;session&nbsp;stop_time &rarr; "
                "(2)&nbsp;feature&nbsp;last&nbsp;timestamp &rarr; "
                "(3)&nbsp;video/audio&nbsp;file&nbsp;length."
                "</span>"
            )

        self._note_label.setText(
            "Trial end: &nbsp;" + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts) + priority
        )
        self._note_label.show()
