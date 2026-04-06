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
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.wizard_multi_codegen import generate_alignment_code
from ethograph.gui.dialog_function_params import _do_open_source
from ethograph.io.time_model import compute_trial_alignment, TimeRange, TrialAlignment
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
    """Draw a timeline from NWB acquisition ImageSeries.

    Each ImageSeries with ``external_file`` gets one row.  Per-file bars
    are drawn using ``starting_frame`` + ``timestamps`` (or ``rate``) to
    determine each file's time span.  Trial boundaries from the trials
    table are shown as dotted vertical lines.
    """
    from ethograph.io.nwb_alignment import NWBAlignment

    sio = dt.nwb_alignment
    items: list = items_out if items_out is not None else []

    nwb = sio.nwb if isinstance(sio, NWBAlignment) else None
    if nwb is None or not nwb.acquisition:
        return 1.0

    # Collect acquisition items with external files
    acq_items: list[tuple[str, str, Any]] = []  # (label, stream, series)
    for acq_name, series in nwb.acquisition.items():
        if not hasattr(series, "external_file") or not series.external_file:
            continue
        stream = acq_name.split("_", 1)[0] if "_" in acq_name else acq_name
        acq_items.append((acq_name, stream, series))

    if not acq_items:
        return 1.0

    n_rows = len(acq_items)
    rows_rev = list(reversed(acq_items))
    y_ticks = [(i + 0.5, rows_rev[i][0]) for i in range(n_rows)]
    plot.getAxis("left").setTicks([y_ticks])
    plot.setYRange(-0.2, n_rows + 0.2)

    max_t = 0.0

    # --- Draw per-file bars from ImageSeries timing ---
    for row_idx, (acq_name, stream, series) in enumerate(rows_rev):
        color = pg.mkColor(MODALITY_COLORS.get(stream, "#888888"))
        color.setAlpha(160)
        bar_brush = pg.mkBrush(color)
        bar_pen = pg.mkPen(color.lighter(130), width=1)
        y_base = row_idx

        files = list(series.external_file)
        sf = (
            [int(f) for f in series.starting_frame]
            if getattr(series, "starting_frame", None) is not None
            else [0] * len(files)
        )
        timestamps = getattr(series, "timestamps", None)
        rate = getattr(series, "rate", None)
        starting_time = float(series.starting_time) if getattr(series, "starting_time", None) is not None else 0.0

        for i in range(len(files)):
            frame_start = sf[i]
            frame_end = sf[i + 1] if i + 1 < len(files) else None

            if timestamps is not None and len(timestamps) > 0:
                ts = np.asarray(timestamps)
                t_start = float(ts[frame_start]) if frame_start < len(ts) else 0.0
                if frame_end is not None and frame_end < len(ts):
                    t_end = float(ts[frame_end - 1])
                else:
                    t_end = float(ts[-1])
            elif rate and rate > 0:
                t_start = starting_time + frame_start / rate
                if frame_end is not None:
                    t_end = starting_time + frame_end / rate
                else:
                    t_end = t_start + 1.0
            else:
                continue

            if t_end <= t_start:
                continue

            bar = _make_rounded_bar(
                t_start, t_end, y_base + 0.3, y_base + 0.7,
                bar_brush, bar_pen,
            )
            plot.addItem(bar)
            items.append(bar)
            max_t = max(max_t, t_end)

    # --- Trial boundary lines ---
    try:
        trials = dt.trials
    except ValueError:
        trials = []

    for trial_id in trials:
        t0 = sio.start_time(trial_id)
        line = pg.InfiniteLine(
            pos=t0, angle=90,
            pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
        )
        plot.addItem(line)
        items.append(line)

        t1 = sio.stop_time(trial_id)
        mid = (t0 + (t1 or t0 + 1.0)) / 2
        lbl = pg.TextItem(str(trial_id), color="#aaaaaa", anchor=(0.5, 1.0))
        lbl.setPos(mid, n_rows + 0.1)
        plot.addItem(lbl)
        items.append(lbl)

        if t1:
            max_t = max(max_t, t1)

    total = max(max_t, 1.0)
    plot.setXRange(0, min(total, 120), padding=0.02)
    return total



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

        # Tab 1: Visualization (stacked: table for aligned, timeline for unaligned)
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        self._viz_stack = QStackedWidget()

        # --- Page 0: Aligned table view ---
        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.addWidget(QLabel(
            "All files are aligned to trials. Each row shows which files belong to each trial."
        ))
        self._aligned_table = QTableWidget()
        self._aligned_table.setStyleSheet(
            "QTableWidget { background-color: #1a1d21; color: #d4d4d4; gridline-color: #3e3e3e; }"
            "QHeaderView::section { background-color: #2d2d2d; color: #d4d4d4; padding: 4px; }"
        )
        self._aligned_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self._aligned_table, stretch=1)
        self._viz_stack.addWidget(table_page)

        # --- Page 1: Timeline view ---
        timeline_page = QWidget()
        timeline_layout = QVBoxLayout(timeline_page)
        timeline_layout.addWidget(QLabel(
            "Review how your files align in time. "
            "Colored bars show file durations; dotted lines mark trial boundaries."
        ))
        timeline_layout.addSpacing(4)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1a1d21")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.getAxis("left").setTicks([])
        timeline_layout.addWidget(self._plot, stretch=1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Pan:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider)
        timeline_layout.addLayout(slider_row)
        self._viz_stack.addWidget(timeline_page)

        viz_layout.addWidget(self._viz_stack, stretch=1)

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

        # Switch view mode: table for aligned, timeline for unaligned
        if state.files_aligned_to_trials:
            self._viz_stack.setCurrentIndex(0)
            self._populate_aligned_table(state)
            return
        self._viz_stack.setCurrentIndex(1)

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
                            regex_pattern=cfg.pattern.regex_pattern,
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

    # ------------------------------------------------------------------
    # Aligned table view
    # ------------------------------------------------------------------

    def _populate_aligned_table(self, state: WizardState):
        """Build a table: rows=trials, columns=stream_device, cells=filenames."""
        if state.trial_table is None:
            return

        trial_ids = state.trial_table["trial"].tolist() if "trial" in state.trial_table.columns else []
        if not trial_ids:
            return

        # Collect stream_device columns from the trial table
        stream_cols = [
            c for c in state.trial_table.columns
            if c not in ("trial", "start_time", "stop_time")
            and not c.endswith("_start")
        ]

        self._aligned_table.setRowCount(len(trial_ids))
        self._aligned_table.setColumnCount(len(stream_cols) + 1)
        self._aligned_table.setHorizontalHeaderLabels(["Trial"] + stream_cols)

        for row, trial_id in enumerate(trial_ids):
            self._aligned_table.setItem(row, 0, QTableWidgetItem(str(trial_id)))
            for col_idx, col_name in enumerate(stream_cols):
                val = state.trial_table.iloc[row].get(col_name, "")
                cell = Path(str(val)).name if val else ""
                self._aligned_table.setItem(row, col_idx + 1, QTableWidgetItem(cell))

        self._aligned_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

    def _populate_aligned_table_from_dt(self, dt):
        """Build a table from TrialTree's NWB session trials_df."""
        sio = dt.nwb_alignment
        df = sio.trials_df
        if df.empty:
            return

        trial_col = df["trial"].tolist() if "trial" in df.columns else list(range(1, len(df) + 1))

        stream_cols = [
            c for c in df.columns
            if c not in ("trial", "start_time", "stop_time")
            and not c.endswith("_start")
        ]

        self._aligned_table.setRowCount(len(trial_col))
        self._aligned_table.setColumnCount(len(stream_cols) + 1)
        self._aligned_table.setHorizontalHeaderLabels(["Trial"] + stream_cols)

        for row, trial_id in enumerate(trial_col):
            self._aligned_table.setItem(row, 0, QTableWidgetItem(str(trial_id)))
            for col_idx, col_name in enumerate(stream_cols):
                val = df.iloc[row].get(col_name, "")
                cell = Path(str(val)).name if val else ""
                self._aligned_table.setItem(row, col_idx + 1, QTableWidgetItem(cell))

        self._aligned_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

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
        """Populate from a loaded TrialTree's NWB session data.

        Aligned mode (table view): trials table has ``{stream}_{device}``
        filename columns.
        Unaligned mode (timeline view): filenames are in acquisition
        ImageSeries, trials table has only timing.
        """
        self._clear()
        self._state = None
        self._code_editor.setPlainText(
            "# Open via the New Dataset Wizard to generate alignment code."
        )

        # Detect mode: table has filename columns → aligned, otherwise → timeline
        sio = dt.nwb_alignment
        df = sio.trials_df
        has_filename_cols = any(
            col not in ("trial", "start_time", "stop_time")
            and not col.endswith("_start")
            for col in df.columns
        ) if not df.empty else False

        if has_filename_cols:
            self._viz_stack.setCurrentIndex(0)
            self._populate_aligned_table_from_dt(dt)
        else:
            self._viz_stack.setCurrentIndex(1)
            self._total_duration = draw_session_timeline(
                self._plot, dt, items_out=self._items,
            )



    @staticmethod
    def _has_pose_data(dt, cam: str) -> bool:
        for trial_id in (dt.trials or [])[:5]:
            try:
                if dt.nwb_alignment.get_media(trial_id, "pose", cam):
                    return True
            except (KeyError, IndexError):
                pass
        return False

    @staticmethod
    def _get_end_source(dt, trial_id, ds, alignment: TrialAlignment | None) -> str:
        nwb_alignment = getattr(dt, "nwb_alignment", None)
        if nwb_alignment is not None:
            try:
                if nwb_alignment.stop_time(trial_id) is not None:
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
