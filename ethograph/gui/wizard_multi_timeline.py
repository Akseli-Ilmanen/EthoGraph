"""Temporal alignment timeline visualization for the NC wizard (Page 4)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
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
from ethograph.utils.xr_utils import get_time_coord

# Colors per modality (matching dialog_media_files.py palette)
MODALITY_COLORS = {
    "video": "#50c8b4",
    "pose": "#e8737a",
    "audio": "#e8c75a",
    "ephys": "#b07ae8",
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
    """Convert Python code with section markers to Jupyter notebook format.
    
    Splits code into cells based on '# ─── N. Section name ───' markers.
    """
    import re
    
    # Split by section markers
    section_pattern = r'^# ─── \d+\. .+ ───$'
    lines = code.split('\n')
    
    cells = []
    current_cell_lines = []
    
    for line in lines:
        if re.match(section_pattern, line):
            # Save previous cell if not empty
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
            # Start new cell with section marker as heading comment
            current_cell_lines = [line]
        else:
            current_cell_lines.append(line)
    
    # Add last cell
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
    
    # Create notebook structure
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
# Duration calculation helpers
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


def _get_video_duration(path: str) -> float | None:
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if stream.frames and stream.average_rate:
                return stream.frames / float(stream.average_rate)
    except Exception:
        pass
    return None


def _get_audio_duration(path: str) -> float | None:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:
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
                dur = _get_video_duration(fp)
            elif name == "audio":
                dur = _get_audio_duration(fp)
            elif name == "pose":
                dur = _estimate_pose_duration(fp, cfg)
            elif name == "ephys":
                dur = _get_ephys_duration(fp, cfg)
            if dur is not None:
                durs[fp] = dur

        if durs:
            durations[name] = durs

    return durations

def _count_csv_headers(path: str) -> int:
    """
    Dynamically count CSV header rows by detecting where data starts.
    Works for DLC (3-4 headers), LightningPose (3 headers), SLEAP (1 header).
    """
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains numeric data (data row)
            parts = line.split(',')
            if len(parts) < 2:
                continue
            
            try:
                # Try to parse as floats (skip first column which is frame index)
                float(parts[0])  # Frame index
                float(parts[1])  # First coordinate value
                # Found data row, so header count is current line number
                return i
            except (ValueError, IndexError):
                # Still in header section
                continue
    
    # Default to 1 if we can't determine
    return 1


def _estimate_pose_duration(path: str, cfg: ModalityConfig) -> float | None:
    try:
        suffix = Path(path).suffix.lower()
        n_frames = None
        
        if suffix == ".csv":
            # Dynamically detect header count
            n_headers = _count_csv_headers(path)
            # Count total lines and subtract headers
            with open(path, 'r') as fh:
                n_frames = sum(1 for _ in fh) - n_headers
 
        elif suffix in (".h5", ".hdf5", ".slp"):
            import h5py
            with h5py.File(path, "r") as f:
                if suffix == ".slp":
                    n_frames = f["instances"].shape[0]
                else:
                    # Generic HDF5/h5 handling - find first 2D dataset
                    for key in f.keys():
                        data = f[key]
                        if hasattr(data, 'shape') and len(data.shape) >= 2:
                            n_frames = data.shape[0]
                            break
            
        if n_frames is not None and n_frames > 0:
            return n_frames / cfg.fps
        else:
            return None

    except Exception as e:
        print(f"Could not estimate duration for pose file {path}: {e}")
        return None


def _get_ephys_duration(path: str, cfg: ModalityConfig) -> float | None:
    try:
        from ethograph.gui.plots_ephystrace import GenericEphysLoader
        loader = GenericEphysLoader(path)
        return len(loader) / loader.rate
    except Exception:
        return None


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
        # Increase tab header spacing
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

        # pyqtgraph plot
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1a1d21")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.getAxis("left").setTicks([])
        viz_layout.addWidget(self._plot, stretch=1)

        # Slider for panning
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
        
        # Button row for code tab
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

        # Output path (shared across tabs) — wrapped so configure_for_standalone() can hide it
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

            # Prefer explicit trial durations from table when available.
            if state.is_fully_aligned() and {
                "start_time", "stop_time",
            }.issubset(state.trial_table.columns):
                starts = pd.to_numeric(state.trial_table["start_time"], errors="coerce")
                stops = pd.to_numeric(state.trial_table["stop_time"], errors="coerce")
                table_durs = (stops - starts).to_numpy(dtype=float)
                trial_durs = [float(d) for d in table_durs if np.isfinite(d) and d > 0]

            # Otherwise derive trial duration from file durations by trial index.
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
            # Filter to this device's files
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
        """Save code to .ethograph folder in home directory and open in user's code editor."""
        from datetime import datetime

        code = self._code_editor.toPlainText()
        if not code.strip():
            return

        # Convert Python code to Jupyter notebook format
        notebook = _code_to_notebook(code)

        # Save to home/.ethograph directory
        wizard_dir = Path.home() / ".ethograph" / "alignment_wizard"
        wizard_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = wizard_dir / f"ethograph_alignment_setup_{timestamp}.ipynb"

        output_file.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

        # Use the editor selection dialog from dialog_function_params.py pattern
        _do_open_source(str(output_file), self)

    # ------------------------------------------------------------------
    # Standalone (non-wizard) entry point
    # ------------------------------------------------------------------

    def configure_for_standalone(self):
        """Hide wizard-specific controls; call before populate_from_trialtree()."""
        self._out_widget.hide()

    def populate_from_trialtree(self, dt, app_state):
        """Populate timeline from a loaded TrialTree.

        Uses :func:`compute_trial_alignment` (from ``plots_timeseriessource``) to
        derive each trial's absolute time range and available modalities.
        """
        self._clear()
        self._state = None
        self._code_editor.setPlainText(
            "# Open via the New Dataset Wizard to generate alignment code."
        )

        cameras = list(getattr(dt, "cameras", None) or [])
        mics = list(getattr(dt, "mics", None) or [])

        has_ephys = bool(getattr(app_state, "ephys_stream_sel", None))

        rows: list[tuple[str, str, str | None]] = []
        for cam in cameras:
            rows.append((f"video: {cam}", "video", cam))
        for cam in cameras:
            if self._has_pose_data(dt, cam):
                rows.append((f"pose: {cam}", "pose", cam))
        for mic in mics:
            rows.append((f"audio: {mic}", "audio", mic))
        if has_ephys:
            rows.append(("ephys", "ephys", None))
        rows.append(("features", "features", None))

        n_rows = len(rows)
        rows_rev = list(reversed(rows))
        y_ticks = [(i + 0.5, rows_rev[i][0]) for i in range(n_rows)]
        self._plot.getAxis("left").setTicks([y_ticks])
        self._plot.setYRange(-0.2, n_rows + 0.2)

        video_folder = getattr(app_state, "video_folder", None)
        audio_folder = getattr(app_state, "audio_folder", None)
        cameras_sel = getattr(app_state, "cameras_sel", None)

        end_sources: list[tuple[str, str]] = []
        max_t = 0.0
        t_cursor = 0.0

        for trial_id in dt.trials:
            ds = None
            try:
                ds = dt.trial(trial_id)
            except Exception:
                pass

            alignment: TrialAlignment | None = None
            try:
                alignment = compute_trial_alignment(
                    dt, trial_id, ds or xr.Dataset(),
                    video_folder=video_folder,
                    audio_folder=audio_folder,
                    cameras_sel=cameras_sel,
                )
                t_start = alignment.ephys_offset
                dur = alignment.trial_range.duration if alignment.trial_range else 0.0
            except Exception:
                t_start = t_cursor
                dur = 0.0

            t_end = t_start + dur if dur > 0 else t_start + 10.0
            t_cursor = t_end
            max_t = max(max_t, t_end)

            src = self._get_end_source(dt, trial_id, ds, alignment)
            end_sources.append((str(trial_id), src))

            for row_idx, (_label, modality, device) in enumerate(rows_rev):
                if not self._trial_has_modality(dt, trial_id, modality, device, ds):
                    continue
                y_base = row_idx
                color = pg.mkColor(MODALITY_COLORS.get(modality, "#888888"))
                color.setAlpha(160)
                bar = _make_rounded_bar(
                    t_start, t_end, y_base + 0.3, y_base + 0.7,
                    pg.mkBrush(color), pg.mkPen(color.lighter(130), width=1),
                )
                self._plot.addItem(bar)
                self._items.append(bar)

            line = pg.InfiniteLine(
                pos=t_start, angle=90,
                pen=pg.mkPen("#aaaaaa", width=1, style=Qt.PenStyle.DotLine),
            )
            self._plot.addItem(line)
            self._items.append(line)

            lbl = pg.TextItem(str(trial_id), color="#aaaaaa", anchor=(0.5, 1.0))
            lbl.setPos(t_start + (t_end - t_start) / 2, n_rows + 0.1)
            self._plot.addItem(lbl)
            self._items.append(lbl)

        self._total_duration = max(max_t, 1.0)
        self._plot.setXRange(0, min(self._total_duration, 120), padding=0.02)
        self._update_note(end_sources)

    @staticmethod
    def _trial_has_modality(dt, trial_id, modality: str, device, ds) -> bool:
        if modality == "features":
            return ds is not None and bool(ds.data_vars)
        if modality == "ephys":
            return True  # row only added when ephys_stream_sel is set
        if device is None:
            return False
        try:
            if modality == "video":
                return bool(dt.get_video(trial_id, device))
            if modality == "audio":
                return bool(dt.get_audio(trial_id, device))
            if modality == "pose":
                return bool(dt.get_pose(trial_id, device))
        except Exception:
            pass
        return False

    @staticmethod
    def _has_pose_data(dt, cam: str) -> bool:
        for trial_id in (dt.trials or [])[:5]:
            try:
                if dt.get_pose(trial_id, cam):
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _get_end_source(dt, trial_id, ds, alignment: TrialAlignment | None) -> str:
        session_io = getattr(dt, "session_io", None)
        if session_io is not None:
            try:
                if session_io.stop_time(trial_id) is not None:
                    return "session stop_time"
            except Exception:
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

