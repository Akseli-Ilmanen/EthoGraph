"""
Napari dock widget for media file discovery and pattern matching.

Run standalone:  python media_discovery_widget.py
Dock in napari:  viewer.window.add_dock_widget(MediaDiscoveryWidget(viewer))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.utils.validation import (
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    POSE_EXTENSIONS,
)

BG = "#1a1d21"
BG_PANEL = "#22262c"
BG_INPUT = "#2c3038"
BORDER = "#383e4a"
TEXT = "#e0e0e0"
TEXT_MID = "#9aa0ac"
TEXT_DIM = "#5e6470"
ACCENT = "#50c8b4"

COLOR_TRIAL = "#50c8b4"
COLOR_CAMERA = "#e8737a"
COLOR_MIC = "#e8c75a"

ROLE_COLORS = {
    "trial": COLOR_TRIAL,
    "camera": COLOR_CAMERA,
    "mic": COLOR_MIC,
    "ignore": TEXT_DIM,
    None: TEXT_DIM,
}

STREAM_ROLES: dict[str, list[str]] = {
    "video": ["ignore", "trial", "camera"],
    "pose": ["ignore", "trial", "camera"],
    "audio": ["ignore", "trial", "mic"],
}

# Convert extension sets to regex patterns for classification
STREAM_RULES: dict[str, list[str]] = {
    "video": [rf"\{ext}$" for ext in VIDEO_EXTENSIONS],
    "pose": [rf"\{ext}$" for ext in POSE_EXTENSIONS],
    "audio": [rf"\{ext}$" for ext in AUDIO_EXTENSIONS],
}

FS = 13
MAX_PREVIEW = 20
FOLDER_POSITION = -1


def _mono(size: int = FS) -> QFont:
    f = QFont("Menlo")
    f.setPointSize(size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


# ─── pattern analysis ────────────────────────────────────────────────────────


@dataclass
class Segment:
    position: int
    text: str
    varying: bool
    values: list[str] = field(default_factory=list)
    role: str | None = None


@dataclass
class FilePattern:
    segments: list[Segment]
    files: list[Path]
    suffix: str
    tokenize_mode: str = "smart"

    def summary(self) -> dict[str, list[str]]:
        return {
            s.role: s.values
            for s in self.segments
            if s.varying and s.role and s.role != "ignore"
        }


def _auto_detect_role(seg: Segment) -> str | None:
    """Auto-detect role from segment values.
    
    Returns 'camera' if values look like cam-X, camera-X
    Returns 'mic' if values look like mic-X, microphone-X
    Returns 'trial' if values look like trial-X
    Returns None otherwise
    """
    if not seg.values:
        return None
    
    # Check if all values match camera patterns
    cam_patterns = [
        re.compile(r'^cam[-_]?\d+$', re.IGNORECASE),
        re.compile(r'^camera[-_]?\d+$', re.IGNORECASE),
        re.compile(r'^cam$', re.IGNORECASE),
        re.compile(r'^camera$', re.IGNORECASE),
    ]
    
    if all(any(p.match(v) for p in cam_patterns) for v in seg.values):
        return "camera"
    
    # Check if all values match mic patterns
    mic_patterns = [
        re.compile(r'^mic[-_]?\d+$', re.IGNORECASE),
        re.compile(r'^microphone[-_]?\d+$', re.IGNORECASE),
        re.compile(r'^mic$', re.IGNORECASE),
        re.compile(r'^microphone$', re.IGNORECASE),
    ]
    
    if all(any(p.match(v) for p in mic_patterns) for v in seg.values):
        return "mic"
    
    # Check if all values match trial patterns
    trial_patterns = [
        re.compile(r'^trial[-_]?\d+$', re.IGNORECASE),
        re.compile(r'^trial$', re.IGNORECASE),
    ]
    
    if all(any(p.match(v) for p in trial_patterns) for v in seg.values):
        return "trial"
    
    # Heuristic: small number of short numeric values (1-2 digits) likely indicates camera/mic IDs
    # e.g., ["1", "2"] or ["1", "2", "3"] when there are only a few cameras/mics
    n = len(seg.values)
    if all(v.isdigit() and len(v) <= 2 for v in seg.values) and 1 < n <= 4:
        return "camera"
    
    return None


_TIMESTAMP_RE = re.compile(
    r"^\d{4}[-/.]\d{2}[-/.]\d{2}$"
    r"|^\d{2}[-/.]\d{2}[-/.]\d{4}$"
    r"|^\d{2}[-/.]\d{2}[-/.]\d{2}$"
)


def _is_timestamp(s: str) -> bool:
    return bool(_TIMESTAMP_RE.match(s))


def _tokenize(name: str, mode: str = "smart") -> list[str]:
    if mode == "smart":
        primary = [t for t in re.split(r"[_\s]+", name) if t]
        result: list[str] = []
        for part in primary:
            if _is_timestamp(part):
                result.append(part)
            else:
                sub = [s for s in part.split("-") if s]
                result.extend(sub)
        return result
    if mode == "_":
        return [t for t in re.split(r"[_\s]+", name) if t]
    if mode == "-":
        return [t for t in re.split(r"[-\s]+", name) if t]
    return name.split("_")


def _find_token_spans(stem: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        idx = stem.find(tok, pos)
        if idx >= 0:
            spans.append((idx, idx + len(tok)))
            pos = idx + len(tok)
        else:
            spans.append((pos, pos))
    return spans


def _build_segments(tokenized: list[list[str]]) -> list[Segment]:
    segments: list[Segment] = []
    for i, grp in enumerate(zip(*tokenized)):
        uniq = sorted(set(grp))
        if len(uniq) == 1:
            segments.append(Segment(i, uniq[0], False))
        else:
            segments.append(Segment(i, "", True, uniq))
    return segments


def analyze_filenames(files: list[Path]) -> FilePattern | None:
    if not files:
        return None
    suffix = files[0].suffix
    stems = [f.stem for f in files]
    for mode in ("smart", "_", "-"):
        tokenized = [_tokenize(s, mode) for s in stems]
        if len({len(t) for t in tokenized}) == 1:
            return FilePattern(
                _build_segments(tokenized), files, suffix, tokenize_mode=mode,
            )
    names = sorted(set(stems))
    return FilePattern([Segment(0, "<varies>", True, names)], files, suffix)


def _guess_role(values: list[str]) -> str | None:
    n = len(values)
    low = [v.lower() for v in values]
    if all(v.isdigit() for v in values):
        return "trial"
    if any(kw in v for v in low for kw in ("view", "cam", "camera")):
        return "camera"
    if any(kw in v for v in low for kw in ("mic", "microphone")):
        return "mic"
    if _values_share_prefix_with_varying_suffix(values):
        return "trial"
    if n > 4:
        return "trial"
    if n <= 4:
        return "camera"
    return None


def _values_share_prefix_with_varying_suffix(values: list[str]) -> bool:
    if len(values) < 2:
        return False
    match = re.match(r"^([a-zA-Z]+)", values[0])
    if not match:
        return False
    prefix = match.group(1)
    return all(v.startswith(prefix) and v[len(prefix):] != values[0][len(prefix):] for v in values[1:])


def classify_stream(filename: str) -> str | None:
    lower = filename.lower()
    for stream, pats in STREAM_RULES.items():
        if any(re.search(p, lower) for p in pats):
            return stream
    return None


def analyze_nested_filenames(folder: Path, stream: str) -> FilePattern | None:
    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    if not subdirs:
        return None
    all_files: list[Path] = []
    subdir_names: set[str] = set()
    for sd in subdirs:
        files = sorted(f for f in sd.iterdir() if f.is_file())
        relevant = [f for f in files if classify_stream(f.name) == stream]
        candidates = relevant or files
        for f in candidates:
            all_files.append(f)
            subdir_names.add(sd.name)
    if not all_files:
        return None
    suffix = all_files[0].suffix
    device_role = "mic" if stream == "audio" else "camera"
    folder_seg = Segment(
        position=FOLDER_POSITION,
        text="",
        varying=True,
        values=sorted(subdir_names),
        role=device_role,
    )
    stems = [f.stem for f in all_files]
    for mode in ("smart", "_", "-"):
        tokenized = [_tokenize(s, mode) for s in stems]
        if len({len(t) for t in tokenized}) == 1:
            segments = [folder_seg] + _build_segments(tokenized)
            return FilePattern(segments, all_files, suffix, tokenize_mode=mode)
    names = sorted(set(stems))
    return FilePattern(
        [folder_seg, Segment(0, "<varies>", True, names)],
        all_files,
        suffix,
    )


def extract_file_row(
    filepath: Path, segments: list[Segment], tokenize_mode: str = "smart",
) -> dict[str, str]:
    tokens = _tokenize(filepath.stem, tokenize_mode)
    row: dict[str, str] = {"path": str(filepath)}
    for seg in segments:
        if not (seg.varying and seg.role and seg.role != "ignore"):
            continue
        if seg.position == FOLDER_POSITION:
            row[seg.role] = filepath.parent.name
        elif seg.position < len(tokens):
            row[seg.role] = tokens[seg.position]
    return row


# ─── config persistence ──────────────────────────────────────────────────────

CONFIG_FILENAME = ".media_discovery.json"


@dataclass
class StreamConfig:
    folder: str
    role_map: dict[int, str]  # segment position → role name
    nested: bool = False


@dataclass
class MediaConfig:
    streams: dict[str, StreamConfig]

    def to_dict(self) -> dict:
        return {
            "streams": {
                name: {
                    "folder": sc.folder,
                    "roles": {str(k): v for k, v in sc.role_map.items()},
                    "nested": sc.nested,
                }
                for name, sc in self.streams.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MediaConfig":
        streams = {}
        for name, sc in d.get("streams", {}).items():
            streams[name] = StreamConfig(
                folder=sc["folder"],
                role_map={int(k): v for k, v in sc.get("roles", {}).items()},
                nested=sc.get("nested", False),
            )
        return cls(streams=streams)

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MediaConfig":
        import json

        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        return path.is_file()


def _apply_roles(pattern: FilePattern, role_map: dict[int, str]) -> None:
    for seg in pattern.segments:
        if seg.varying and seg.position in role_map:
            seg.role = role_map[seg.position]


# ─── filename list ───────────────────────────────────────────────────────────


class FilenameList(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(20, 10, 20, 10)
        self._lay.setSpacing(4)
        self._rows: list[QLabel] = []

    def refresh(self, pattern: FilePattern | None):
        for w in self._rows:
            w.deleteLater()
        self._rows.clear()
        if not pattern:
            return

        mono = _mono(FS)
        segs = [s for s in pattern.segments if s.position != FOLDER_POSITION]
        folder_seg = next((s for s in pattern.segments if s.position == FOLDER_POSITION), None)
        for fp in pattern.files[:MAX_PREVIEW]:
            stem = fp.stem
            tokens = _tokenize(stem, pattern.tokenize_mode)
            spans = _find_token_spans(stem, tokens)
            parts: list[str] = []
            if folder_seg:
                tok = fp.parent.name
                if folder_seg.varying and folder_seg.role and folder_seg.role != "ignore":
                    c = ROLE_COLORS.get(folder_seg.role, TEXT_DIM)
                    parts.append(f"<span style='color:{c};font-weight:700'>{tok}</span>")
                else:
                    parts.append(f"<span style='color:{TEXT_DIM}'>{tok}</span>")
                parts.append(f"<span style='color:{TEXT_DIM}'>/</span>")
            prev_end = 0
            for idx, seg in enumerate(segs):
                if idx >= len(spans):
                    break
                start, end = spans[idx]
                if start > prev_end:
                    delim_text = stem[prev_end:start]
                    parts.append(f"<span style='color:{TEXT_DIM}'>{delim_text}</span>")
                tok_text = stem[start:end]
                if seg.varying and seg.role and seg.role != "ignore":
                    c = ROLE_COLORS.get(seg.role, TEXT_DIM)
                    parts.append(f"<span style='color:{c};font-weight:700'>{tok_text}</span>")
                else:
                    parts.append(f"<span style='color:{TEXT_DIM}'>{tok_text}</span>")
                prev_end = end
            parts.append(f"<span style='color:{TEXT_DIM}'>{pattern.suffix}</span>")

            lbl = QLabel("".join(parts))
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent; padding:2px 0;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)

        rest = len(pattern.files) - MAX_PREVIEW
        if rest > 0:
            lbl = QLabel(f"<span style='color:{TEXT_DIM};font-style:italic'>… {rest} more</span>")
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)


# ─── pattern bar ─────────────────────────────────────────────────────────────


class PatternBar(QWidget):
    role_changed = Signal()

    def __init__(self, pattern: FilePattern, stream: str, parent: QWidget | None = None, allowed_roles: list[str] | None = None):
        super().__init__(parent)
        self._pattern = pattern
        self._stream = stream
        self._allowed_roles = allowed_roles
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 6, 20, 6)
        lay.setSpacing(3)
        roles = self._allowed_roles if self._allowed_roles is not None else STREAM_ROLES.get(self._stream, ["ignore", "trial", "camera"])
        mono = _mono(FS)
        segs = self._pattern.segments
        segs_no_folder = [s for s in segs if s.position != FOLDER_POSITION]

        # Extract delimiters from first file's original stem
        ref_delims: list[str] = []
        if self._pattern.files:
            ref_stem = self._pattern.files[0].stem
            ref_tokens = _tokenize(ref_stem, self._pattern.tokenize_mode)
            ref_spans = _find_token_spans(ref_stem, ref_tokens)
            for i in range(len(ref_spans) - 1):
                ref_delims.append(ref_stem[ref_spans[i][1]:ref_spans[i + 1][0]])

        for idx, seg in enumerate(segs):
            if not seg.varying:
                lbl = QLabel(seg.text)
                lbl.setFont(mono)
                lbl.setStyleSheet(f"color:{TEXT_MID}; padding:0 1px;")
                lay.addWidget(lbl)
            else:
                cb = QComboBox()
                cb.addItems(roles)
                if seg.role and seg.role in roles:
                    cb.setCurrentText(seg.role)
                elif seg.role == "camera" and "mic" in roles and "camera" not in roles:
                    seg.role = "mic"
                    cb.setCurrentText("mic")
                else:
                    # Auto-detect role if not set
                    detected = _auto_detect_role(seg)
                    if detected and detected in roles:
                        seg.role = detected
                        cb.setCurrentText(detected)
                    else:
                        cb.setCurrentIndex(0)
                self._style(cb, seg.role)
                cb.setFixedHeight(30)
                cb.setMinimumWidth(100)
                cb.currentTextChanged.connect(
                    lambda text, s=seg, c=cb: self._changed(s, text, c)
                )
                lay.addWidget(cb)
            if seg.position == FOLDER_POSITION:
                sep = QLabel("/")
                sep.setFont(mono)
                sep.setStyleSheet(f"color:{TEXT_MID}; padding:0 1px;")
                lay.addWidget(sep)
            else:
                nf_idx = segs_no_folder.index(seg) if seg in segs_no_folder else -1
                if 0 <= nf_idx < len(ref_delims):
                    sep = QLabel(ref_delims[nf_idx])
                    sep.setFont(mono)
                    sep.setStyleSheet(f"color:{TEXT_MID}; padding:0 1px;")
                    lay.addWidget(sep)
        lay.addStretch()

    def _style(self, cb: QComboBox, role: str | None):
        c = ROLE_COLORS.get(role, TEXT_DIM)
        cb.setStyleSheet(
            f"QComboBox {{ background:{BG_INPUT}; color:{c}; font-weight:700; "
            f"border:2px solid {c}; border-radius:4px; padding:3px 10px; "
            f"font-size:{FS}px; }}"
            f"QComboBox::drop-down {{ border:none; width:20px; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_PANEL}; color:{TEXT}; "
            f"selection-background-color:{BORDER}; font-size:{FS}px; }}"
        )

    def _changed(self, seg: Segment, text: str, cb: QComboBox):
        seg.role = text if text != "ignore" else None
        self._style(cb, seg.role)
        self.role_changed.emit()


# ─── stream panel ────────────────────────────────────────────────────────────


class StreamPanel(QWidget):
    changed = Signal()

    def __init__(self, stream: str, parent: QWidget | None = None, allowed_roles: list[str] | None = None):
        super().__init__(parent)
        self._stream = stream
        self._allowed_roles = allowed_roles
        self._pattern: FilePattern | None = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 14, 0, 0)
        outer.setSpacing(10)

        # folder row
        row = QHBoxLayout()
        row.setContentsMargins(14, 0, 14, 0)
        self._folder = QLineEdit()
        self._folder.setPlaceholderText(f"select {self._stream} folder …")
        self._folder.setStyleSheet(
            f"QLineEdit {{ background:{BG_INPUT}; color:{TEXT}; "
            f"border:1px solid {BORDER}; border-radius:5px; "
            f"padding:9px 14px; font-size:{FS}px; }}"
            f"QLineEdit:focus {{ border-color:{ACCENT}; }}"
        )
        self._folder.textChanged.connect(self._on_folder)
        btn = QPushButton("…")
        btn.setFixedSize(38, 38)
        btn.setStyleSheet(
            f"QPushButton {{ background:{BG_INPUT}; color:{TEXT_MID}; "
            f"border:1px solid {BORDER}; border-radius:5px; "
            f"font-size:18px; font-weight:bold; }}"
            f"QPushButton:hover {{ border-color:{ACCENT}; color:{TEXT}; }}"
        )
        btn.clicked.connect(self._browse)
        row.addWidget(self._folder, stretch=1)
        row.addWidget(btn)
        outer.addLayout(row)

        # nested subfolder checkbox
        self._nested_cb = QCheckBox("Scan subfolders (e.g. one folder per camera)")
        self._nested_cb.setStyleSheet(
            f"QCheckBox {{ color:{TEXT_MID}; font-size:{FS - 1}px; padding:0 14px; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; }}"
        )
        self._nested_cb.toggled.connect(lambda: self._on_folder(self._folder.text()))
        outer.addWidget(self._nested_cb)

        # pattern label
        lbl = QLabel("pattern")
        lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; padding:0 20px;")
        outer.addWidget(lbl)

        # pattern bar area
        self._pat_area = QVBoxLayout()
        self._pat_area.setContentsMargins(0, 0, 0, 0)
        ph = QLabel("no folder selected")
        ph.setStyleSheet(f"color:{TEXT_DIM}; font-style:italic; padding:4px 20px; font-size:{FS}px;")
        self._pat_area.addWidget(ph)
        outer.addLayout(self._pat_area)

        # summary
        self._summary = QLabel()
        self._summary.setStyleSheet(f"color:{TEXT_MID}; font-size:{FS}px; padding:0 20px;")
        outer.addWidget(self._summary)

        # scrollable filename list — takes remaining space
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:transparent; }}"
            f"QScrollBar:vertical {{ background:{BG}; width:7px; border:none; }}"
            f"QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; min-height:30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}"
        )
        self._flist = FilenameList()
        scroll.setWidget(self._flist)
        outer.addWidget(scroll, stretch=1)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, f"select {self._stream} folder")
        if d:
            self._folder.setText(d)

    def _on_folder(self, text: str):
        p = Path(text)
        if not p.is_dir():
            self._set_pattern(None)
            return
        if self._nested_cb.isChecked():
            self._set_pattern(analyze_nested_filenames(p, self._stream))
        else:
            files = sorted(f for f in p.iterdir() if f.is_file())
            relevant = [f for f in files if classify_stream(f.name) == self._stream]
            self._set_pattern(analyze_filenames(relevant or files))

    def _set_pattern(self, pat: FilePattern | None):
        self._pattern = pat
        while self._pat_area.count():
            w = self._pat_area.takeAt(0).widget()
            if w:
                w.deleteLater()
        if pat is None:
            lbl = QLabel("no matching files")
            lbl.setStyleSheet(f"color:{TEXT_DIM}; font-style:italic; padding:4px 20px; font-size:{FS}px;")
            self._pat_area.addWidget(lbl)
            self._flist.refresh(None)
            self._summary.setText("")
            self.changed.emit()
            return
        bar = PatternBar(pat, self._stream, allowed_roles=self._allowed_roles)
        bar.role_changed.connect(self._refresh)
        self._pat_area.addWidget(bar)
        self._refresh()

    def _refresh(self):
        self._flist.refresh(self._pattern)
        info = self._pattern.summary() if self._pattern else {}
        bits: list[str] = []
        if self._pattern:
            bits.append(f"{len(self._pattern.files)} files")
        for key, label in [("trial", "trials"), ("camera", "cameras"), ("mic", "mics")]:
            if key in info:
                vals = info[key]
                extra = f" ({', '.join(vals)})" if len(vals) <= 6 else ""
                bits.append(f"{len(vals)} {label}{extra}")
        self._summary.setText("  ·  ".join(bits))
        self.changed.emit()

    @property
    def pattern(self) -> FilePattern | None:
        return self._pattern

    def get_config(self) -> StreamConfig | None:
        folder = self._folder.text()
        if not folder or self._pattern is None:
            return None
        role_map = {
            seg.position: seg.role
            for seg in self._pattern.segments
            if seg.varying and seg.role and seg.role != "ignore"
        }
        return StreamConfig(folder=folder, role_map=role_map, nested=self._nested_cb.isChecked())

    def apply_config(self, cfg: StreamConfig) -> None:
        self._nested_cb.setChecked(cfg.nested)
        self._folder.setText(cfg.folder)
        if self._pattern is not None:
            _apply_roles(self._pattern, cfg.role_map)
            self._set_pattern(self._pattern)


# ─── session table ───────────────────────────────────────────────────────────


class SessionPreview(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.setSpacing(6)

        hdr = QLabel("session table")
        hdr.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; padding:0 14px;")
        lay.addWidget(hdr)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ gridline-color:{BORDER}; background:{BG}; "
            f"alternate-background-color:{BG_PANEL}; color:{TEXT}; "
            f"font-size:{FS}px; border:none; }}"
            f"QHeaderView::section {{ background:{BG_PANEL}; color:{TEXT_MID}; "
            f"padding:6px 10px; border:none; border-bottom:1px solid {BORDER}; "
            f"font-size:{FS - 1}px; }}"
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, stretch=1)

        self._status = QLabel()
        self._status.setStyleSheet(f"color:{TEXT_MID}; font-size:{FS}px; padding:4px 14px;")
        lay.addWidget(self._status)

    def update_from_panels(self, panels: list[StreamPanel]):
        import pandas as pd

        dfs: list[pd.DataFrame] = []
        for panel in panels:
            pat = panel.pattern
            if not pat:
                continue
            stream = panel._stream
            rows = [extract_file_row(f, pat.segments, pat.tokenize_mode) for f in pat.files]
            df = pd.DataFrame(rows)
            if "trial" not in df.columns:
                continue
            dev = next((c for c in ("camera", "mic") if c in df.columns), None)
            if dev:
                piv = df.pivot(index="trial", columns=dev, values="path")
                piv.columns = [f"{stream}_{c}" for c in piv.columns]
                piv = piv.reset_index()
            else:
                piv = df[["trial", "path"]].rename(columns={"path": f"{stream}_0"})
            dfs.append(piv)

        if not dfs:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._status.setText("assign 'trial' in at least one stream")
            return

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="trial", how="outer")

        # Sort naturally: numeric if all trial IDs are digits, else alphabetical
        trials = merged["trial"]
        if trials.apply(lambda v: str(v).isdigit()).all():
            merged = merged.assign(_sort=merged["trial"].astype(int))
        else:
            merged = merged.assign(_sort=merged["trial"].astype(str).str.lower())
        merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

        cols = list(merged.columns)
        show = min(len(merged), 10)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(show)

        stream_bg = {"video": COLOR_TRIAL, "pose": COLOR_CAMERA, "audio": COLOR_MIC}
        for r in range(show):
            for c, col in enumerate(cols):
                val = merged.iloc[r][col]
                txt = Path(str(val)).name if pd.notna(val) and str(val) else ""
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                for sn, clr in stream_bg.items():
                    if col.startswith(sn):
                        qc = QColor(clr)
                        qc.setAlpha(25)
                        item.setBackground(qc)
                self._table.setItem(r, c, item)

        self._table.resizeColumnsToContents()
        nt = merged["trial"].nunique()
        mc = len([c for c in cols if c != "trial"])
        extra = f"  (showing {show}/{len(merged)})" if len(merged) > show else ""
        self._status.setText(f"{nt} trials  ·  {mc} streams{extra}")


# ─── main widget ─────────────────────────────────────────────────────────────


class MediaDiscoveryWidget(QWidget):
    def __init__(self, napari_viewer=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._panels: list[StreamPanel] = []
        self._build()

    def _build(self):
        self.setWindowTitle("media discovery")
        self.setMinimumWidth(500)
        self.setStyleSheet(f"background:{BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 10)
        root.setSpacing(0)

        # ── tab labels (plain text, spaced by font size) ──
        self._tab_labels: list[QLabel] = []
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(16, 12, 16, 10)
        tab_row.setSpacing(FS * 2)

        for i, name in enumerate(("Video", "Pose", "Audio")):
            lbl = QLabel(name)
            lbl.setStyleSheet(
                f"color:{TEXT_DIM}; font-size:{FS + 1}px; "
                f"padding:0; background:transparent; border:none;"
            )
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl.mousePressEvent = lambda _, idx=i: self._show_tab(idx)
            tab_row.addWidget(lbl)
            self._tab_labels.append(lbl)

        tab_row.addStretch()
        tab_bg = QWidget()
        tab_bg.setStyleSheet(f"background:{BG_PANEL};")
        tab_bg.setLayout(tab_row)
        root.addWidget(tab_bg)

        # ── stream panels (stacked) ──
        for stream in ("video", "pose", "audio"):
            p = StreamPanel(stream)
            p.changed.connect(self._rebuild_session)
            p.setVisible(False)
            self._panels.append(p)
            root.addWidget(p, stretch=3)

        # ── divider ──
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background:{BORDER};")
        root.addWidget(div)

        # ── session table (always visible, bottom) ──
        self._session = SessionPreview()
        root.addWidget(self._session, stretch=2)

        # ── save / load row ──
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 0)

        self._config_status = QLabel()
        self._config_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        bar.addWidget(self._config_status)
        bar.addStretch()

        load_btn = QPushButton("Load config")
        load_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_MID}; "
            f"border:1px solid {BORDER}; border-radius:4px; "
            f"padding:7px 16px; font-size:{FS}px; }}"
            f"QPushButton:hover {{ border-color:{TEXT_MID}; color:{TEXT}; }}"
        )
        load_btn.clicked.connect(self._load_config)
        bar.addWidget(load_btn)

        save_btn = QPushButton("Save config")
        save_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{ACCENT}; "
            f"border:1px solid {ACCENT}; border-radius:4px; "
            f"padding:7px 16px; font-size:{FS}px; }}"
            f"QPushButton:hover {{ background:{ACCENT}; color:{BG}; }}"
        )
        save_btn.clicked.connect(self._save_config)
        bar.addWidget(save_btn)

        root.addLayout(bar)

        self._show_tab(0)

    def _show_tab(self, idx: int):
        for i, p in enumerate(self._panels):
            p.setVisible(i == idx)
        for i, lbl in enumerate(self._tab_labels):
            if i == idx:
                lbl.setStyleSheet(
                    f"color:{TEXT}; font-size:{FS + 1}px; font-weight:bold; "
                    f"padding:0; background:transparent; border:none;"
                )
            else:
                lbl.setStyleSheet(
                    f"color:{TEXT_DIM}; font-size:{FS + 1}px; "
                    f"padding:0; background:transparent; border:none;"
                )

    def _rebuild_session(self):
        self._session.update_from_panels(self._panels)

    # ── config persistence ──

    def get_config(self) -> MediaConfig:
        streams = {}
        stream_names = ("video", "pose", "audio")
        for panel, name in zip(self._panels, stream_names):
            cfg = panel.get_config()
            if cfg is not None:
                streams[name] = cfg
        return MediaConfig(streams=streams)

    def apply_config(self, config: MediaConfig) -> None:
        stream_names = ("video", "pose", "audio")
        for panel, name in zip(self._panels, stream_names):
            if name in config.streams:
                panel.apply_config(config.streams[name])
        self._rebuild_session()

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save media config", CONFIG_FILENAME, "JSON (*.json)"
        )
        if not path:
            return
        config = self.get_config()
        config.save(path)
        self._config_status.setText(f"saved → {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load media config", "", "JSON (*.json)"
        )
        if not path:
            return
        config = MediaConfig.load(path)
        self.apply_config(config)
        self._config_status.setText(f"loaded ← {Path(path).name}")


# ─── demo ────────────────────────────────────────────────────────────────────


def create_demo_files(root: Path):
    for d in ("video", "pose", "audio"):
        (root / d).mkdir(parents=True, exist_ok=True)
    # flat layout (default)
    for t in range(1, 13):
        for cam in ("left", "right"):
            (root / "video" / f"{cam}_trial{t:03d}.mp4").touch()
        (root / "pose" / f"dlc_left_trial{t:03d}.h5").touch()
        if t <= 8:
            (root / "pose" / f"dlc_right_trial{t:03d}.h5").touch()
        (root / "audio" / f"mic1_trial{t:03d}.wav").touch()
    # nested layout (one subfolder per camera)
    for cam in ("left", "right"):
        (root / "video_nested" / cam).mkdir(parents=True, exist_ok=True)
    for t in range(1, 13):
        for cam in ("left", "right"):
            (root / "video_nested" / cam / f"trial{t:03d}.mp4").touch()


def main():
    import sys, tempfile

    app = QApplication.instance() or QApplication(sys.argv)
    root = Path(tempfile.mkdtemp(prefix="media_demo_"))
    create_demo_files(root)
    print(f"Demo files: {root}")

    w = MediaDiscoveryWidget()
    w.resize(640, 860)
    w.show()

    w._panels[0]._folder.setText(str(root / "video"))
    w._panels[1]._folder.setText(str(root / "pose"))
    w._panels[2]._folder.setText(str(root / "audio"))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()