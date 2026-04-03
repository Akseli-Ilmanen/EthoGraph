"""Screen recording: capture napari viewer to MP4 or GIF."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import imageio.v3 as iio
from PIL import Image
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

CROP_MODES = {
    "Full window": None,
    "Square (center crop)": "square",
    "16:9": (16, 9),
    "4:3": (4, 3),
}

SCALE_PRESETS = {
    "Native": 1.0,
    "720p": 720,
    "1080p": 1080,
    "480p": 480,
}


def _crop_frame(frame: np.ndarray, mode) -> np.ndarray:
    h, w = frame.shape[:2]
    if mode == "square":
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return frame[y0 : y0 + s, x0 : x0 + s]
    if isinstance(mode, tuple):
        ar_w, ar_h = mode
        target_ratio = ar_w / ar_h
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            return frame[:, x0 : x0 + new_w]
        else:
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            return frame[y0 : y0 + new_h, :]
    return frame


def _scale_frame(frame: np.ndarray, scale) -> np.ndarray:
    if scale is None or scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    if isinstance(scale, (int, float)) and scale > 1:
        target_h = int(scale)
        if h <= target_h:
            return frame
        ratio = target_h / h
        target_w = int(w * ratio)
    else:
        target_w = int(w * scale)
        target_h = int(h * scale)
    img = Image.fromarray(frame)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.asarray(img)


def _ensure_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


class ScreenRecorder:
    """Captures napari viewer frames to MP4 and/or GIF via a QTimer."""

    def __init__(
        self,
        viewer,
        fps: int = 24,
        canvas_only: bool = False,
        crop_mode=None,
        scale=None,
        save_mp4: bool = True,
        save_gif: bool = False,
    ):
        self._viewer = viewer
        self._fps = fps
        self._canvas_only = canvas_only
        self._crop_mode = crop_mode
        self._scale = scale
        self._save_mp4 = save_mp4
        self._save_gif = save_gif
        self._mp4_writer = None
        self._gif_frames: list[np.ndarray] | None = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._capture_frame)
        self._recording = False
        self._base_path: Optional[Path] = None
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, base_path: Path) -> None:
        """Start recording. *base_path* is used without extension; .mp4/.gif are appended."""
        self._base_path = Path(base_path).with_suffix("")
        if self._save_mp4:
            mp4_path = self._base_path.with_suffix(".mp4")
            self._mp4_writer = iio.imopen(mp4_path, "w", plugin="pyav")
            self._mp4_writer.init_video_stream("libx264", fps=self._fps)
        if self._save_gif:
            self._gif_frames = []
        self._frame_count = 0
        self._recording = True
        self._timer.start(int(1000 / self._fps))

    def stop_capture(self) -> list[Path]:
        """Stop capturing frames. Close MP4 (fast). Return immediate outputs.

        GIF frames are kept in memory — call `finalize_gif()` separately.
        """
        self._recording = False
        self._timer.stop()
        outputs: list[Path] = []
        if self._mp4_writer is not None:
            self._mp4_writer.close()
            self._mp4_writer = None
            outputs.append(self._base_path.with_suffix(".mp4"))
        return outputs

    @property
    def needs_gif_render(self) -> bool:
        return self._gif_frames is not None and len(self._gif_frames) > 0

    def finalize_gif(self) -> Optional[Path]:
        """Write accumulated GIF frames to disk (slow). Call from a thread."""
        if not self._gif_frames:
            return None
        gif_path = self._base_path.with_suffix(".gif")
        duration_ms = int(1000 / self._fps)
        iio.imwrite(
            gif_path,
            self._gif_frames,
            duration=duration_ms,
            loop=0,
            plugin="pillow",
        )
        self._gif_frames = None
        return gif_path

    def _capture_frame(self) -> None:
        if not self._recording:
            return
        frame = self._viewer.screenshot(
            canvas_only=self._canvas_only, flash=False
        )
        rgb = frame[:, :, :3]
        rgb = _crop_frame(rgb, self._crop_mode)
        rgb = _scale_frame(rgb, self._scale)
        rgb = _ensure_even(rgb)

        if self._mp4_writer is not None:
            self._mp4_writer.write_frame(rgb)
        if self._gif_frames is not None:
            self._gif_frames.append(rgb)
        self._frame_count += 1


# ── Dialog ───────────────────────────────────────────────────────────


class RecordDialog(QDialog):
    """Settings dialog shown before recording starts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Movie")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        row = 0

        # Format checkboxes
        grid.addWidget(QLabel("Format:"), row, 0)
        fmt_row = QHBoxLayout()
        self.mp4_cb = QCheckBox("MP4")
        self.mp4_cb.setChecked(True)
        self.gif_cb = QCheckBox("GIF")
        fmt_row.addWidget(self.mp4_cb)
        fmt_row.addWidget(self.gif_cb)
        grid.addLayout(fmt_row, row, 1)
        row += 1

        # FPS
        grid.addWidget(QLabel("FPS:"), row, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(24)
        grid.addWidget(self.fps_spin, row, 1)
        row += 1

        # Crop / aspect ratio
        grid.addWidget(QLabel("Crop:"), row, 0)
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(CROP_MODES.keys())
        grid.addWidget(self.crop_combo, row, 1)
        row += 1

        # Scale / resolution
        grid.addWidget(QLabel("Resolution:"), row, 0)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALE_PRESETS.keys())
        grid.addWidget(self.scale_combo, row, 1)
        row += 1

        # Canvas only
        self.canvas_only_cb = QCheckBox("Video canvas only (hide widgets)")
        grid.addWidget(self.canvas_only_cb, row, 0, 1, 2)

        layout.addLayout(grid)

        hint = QLabel(
            "Navigate as usual — <b>Space</b> to play/pause, arrow keys to "
            "step, etc. Everything on screen is captured.<br>"
            "To stop: press <b>Ctrl+Space</b> or click the red "
            "<b>Recording</b> button."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton("Start Recording", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def fps(self) -> int:
        return self.fps_spin.value()

    @property
    def save_mp4(self) -> bool:
        return self.mp4_cb.isChecked()

    @property
    def save_gif(self) -> bool:
        return self.gif_cb.isChecked()

    @property
    def canvas_only(self) -> bool:
        return self.canvas_only_cb.isChecked()

    @property
    def crop_mode(self):
        return CROP_MODES[self.crop_combo.currentText()]

    @property
    def scale(self):
        return SCALE_PRESETS[self.scale_combo.currentText()]


# ── Helpers ──────────────────────────────────────────────────────────


def _reveal_in_explorer(path: Path) -> None:
    path = Path(path)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


# ── Record button ────────────────────────────────────────────────────


class RecordButton(QWidget):
    """Toggle button that manages the full record lifecycle."""

    recording_started = Signal()
    recording_stopped = Signal(str)  # emits output path

    _STYLE_NORMAL = ""
    _STYLE_RECORDING = "background-color: #cc3333; color: white;"
    _STYLE_RENDERING = "background-color: #2e8b2e; color: white;"

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._recorder: Optional[ScreenRecorder] = None
        self._rendering = False

        self._btn = QPushButton("Record")
        self._btn.clicked.connect(self._on_click)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._btn)

        self._render_poll = QTimer(self)
        self._render_poll.setInterval(200)
        self._render_poll.timeout.connect(self._check_render_done)
        self._render_thread: Optional[threading.Thread] = None
        self._render_outputs: list[Path] = []

    def _on_click(self) -> None:
        if self._rendering:
            return
        if self._recorder and self._recorder.is_recording:
            self._stop_recording()
            return

        dlg = RecordDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        if not dlg.save_mp4 and not dlg.save_gif:
            return

        filters = []
        if dlg.save_mp4:
            filters.append("MP4 (*.mp4)")
        if dlg.save_gif:
            filters.append("GIF (*.gif)")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recording as", str(Path.home()), ";;".join(filters)
        )
        if not path:
            return

        self._recorder = ScreenRecorder(
            self._viewer,
            fps=dlg.fps,
            canvas_only=dlg.canvas_only,
            crop_mode=dlg.crop_mode,
            scale=dlg.scale,
            save_mp4=dlg.save_mp4,
            save_gif=dlg.save_gif,
        )

        self._btn.setText("Recording")
        self._btn.setStyleSheet(self._STYLE_RECORDING)
        self._recorder.start(Path(path))
        self.recording_started.emit()

    def _stop_recording(self) -> None:
        if not self._recorder:
            return

        # Stop capture — MP4 closes immediately
        immediate = self._recorder.stop_capture()

        if self._recorder.needs_gif_render:
            # GIF rendering in background thread
            self._rendering = True
            self._btn.setText("Rendering...")
            self._btn.setStyleSheet(self._STYLE_RENDERING)
            self._btn.setEnabled(False)
            self._render_outputs = list(immediate)
            recorder = self._recorder
            self._recorder = None

            def _render():
                gif_path = recorder.finalize_gif()
                if gif_path:
                    self._render_outputs.append(gif_path)

            self._render_thread = threading.Thread(target=_render, daemon=True)
            self._render_thread.start()
            self._render_poll.start()
        else:
            self._recorder = None
            self._finish(immediate)

    def _check_render_done(self):
        if self._render_thread is not None and not self._render_thread.is_alive():
            self._render_poll.stop()
            self._render_thread = None
            outputs = self._render_outputs
            self._render_outputs = []
            self._rendering = False
            self._btn.setEnabled(True)
            self._finish(outputs)

    def _finish(self, outputs: list[Path]):
        self._btn.setText("Record")
        self._btn.setStyleSheet(self._STYLE_NORMAL)
        if outputs:
            self.recording_stopped.emit(str(outputs[0]))
            _reveal_in_explorer(outputs[0])
