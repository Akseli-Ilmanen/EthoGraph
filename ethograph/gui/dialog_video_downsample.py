"""Scan a video folder for high-resolution files and offer to downsample via ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

from qtpy.QtCore import QThread, Signal, Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QProgressDialog,
    QVBoxLayout,
)

from ethograph.io.validation import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

RESOLUTION_THRESHOLD = 1080
DOWNSAMPLE_FOLDER = "videos_downsampled"
DOWNSAMPLE_META = "_downsample_info.json"

RESOLUTION_PRESETS = {
    "720p (recommended)": 720,
    "480p (smaller)": 480,
    "1080p": 1080,
    "540p": 540,
}


def get_downsample_scale(video_folder: str, video_filename: str) -> tuple[float, float]:
    """Return (scale_y, scale_x) to convert original-resolution coords to display coords.

    If the video was not downsampled, returns (1.0, 1.0).
    """
    meta_path = os.path.join(video_folder, DOWNSAMPLE_META)
    if not os.path.isfile(meta_path):
        return 1.0, 1.0
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 1.0, 1.0
    orig = meta.get("original_resolutions", {}).get(video_filename)
    if orig is None:
        return 1.0, 1.0
    target_h = meta["target_height"]
    scale = target_h / orig["h"]
    return scale, scale


def _get_video_resolution(path: str) -> tuple[int, int] | None:
    """Return (width, height) using ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def scan_large_videos(
    folder: str, threshold_height: int = RESOLUTION_THRESHOLD, sample: int = 5,
) -> list[tuple[str, int, int]]:
    """Return list of (filename, width, height) for videos above the threshold.

    Probes a random sample of up to *sample* files to infer whether the folder
    contains high-resolution videos, then returns matching entries from that sample.
    """
    folder_path = Path(folder)
    all_videos = sorted(
        f for f in folder_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    probed = random.sample(all_videos, min(sample, len(all_videos)))
    large = []
    for f in probed:
        res = _get_video_resolution(str(f))
        if res is None:
            continue
        w, h = res
        if h > threshold_height:
            large.append((f.name, w, h))
    return large


class _DownsampleWorker(QThread):
    """Runs ffmpeg downsampling in a background thread."""

    file_started = Signal(int, str)
    file_done = Signal(int)
    all_done = Signal()
    error = Signal(str)

    def __init__(self, files: list[str], src_folder: str, dst_folder: str, target_height: int):
        super().__init__()
        self.files = files
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.target_height = target_height
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        os.makedirs(self.dst_folder, exist_ok=True)
        for i, name in enumerate(self.files):
            if self._cancelled:
                return
            src = os.path.join(self.src_folder, name)
            dst = os.path.join(self.dst_folder, name)
            self.file_started.emit(i, name)
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", src,
                        "-vf", f"scale=-2:{self.target_height}",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-c:a", "copy",
                        dst,
                    ],
                    capture_output=True, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if not os.path.isfile(dst):
                    self.error.emit(f"ffmpeg failed for {name}")
                    return
            except FileNotFoundError:
                self.error.emit(
                    "ffmpeg not found. Install ffmpeg and ensure it is on your PATH."
                )
                return
            except Exception as exc:
                self.error.emit(str(exc))
                return
            self.file_done.emit(i)
        self.all_done.emit()


def offer_downsample(
    folder: str, parent=None,
) -> str:
    """Check for large videos, prompt user, downsample if accepted.

    Returns the folder path to use — either the downsampled subfolder or the
    original folder (if user declines, cancels, or ffmpeg is unavailable).
    """
    if not shutil.which("ffmpeg"):
        return folder

    large = scan_large_videos(folder)
    if not large:
        return folder

    max_h = max(h for _, _, h in large)
    file_list = "\n".join(f"  {name} ({w}x{h})" for name, w, h in large[:8])
    if len(large) > 8:
        file_list += f"\n  ... and {len(large) - 8} more"

    dialog = QDialog(parent)
    dialog.setWindowTitle("High-resolution videos detected")
    layout = QVBoxLayout(dialog)

    layout.addWidget(QLabel(
        f"<b>{len(large)} video(s)</b> have resolution above {RESOLUTION_THRESHOLD}p "
        f"(max {max_h}p).<br><br>"
        "Downsampling image resolution will make playback significantly faster "
        "without affecting temporal precision for labelling.<br>"
        "<i>Note: high frame rate is not a problem (use frame skipping). "
        "Only high image resolution affects performance.</i><br><br>"
        "Pose overlays will be rescaled automatically to match the "
        "downsampled resolution."
    ))
    layout.addWidget(QLabel(f"<pre>{file_list}</pre>"))

    form = QFormLayout()
    combo = QComboBox()
    for label in RESOLUTION_PRESETS:
        combo.addItem(label)
    combo.setCurrentIndex(0)
    form.addRow("Target resolution:", combo)
    layout.addLayout(form)

    dst_folder = os.path.join(folder, DOWNSAMPLE_FOLDER)
    layout.addWidget(QLabel(f"<i>Output folder: {dst_folder}</i>"))

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("Downsample")
    buttons.button(QDialogButtonBox.Cancel).setText("Use originals")
    layout.addWidget(buttons)

    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec_() != QDialog.Accepted:
        return folder

    target_height = RESOLUTION_PRESETS[combo.currentText()]
    filenames = [name for name, _, _ in large]

    progress = QProgressDialog(
        "Starting downsample...", "Cancel", 0, len(filenames), parent,
    )
    progress.setWindowTitle("Downsampling videos")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setMinimumWidth(400)
    progress.setValue(0)

    worker = _DownsampleWorker(filenames, folder, dst_folder, target_height)
    had_error = False

    def on_file_started(idx, name):
        if not progress.wasCanceled():
            progress.setLabelText(f"Downsampling ({idx + 1}/{len(filenames)}): {name}")

    def on_file_done(idx):
        if not progress.wasCanceled():
            progress.setValue(idx + 1)

    def on_error(msg):
        nonlocal had_error
        had_error = True
        progress.close()
        from .notify import notify_dialog
        notify_dialog(msg, "error", parent=parent)

    def on_all_done():
        progress.setValue(len(filenames))
        progress.close()

    worker.file_started.connect(on_file_started)
    worker.file_done.connect(on_file_done)
    worker.error.connect(on_error)
    worker.all_done.connect(on_all_done)
    progress.canceled.connect(worker.cancel)

    worker.start()
    while worker.isRunning():
        QApplication.processEvents()

    if progress.wasCanceled() or worker._cancelled or had_error:
        return folder

    # Copy over any videos that were below threshold (not downsampled)
    all_videos = [
        f.name for f in Path(folder).iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.name not in filenames
    ]
    for name in all_videos:
        src = os.path.join(folder, name)
        dst = os.path.join(dst_folder, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Save metadata so pose overlay can compute the correct scale factor
    meta = {
        "target_height": target_height,
        "original_resolutions": {name: {"w": w, "h": h} for name, w, h in large},
    }
    meta_path = os.path.join(dst_folder, DOWNSAMPLE_META)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    logger.info(
        "Downsampled %d videos to %dp in %s", len(filenames), target_height, dst_folder,
    )
    return dst_folder
