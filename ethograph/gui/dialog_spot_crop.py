"""Tools ▸ *Video: Pick a crop for spot config…* — read off a `labels.crop:` box.

A one-shot, standalone tool: drag a rectangle on a video panel and get back
the box in exactly the spelling `ethograph.spot.config.CropConfig` takes, so
it can be pasted straight into a spot config's `labels.crop:` section. It
reuses the same rectangle-drag mechanics as the existing per-camera *display*
crop (`widgets_data.py: _on_crop_video_clicked`), but is otherwise unrelated:
it never touches `VideoManager.set_camera_crop` or any display-crop state.

**Displayed pixels are not source pixels.** `CameraView.screen_to_image`
reports a rectangle in whatever texture is on screen — under
`app_state.video_quality_mode == "proxy"` that is a downscaled proxy, while
`ethograph.spot.dataset` always decodes the *source* file. So the dragged
rectangle is rescaled from `view.image_size()` (displayed) to the source
video's own probed size before it is shown to the user — a crop config wrong
by the proxy's downscale factor would silently train on the wrong region.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.notify import notify
from ethograph.io.video_probe import VideoProbe, probe_video

_ARM_MESSAGE = "Click a corner of the region on the video, drag, then click again to crop."


def pick_spot_crop(data_widget, parent=None) -> QDialog | None:
    """Arm the rectangle tool on the clicked camera and open the report dialog.

    Returns ``None`` (and notifies) when there is nothing to crop: no video
    loaded, or the active panel is not a video panel.
    """
    view = data_widget._crop_target_view()
    if view is None:
        notify("No video is loaded.", "warning")
        return None
    source = getattr(view, "source_video_path", None)
    if not source:
        notify("This camera has no source video file to probe.", "warning")
        return None
    probe = probe_video(source)
    camera = getattr(view, "camera_name", None)
    dlg = SpotCropDialog(view, camera, probe, parent=parent)
    dlg.show()
    notify(_ARM_MESSAGE)
    return dlg


class SpotCropDialog(QDialog):
    """Shows the dragged rectangle as source pixels and a paste-ready YAML snippet."""

    def __init__(self, view, camera_name: str | None, probe: VideoProbe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick a crop for spot config")
        self.setModal(False)
        self._view = view
        self._camera_name = camera_name
        self._probe = probe

        layout = QVBoxLayout(self)

        instructions = QLabel(_ARM_MESSAGE)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # The model's backbone ends in global average pooling (timm's
        # RegNetY), which is indifferent to aspect ratio -- there is no
        # correctness or speed reason to make this box square.
        note = QLabel("The crop doesn't need to be square — any width × height works.")
        note.setWordWrap(True)
        note.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(note)

        self.status_label = QLabel("Armed — no region picked yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.yaml_edit = QPlainTextEdit()
        self.yaml_edit.setReadOnly(True)
        self.yaml_edit.setPlaceholderText("labels.crop: … — appears once a region is picked")
        self.yaml_edit.setMaximumHeight(90)
        layout.addWidget(self.yaml_edit)

        button_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy)
        button_row.addWidget(self.copy_btn)

        self.pick_again_btn = QPushButton("Pick again")
        self.pick_again_btn.setAutoDefault(False)
        self.pick_again_btn.clicked.connect(self._arm)
        button_row.addWidget(self.pick_again_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.setAutoDefault(False)
        self.close_btn.clicked.connect(self.close)
        button_row.addWidget(self.close_btn)
        layout.addLayout(button_row)

        self.resize(420, 320)
        self._arm()

    # ------------------------------------------------------------------

    def _arm(self) -> None:
        self.status_label.setText("Armed — click a corner of the region on the video, drag, then click again.")
        self._view.start_crop_selection(self._on_selected)

    def _on_selected(self, rect: tuple[float, float, float, float] | None) -> None:
        if rect is None:
            self.status_label.setText("Crop selection cancelled — click “Pick again” to retry.")
            notify("Crop selection cancelled.", "warning")
            return
        disp_w, disp_h = self._view.image_size()
        if not disp_w or not disp_h:
            self.status_label.setText("Could not read the displayed video size — click “Pick again” to retry.")
            notify("Could not read the displayed video size.", "warning")
            return
        # Rescale from displayed (possibly proxy) pixels to the source video's
        # own probed size -- see the module docstring.
        sx = self._probe.width / disp_w
        sy = self._probe.height / disp_h
        x0 = round(rect[0] * sx)
        y0 = round(rect[1] * sy)
        x1 = round(rect[2] * sx)
        y1 = round(rect[3] * sy)
        x0 = max(0, min(x0, self._probe.width))
        x1 = max(0, min(x1, self._probe.width))
        y0 = max(0, min(y0, self._probe.height))
        y1 = max(0, min(y1, self._probe.height))
        self._show_result(x0, y0, x1, y1)

    def _show_result(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.status_label.setText("Picked — click “Pick again” to redo it.")
        self.result_label.setText(f"{x1 - x0}×{y1 - y0} px at ({x0}, {y0})-({x1}, {y1})")
        self.yaml_edit.setPlainText(self._yaml_text(x0, y0, x1, y1))
        self.copy_btn.setEnabled(True)

    def _yaml_text(self, x0: int, y0: int, x1: int, y1: int) -> str:
        if self._camera_name:
            camera_line = f"  camera: {self._camera_name}\n"
        else:
            camera_line = "  camera: null  # name this camera\n"
        return "labels:\n" f"{camera_line}" f"  crop: {{x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}}}\n"

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.yaml_edit.toPlainText())
        notify("Copied the crop YAML to the clipboard.")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._view.crop_selection_active:
            self._view.cancel_crop_selection()
        super().closeEvent(event)
