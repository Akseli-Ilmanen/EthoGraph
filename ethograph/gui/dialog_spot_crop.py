"""Tools ▸ *Video: Pick a crop…* — read off a crop box for a config.

A one-shot, standalone tool: drag a rectangle on a video panel and get back
the box in exactly the spelling `ethograph.spot.config.CropConfig` and
`ethograph.video_features.CropBox` take, so it can be pasted straight into a
spot config's `labels.crop:` or a segment config's `video_features.crop:`.
It reuses the same rectangle-drag mechanics as the existing per-camera
*display* crop (`widgets_data.py: _on_crop_video_clicked`), but is otherwise
unrelated: it never touches `VideoManager.set_camera_crop` or any
display-crop state.

**Square is a choice, and the tool can make it for you.** The spot model's
backbone pools spatially and is indifferent to aspect ratio. The video-feature
extractors are not: each scales the box's shorter side to the network's input
(224 px for S3D, the model's own side for timm) and takes the *centre square*,
so the long side of a non-square box is thrown away. With *Square* ticked the
drag itself is held to a square (``CameraView.start_crop_selection(square=True)``:
the cursor's far corner moves the same amount on both axes), and the snapped
result is squared again about its centre (:func:`square_box`) in case the
frame edge clipped it — so what you drew is exactly what the network sees,
only rescaled.

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
    QCheckBox,
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


def square_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow the box to a square about its centre, kept inside a ``width`` x ``height`` frame.

    The side is the longer of the two, so nothing that was drawn is lost;
    the square is shifted back into the frame when it overhangs, and shrunk
    to the frame's shorter side only when the frame itself is too small.
    """
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    nx0 = int(round(cx - side / 2))
    ny0 = int(round(cy - side / 2))
    nx0 = max(0, min(nx0, width - side))
    ny0 = max(0, min(ny0, height - side))
    return nx0, ny0, nx0 + side, ny0 + side


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
        self.setWindowTitle("Pick a crop for a config")
        self.setModal(False)
        self._view = view
        self._camera_name = camera_name
        self._probe = probe

        layout = QVBoxLayout(self)

        instructions = QLabel(_ARM_MESSAGE)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # The spot model's backbone ends in global average pooling (timm's
        # RegNetY) and is indifferent to aspect ratio; the video-feature
        # extractors centre-crop a square, so there the box should be one.
        self.square_check = QCheckBox("Square (for video features: S3D takes 224×224, timm its model's square)")
        self.square_check.setChecked(True)
        self.square_check.toggled.connect(lambda _checked: self._arm())
        self.square_check.setToolTip(
            "Hold the drag to a square: the far corner moves the same amount on both axes. "
            "The spot model does not need it; S3D and timm extractors cut the centre square of the box, "
            "so a non-square box loses its long side."
        )
        layout.addWidget(self.square_check)

        self.status_label = QLabel("Armed — no region picked yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.yaml_edit = QPlainTextEdit()
        self.yaml_edit.setReadOnly(True)
        self.yaml_edit.setPlaceholderText("labels.crop: / video_features.crop: … — appears once a region is picked")
        self.yaml_edit.setMaximumHeight(130)
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

        self.resize(460, 380)
        self._arm()

    # ------------------------------------------------------------------

    def _arm(self) -> None:
        self.status_label.setText("Armed — click a corner of the region on the video, drag, then click again.")
        self._view.start_crop_selection(self._on_selected, square=self.square_check.isChecked())

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
        if self.square_check.isChecked():
            x0, y0, x1, y1 = square_box(x0, y0, x1, y1, self._probe.width, self._probe.height)
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
        box = f"{{x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}}}"
        # Both configs take the camera the box was drawn on: a crop is only
        # right for the video it was picked from.
        return (
            f"# spot config\nlabels:\n{camera_line}  crop: {box}\n"
            f"# segment config\nvideo_features:\n{camera_line}  crop: {box}\n"
        )

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.yaml_edit.toPlainText())
        notify("Copied the crop YAML to the clipboard.")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._view.crop_selection_active:
            self._view.cancel_crop_selection()
        super().closeEvent(event)
