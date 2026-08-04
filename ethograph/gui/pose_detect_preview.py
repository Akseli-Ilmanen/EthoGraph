"""Seeing what the detector sees, on the frame already on screen.

Tuning a tag detector by numbers alone is guesswork — you set a minimum size,
run over thirty thousand frames, and find out afterwards that every tag was
half a pixel too small to decode. :class:`PreviewPanel` closes that loop on the
current frame, in milliseconds, with no run.

It draws the frame **at the resolution the detector actually sees it**, which
matters more than it sounds: previewing over a full-resolution frame would make
a tag look readable that the detector will never resolve at all. Over it go
every shape the detector found — decoded tags outlined in their keypoint's
colour, **rejected ones dashed in red with the reason**.

That last distinction is the point of the widget. "Nothing was detected" has two
causes that look identical from outside: no tag was there, or one was there and
could not be decoded (too small, too blurred, glossy paper reflecting the light).
Only the second is worth changing a setting for.

Nothing here computes anything: the panel is handed a
:class:`~ethograph.gui.pose_detect.DetectionPreview` produced by the same
``detect`` code path a run uses. A preview that can disagree with the detector
would be worse than none.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget

#: Height the preview image is drawn at. Tall enough to judge a tag, short
#: enough that the parameters stay on screen beside it.
PREVIEW_HEIGHT = 220

#: Rejected shapes are drawn in one colour whatever they were: the question they
#: answer is "why was this thrown away", not "which marker was it".
REJECT_COLOR = QColor(230, 70, 70)

#: Radius of the circle drawn around a shape with no outline of its own.
MARKER_RADIUS = 7.0


def _to_qimage(rgb: np.ndarray) -> QImage:
    """A contiguous ``(H, W, 3)`` uint8 array as a QImage that owns its data."""
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    height, width = array.shape[:2]
    return QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()


class PreviewPanel(QLabel):
    """The current frame as the detector sees it, with its near misses named."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(PREVIEW_HEIGHT)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(
            "The current frame at the size the detector scans it (never the full\n"
            "video resolution — that would make a tag look readable when it is\n"
            "not).\n\n"
            "A solid outline is a tag that decoded cleanly; a dashed red one\n"
            "needed bit corrections and was thrown away — a wrong ID accepted\n"
            "silently is the one failure that survives every later stage.\n\n"
            "Nothing outlined at all is the other failure: the tag is too small,\n"
            "too blurred or printed on gloss. Compare it with the 'must be ≥N px\n"
            "per side' figure below."
        )
        self.setText("No video frame to preview.")

    def show_message(self, message: str) -> None:
        """Say why there is nothing to draw, rather than going blank."""
        self.setPixmap(QPixmap())
        self.setText(message)

    def show_preview(
        self,
        frame: np.ndarray,
        preview,
        colors: dict[int, tuple[float, float, float]],
        names: dict[int, str] | None = None,
    ) -> None:
        """Draw *frame* with *preview*'s shapes over it."""
        names = names or {}
        image = _to_qimage(frame)
        scale = PREVIEW_HEIGHT / max(image.height(), 1)
        pixmap = QPixmap.fromImage(
            image.scaled(
                max(int(image.width() * scale), 1),
                PREVIEW_HEIGHT,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont(painter.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.0, 6.0))
        painter.setFont(font)
        for shape in preview.shapes:
            self._draw_shape(painter, shape, scale, colors, names)
        painter.end()

        self.setText("")
        self.setPixmap(pixmap)

    def _draw_shape(self, painter: QPainter, shape, scale: float, colors: dict, names: dict) -> None:
        centre = QPointF(float(shape.xy[0]) * scale, float(shape.xy[1]) * scale)
        if shape.accepted:
            rgb = colors.get(shape.label, (1.0, 1.0, 1.0))
            pen = QPen(QColor.fromRgbF(*rgb), 2.0)
        else:
            pen = QPen(REJECT_COLOR, 1.6, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.NoBrush))

        if shape.outline is not None:
            painter.drawPolygon(QPolygonF([QPointF(float(x) * scale, float(y) * scale) for x, y in shape.outline]))
        else:
            painter.drawEllipse(centre, MARKER_RADIUS, MARKER_RADIUS)

        caption = names.get(shape.label, "") if shape.accepted else shape.reason
        if not caption:
            return
        # Offset so the caption never sits on the marker being judged.
        painter.drawText(QPointF(centre.x() + MARKER_RADIUS + 3, centre.y() - MARKER_RADIUS), caption)
