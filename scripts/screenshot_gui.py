#!/usr/bin/env python
"""High-resolution screenshots of the running GUI, at the layout you see.

``QT_SCALE_FACTOR`` makes Qt lay the window out in bigger pixels, which
rearranges (and overlaps) the docks. This captures the opposite way round: the
window keeps its on-screen geometry and is *re-rendered* into a pixmap with a
device pixel ratio of ``scale``, so text, plot curves and icons are redrawn as
vectors at N times the resolution while every widget stays exactly where it is.

Run it instead of ``ethograph launch`` and press the hotkey whenever the GUI
shows what you want::

    python scripts/screenshot_gui.py                 # Ctrl+Shift+S, 3x
    python scripts/screenshot_gui.py --scale 4 --out docs/source/_static/media

The scale is bounded by :func:`max_scale` — Qt's raster limit, the GPU texture
limit and a RAM budget — so an over-large request is clamped and reported
instead of taking the process down with it.

The pygfx video canvases render on the GPU and never appear in a Qt widget
repaint, so each one is snapshotted through its own renderer at the same scale
and composited back into place.

``Ctrl+Shift+M`` toggles :class:`MinimalMode`, which strips plot titles, axis
label text and panel headers and enlarges the menu and playback bars — a
reversible view over the running session, saved and restored in full.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QFont, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from qtpy.QtWidgets import QApplication, QDockWidget, QWidget

from ethograph.gui.grid_section_container import GridSectionContainer
from ethograph.gui.pygfx_video import CameraView
from ethograph.gui.widgets_changepoints import ChangepointsWidget

logger = logging.getLogger("screenshot")

#: Hotkey that captures the whole main window.
DEFAULT_HOTKEY = "Ctrl+Shift+S"

#: Hotkey that toggles the stripped-down look.
DEFAULT_MINIMAL_HOTKEY = "Ctrl+Shift+M"

#: Points added to the menu bar, bottom bar and section-button fonts.
MINIMAL_FONT_DELTA_PT = 4

#: Height the bottom bar is given so the grown fonts have room.
MINIMAL_BAR_HEIGHT_PX = 92

#: Point size and row height for the Data / Labels / Trials buttons, whose
#: 7pt is pinned by a stylesheet and so cannot be grown with a font alone.
MINIMAL_SECTION_PT = 12
MINIMAL_SECTION_MIN_HEIGHT_PX = 34

#: Point size for the bottom bar's readouts — playback mode, speed, effective
#: fps and the trial label.
MINIMAL_BAR_PT = 13

#: Bottom-bar controls minimal mode hides: a screenshot is not going to show
#: anyone toggling them, and the space pays for the larger text.
MINIMAL_BAR_HIDDEN = ("proxy_cb", "center_playback_cb", "rotate_btn")

#: Bottom-bar widgets pinned to a size that the larger font would overflow, as
#: ``(attribute, width, height)``; ``None`` leaves that dimension as it is.
MINIMAL_BAR_RESIZED = (
    ("play_pause_btn", 52, None),
    ("speed_display", 68, None),
    ("prev_btn", 40, 30),
    ("next_btn", 40, 30),
)

#: Resolution multiplier used when none is asked for. Plenty for print, and
#: cheap enough that a capture never risks the process.
DEFAULT_SCALE = 3

#: Qt's raster paint engine refuses a QPixmap with a side beyond this.
QT_MAX_SIDE = 32767

#: Used only when a camera's renderer will not name its own device limit.
FALLBACK_GPU_MAX_SIDE = 8192

#: Peak RAM one capture may hold. The pixmap, its QImage copy and the PNG
#: encoder are all live at once, so the true footprint is a few times one
#: buffer — this budget counts a single buffer and stays well clear.
CAPTURE_BUDGET_BYTES = 512_000_000

#: Bytes per pixel of the RGBA buffers a capture allocates.
BYTES_PER_PIXEL = 4

#: PlotItem methods minimal mode neutralises while it is on. Panels re-apply
#: their labels on every redraw (``plots_lineplot.render_plot_data``), and
#: ``setLabel`` calls ``showAxis`` internally, so a one-shot strip is undone by
#: the next zoom; these are shadowed per instance and deleted on the way out.
FROZEN_PLOT_METHODS = ("setLabel", "setTitle", "showAxis", "showLabel")


def _ignore(*_args, **_kwargs) -> None:
    """Stand-in for a PlotItem method while minimal mode is on."""


def max_scale(widget: QWidget, budget_bytes: int = CAPTURE_BUDGET_BYTES) -> int:
    """The largest whole multiplier *widget* can safely be rendered at.

    Three ceilings bound it: Qt's raster limit on the window-sized pixmap; for
    every visible camera, the GPU's maximum 2D texture dimension, read off that
    renderer's own device rather than assumed; and *budget_bytes* of RAM, which
    in practice binds first and is what keeps a capture from killing the
    process on a large window.
    """
    size = widget.size()
    longest = max(size.width(), size.height(), 1)
    limit = QT_MAX_SIDE // longest

    pixels = max(size.width() * size.height(), 1)
    affordable = int((budget_bytes / (pixels * BYTES_PER_PIXEL)) ** 0.5)
    limit = min(limit, affordable)

    for view in _iter_camera_views(widget):
        canvas = view.canvas_widget()
        renderer, _, _ = view._render_target()
        if canvas is None or renderer is None or not canvas.isVisible():
            continue
        try:
            gpu_max = int(renderer.device.limits["max-texture-dimension-2d"])
        except (AttributeError, KeyError, TypeError):
            gpu_max = FALLBACK_GPU_MAX_SIDE
        canvas_longest = max(canvas.width(), canvas.height(), 1)
        limit = min(limit, gpu_max // canvas_longest)

    return max(1, limit)


def _iter_camera_views(root: QWidget):
    """Every live :class:`CameraView` under *root*, primary and extra alike."""
    for view in root.findChildren(CameraView):
        if view.isVisible():
            yield view


def _snapshot_camera(view: CameraView, scale: int) -> QImage | None:
    """Re-render one camera's pygfx scene at *scale* and return it as a QImage.

    Goes through ``CameraView``'s own accessors, which resolve either a loaded
    video or a static image; ``renderer.pixel_ratio`` is restored and a normal
    draw re-armed afterwards, so the on-screen canvas is left as it was found.
    """
    renderer, camera, _ = view._render_target()
    scene = view.scene()
    if renderer is None or camera is None or scene is None:
        return None
    previous = renderer.pixel_ratio
    try:
        renderer.pixel_ratio = scale
        renderer.render(scene, camera)
        frame = renderer.snapshot()
    finally:
        renderer.pixel_ratio = previous
        view.request_draw()

    frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    height, width = frame.shape[:2]
    if frame.shape[2] == 3:
        image = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        image = QImage(frame.data, width, height, 4 * width, QImage.Format_RGBA8888)
    # The numpy buffer dies with this frame; QImage does not copy by default.
    return image.copy()


def resolve_scale(widget: QWidget, scale: int | str | None) -> int:
    """Turn a requested scale into one *widget* can actually be rendered at.

    ``"max"`` asks for the largest that fits; a number is honoured unless it
    exceeds what is safe, in which case it is clamped and the reduction said
    out loud rather than silently applied.
    """
    ceiling = max_scale(widget)
    if scale == "max":
        return ceiling
    wanted = DEFAULT_SCALE if scale is None else int(scale)
    if wanted > ceiling:
        logger.warning(
            "Scale %dx would need more memory than this window can afford — capturing at %dx.",
            wanted,
            ceiling,
        )
        return ceiling
    return max(1, wanted)


def capture_widget(widget: QWidget, scale: int | str | None = None) -> QImage:
    """Render *widget* at *scale* times its on-screen resolution.

    ``None`` uses :data:`DEFAULT_SCALE`; ``"max"`` uses the largest that fits.
    """
    scale = resolve_scale(widget, scale)
    size = widget.size()
    pixmap = QPixmap(QSize(size.width() * scale, size.height() * scale))
    pixmap.setDevicePixelRatio(float(scale))
    pixmap.fill(Qt.transparent)
    widget.render(pixmap)

    painter = QPainter(pixmap)
    try:
        for view in _iter_camera_views(widget):
            canvas = view.canvas_widget()
            if canvas is None or not canvas.isVisible():
                continue
            frame = _snapshot_camera(view, scale)
            if frame is None:
                continue
            origin = canvas.mapTo(widget, canvas.rect().topLeft())
            frame.setDevicePixelRatio(float(scale))
            painter.drawImage(origin, frame)
    finally:
        painter.end()
    return pixmap.toImage()


def save_capture(widget: QWidget, out_dir: Path, scale: int | str | None = None) -> Path:
    """Capture *widget* and write a timestamped PNG into *out_dir*."""
    scale = resolve_scale(widget, scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ethograph_{datetime.now():%Y%m%d_%H%M%S}_{scale}x.png"
    image = capture_widget(widget, scale)
    if not image.save(str(path)):
        raise RuntimeError(f"Could not write {path}")
    megabytes = image.sizeInBytes() / 1e6
    logger.info(
        "Saved %s — %dx at %d x %d px (%.0f MB in memory)",
        path,
        scale,
        image.width(),
        image.height(),
        megabytes,
    )
    return path


class MinimalMode:
    """Strips the GUI down to what a screenshot is actually about.

    Everything is stored before it is changed and put back verbatim on the way
    out, so the mode is a view over the running session rather than an edit of
    it: no app_state is touched and nothing survives the toggle.

    What goes: plot titles and axis label text (the ticks stay — they are data),
    every panel header, and the bottom bar's Proxy checkbox. What grows: the
    menu bar and the bottom bar, both in font size and, for the bar, height.
    """

    def __init__(self, window, font_delta: int = MINIMAL_FONT_DELTA_PT, bar_height: int = MINIMAL_BAR_HEIGHT_PX):
        self.window = window
        self.font_delta = font_delta
        self.bar_height = bar_height
        self.active = False
        self._plots: list[tuple] = []
        self._headers: list[tuple] = []
        self._fonts: list[tuple] = []
        self._sheets: list[tuple] = []
        self._sizes: list[tuple] = []
        self._checkboxes: list = []
        self._bar: tuple | None = None

    def _restyle(self, widget: QWidget, extra: str) -> None:
        """Append a stylesheet rule to *widget*, remembering what it replaced."""
        self._sheets.append((widget, widget.styleSheet()))
        widget.setStyleSheet(widget.styleSheet() + extra)

    # -- plots ---------------------------------------------------------
    def _strip_plots(self) -> None:
        """Drop each plot's title and its axes entirely — ticks included.

        Hiding the whole axis, rather than only its label, is what actually
        frees the margin: the tick strip is most of the width a y-axis costs.
        """
        for widget in self.window.findChildren(pg.PlotWidget):
            item = widget.getPlotItem()
            axes = []
            for name in ("left", "bottom", "right", "top"):
                axis = item.getAxis(name)
                if axis is None:
                    continue
                axes.append((name, axis, axis.isVisible(), axis.label.isVisible(), axis.labelText, axis.labelUnits))
                item.hideAxis(name)
            title = item.titleLabel.text if item.titleLabel.isVisible() else None
            self._plots.append((item, title, axes))
            item.setTitle(None)
            for name in FROZEN_PLOT_METHODS:
                setattr(item, name, _ignore)

    def _restore_plots(self) -> None:
        for item, title, axes in self._plots:
            for name in FROZEN_PLOT_METHODS:
                item.__dict__.pop(name, None)
            for name, axis, axis_shown, label_shown, text, units in axes:
                if axis_shown:
                    item.showAxis(name)
                if label_shown:
                    axis.setLabel(text, units)
                axis.showLabel(label_shown)
            if title is not None:
                item.setTitle(title)
        self._plots.clear()

    # -- changepoints --------------------------------------------------
    def _strip_changepoints(self) -> None:
        """Untick "Show changepoints", the way a user would.

        Driving the widget rather than writing ``app_state.show_changepoints``
        reuses the app's own handler — the audio lines are cleared and the
        panels redraw — and it survives redraws for free, since every panel
        reads the flag on each draw.
        """
        for widget in self.window.findChildren(ChangepointsWidget):
            checkbox = getattr(widget, "show_cp_checkbox", None)
            if checkbox is not None and checkbox.isChecked():
                checkbox.setChecked(False)
                self._checkboxes.append(checkbox)

    def _restore_changepoints(self) -> None:
        for checkbox in self._checkboxes:
            checkbox.setChecked(True)
        self._checkboxes.clear()

    # -- panel headers -------------------------------------------------
    def _strip_headers(self) -> None:
        for dock in self.window.findChildren(QDockWidget):
            existing = dock.titleBarWidget()
            if existing is not None:
                # A custom header (``_PanelDockTitleBar``) — hiding it collapses
                # the row without dropping the widget the dock still owns.
                if existing.isVisible():
                    existing.hide()
                    self._headers.append((dock, existing))
                continue
            blank = QWidget(dock)
            blank.setFixedHeight(0)
            dock.setTitleBarWidget(blank)
            self._headers.append((dock, None))

    def _restore_headers(self) -> None:
        for dock, existing in self._headers:
            if existing is None:
                blank = dock.titleBarWidget()
                dock.setTitleBarWidget(None)
                if blank is not None:
                    blank.deleteLater()
            else:
                existing.show()
        self._headers.clear()

    # -- bars ----------------------------------------------------------
    def _enlarge(self, widget: QWidget) -> None:
        font = widget.font()
        self._fonts.append((widget, QFont(font)))
        grown = QFont(font)
        grown.setPointSize(max(1, font.pointSize()) + self.font_delta)
        widget.setFont(grown)

    def _strip_bars(self) -> None:
        # Only a QMainWindow has a menu bar; the mode also runs on a bare
        # widget, which simply has no bars to grow.
        menu_bar = self.window.menuBar() if hasattr(self.window, "menuBar") else None
        if menu_bar is not None:
            self._enlarge(menu_bar)

        bar = getattr(self.window, "bottom_bar", None)
        if bar is None:
            return
        self._enlarge(bar)

        hidden = []
        for name in MINIMAL_BAR_HIDDEN:
            control = getattr(bar, name, None)
            if control is not None and control.isVisible():
                control.hide()
                hidden.append(control)

        # A font alone will not move these: the trial arrows pin font-size in
        # their own stylesheet, which outranks the widget font, and the rest
        # are pinned to a width the larger text would overflow.
        self._restyle(bar, f"QLabel, QCheckBox, QComboBox, QLineEdit {{ font-size: {MINIMAL_BAR_PT}pt; }}")
        for name in ("prev_btn", "next_btn"):
            button = getattr(bar, name, None)
            if button is not None:
                self._restyle(button, f"QPushButton {{ font-size: {MINIMAL_BAR_PT}pt; }}")
        for name, width, height in MINIMAL_BAR_RESIZED:
            widget = getattr(bar, name, None)
            if widget is None:
                continue
            self._sizes.append((widget, widget.minimumSize(), widget.maximumSize()))
            if width is not None:
                widget.setFixedWidth(width)
            if height is not None:
                widget.setFixedHeight(height)

        self._bar = (bar, bar.height(), hidden)
        bar.setFixedHeight(self.bar_height)
        self._resync_bar_host()

    def _restore_bars(self) -> None:
        for widget, font in self._fonts:
            widget.setFont(font)
        self._fonts.clear()
        for widget, minimum, maximum in self._sizes:
            widget.setMinimumSize(minimum)
            widget.setMaximumSize(maximum)
        self._sizes.clear()
        if self._bar is not None:
            bar, height, hidden = self._bar
            bar.setFixedHeight(height)
            for control in hidden:
                control.show()
            self._bar = None
            self._resync_bar_host()

    # -- section buttons -----------------------------------------------
    def _strip_sections(self) -> None:
        """Grow the Data / Labels / Trials buttons.

        Button size is a stylesheet property, which outranks any font set on the
        widget, so the override must be a stylesheet too. It *replaces* rather
        than extends the existing sheet: ``grid_section_container._BTN_STYLE``
        is written with doubled braces, which Qt rejects wholesale, so anything
        appended to it is discarded along with it. The original string is put
        back on exit regardless.
        """
        override = (
            f"QPushButton {{ font-size: {MINIMAL_SECTION_PT}pt;"
            f" min-height: {MINIMAL_SECTION_MIN_HEIGHT_PX}px;"
            " border: 1px solid rgba(255,255,255,35); border-radius: 3px; }"
        )
        for container in self.window.findChildren(GridSectionContainer):
            for button in container._buttons:
                self._sheets.append((button, button.styleSheet()))
                button.setStyleSheet(override)  # replaces, see docstring

    def _restore_styles(self) -> None:
        """Put back every stylesheet touched, section buttons and bar alike."""
        for widget, sheet in self._sheets:
            widget.setStyleSheet(sheet)
        self._sheets.clear()

    def _resync_bar_host(self) -> None:
        """Let the scroll host re-measure — it caps its own height to the bar's."""
        host = getattr(self.window, "_bottom_bar_host", None)
        if host is not None:
            host._sync_height()

    # -- public --------------------------------------------------------
    def toggle(self) -> bool:
        if self.active:
            self._restore_plots()
            self._restore_changepoints()
            self._restore_headers()
            self._restore_bars()
            self._restore_styles()
        else:
            self._strip_plots()
            self._strip_changepoints()
            self._strip_headers()
            self._strip_sections()
            self._strip_bars()
        self.active = not self.active
        logger.info("Minimal mode %s", "on" if self.active else "off")
        return self.active


def install_minimal_shortcut(window: QWidget, hotkey: str) -> MinimalMode:
    """Bind *hotkey* on *window* to toggling :class:`MinimalMode`.

    The mode is parked on *window* because PyQt holds a bound-method slot
    weakly: with no strong reference the ``MinimalMode`` is collected and the
    key silently stops doing anything.
    """
    mode = MinimalMode(window)
    window._minimal_mode = mode
    shortcut = QShortcut(QKeySequence(hotkey), window)
    shortcut.setContext(Qt.ApplicationShortcut)
    shortcut.activated.connect(mode.toggle)

    def restore_on_quit() -> None:
        """Never let the mode outlive the session.

        "Show changepoints" is a saved setting, so quitting mid-mode would
        write the stripped-down state into gui_settings.yaml and it would
        still be off at the next launch.
        """
        if mode.active:
            mode.toggle()

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(restore_on_quit)

    logger.info("Press %s to toggle minimal mode", hotkey)
    return mode


def install_capture_shortcut(window: QWidget, out_dir: Path, scale: int | str | None, hotkey: str) -> None:
    """Bind *hotkey* on *window* to a full-window capture.

    The scale is resolved per capture, not now: what fits depends on the window
    size and on which cameras are open at the time.
    """
    shortcut = QShortcut(QKeySequence(hotkey), window)
    shortcut.setContext(Qt.ApplicationShortcut)
    shortcut.activated.connect(lambda: save_capture(window, out_dir, scale))
    how = "the largest scale that fits" if scale == "max" else f"{scale or DEFAULT_SCALE}x"
    logger.info("Press %s to capture the window at %s into %s", hotkey, how, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        default=DEFAULT_SCALE,
        help=f"Resolution multiplier, or 'max' for the largest that fits (default: {DEFAULT_SCALE}).",
    )
    parser.add_argument("--out", type=Path, default=Path("screenshots"), help="Output folder.")
    parser.add_argument("--hotkey", default=DEFAULT_HOTKEY, help="Capture shortcut.")
    parser.add_argument("--minimal-hotkey", default=DEFAULT_MINIMAL_HOTKEY, help="Minimal-mode toggle shortcut.")
    args = parser.parse_args()

    from ethograph import cli
    from ethograph.gui.main_window import EthographMainWindow

    original_show = EthographMainWindow.show
    installed: set[int] = set()

    def show_with_capture(self: EthographMainWindow) -> None:
        original_show(self)
        if id(self) not in installed:
            installed.add(id(self))
            install_capture_shortcut(self, args.out, args.scale, args.hotkey)
            install_minimal_shortcut(self, args.minimal_hotkey)

    EthographMainWindow.show = show_with_capture  # type: ignore[method-assign]
    try:
        cli.launch()
    finally:
        EthographMainWindow.show = original_show  # type: ignore[method-assign]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    if QApplication.instance() is not None:
        sys.exit("Run this as a script, not inside an existing Qt session.")
    main()
