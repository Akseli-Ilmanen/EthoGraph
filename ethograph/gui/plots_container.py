"""Unified flexible panel container for all layout scenarios.

Every panel is a QDockWidget inside a nested QMainWindow (pynaviz-style), so
panels can be arranged freely: side by side, stacked, tabbed, or floated.
The default arrangement is a vertical stack in ``_PANEL_ORDER`` with line
plots at the bottom; drag a panel's title bar to rearrange.

Panels (every one optional):
  - AudioTrace / Spectrogram (any number; instances created via
    :meth:`add_audio_panel`, removed via each panel's ✕ — duplicates are
    allowed, each instance may pin its own mic/channel)
  - EphysTrace / Raster (only if neural data)
  - Heatmap     (singleton, like the other fixed panels)
  - Line plots  (any number; ALL equal — created via :meth:`add_lineplot`,
    removed via each panel's ✕ / :meth:`remove_lineplot`)
"""

import base64
from typing import Any, Dict

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QByteArray, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.time_model import TimeRange
from ethograph.labels.intervals import find_interval_at, get_interval_bounds

from ..io.plot_sources import build_audio_source
from .app_constants import (
    ENVELOPE_OVERLAY_COLOR,
    ENVELOPE_OVERLAY_DEBOUNCE_MS,
    ENVELOPE_OVERLAY_WIDTH,
    PLOT_CONTAINER_SIZE_HINT_HEIGHT,
)
from .audio_player import AudioPlayer
from .label_drawing_mixin import LabelDrawingMixin
from .plots_audiotrace import AudioTracePlot
from .plots_base import ThrottleDebounce
from .plots_ephystrace import EphysTracePlot
from .plots_heatmap import HeatmapPlot
from .plots_lineplot import LinePlot
from .plots_overlay import OverlayManager
from .plots_raster import RasterPlot
from .plots_spectrogram import SharedAudioCache, SpectrogramPlot
from .widgets_transform import compute_energy_envelope


class TimeSlider(QWidget):
    """Horizontal slider mapped to a time range, emitting time in seconds."""

    time_changed = Signal(float)

    _SLIDER_STEPS = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._SLIDER_STEPS)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._label = QLabel("0.00 s")
        self._label.setFixedWidth(80)

        layout.addWidget(self._slider)
        layout.addWidget(self._label)

        self._t_min = 0.0
        self._t_max = 1.0

    def set_time_range(self, t_min: float, t_max: float):
        self._t_min = t_min
        self._t_max = max(t_min + 1e-6, t_max)

    def set_slider_time(self, t: float):
        if self._t_max <= self._t_min:
            return
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        frac = max(0.0, min(1.0, frac))
        self._slider.blockSignals(True)
        self._slider.setValue(int(frac * self._SLIDER_STEPS))
        self._slider.blockSignals(False)
        self._update_label(t)

    def _on_slider_moved(self, value: int):
        frac = value / self._SLIDER_STEPS
        t = self._t_min + frac * (self._t_max - self._t_min)
        self._update_label(t)
        self.time_changed.emit(t)

    @property
    def current_time(self) -> float:
        frac = self._slider.value() / self._SLIDER_STEPS
        return self._t_min + frac * (self._t_max - self._t_min)

    def _update_label(self, t: float):
        minutes = int(abs(t) // 60)
        seconds = abs(t) % 60
        sign = "-" if t < 0 else ""
        if minutes:
            self._label.setText(f"{sign}{minutes}:{seconds:05.2f}")
        else:
            self._label.setText(f"{sign}{seconds:.2f} s")


# Panel size ratios keyed by (has_audio, has_neurons_or_neo)
# Values: dict mapping panel_name -> fraction of splitter height
_PANEL_RATIOS = {
    # audio + ephys
    (True, True): {
        "audiotrace": 0.10,
        "spectrogram": 0.15,
        "neo": 0.15,
        "ephys": 0.20,
        "raster": 0.10,
        "feature": 0.30,
    },
    # audio only
    (True, False): {"audiotrace": 0.20, "spectrogram": 0.30, "feature": 0.50},
    # ephys only
    (False, True): {"neo": 0.20, "ephys": 0.30, "raster": 0.15, "feature": 0.35},
    # nothing extra
    (False, False): {"feature": 1.0},
}

# Ordered list of (panel_name, app_state_guard_attr | None) for the fixed
# singleton panels. Line plots and audio panels (audiotrace/spectrogram) are
# not listed: they are dynamic instances.
# guard_attr: app_state boolean that must be True for the panel to appear; None = always allowed
_PANEL_ORDER = [
    ("neo", None),
    ("ephys", "has_neurons"),
    ("raster", "has_neurons"),
    ("heatmap", None),
]

# Maps fixed panel name -> widget attribute name on the container
_PANEL_PLOT_ATTR = {
    "neo": "neo_trace_plot",
    "ephys": "ephys_trace_plot",
    "raster": "raster_plot",
    "heatmap": "heatmap_plot",
}


class CurrentLabelIndicator(QLabel):
    """Floating badge showing the label name + color at the current time position."""

    _MARGIN = 8
    _PAD_H = 10
    _PAD_V = 4

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.hide()

    def update_label(self, name: str, color_rgb: tuple | list | None):
        if not name:
            self.hide()
            return
        self._apply_style(name, color_rgb)
        self.show()
        self.raise_()
        self._reposition()

    def _apply_style(self, text: str, color_rgb: tuple | list | None):
        if color_rgb is not None:
            scale = max(color_rgb[:3]) <= 1.0
            r, g, b = (int(c * 255) if scale else int(c) for c in color_rgb[:3])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "#000" if lum > 140 else "#fff"
            bg = f"rgb({r},{g},{b})"
        else:
            fg, bg = "#000", "rgba(255,255,255,200)"
        self.setText(text)
        self.setStyleSheet(
            f"QLabel {{ color: {fg}; background: {bg}; border: 1px solid #555;"
            f" border-radius: 4px; padding: {self._PAD_V}px {self._PAD_H}px;"
            f" font-size: 13px; font-weight: bold; }}"
        )
        self.adjustSize()

    def _reposition(self):
        p = self.parent()
        if p is None:
            return
        x = p.width() - self.width() - self._MARGIN
        self.move(max(0, x), self._MARGIN)


class _PanelDockTitleBar(QWidget):
    """Slim dock title bar: drag handle + panel name + move (⠿) + close (✕)."""

    _BTN_STYLE = (
        "QPushButton {{ color:#ddd; background:rgba(40,40,40,160); border:none;"
        " border-radius:3px; font-size:9px; }}"
        "QPushButton:hover {{ color:#fff; background:{hover}; }}"
    )

    def __init__(self, dock: QDockWidget, title: str, on_close, on_move):
        super().__init__(dock)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(4)

        self._label = QLabel(title)
        self._label.setStyleSheet("color: rgba(255,255,255,130); font-size: 8pt;")
        layout.addWidget(self._label)
        layout.addStretch()

        move_btn = QPushButton("⠿")
        move_btn.setObjectName("panel_move_btn")
        move_btn.setFixedSize(14, 14)
        move_btn.setToolTip("Move this panel next to another panel…")
        move_btn.setStyleSheet(self._BTN_STYLE.format(hover="rgba(80,120,200,200)"))
        move_btn.clicked.connect(on_move)
        layout.addWidget(move_btn)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("panel_close_btn")
        close_btn.setFixedSize(14, 14)
        close_btn.setToolTip("Remove this panel")
        close_btn.setStyleSheet(self._BTN_STYLE.format(hover="rgba(200,60,60,200)"))
        close_btn.clicked.connect(on_close)
        layout.addWidget(close_btn)
        self.setFixedHeight(17)

    def title(self) -> str:
        return self._label.text()

    def set_title(self, text: str):
        self._label.setText(str(text))


class UnifiedPanelContainer(LabelDrawingMixin, QWidget):
    """Unified container with dynamic panel visibility.

    All panels share the same x-axis via pyqtgraph linking.
    Labels, changepoints, and time markers are drawn on all visible panels.
    """

    plot_changed = Signal(str)
    labels_redraw_needed = Signal()
    spectrogram_overlay_shown = Signal()
    time_marker_updated = Signal(float)
    #: Emitted with the plot widget whenever a dynamic panel (line plot or
    #: audio panel) is created.
    panel_added = Signal(object)
    #: Relays bufferUpdated from every spectrogram instance (auto-levels).
    spectrogram_buffer_updated = Signal()

    #: MIME type used by the add-panel popup's drag-and-drop panel creator.
    SOURCE_MIME = "application/x-ethograph-source"

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._data_widget = None  # set by widgets_meta after construction

        # Drag-and-drop panel creation: the add-panel popup drags a
        # Media/Feature source onto this container; ``_on_source_drop(kind,
        # name)`` (set by MetaWidget) opens the plot-type picker and creates
        # a panel.
        self._on_source_drop = None
        self.setAcceptDrops(True)

        # Coalesces deferred label redraws (see schedule_labels_redraw).
        self._labels_redraw_scheduled = False

        # --- Plots ---
        # Fixed singleton panels; line plots and audio panels are dynamic
        # instances (all equal, duplicates allowed).
        self.heatmap_plot = HeatmapPlot(app_state)
        self.neo_trace_plot = EphysTracePlot(app_state)  # Neo-Viewer panel
        self.ephys_trace_plot = EphysTracePlot(app_state)  # Phy-Viewer panel
        self.raster_plot = RasterPlot(app_state)

        #: All line-plot panels, equal peers (create: add_lineplot, remove: ✕).
        self.line_plots: list[LinePlot] = []
        #: Audio panels, equal peers (create: add_audio_panel, remove: ✕).
        #: Each instance may pin its own mic/channel via ``plot.mic_name``.
        self.audio_trace_plots: list[AudioTracePlot] = []
        self.spectrogram_plots: list[SpectrogramPlot] = []

        # The feature plot the right sidebar currently controls (last clicked).
        self.active_feature_plot = None

        # --- Panel visibility state (fixed panels; dynamic panels exist or don't) ---
        self._panel_visible: dict[str, bool] = {
            "neo": False,
            "ephys": False,
            "raster": False,
            "heatmap": False,
        }

        # --- Mixin state ---
        self.label_mappings: Dict[int, Dict[str, Any]] = {}
        self.audio_overlay_type = None
        self.audio_cp_items: list = []
        self.osc_event_items: list = []
        self.dataset_cp_items: list = []

        self.overlay_manager = OverlayManager()
        self.neo_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.neo_trace_plot)
        )
        self.ephys_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.ephys_trace_plot)
        )

        # Envelope throttle+debounce for x-range data refresh
        self._envelope_td = None
        self._envelope_xrange_updater = None
        self._envelope_y_updater = None
        self._envelope_host = None

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        # Dock host: a nested QMainWindow whose docks are the panels
        # (pynaviz-style free arrangement: side-by-side, tabs, floating).
        self._dock_host = QMainWindow()
        self._dock_host.setWindowFlags(Qt.Widget)
        self._dock_host.setDockNestingEnabled(True)
        self._dock_host.setDockOptions(
            QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks
        )
        main_layout.addWidget(self._dock_host)

        # Audio playback (no-video mode)
        self.audio_player = AudioPlayer(
            app_state,
            get_xlim=self.get_current_xlim,
            get_visible_time=self._get_first_visible_time,
            update_marker=self.update_time_marker_by_time,
        )

        # --- Current-label floating indicator (top-right corner) ---
        self._label_indicator = CurrentLabelIndicator(self)

        # --- X-axis linking: all panels link to the x-axis master ---
        # The master is whichever panel is first visible; we use audio_trace_plot
        # if audio is present, otherwise the feature plot.
        self._xlink_master = None

        # Connect zoom events for changepoint line style updates
        for plot in (
            self.heatmap_plot,
            self.neo_trace_plot,
            self.ephys_trace_plot,
            self.raster_plot,
        ):
            plot.vb.sigRangeChanged.connect(self._on_plot_zoom)

        # Bidirectional y-axis sync between ephys trace and raster
        self._syncing_y = False
        self.ephys_trace_plot.vb.sigYRangeChanged.connect(self._sync_raster_y_from_ephys)
        self.raster_plot.y_range_changed.connect(self._sync_ephys_y_from_raster)
        self.ephys_trace_plot.y_space_changed.connect(self._on_ephys_y_space_changed)
        self.ephys_trace_plot.seek_time_requested.connect(self._on_seek_time_requested)

        # Track which panel was last clicked (for changepoint navigation).
        # The green-edge highlight + sidebar context is handled by the
        # ActivePanelManager (set as self.active_panels by MetaWidget).
        self._last_clicked_panel = "feature"
        self.active_panels = None
        self.heatmap_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "feature"))
        self.neo_trace_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "neo"))
        self.ephys_trace_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "ephys"))
        self.raster_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "raster"))

        # Create a dock per fixed panel, hidden, in the default vertical stack.
        # Line plots and audio panels get their docks dynamically in
        # add_lineplot() / add_audio_panel().
        self._lineplot_docks: dict = {}
        self._lineplot_dock_counter = 0
        self._audio_docks: dict = {}
        self._audio_dock_counter = 0
        self._create_panel_docks()

    def _create_panel_docks(self):
        """One QDockWidget per fixed panel; ✕ hides it (line-plot ✕ removes)."""
        closers = {
            "heatmap": lambda: self.set_heatmap_visible(False),
            "neo": lambda: self.set_neo_visible(False),
            "ephys": lambda: self.set_ephys_visible(False),
            "raster": lambda: self.set_raster_visible(False),
        }
        self._panel_docks: dict[str, QDockWidget] = {}
        prev = None
        for name, _ in _PANEL_ORDER:
            dock = self._make_dock(name, self._get_panel_widget(name), closers[name])
            dock.setObjectName(f"panel_{name}")
            self._panel_docks[name] = dock
            if prev is None:
                self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
            else:
                self._dock_host.splitDockWidget(prev, dock, Qt.Vertical)
            dock.hide()
            prev = dock

    def _make_dock(self, title: str, widget: QWidget, on_close) -> QDockWidget:
        dock = QDockWidget(title, self._dock_host)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setTitleBarWidget(
            _PanelDockTitleBar(dock, title, on_close, on_move=lambda: self._show_move_menu(dock))
        )
        return dock

    def _audio_plots(self) -> list:
        return list(self.audio_trace_plots) + list(self.spectrogram_plots)

    def _open_docks(self) -> list[QDockWidget]:
        docks = [self._audio_docks[p] for p in self._audio_plots() if not self._audio_docks[p].isHidden()]
        docks += [self._panel_docks[n] for n, _ in _PANEL_ORDER if not self._panel_docks[n].isHidden()]
        docks += [self._lineplot_docks[p] for p in self.line_plots if not self._lineplot_docks[p].isHidden()]
        return docks

    def _show_move_menu(self, dock: QDockWidget):
        """Click-driven panel placement: pick a target panel and a side."""
        targets = [d for d in self._open_docks() if d is not dock]
        if not targets:
            return
        menu = QMenu(self)

        def _place(target: QDockWidget, orient=None, tab=False):
            dock.setFloating(False)
            if tab:
                self._dock_host.tabifyDockWidget(target, dock)
            else:
                self._dock_host.splitDockWidget(target, dock, orient)
            dock.show()
            dock.raise_()

        for target in targets:
            sub = menu.addMenu(target.titleBarWidget().title())
            sub.addAction("Below", lambda _=False, t=target: _place(t, Qt.Vertical))
            sub.addAction("Right of", lambda _=False, t=target: _place(t, Qt.Horizontal))
            sub.addAction("Tab with", lambda _=False, t=target: _place(t, tab=True))
        menu.exec_(QCursor.pos())

    def _dock_of(self, plot) -> QDockWidget | None:
        for name, dock in self._panel_docks.items():
            if self._get_panel_widget(name) is plot:
                return dock
        return self._lineplot_docks.get(plot) or self._audio_docks.get(plot)

    def set_panel_title(self, plot, title: str) -> None:
        """Update a panel dock's title (e.g. when its feature changes)."""
        dock = self._dock_of(plot)
        if dock is not None:
            dock.titleBarWidget().set_title(title)

    # ------------------------------------------------------------------
    # Drag-and-drop panel creation (add-panel popup → plot area)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.SOURCE_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.SOURCE_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(self.SOURCE_MIME):
            return
        payload = bytes(event.mimeData().data(self.SOURCE_MIME)).decode("utf-8")
        kind, _, name = payload.partition("|")
        if callable(self._on_source_drop):
            self._on_source_drop(kind, name)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Line-plot panels (all equal — no built-in/extra distinction)
    # ------------------------------------------------------------------

    def _available_features(self) -> list[str]:
        data_widget = self._data_widget
        catalog = getattr(data_widget, "catalog", None) if data_widget else None
        if catalog is not None:
            choices = catalog.feature_choices()
            if choices:
                return choices
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            return list(ds.data_vars)
        return []

    def add_lineplot(self, feature: str | None = None):
        """Create a line-plot panel. Every line plot goes through here — they
        are all equal peers with their own ✕ (which removes the instance),
        per-panel state, and the same sidebar controls when clicked."""
        features = self._available_features()
        if not features:
            return None

        plot = LinePlot(self.app_state)
        if feature in features:
            plot.set_panel_control("features", feature)

        self._lineplot_dock_counter += 1
        dock = self._make_dock(feature or "lineplot", plot, lambda: self.remove_lineplot(plot))
        dock.setObjectName(f"panel_lineplot_{self._lineplot_dock_counter}")
        anchor = self._last_open_dock()
        self._lineplot_docks[plot] = dock
        if anchor is None:
            self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
        else:
            self._dock_host.splitDockWidget(anchor, dock, Qt.Vertical)
        dock.show()

        self.line_plots.append(plot)
        plot.vb.sigRangeChanged.connect(self._on_plot_zoom)
        plot.vb.sigYRangeChanged.connect(lambda *_, p=plot: self.overlay_manager.rescale_for_plot(p))
        plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "feature"))
        # Register with the active-panel manager so it highlights + shows controls.
        if self.active_panels is not None:
            self.active_panels.register(plot, "lineplot", clicked_signal=plot.plot_clicked, plot=plot)
        if self.active_feature_plot is None:
            self.active_feature_plot = plot
        self._update_panel_visibility()
        if self.app_state.ready:
            plot.update_plot()
        self.panel_added.emit(plot)
        return plot

    def _last_open_dock(self) -> QDockWidget | None:
        """The bottom anchor for a new line-plot dock (default vertical stack)."""
        for plot in reversed(self.line_plots):
            dock = self._lineplot_docks[plot]
            if not dock.isHidden() and not dock.isFloating():
                return dock
        for name, _ in reversed(_PANEL_ORDER):
            dock = self._panel_docks[name]
            if not dock.isHidden() and not dock.isFloating():
                return dock
        for plot in reversed(self._audio_plots()):
            dock = self._audio_docks[plot]
            if not dock.isHidden() and not dock.isFloating():
                return dock
        return None

    def remove_lineplot(self, plot) -> None:
        """Remove any line-plot panel (they are all removable)."""
        if plot not in self.line_plots:
            return
        if self.active_panels is not None:
            self.active_panels.unregister(plot)
        plot._td.stop()
        self.line_plots.remove(plot)
        dock = self._lineplot_docks.pop(plot, None)
        if self.active_feature_plot is plot:
            self.active_feature_plot = self.line_plots[0] if self.line_plots else None
        if dock is not None:
            self._dock_host.removeDockWidget(dock)
            dock.deleteLater()
        else:
            plot.setParent(None)
            plot.deleteLater()
        self._update_panel_visibility()

    # ------------------------------------------------------------------
    # Audio panels (audiotrace / spectrogram — instances, duplicates allowed)
    # ------------------------------------------------------------------

    @property
    def audio_trace_plot(self) -> AudioTracePlot | None:
        """First audio-trace instance (backwards-compat accessor)."""
        return self.audio_trace_plots[0] if self.audio_trace_plots else None

    @property
    def spectrogram_plot(self) -> SpectrogramPlot | None:
        """First spectrogram instance (backwards-compat accessor)."""
        return self.spectrogram_plots[0] if self.spectrogram_plots else None

    def add_audio_panel(self, panel_type: str, mic_name: str | None = None):
        """Create an audio panel instance: ``"audiotrace"`` or ``"spectrogram"``.

        Duplicates are allowed — every call creates a new panel; the user
        removes extras via each panel's ✕. *mic_name* pins the panel to one
        mic/channel (an ``audio_source_map`` key); ``None`` follows the
        global Mic combo.
        """
        if panel_type == "spectrogram":
            plot = SpectrogramPlot(self.app_state)
            plots = self.spectrogram_plots
            plot.bufferUpdated.connect(self.spectrogram_buffer_updated)
        else:
            panel_type = "audiotrace"
            plot = AudioTracePlot(self.app_state)
            plots = self.audio_trace_plots
            plot.vb.sigYRangeChanged.connect(lambda *_, p=plot: self.overlay_manager.rescale_for_plot(p))
        plot.mic_name = mic_name

        self._audio_dock_counter += 1
        title = f"{panel_type} — {mic_name}" if mic_name else panel_type
        dock = self._make_dock(title, plot, lambda: self.remove_audio_panel(plot))
        dock.setObjectName(f"panel_{panel_type}_{self._audio_dock_counter}")
        self._audio_docks[plot] = dock

        # Keep audio panels grouped at the top of the default vertical stack:
        # anchor below the last open audio dock, else become the first dock.
        anchor = None
        for other in reversed(self._audio_plots()):
            other_dock = self._audio_docks[other]
            if not other_dock.isHidden() and not other_dock.isFloating():
                anchor = other_dock
                break
        if anchor is None:
            self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
        else:
            self._dock_host.splitDockWidget(anchor, dock, Qt.Vertical)
        dock.show()

        plots.append(plot)
        plot.vb.sigRangeChanged.connect(self._on_plot_zoom)
        plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "audio"))
        if self.active_panels is not None:
            self.active_panels.register(plot, panel_type, clicked_signal=plot.plot_clicked)

        self._update_panel_visibility()

        source = build_audio_source(self.app_state, plot.mic_name)
        plot.set_source(source)
        if self.app_state.ready and source is not None:
            t0, t1 = self.get_current_xlim()
            plot.update_plot(t0=t0, t1=t1)
        self.panel_added.emit(plot)
        self.schedule_labels_redraw()
        return plot

    def set_audio_panel_mic(self, plot, mic_name: str | None) -> None:
        """Re-pin an audio panel to another mic/channel and refresh it."""
        if plot in self.spectrogram_plots:
            panel_type = "spectrogram"
        elif plot in self.audio_trace_plots:
            panel_type = "audiotrace"
        else:
            return
        plot.mic_name = mic_name
        plot.set_source(build_audio_source(self.app_state, mic_name))
        dock = self._audio_docks.get(plot)
        if dock is not None:
            title = f"{panel_type} — {mic_name}" if mic_name else panel_type
            dock.setWindowTitle(title)
            bar = dock.titleBarWidget()
            if bar is not None:
                bar.set_title(title)
        t0, t1 = self.get_current_xlim()
        plot.update_plot(t0=t0, t1=t1)
        self.schedule_labels_redraw()

    def remove_audio_panel(self, plot) -> None:
        """Remove any audio panel instance (they are all removable)."""
        for plots in (self.audio_trace_plots, self.spectrogram_plots):
            if plot in plots:
                plots.remove(plot)
                break
        else:
            return
        if self.active_panels is not None:
            self.active_panels.unregister(plot)
        # The throttle/debounce QTimers are not parented to the widget — stop
        # them so no callback fires into the deleted plot.
        plot._td.stop()
        plot.set_source(None)
        dock = self._audio_docks.pop(plot, None)
        if dock is not None:
            self._dock_host.removeDockWidget(dock)
            dock.deleteLater()
        else:
            plot.setParent(None)
            plot.deleteLater()
        self._update_panel_visibility()

    # ------------------------------------------------------------------
    # Panel layout persistence (app_state.panel_layout → local_settings.yaml)
    # ------------------------------------------------------------------

    def _canonicalize_lineplot_dock_names(self):
        """Name dynamic-panel docks by list position so QMainWindow.saveState /
        restoreState blobs match across sessions."""
        for i, plot in enumerate(self.line_plots):
            self._lineplot_docks[plot].setObjectName(f"panel_lineplot_{i}")
        for i, plot in enumerate(self.audio_trace_plots):
            self._audio_docks[plot].setObjectName(f"panel_audiotrace_{i}")
        for i, plot in enumerate(self.spectrogram_plots):
            self._audio_docks[plot].setObjectName(f"panel_spectrogram_{i}")

    def layout_state(self) -> dict:
        """Serializable panel layout: the open panels (type + full per-panel
        settings: feature, dim selections — a dim absent means "All" — and
        color) plus the dock-host state blob that encodes the free 2D
        arrangement (positions, sizes, tabs, floating)."""
        self._canonicalize_lineplot_dock_names()
        panels = []
        for plot in self.audio_trace_plots:
            panels.append({"type": "audiotrace", "mic": plot.mic_name})
        for plot in self.spectrogram_plots:
            panels.append({"type": "spectrogram", "mic": plot.mic_name})
        for name, _ in _PANEL_ORDER:
            if not self._panel_visible[name]:
                continue
            entry: dict = {"type": name}
            if name == "heatmap":
                entry.update(self.heatmap_plot.panel_settings())
            panels.append(entry)
        for plot in self.line_plots:
            panels.append({"type": "lineplot", **plot.panel_settings()})
        return {
            "panels": panels,
            "dock_state_b64": base64.b64encode(bytes(self._dock_host.saveState())).decode("ascii"),
        }

    def apply_layout_state(self, state: dict) -> None:
        """Recreate the panel layout captured by :meth:`layout_state`."""
        entries = state.get("panels")
        if not isinstance(entries, list):
            return

        types = {e.get("type") for e in entries}

        for plot in self._audio_plots():
            self.remove_audio_panel(plot)
        for e in entries:
            if e.get("type") in ("audiotrace", "spectrogram"):
                self.add_audio_panel(e["type"], mic_name=e.get("mic"))
        self.set_neo_visible("neo" in types)
        self.set_heatmap_visible("heatmap" in types)
        if "raster" in types:
            self.set_neural_panel_mode("raster")
        elif "ephys" in types:
            self.set_neural_panel_mode("trace")
        else:
            self.set_ephys_visible(False)

        for plot in list(self.line_plots):
            self.remove_lineplot(plot)
        for e in entries:
            if e.get("type") == "lineplot":
                plot = self.add_lineplot(feature=e.get("feature"))
                if plot is not None:
                    plot.apply_panel_settings(e)
                    if self.app_state.ready:
                        plot.update_plot()
            elif e.get("type") == "heatmap":
                self.heatmap_plot.apply_panel_settings(e)
        self._canonicalize_lineplot_dock_names()

        # Showing an audio panel requires re-wiring its source.
        if "audiotrace" in types or "spectrogram" in types:
            self.update_audio_panels()

        blob = state.get("dock_state_b64")
        if blob:
            # Deferred so it runs after the _apply_panel_sizes singleShot that
            # update_audio_panels / visibility changes schedule.
            data = QByteArray(base64.b64decode(blob))
            QTimer.singleShot(0, lambda: self._dock_host.restoreState(data))

    def update_feature_plots(self, **kwargs) -> None:
        """Re-render every feature panel: all line plots + heatmap if shown."""
        if self._panel_visible["heatmap"]:
            self.heatmap_plot.update_plot(**kwargs)
        for plot in self.line_plots:
            plot.update_plot(**kwargs)

    def _get_all_plots(self) -> list:
        return super()._get_all_plots() + list(self.line_plots)

    def sizeHint(self):
        return QSize(self.width(), PLOT_CONTAINER_SIZE_HINT_HEIGHT)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._label_indicator._reposition()

    def _update_label_indicator(self, time_s: float):
        # Label indicator (text badge) is only shown on video, not on plots
        self._label_indicator.hide()

    def _on_plot_zoom(self):
        self.update_audio_changepoint_styles()
        self.update_oscillatory_event_styles()

    # ------------------------------------------------------------------
    # Panel configuration
    # ------------------------------------------------------------------

    def configure_panels(self):
        """Called after data load to set up which panels are available."""
        self._update_panel_visibility()

    def _get_panel_widget(self, name: str):
        return getattr(self, _PANEL_PLOT_ATTR[name])

    def _visible_panel_names(self) -> list[str]:
        result = []
        for name, guard in _PANEL_ORDER:
            if guard and not getattr(self.app_state, guard, False):
                continue
            if self._panel_visible[name]:
                result.append(name)
        return result

    def _visible_panel_widgets(self) -> list:
        """All open panels in visual order: audio + fixed panels + line plots."""
        return (
            self._audio_plots()
            + [self._get_panel_widget(n) for n in self._visible_panel_names()]
            + list(self.line_plots)
        )

    def _update_panel_visibility(self):
        """Show/hide fixed panel docks in-place; never reparents widgets."""
        visible_names = self._visible_panel_names()
        visible_set = set(visible_names)

        for name, guard in _PANEL_ORDER:
            self._panel_docks[name].setVisible(name in visible_set)

        self._setup_xlinks_from_visible(visible_names)

        # Every panel keeps its own x-axis ticks: with free 2D arrangement
        # there is no single "bottom" panel to delegate them to.
        for widget in self._visible_panel_widgets():
            widget.plotItem.getAxis("bottom").setStyle(showValues=True)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.schedule_labels_redraw()

    def schedule_labels_redraw(self) -> None:
        """Redraw labels once the pending panel-content renders are done.

        General rule: any path that creates or shows a panel must end with a
        label redraw that runs AFTER the panel's content render (update_plot /
        set_source), because "bottom"-strip overlay modes position rectangles
        from the plot's y viewRange — drawing before the content sets the real
        range leaves the labels invisible. Deferring to the next event-loop
        tick (coalesced) guarantees that ordering for every creation path.
        """
        if self._labels_redraw_scheduled:
            return
        self._labels_redraw_scheduled = True
        QTimer.singleShot(0, self._emit_labels_redraw)

    def _emit_labels_redraw(self) -> None:
        self._labels_redraw_scheduled = False
        self.labels_redraw_needed.emit()

    def _setup_xlinks_from_visible(self, visible_names: list[str] | None = None):
        """Link all panels to the first open panel's x-axis."""
        widgets = self._visible_panel_widgets()
        if not widgets:
            self._xlink_master = None
            return

        master = widgets[0]
        self._xlink_master = master
        for widget in widgets[1:]:
            widget.plotItem.setXLink(master.plotItem)

        # Keep hidden singletons linked too so they are in sync when shown.
        for plot in (self.heatmap_plot, self.neo_trace_plot):
            if plot is not master:
                plot.plotItem.setXLink(master.plotItem)

    def _apply_panel_sizes(self):
        """Default vertical sizing from `_PANEL_RATIOS` via resizeDocks.

        Only a best-effort hint: the dock system preserves whatever
        arrangement/sizes the user drags afterwards.
        """
        total = self._dock_host.height()
        if total <= 0:
            return

        v = self._panel_visible
        has_audio_panel = self.app_state.has_audio and bool(self._audio_plots())
        has_neural_panel = v["neo"] or (self.app_state.has_neurons and (v["ephys"] or v["raster"]))
        ratios = _PANEL_RATIOS.get((has_audio_panel, has_neural_panel), {"feature": 1.0})

        visible_names = self._visible_panel_names()

        # The "feature" ratio is shared equally by the feature group:
        # heatmap (if shown) + every line plot.
        n_feature = len(self.line_plots) + (1 if "heatmap" in visible_names else 0)
        feature_share = ratios.get("feature", 0.3) * total if n_feature else 0.0

        # Every audio instance gets its panel type's ratio share.
        audio_raw = [
            (self._audio_docks[plot], ratios.get("audiotrace", 0.2) * total) for plot in self.audio_trace_plots
        ] + [(self._audio_docks[plot], ratios.get("spectrogram", 0.2) * total) for plot in self.spectrogram_plots]

        raw = {}
        for name in visible_names:
            if name == "heatmap":
                continue
            raw[name] = ratios.get(name, 0.2) * total

        # When neo and phy panels coexist, enforce a 1:5 size ratio between them.
        phy_present = [n for n in ("ephys", "raster") if n in raw]
        if "neo" in raw and phy_present:
            neo_phy_total = raw["neo"] + sum(raw[n] for n in phy_present)
            raw["neo"] = neo_phy_total / 6
            phy_raw_total = sum(raw[n] for n in phy_present)
            for n in phy_present:
                raw[n] = raw[n] / phy_raw_total * (neo_phy_total * 5 / 6)

        total_alloc = sum(raw.values()) + feature_share + sum(size for _, size in audio_raw)
        if total_alloc <= 0:
            return
        scale = total / total_alloc
        member_size = max(1, int(feature_share / n_feature * scale)) if n_feature else 0

        docks, sizes = [], []
        for dock, size in audio_raw:
            docks.append(dock)
            sizes.append(max(1, int(size * scale)))
        for name in visible_names:
            docks.append(self._panel_docks[name])
            sizes.append(member_size if name == "heatmap" else max(1, int(raw[name] * scale)))
        for plot in self.line_plots:
            docks.append(self._lineplot_docks[plot])
            sizes.append(member_size)
        if docks:
            self._dock_host.resizeDocks(docks, sizes, Qt.Vertical)

    # ------------------------------------------------------------------
    # Panel visibility toggles
    # ------------------------------------------------------------------

    def _set_panel_visible(self, name: str, visible: bool):
        if self._panel_visible[name] == visible:
            return
        self._panel_visible[name] = visible
        self._update_panel_visibility()

    def set_audiotrace_visible(self, visible: bool):
        """Backwards-compat shim: True ensures at least one audio-trace
        instance exists; False removes every instance."""
        if visible:
            if not self.audio_trace_plots:
                self.add_audio_panel("audiotrace")
        else:
            for plot in list(self.audio_trace_plots):
                self.remove_audio_panel(plot)

    def set_spectrogram_visible(self, visible: bool):
        """Backwards-compat shim: True ensures at least one spectrogram
        instance exists; False removes every instance."""
        if visible:
            if not self.spectrogram_plots:
                self.add_audio_panel("spectrogram")
        else:
            for plot in list(self.spectrogram_plots):
                self.remove_audio_panel(plot)

    def set_heatmap_visible(self, visible: bool):
        self._set_panel_visible("heatmap", visible)

    def set_feature_view(self, mode: str):
        """Switch how the selected feature is shown: "lineplot" or "heatmap".

        The heatmap is a singleton panel like any other; line plots are
        instances. Heatmap mode shows the heatmap panel; lineplot mode hides
        it (creating a line plot if none exists).
        """
        if mode not in ("lineplot", "heatmap") or mode == self._feature_type:
            return
        if mode == "heatmap":
            self.set_heatmap_visible(True)
        else:
            self.set_heatmap_visible(False)
            if not self.line_plots:
                self.add_lineplot()
        self.plot_changed.emit(mode)
        self.schedule_labels_redraw()

    # ------------------------------------------------------------------
    # Ephys panel show/hide
    # ------------------------------------------------------------------

    def set_neo_visible(self, visible: bool):
        self._set_panel_visible("neo", visible)
        if not visible:
            self.neo_trace_plot.buffer.loader = None
            self.neo_trace_plot.set_source(None)

    def set_ephys_visible(self, visible: bool):
        if self._panel_visible["ephys"] == visible:
            return
        self._panel_visible["ephys"] = visible
        if not visible:
            self._panel_visible["raster"] = False
            self.ephys_trace_plot.buffer.loader = None
            self.ephys_trace_plot.set_source(None)
        self._update_panel_visibility()

    def set_raster_visible(self, visible: bool):
        self._set_panel_visible("raster", visible)

    def set_neural_panel_mode(self, mode: str):
        """Switch between 'trace' and 'raster' for the neural panel slot."""
        if mode == "trace":
            show_ephys, show_raster = True, False
        elif mode == "raster":
            show_ephys, show_raster = False, True
        else:
            return

        v = self._panel_visible
        if v["ephys"] != show_ephys or v["raster"] != show_raster:
            v["ephys"] = show_ephys
            v["raster"] = show_raster
            self._update_panel_visibility()

    # ------------------------------------------------------------------
    # Bidirectional y-axis sync: ephys <-> raster
    # ------------------------------------------------------------------

    def _sync_raster_y_from_ephys(self):
        if self._syncing_y or not self._panel_visible["raster"]:
            return
        self._syncing_y = True
        try:
            y_lo, y_hi = self.ephys_trace_plot.vb.viewRange()[1]
            self.raster_plot.vb.setYRange(y_lo, y_hi, padding=0)
        finally:
            self._syncing_y = False

    def _sync_ephys_y_from_raster(self):
        if self._syncing_y or not self._panel_visible["raster"]:
            return
        self._syncing_y = True
        try:
            y_lo, y_hi = self.raster_plot.vb.viewRange()[1]
            self.ephys_trace_plot.vb.setYRange(y_lo, y_hi, padding=0)
        finally:
            self._syncing_y = False

    def _on_ephys_y_space_changed(self):
        ep = self.ephys_trace_plot
        total = len(ep._total_ordered_channels)
        if total == 0:
            return
        spacing = ep.buffer.channel_spacing
        self.raster_plot.sync_y_axis(ep._hw_to_global_y, spacing, total)

    # ------------------------------------------------------------------
    # Public API (compatible with PlotContainer + MultiPanelContainer)
    # ------------------------------------------------------------------

    def get_current_plot(self):
        """The feature plot the user is working with: the active (last
        clicked) line plot or heatmap, else the first line plot, else the
        heatmap (which always exists as a widget)."""
        active = self.active_feature_plot
        if active is self.heatmap_plot and self._panel_visible["heatmap"]:
            return active
        if active in self.line_plots:
            return active
        if self.line_plots:
            return self.line_plots[0]
        return self.heatmap_plot

    @property
    def _feature_plot(self):
        return self.get_current_plot()

    @property
    def _feature_type(self) -> str:
        return "heatmap" if self._panel_visible["heatmap"] else "lineplot"

    @property
    def current_plot(self):
        return self.get_current_plot()

    @property
    def current_plot_type(self) -> str:
        return self._feature_type

    def get_current_xlim(self):
        master = self._xlink_master or self.get_current_plot()
        return master.get_current_xlim()

    def set_x_range(self, mode="default", curr_xlim=None, center_on_frame=None):
        master = self._xlink_master or self.get_current_plot()
        return master.set_x_range(
            mode=mode,
            curr_xlim=curr_xlim,
            center_on_frame=center_on_frame,
        )

    @property
    def vb(self):
        return self.get_current_plot().vb

    def get_hovered_plot(self):
        for plot in self._visible_plots():
            if plot.underMouse():
                return plot
        return self.get_current_plot()

    def _visible_plots(self):
        for plot in self._audio_plots():
            if not self._audio_docks[plot].isHidden():
                yield plot
        for name, _ in _PANEL_ORDER:
            if not self._panel_docks[name].isHidden():
                yield self._get_panel_widget(name)
        for plot in self.line_plots:
            if not self._lineplot_docks[plot].isHidden():
                yield plot

    def update_time_marker_by_time(self, time_s: float):
        for plot in self._visible_plots():
            plot.update_time_marker(time_s)
        self._update_label_indicator(time_s)
        self.time_marker_updated.emit(time_s)

    def _on_seek_time_requested(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        video = getattr(self.app_state, "video", None)
        if video:
            frame = video.time_to_frame(time_s)
            video.blockSignals(True)
            video.seek_to_frame(frame)
            video.blockSignals(False)
            self.app_state.current_frame = frame

    def update_time_marker_and_window(self, frame_number):
        video = getattr(self.app_state, "video", None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps
        for plot in self._visible_plots():
            plot.update_time_marker(current_time)
        self._update_label_indicator(current_time)
        self.time_marker_updated.emit(current_time)

    def apply_y_range(self, ymin, ymax):
        return self.get_current_plot().apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot.toggle_axes_lock(x_bounds_override=bounds)

    def _apply_all_zoom_constraints(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot._apply_zoom_constraints(x_bounds_override=bounds)

    def _trial_bounds_tuple(self):
        tr = self.app_state.padded_bounds
        return (tr.start_s, tr.end_s) if tr is not None else None

    # --- Feature view switching ---

    def switch_to_lineplot(self):
        self.set_feature_view("lineplot")

    def switch_to_heatmap(self):
        self.set_feature_view("heatmap")

    # --- Type checking ---

    def is_lineplot(self):
        return self._feature_type == "lineplot"

    def is_heatmap(self):
        return self._feature_type == "heatmap"

    def is_spectrogram(self):
        return False  # spectrogram is always its own panel when audio loaded

    def is_audiotrace(self):
        return False  # audio trace is always its own panel

    def is_ephystrace(self):
        return self._panel_visible["ephys"] or self._panel_visible["raster"]

    def has_spectrogram_overlay(self) -> bool:
        return False  # no overlay system — dedicated panel instead

    # --- Audio overlay stubs (dedicated panels instead) ---

    def update_audio_overlay(self):
        pass

    def apply_overlay_levels(self, vmin: float, vmax: float):
        pass

    def apply_overlay_colormap(self, colormap_name: str):
        pass

    # --- Audio panel updates ---

    def update_audio_panels(self):
        """Refresh audio-driven panels (waveform + spectrogram) after mic change.

        Panels pinned to a mic/channel (``plot.mic_name``) keep their own
        source; unpinned panels follow the global Mic combo.
        """
        audio_plots = self._audio_plots()
        for plot in audio_plots:
            plot.set_source(build_audio_source(self.app_state, plot.mic_name))

        t0, t1 = self.get_current_xlim()
        time = self.app_state.time

        if time is not None:
            vals = np.asarray(time)
            data_t0, data_t1 = float(vals[0]), float(vals[-1])
            if t1 - t0 < 0.01 or t0 < data_t0 - 1000 or t1 > data_t1 + 1000:
                view_span = self.app_state.view_span
                t0 = data_t0
                t1 = min(data_t0 + view_span, data_t1)
                master = self._xlink_master or self._feature_plot
                master.vb.setXRange(t0, t1, padding=0)

        for plot in audio_plots:
            plot.update_plot(t0=t0, t1=t1)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.schedule_labels_redraw()

    # --- Time slider ---

    def _on_slider_time(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        center = getattr(self.app_state, "center_playback", False)
        visible = TimeRange(*self.get_current_xlim())
        if center or not visible.contains(time_s):
            half = self.app_state.view_span / 2.0
            master = self._xlink_master or self._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    # --- Audio playback (space key) ---

    def toggle_pause_resume(self):
        self.audio_player.toggle()

    def _get_first_visible_time(self) -> float:
        for plot in self._visible_plots():
            return plot.time_marker.value()
        return 0.0

    # --- Confidence overlay ---

    def show_confidence_plot(self, confidence_data, time_coord=None):
        self.overlay_manager.remove_overlay("confidence")

        if confidence_data is None or len(confidence_data) == 0:
            return

        if time_coord is None:
            time_coord = self.app_state.time_coord.values

        item = pg.PlotCurveItem(pen=pg.mkPen(color="k", width=2, style=pg.QtCore.Qt.DashLine))
        self.overlay_manager.add_scaled_overlay(
            "confidence",
            self.current_plot,
            item,
            time_coord,
            np.asarray(confidence_data, dtype=np.float64),
            tick_format="{:.2f}",
        )

    def hide_confidence_plot(self):
        self.overlay_manager.remove_overlay("confidence")

    # --- Amplitude envelope ---

    def draw_amplitude_envelope(
        self,
        time: np.ndarray,
        envelope: np.ndarray,
        threshold: float | None = None,
        thresholds: list[tuple[float, Any]] | None = None,
    ):
        self.clear_amplitude_envelope()

        host = self._get_amp_envelope_host()
        if host is None:
            return

        if thresholds is None and threshold is not None:
            default_pen = pg.mkPen(color=(255, 50, 50, 200), width=2, style=Qt.DashLine)
            if isinstance(threshold, (tuple, list)):
                thresholds = [(v, default_pen) for v in threshold]
            else:
                thresholds = [(threshold, default_pen)]

        vb = self.overlay_manager.add_viewbox_overlay(
            "amplitude_envelope",
            host,
            axis_label="Envelope",
            axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        vb.setZValue(1000)

        item = pg.PlotDataItem(
            time,
            envelope,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=2),
            downsample=10,
            downsampleMethod="peak",
        )
        vb.addItem(item)

        max_thresh = 0.0
        if thresholds:
            for value, pen in thresholds:
                vb.addItem(pg.InfiniteLine(pos=value, angle=0, pen=pen))
                max_thresh = max(max_thresh, float(value))

        env_max = max(float(envelope.max()), max_thresh * 1.5) if max_thresh > 0 else float(envelope.max())
        env_min = float(envelope.min())
        if env_min >= env_max:
            env_max = env_min + 1.0
        vb.setYRange(env_min, env_max, padding=0.05)

        t0, t1 = host.get_current_xlim()
        vb.setXRange(t0, t1, padding=0)

    def clear_amplitude_envelope(self):
        self.overlay_manager.remove_overlay("amplitude_envelope")

    def _get_amp_envelope_host(self):
        host = self._get_envelope_host_plot()
        if host is not None:
            return host
        if self._panel_visible["ephys"] and self.ephys_trace_plot.isVisible():
            return self.ephys_trace_plot
        plot = self.get_current_plot()
        if plot in self.line_plots:
            return plot
        return None

    # --- Envelope sibling trace ---

    def _get_envelope_host_plot(self):
        for plot in self.audio_trace_plots:
            if plot.isVisible():
                return plot
        return None

    def show_envelope_overlay(self):
        host = self._get_envelope_host_plot()
        if host is None:
            return

        self.hide_envelope_overlay()

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        metric = self.app_state.get_with_default("energy_metric")
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return

        env_time = env_time + buf_t0

        item = pg.PlotCurveItem(
            env_time,
            env_data,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=ENVELOPE_OVERLAY_WIDTH),
        )

        vb = self.overlay_manager.add_viewbox_overlay(
            "energy_envelope",
            host,
            host_items=[item],
            axis_label="Envelope",
            axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        host.addItem(item)

        self._sync_envelope_axis_to_host(host, vb)

        def on_host_y_changed():
            env_vb = self.overlay_manager.get_viewbox("energy_envelope")
            if env_vb is not None:
                self._sync_envelope_axis_to_host(host, env_vb)

        host.vb.sigYRangeChanged.connect(on_host_y_changed)
        self._envelope_y_updater = on_host_y_changed

        self._envelope_td = ThrottleDebounce(
            debounce_ms=ENVELOPE_OVERLAY_DEBOUNCE_MS,
            throttle_cb=self._refresh_envelope_data,
            debounce_cb=self._refresh_envelope_data,
        )

        def on_x_range_changed():
            if self.overlay_manager.has_overlay("energy_envelope"):
                self._envelope_td.trigger()

        host.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._envelope_xrange_updater = on_x_range_changed
        self._envelope_host = host

    @staticmethod
    def _sync_envelope_axis_to_host(host, env_vb):
        ymin, ymax = host.vb.viewRange()[1]
        if ymax > ymin:
            env_vb.setYRange(ymin, ymax, padding=0)

    def hide_envelope_overlay(self):
        host = self._envelope_host

        updater = self._envelope_xrange_updater
        if updater and host:
            try:
                host.vb.sigXRangeChanged.disconnect(updater)
            except (RuntimeError, TypeError):
                pass
        self._envelope_xrange_updater = None

        y_updater = self._envelope_y_updater
        if y_updater and host:
            try:
                host.vb.sigYRangeChanged.disconnect(y_updater)
            except (RuntimeError, TypeError):
                pass
        self._envelope_y_updater = None
        self._envelope_host = None

        td = self._envelope_td
        if td:
            td.stop()
            self._envelope_td = None

        self.overlay_manager.remove_overlay("energy_envelope")

    def _compute_current_envelope(self):
        if not self.overlay_manager.has_overlay("energy_envelope"):
            return None

        host = self._get_envelope_host_plot()
        if host is None:
            return None

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return None

        metric = self.app_state.get_with_default("energy_metric")
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return None

        env_time = env_time + buf_t0
        return env_time, env_data

    def _refresh_envelope_data(self):
        result = self._compute_current_envelope()
        if result is None:
            return
        env_time, env_data = result

        vb_entry = self.overlay_manager._vb_entries.get("energy_envelope")
        if vb_entry and vb_entry.host_items:
            vb_entry.host_items[0].setData(env_time, env_data)

        env_vb = self.overlay_manager.get_viewbox("energy_envelope")
        if env_vb is not None:
            host = self._get_envelope_host_plot()
            if host is not None:
                self._sync_envelope_axis_to_host(host, env_vb)

    def _load_envelope_data(self, host, t0, t1):
        audio_path = getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return None, None, None
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return None, None, None
        fs = loader.rate
        _, channel_idx = self.app_state.get_audio_source()
        start_idx = max(0, int(t0 * fs))
        stop_idx = min(len(loader), int(t1 * fs))
        if stop_idx <= start_idx:
            return None, None, None
        audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if audio_data.ndim > 1:
            ch = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, ch]
        return audio_data, fs, t0

    # --- Cache management ---

    def clear_audio_cache(self):
        SharedAudioCache.clear_cache()
        for plot in self.spectrogram_plots:
            plot.buffer._clear_buffer()
        for plot in self.audio_trace_plots:
            plot.set_source(None)
