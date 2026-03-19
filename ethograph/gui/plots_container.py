"""Unified flexible panel container for all layout scenarios.

Replaces both PlotContainer (video mode) and MultiPanelContainer (no-video mode)
with a single container that dynamically shows/hides panels based on loaded data.

Panel stack (top to bottom, each optional except Feature Plot):
  - AudioTrace  (only if audio; toggleable)
  - Spectrogram (only if audio; toggleable)
  - EphysTrace  (only if ephys folder)
  - Raster      (only if Kilosort spike data; toggleable)
  - Feature Plot (always present; switches between LinePlot / HeatmapPlot)
"""

from typing import Any, Dict

import numpy as np
from ethograph.gui.plots_timeseriessource import TimeRange
import pyqtgraph as pg
from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .app_constants import (
    ENVELOPE_OVERLAY_COLOR,
    ENVELOPE_OVERLAY_DEBOUNCE_MS,
    ENVELOPE_OVERLAY_WIDTH,
    PLOT_CONTAINER_SIZE_HINT_HEIGHT,
)

import ethograph as eto
from .audio_player import AudioPlayer
from .data_sources import build_audio_source_from_alignment
from .label_drawing_mixin import LabelDrawingMixin
from .plots_audiotrace import AudioTracePlot
from .plots_ephystrace import EphysTracePlot, get_loader as get_ephys_loader
from .plots_heatmap import HeatmapPlot
from .plots_lineplot import LinePlot
from .plots_base import ThrottleDebounce
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


# Panel size ratios keyed by (has_audio, has_ephys)
# Values: dict mapping panel_name -> fraction of splitter height
_PANEL_RATIOS = {
    # audio + ephys
    (True, True): {"audiotrace": 0.10, "spectrogram": 0.15, "neo": 0.15, "ephys": 0.20, "raster": 0.10, "feature": 0.30},
    # audio only
    (True, False): {"audiotrace": 0.20, "spectrogram": 0.30, "feature": 0.50},
    # ephys only
    (False, True): {"neo": 0.20, "ephys": 0.30, "raster": 0.15, "feature": 0.35},
    # nothing extra
    (False, False): {"feature": 1.0},
}


class UnifiedPanelContainer(LabelDrawingMixin, QWidget):
    """Unified container with dynamic panel visibility.

    All panels share the same x-axis via pyqtgraph linking.
    Labels, changepoints, and time markers are drawn on all visible panels.
    """

    plot_changed = Signal(str)
    labels_redraw_needed = Signal()
    spectrogram_overlay_shown = Signal()

    def __init__(self, napari_viewer, app_state, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.app_state = app_state

        # --- Plots ---
        self.audio_trace_plot = AudioTracePlot(app_state)
        self.spectrogram_plot = SpectrogramPlot(app_state)
        self.line_plot = LinePlot(napari_viewer, app_state)
        self.heatmap_plot = HeatmapPlot(app_state)
        self.neo_trace_plot = EphysTracePlot(app_state)   # Neo-Viewer panel
        self.ephys_trace_plot = EphysTracePlot(app_state)  # Phy-Viewer panel
        self.raster_plot = RasterPlot(app_state)

        # Feature panel: line_plot or heatmap_plot
        self._feature_plot = self.line_plot
        self._feature_type = "lineplot"

        # current_plot semantics: always the feature (bottom) panel
        self.current_plot = self._feature_plot
        self.current_plot_type = self._feature_type

        # --- Panel visibility state ---
        self._audiotrace_visible = False
        self._spectrogram_visible = False
        self._neo_visible = False
        self._ephys_visible = False
        self._raster_visible = False
        self._featureplot_visible = True

        # --- Mixin state ---
        self.label_mappings: Dict[int, Dict[str, Any]] = {}
        self.audio_overlay_type = None
        self.audio_cp_items: list = []
        self.osc_event_items: list = []
        self.dataset_cp_items: list = []

        self.overlay_manager = OverlayManager()
        self.line_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.line_plot)
        )
        self.audio_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.audio_trace_plot)
        )
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

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self._splitter)

        # Time slider (shown when no video)
        self.time_slider = TimeSlider()
        self.time_slider.time_changed.connect(self._on_slider_time)
        self.time_slider.hide()
        main_layout.addWidget(self.time_slider)

        # Audio playback (no-video mode)
        self.audio_player = AudioPlayer(
            app_state,
            get_xlim=self.get_current_xlim,
            get_visible_time=self._get_first_visible_time,
            update_marker=self.update_time_marker_by_time,
        )

        # --- X-axis linking: all panels link to the x-axis master ---
        # The master is whichever panel is first visible; we use audio_trace_plot
        # if audio is present, otherwise the feature plot.
        self._xlink_master = None

        # Connect zoom events for changepoint line style updates
        for plot in (self.spectrogram_plot, self.audio_trace_plot,
                     self.heatmap_plot, self.neo_trace_plot, self.ephys_trace_plot, self.raster_plot):
            plot.vb.sigRangeChanged.connect(self._on_plot_zoom)

        # Bidirectional y-axis sync between ephys trace and raster
        self._syncing_y = False
        self.ephys_trace_plot.vb.sigYRangeChanged.connect(self._sync_raster_y_from_ephys)
        self.raster_plot.y_range_changed.connect(self._sync_ephys_y_from_raster)
        self.ephys_trace_plot.y_space_changed.connect(self._on_ephys_y_space_changed)

        # Track which panel was last clicked (for changepoint navigation)
        self._last_clicked_panel = "feature"
        self.audio_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'audio'))
        self.spectrogram_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'audio'))
        self.line_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'feature'))
        self.heatmap_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'feature'))
        self.neo_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'neo'))
        self.ephys_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'ephys'))
        self.raster_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'raster'))

        # Initially only the feature plot is in the splitter
        self._splitter.addWidget(self.line_plot)
        self.heatmap_plot.hide()
        self.audio_trace_plot.hide()
        self.spectrogram_plot.hide()
        self.neo_trace_plot.hide()
        self.ephys_trace_plot.hide()
        self.raster_plot.hide()

    def sizeHint(self):
        return QSize(self.width(), PLOT_CONTAINER_SIZE_HINT_HEIGHT)

    def _on_plot_zoom(self):
        self.update_audio_changepoint_styles()
        self.update_oscillatory_event_styles()

    # ------------------------------------------------------------------
    # Panel configuration
    # ------------------------------------------------------------------

    def configure_panels(self):
        """Called after data load to set up which panels are available."""
        # Show/hide time slider (shown when no video)
        if not self.app_state.has_video:
            self.time_slider.show()
        else:
            self.time_slider.hide()

        # Rebuild splitter contents
        self._rebuild_splitter()

    def _rebuild_splitter(self):
        """Rebuild the splitter with currently visible panels."""
        # Remove all widgets from splitter
        while self._splitter.count():
            w = self._splitter.widget(0)
            w.hide()
            w.setParent(None)

        panels_in_order = []

        if self.app_state.has_audio and self._audiotrace_visible:
            panels_in_order.append(("audiotrace", self.audio_trace_plot))
        if self.app_state.has_audio and self._spectrogram_visible:
            panels_in_order.append(("spectrogram", self.spectrogram_plot))
        if self._neo_visible:
            panels_in_order.append(("neo", self.neo_trace_plot))
        if self.app_state.has_ephys and self._ephys_visible:
            panels_in_order.append(("ephys", self.ephys_trace_plot))
        if self.app_state.has_ephys and self._raster_visible:
            panels_in_order.append(("raster", self.raster_plot))

        if self._featureplot_visible:
            panels_in_order.append(("feature", self._feature_plot))

        for _, widget in panels_in_order:
            self._splitter.addWidget(widget)
            widget.show()

        # Set up x-axis linking
        self._setup_xlinks(panels_in_order)

        # Hide x-axis labels on non-bottom panels
        for i, (_, widget) in enumerate(panels_in_order):
            if i < len(panels_in_order) - 1:
                widget.plotItem.getAxis("bottom").setStyle(showValues=False)
                widget.plotItem.setLabel("bottom", "")
            else:
                widget.plotItem.getAxis("bottom").setStyle(showValues=True)

        # Apply zoom constraints to all newly added panels (using tight bounds)
        self._apply_all_zoom_constraints()

        # Apply panel size ratios
        QTimer.singleShot(0, self._apply_panel_sizes)

    def _setup_xlinks(self, panels_in_order):
        """Link all panels to the first panel's x-axis."""
        if not panels_in_order:
            return

        master = panels_in_order[0][1]
        self._xlink_master = master

        for i, (_, widget) in enumerate(panels_in_order):
            if i > 0:
                widget.plotItem.setXLink(master.plotItem)

        # Also link hidden plots that might be swapped in later
        for plot in (self.line_plot, self.heatmap_plot, self.neo_trace_plot):
            if plot not in [w for _, w in panels_in_order]:
                plot.plotItem.setXLink(master.plotItem)

    def _apply_panel_sizes(self):
        total = self._splitter.height()
        if total <= 0:
            return

        visible_audio = self.app_state.has_audio and self._audiotrace_visible
        visible_spec = self.app_state.has_audio and self._spectrogram_visible
        visible_ephys = self.app_state.has_ephys and self._ephys_visible
        visible_neo = self._neo_visible
        visible_neural = visible_ephys or visible_neo or (self.app_state.has_ephys and self._raster_visible)

        ratios = _PANEL_RATIOS.get(
            (visible_audio or visible_spec, visible_neural),
            {"feature": 1.0}
        )

        visible_raster = self.app_state.has_ephys and self._raster_visible

        raw = []
        if visible_audio:
            raw.append(ratios.get("audiotrace", 0.2))
        if visible_spec:
            raw.append(ratios.get("spectrogram", 0.3))
        if visible_neo:
            raw.append(ratios.get("neo", 0.2))
        if visible_ephys:
            raw.append(ratios.get("ephys", 0.3))
        if visible_raster:
            raw.append(ratios.get("raster", 0.15))
        if self._featureplot_visible:
            raw.append(ratios.get("feature", 0.5))

        if raw and len(raw) == self._splitter.count():
            scale = 1.0 / sum(raw)
            sizes = [int(total * r * scale) for r in raw]
            self._splitter.setSizes(sizes)

    # ------------------------------------------------------------------
    # Panel visibility toggles
    # ------------------------------------------------------------------

    def set_audiotrace_visible(self, visible: bool):
        if self._audiotrace_visible == visible:
            return
        self._audiotrace_visible = visible
        self._rebuild_splitter()

    def set_spectrogram_visible(self, visible: bool):
        if self._spectrogram_visible == visible:
            return
        self._spectrogram_visible = visible
        self._rebuild_splitter()

    def set_feature_view(self, mode: str):
        """Switch the feature (bottom) panel.

        Args:
            mode: "lineplot" or "heatmap"
        """
        if mode == "heatmap" and self._feature_type != "heatmap":
            self._swap_feature_panel(self.heatmap_plot, "heatmap")
        elif mode == "lineplot" and self._feature_type != "lineplot":
            self._swap_feature_panel(self.line_plot, "lineplot")

    def _swap_feature_panel(self, new_plot, new_type):
        prev_xlim = self._feature_plot.get_current_xlim()
        prev_marker = self._feature_plot.time_marker.value()
        sizes = self._splitter.sizes()

        idx = self._splitter.indexOf(self._feature_plot)
        self._feature_plot.hide()

        self._splitter.insertWidget(idx, new_plot)
        new_plot.show()

        # Re-link x-axis
        if self._xlink_master and new_plot is not self._xlink_master:
            new_plot.plotItem.setXLink(self._xlink_master.plotItem)

        self._feature_plot = new_plot
        self._feature_type = new_type
        self.current_plot = new_plot
        self.current_plot_type = new_type

        self._splitter.setSizes(sizes)
        new_plot.set_x_range(mode="preserve", curr_xlim=prev_xlim)
        new_plot.update_time_marker(prev_marker)
        tight_bounds = self._compute_tight_x_bounds()
        new_plot._apply_zoom_constraints(x_bounds_override=tight_bounds)

        self.plot_changed.emit(new_type)
        self.labels_redraw_needed.emit()

    # ------------------------------------------------------------------
    # Ephys panel show/hide
    # ------------------------------------------------------------------

    def set_neo_visible(self, visible: bool):
        if self._neo_visible == visible:
            return
        self._neo_visible = visible
        self._rebuild_splitter()

    def show_ephys_panel(self):
        if self._ephys_visible or self._raster_visible:
            return
        self._ephys_visible = True
        self._rebuild_splitter()

    def hide_ephys_panel(self):
        if not self._ephys_visible and not self._raster_visible:
            return
        self._ephys_visible = False
        self._raster_visible = False
        self._rebuild_splitter()

    def set_raster_visible(self, visible: bool):
        if self._raster_visible == visible:
            return
        self._raster_visible = visible
        self._rebuild_splitter()

    def set_neural_panel_mode(self, mode: str):
        """Switch between 'trace' and 'raster' for the neural panel slot.

        Both panels share the same y-axis space. Only one is visible at a time.
        """
        if mode == "trace":
            show_ephys, show_raster = True, False
        elif mode == "raster":
            show_ephys, show_raster = False, True
        else:
            return

        changed = False
        if self._ephys_visible != show_ephys:
            self._ephys_visible = show_ephys
            changed = True
        if self._raster_visible != show_raster:
            self._raster_visible = show_raster
            changed = True
        if changed:
            self._rebuild_splitter()

    def set_featureplot_visible(self, visible: bool):
        if self._featureplot_visible == visible:
            return
        self._featureplot_visible = visible
        self._rebuild_splitter()

    # ------------------------------------------------------------------
    # Bidirectional y-axis sync: ephys <-> raster
    # ------------------------------------------------------------------

    def _sync_raster_y_from_ephys(self):
        if self._syncing_y or not self._raster_visible:
            return
        self._syncing_y = True
        try:
            y_lo, y_hi = self.ephys_trace_plot.vb.viewRange()[1]
            self.raster_plot.vb.setYRange(y_lo, y_hi, padding=0)
        finally:
            self._syncing_y = False

    def _sync_ephys_y_from_raster(self):
        if self._syncing_y or not self._raster_visible:
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
        return self._feature_plot

    def get_current_xlim(self):
        # Use the x-axis master if available, otherwise feature plot
        master = self._xlink_master or self._feature_plot
        return master.get_current_xlim()

    def set_x_range(self, mode="default", curr_xlim=None, center_on_frame=None):
        master = self._xlink_master or self._feature_plot
        return master.set_x_range(
            mode=mode, curr_xlim=curr_xlim, center_on_frame=center_on_frame,
        )

    @property
    def vb(self):
        return self._feature_plot.vb

    def get_hovered_plot(self):
        for plot in self._visible_plots():
            if plot.underMouse():
                return plot
        return self._feature_plot

    def _visible_plots(self):
        for i in range(self._splitter.count()):
            w = self._splitter.widget(i)
            if w and w.isVisible():
                yield w

    def update_time_marker_by_time(self, time_s: float):
        for plot in self._visible_plots():
            plot.update_time_marker(time_s)
        self.time_slider.set_slider_time(time_s)

    def update_time_marker_and_window(self, frame_number):
        video = getattr(self.app_state, 'video', None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps
        for plot in self._visible_plots():
            plot.update_time_marker(current_time)
        self.time_slider.set_slider_time(current_time)

    def apply_y_range(self, ymin, ymax):
        return self._feature_plot.apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        tight_bounds = self._compute_tight_x_bounds()
        for plot in self._visible_plots():
            plot.toggle_axes_lock(x_bounds_override=tight_bounds)

    def _apply_all_zoom_constraints(self):
        
        tight_bounds = self._compute_tight_x_bounds()
        for plot in self._visible_plots():
            plot._apply_zoom_constraints(x_bounds_override=tight_bounds)

    def _compute_tight_x_bounds(self):
        """Compute the tightest x-bounds across all visible panels.

        Uses the intersection so no panel scrolls past another's data.
        """
        result: TimeRange | None = None
        for plot in self._visible_plots():
            bounds = plot._get_time_bounds()
            if bounds is None:
                continue
            tr = TimeRange(*bounds)
            result = tr if result is None else result.intersect(tr)
            if result is None:
                return None
        if result is not None:
            return (result.start_s, result.end_s)
        return None

    # --- Bottom panel switching ---

    def switch_to_lineplot(self):
        if self._feature_type == "lineplot":
            return
        self._swap_feature_panel(self.line_plot, "lineplot")

    def switch_to_heatmap(self):
        if self._feature_type == "heatmap":
            return
        self._swap_feature_panel(self.heatmap_plot, "heatmap")

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
        return self._ephys_visible or self._raster_visible

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
        """Refresh audio-driven panels (waveform + spectrogram) after mic change."""
        source = build_audio_source_from_alignment(self.app_state)
        self.spectrogram_plot.set_source(source)

        alignment = getattr(self.app_state, 'trial_alignment', None)
        if alignment and "audio" in alignment.continuous:
            self.audio_trace_plot.set_source(alignment.continuous["audio"])
        else:
            self.audio_trace_plot.set_source(None)

        t0, t1 = self.get_current_xlim()
        time = self.app_state.time
        
        
        if time is not None:
            vals = np.asarray(time)
            data_t0, data_t1 = float(vals[0]), float(vals[-1])
            if t1 - t0 < 0.01 or t0 < data_t0 - 1000 or t1 > data_t1 + 1000:
                window = self.app_state.get_with_default("window_size")
                t0 = data_t0
                t1 = min(data_t0 + float(window), data_t1)
                master = self._xlink_master or self._feature_plot
                master.vb.setXRange(t0, t1, padding=0)

        if self._audiotrace_visible:
            self.audio_trace_plot.update_plot(t0=t0, t1=t1)
            self.audio_trace_plot.vb.enableAutoRange(x=False, y=True)
            self.audio_trace_plot._apply_y_constraints()
        if self._spectrogram_visible:
            self.spectrogram_plot.update_plot(t0=t0, t1=t1)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.update_time_range_from_data()

    # --- Time slider ---

    def _on_slider_time(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        visible = TimeRange(*self.get_current_xlim())
        if not visible.contains(time_s):
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            master = self._xlink_master or self._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    def update_time_range_from_data(self):
        time = self.app_state.time
        if time is not None:
            vals = np.asarray(time)
            self.time_slider.set_time_range(float(vals[0]), float(vals[-1]))
            return

        audio_path = getattr(self.app_state, 'audio_path', None)
        if audio_path:
            loader = SharedAudioCache.get_loader(audio_path)
            if loader is not None and len(loader) > 0:
                duration = len(loader) / loader.rate
                self.time_slider.set_time_range(0.0, duration)
                return

        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if ephys_path:
            loader = get_ephys_loader(ephys_path, stream_id=stream_id)
            if loader is not None and len(loader) > 0:
                duration = len(loader) / loader.rate
                self.time_slider.set_time_range(0.0, duration)

    # --- Audio playback (space key) ---

    def toggle_pause_resume(self):
        self.audio_player.toggle()

    def _get_first_visible_time(self) -> float:
        for plot in self._visible_plots():
            return plot.time_marker.value()
        return 0.0

    # --- Confidence overlay ---

    def show_confidence_plot(self, confidence_data):
        self.overlay_manager.remove_overlay('confidence')

        if confidence_data is None or len(confidence_data) == 0:
            return

        time_coord = self.app_state
        
        
        item = pg.PlotCurveItem(
            pen=pg.mkPen(color='k', width=2, style=pg.QtCore.Qt.DashLine)
        )
        self.overlay_manager.add_scaled_overlay(
            'confidence',
            self.current_plot,
            item,
            time_coord.values,
            np.asarray(confidence_data, dtype=np.float64),
            tick_format="{:.2f}",
        )

    def hide_confidence_plot(self):
        self.overlay_manager.remove_overlay('confidence')

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
            'amplitude_envelope', host,
            axis_label='Envelope', axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        vb.setZValue(1000)

        item = pg.PlotDataItem(
            time, envelope,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=2),
            downsample=10, downsampleMethod='peak',
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
        self.overlay_manager.remove_overlay('amplitude_envelope')

    def _get_amp_envelope_host(self):
        if self._audiotrace_visible and self.audio_trace_plot.isVisible():
            return self.audio_trace_plot
        if self._ephys_visible and self.ephys_trace_plot.isVisible():
            return self.ephys_trace_plot
        if self._feature_type == "lineplot":
            return self.line_plot
        return None

    # --- Envelope sibling trace ---

    def _get_envelope_target(self) -> str:
        return getattr(self.app_state, '_envelope_target', 'audio')

    def _get_envelope_host_plot(self):
        target = self._get_envelope_target()
        if target == "audio":
            if self._audiotrace_visible and self.audio_trace_plot.isVisible():
                return self.audio_trace_plot
            return None
        if target == "ephys":
            if not self._ephys_visible or not self.ephys_trace_plot.isVisible():
                return None
            if self.ephys_trace_plot._multichannel:
                return None
            return self.ephys_trace_plot
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

        metric = self.app_state.get_with_default('energy_metric')
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return

        env_time = env_time + buf_t0

        item = pg.PlotCurveItem(
            env_time, env_data,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=ENVELOPE_OVERLAY_WIDTH),
        )

        vb = self.overlay_manager.add_viewbox_overlay(
            'energy_envelope', host,
            host_items=[item],
            axis_label='Envelope', axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        host.addItem(item)

        self._sync_envelope_axis_to_host(host, vb)

        def on_host_y_changed():
            env_vb = self.overlay_manager.get_viewbox('energy_envelope')
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
            if self.overlay_manager.has_overlay('energy_envelope'):
                self._envelope_td.trigger()

        host.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._envelope_xrange_updater = on_x_range_changed

    @staticmethod
    def _sync_envelope_axis_to_host(host, env_vb):
        ymin, ymax = host.vb.viewRange()[1]
        if ymax > ymin:
            env_vb.setYRange(ymin, ymax, padding=0)

    def hide_envelope_overlay(self):
        updater = getattr(self, '_envelope_xrange_updater', None)
        if updater:
            for plot in (self.audio_trace_plot, self.neo_trace_plot, self.ephys_trace_plot):
                try:
                    plot.vb.sigXRangeChanged.disconnect(updater)
                except (RuntimeError, TypeError):
                    pass
            self._envelope_xrange_updater = None

        y_updater = getattr(self, '_envelope_y_updater', None)
        if y_updater:
            for plot in (self.audio_trace_plot, self.neo_trace_plot, self.ephys_trace_plot):
                try:
                    plot.vb.sigYRangeChanged.disconnect(y_updater)
                except (RuntimeError, TypeError):
                    pass
            self._envelope_y_updater = None

        td = getattr(self, '_envelope_td', None)
        if td:
            td.stop()
            self._envelope_td = None

        self.overlay_manager.remove_overlay('energy_envelope')

    def _compute_current_envelope(self):
        if not self.overlay_manager.has_overlay('energy_envelope'):
            return None

        host = self._get_envelope_host_plot()
        if host is None:
            return None

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return None

        metric = self.app_state.get_with_default('energy_metric')
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

        vb_entry = self.overlay_manager._vb_entries.get('energy_envelope')
        if vb_entry and vb_entry.host_items:
            vb_entry.host_items[0].setData(env_time, env_data)

        env_vb = self.overlay_manager.get_viewbox('energy_envelope')
        if env_vb is not None:
            host = self._get_envelope_host_plot()
            if host is not None:
                self._sync_envelope_axis_to_host(host, env_vb)

    def _load_envelope_data(self, host, t0, t1):
        target = self._get_envelope_target()
        if target == "audio":
            audio_path = getattr(self.app_state, 'audio_path', None)
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
        elif target == "ephys":
            ephys_path, stream_id, _ = self.app_state.get_ephys_source()
            if not ephys_path:
                return None, None, None
            loader = get_ephys_loader(ephys_path, stream_id=stream_id)
            if loader is None:
                return None, None, None
            fs = loader.rate
            channel = self.ephys_trace_plot.buffer.channel
            start_idx = max(0, int(t0 * fs))
            stop_idx = min(len(loader), int(t1 * fs))
            if stop_idx <= start_idx:
                return None, None, None
            raw = loader[start_idx:stop_idx]
            if raw.ndim == 1:
                ephys_data = raw.astype(np.float64)
            else:
                ch = min(channel, raw.shape[1] - 1)
                ephys_data = np.asarray(raw[:, ch], dtype=np.float64)
            return ephys_data, fs, t0
        return None, None, None

    # --- Cache management ---

    def clear_audio_cache(self):
        SharedAudioCache.clear_cache()
        if hasattr(self.spectrogram_plot, "buffer"):
            self.spectrogram_plot.buffer._clear_buffer()
        if hasattr(self.audio_trace_plot, "buffer"):
            self.audio_trace_plot.buffer.set_source(None)
