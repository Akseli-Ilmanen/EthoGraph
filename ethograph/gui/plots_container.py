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
from ethograph.labels.intervals import find_interval_at, get_interval_bounds
from .audio_player import AudioPlayer
from .data_sources import build_audio_source
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


# Panel size ratios keyed by (has_audio, has_kilosort_or_neo)
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

# Ordered list of (panel_name, app_state_guard_attr | None)
# guard_attr: app_state boolean that must be True for the panel to appear; None = always allowed
_PANEL_ORDER = [
    ("audiotrace", "has_audio"),
    ("spectrogram", "has_audio"),
    ("neo", None),
    ("ephys", "has_kilosort"),
    ("raster", "has_kilosort"),
    ("feature", None),
]

# Maps panel name -> widget attribute name on the container (except "feature" which is dynamic)
_PANEL_PLOT_ATTR = {
    "audiotrace": "audio_trace_plot",
    "spectrogram": "spectrogram_plot",
    "neo": "neo_trace_plot",
    "ephys": "ephys_trace_plot",
    "raster": "raster_plot",
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
        self._panel_visible: dict[str, bool] = {
            "audiotrace": False,
            "spectrogram": False,
            "neo": False,
            "ephys": False,
            "raster": False,
            "feature": True,
        }

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
        self._envelope_host = None

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

        # --- Current-label floating indicator (top-right corner) ---
        self._label_indicator = CurrentLabelIndicator(self)

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
        self.ephys_trace_plot.seek_time_requested.connect(self._on_seek_time_requested)

        # Track which panel was last clicked (for changepoint navigation)
        self._last_clicked_panel = "feature"
        self.audio_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'audio'))
        self.spectrogram_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'audio'))
        self.line_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'feature'))
        self.heatmap_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'feature'))
        self.neo_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'neo'))
        self.ephys_trace_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'ephys'))
        self.raster_plot.plot_clicked.connect(lambda _: setattr(self, '_last_clicked_panel', 'raster'))

        # Add all panels to the splitter once, hidden. We never reparent them.
        # setVisible() is used to toggle; QSplitter collapses hidden children to zero.
        # Both line_plot and heatmap_plot live at the "feature" slot; only one is visible.
        for name, _ in _PANEL_ORDER:
            w = self._get_panel_widget(name)
            self._splitter.addWidget(w)
            w.setVisible(name == "feature")
        self._splitter.addWidget(self.heatmap_plot)
        self.heatmap_plot.hide()

        # Track user-adjusted sizes so they survive panel toggles
        self._user_sizes: dict[str, int] = {}
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

    def sizeHint(self):
        return QSize(self.width(), PLOT_CONTAINER_SIZE_HINT_HEIGHT)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._label_indicator._reposition()

    def _update_label_indicator(self, time_s: float):
        df = self.app_state.label_intervals
        mappings = self.label_mappings
        if df is None or df.empty or not mappings:
            self._label_indicator.hide()
            return

        ind = getattr(self.app_state, 'individuals_sel', None)
        if ind is None or ind in ("", "None"):
            ds = self.app_state.ds
            if ds is not None and 'individuals' in ds.coords:
                ind = str(ds.coords['individuals'].values[0])
            else:
                ind = "default"

        idx = find_interval_at(df, time_s, ind)
        if idx is not None:
            _, _, label_id = get_interval_bounds(df, idx)
            if label_id in mappings and label_id != 0:
                entry = mappings[label_id]
                color = entry["color"]
                color_list = color.tolist() if hasattr(color, 'tolist') else list(color)
                self._label_indicator.update_label(entry["name"], color_list)
                return

        self._label_indicator.hide()

    def _on_plot_zoom(self):
        self.update_audio_changepoint_styles()
        self.update_oscillatory_event_styles()

    # ------------------------------------------------------------------
    # Panel configuration
    # ------------------------------------------------------------------

    def configure_panels(self):
        """Called after data load to set up which panels are available."""
        if not self.app_state.has_video:
            self.time_slider.show()
        else:
            self.time_slider.hide()

        # Clear saved user sizes so defaults apply for the new dataset
        self._user_sizes.clear()
        self._update_panel_visibility()

    def _get_panel_widget(self, name: str):
        if name == "feature":
            return self._feature_plot
        return getattr(self, _PANEL_PLOT_ATTR[name])

    def _on_splitter_moved(self, pos: int, index: int):
        """Save user-adjusted sizes keyed by panel name."""
        sizes = self._splitter.sizes()
        for name, _ in _PANEL_ORDER:
            widget = self._get_panel_widget(name)
            idx = self._splitter.indexOf(widget)
            if idx >= 0 and widget.isVisible():
                self._user_sizes[name] = sizes[idx]
        # heatmap_plot at last slot
        hm_idx = self._splitter.indexOf(self.heatmap_plot)
        if hm_idx >= 0 and self.heatmap_plot.isVisible():
            self._user_sizes["feature"] = sizes[hm_idx]

    def _visible_panel_names(self) -> list[str]:
        result = []
        for name, guard in _PANEL_ORDER:
            if guard and not getattr(self.app_state, guard, False):
                continue
            if self._panel_visible[name]:
                result.append(name)
        return result

    def _update_panel_visibility(self):
        """Show/hide panels in-place; never reparents widgets."""
        visible_names = self._visible_panel_names()
        visible_set = set(visible_names)

        for name, guard in _PANEL_ORDER:
            widget = self._get_panel_widget(name)
            should_show = name in visible_set
            widget.setVisible(should_show)

        self._setup_xlinks_from_visible(visible_names)

        # Hide x-axis ticks on all panels except the bottom-most visible one
        for i, name in enumerate(visible_names):
            widget = self._get_panel_widget(name)
            is_last = (i == len(visible_names) - 1)
            widget.plotItem.getAxis("bottom").setStyle(showValues=is_last)
            if not is_last:
                widget.plotItem.setLabel("bottom", "")

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)

    def _setup_xlinks_from_visible(self, visible_names: list[str] | None = None):
        """Link all panels to the first visible panel's x-axis."""
        if visible_names is None:
            visible_names = self._visible_panel_names()
        if not visible_names:
            return

        master = self._get_panel_widget(visible_names[0])
        self._xlink_master = master

        for i, name in enumerate(visible_names):
            widget = self._get_panel_widget(name)
            if i > 0:
                widget.plotItem.setXLink(master.plotItem)

        # Keep hidden swap-candidates linked too
        for plot in (self.line_plot, self.heatmap_plot, self.neo_trace_plot):
            if plot is not master:
                plot.plotItem.setXLink(master.plotItem)

    def _apply_panel_sizes(self):
        total = self._splitter.height()
        if total <= 0:
            return

        v = self._panel_visible
        has_audio_panel = self.app_state.has_audio and (v["audiotrace"] or v["spectrogram"])
        has_neural_panel = v["neo"] or (self.app_state.has_kilosort and (v["ephys"] or v["raster"]))
        ratios = _PANEL_RATIOS.get((has_audio_panel, has_neural_panel), {"feature": 1.0})

        visible_names = self._visible_panel_names()

        # Build raw sizes: prefer user-saved sizes, fall back to ratio defaults.
        raw = []
        for name in visible_names:
            if name in self._user_sizes:
                raw.append(self._user_sizes[name])
            else:
                raw.append(ratios.get(name, 0.2) * total)

        # When neo and phy panels coexist, enforce a 1:5 size ratio between them
        # only if neither has a user-saved size yet.
        phy_names = {"ephys", "raster"}
        if "neo" in visible_names and any(n in phy_names for n in visible_names):
            if "neo" not in self._user_sizes and not any(n in self._user_sizes for n in phy_names if n in visible_names):
                neo_i = visible_names.index("neo")
                phy_indices = [i for i, n in enumerate(visible_names) if n in phy_names]
                neo_phy_total = raw[neo_i] + sum(raw[j] for j in phy_indices)
                raw[neo_i] = neo_phy_total / 6
                phy_raw_total = sum(raw[j] for j in phy_indices)
                for j in phy_indices:
                    raw[j] = raw[j] / phy_raw_total * (neo_phy_total * 5 / 6)

        if not raw:
            return

        # Scale to fill total height
        scale = total / sum(raw)
        sizes = [int(r * scale) for r in raw]

        # Build full sizes list for all splitter children (hidden = 0).
        # Map from panel name → computed size for visible panels.
        size_map = dict(zip(visible_names, sizes))
        all_sizes = []
        for name, _ in _PANEL_ORDER:
            all_sizes.append(size_map.get(name, 0))
        # heatmap_plot is last splitter child; gets the feature size when it's active
        all_sizes.append(size_map.get("feature", 0) if self.heatmap_plot.isVisible() else 0)
        # When heatmap is active, line_plot slot should be 0
        if self.heatmap_plot.isVisible():
            feature_idx = next(i for i, (n, _) in enumerate(_PANEL_ORDER) if n == "feature")
            all_sizes[feature_idx] = 0
        self._splitter.setSizes(all_sizes)

    # ------------------------------------------------------------------
    # Panel visibility toggles
    # ------------------------------------------------------------------

    def _set_panel_visible(self, name: str, visible: bool):
        if self._panel_visible[name] == visible:
            return
        self._panel_visible[name] = visible
        self._update_panel_visibility()

    def set_audiotrace_visible(self, visible: bool):
        self._set_panel_visible("audiotrace", visible)

    def set_spectrogram_visible(self, visible: bool):
        self._set_panel_visible("spectrogram", visible)

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

        old_idx = self._splitter.indexOf(self._feature_plot)
        new_idx = self._splitter.indexOf(new_plot)
        sizes = list(self._splitter.sizes())

        # Transfer the old feature's height to the new one
        if old_idx >= 0 and new_idx >= 0:
            sizes[new_idx] = sizes[old_idx]
            sizes[old_idx] = 0

        self._feature_plot.hide()
        new_plot.show()

        if self._xlink_master and new_plot is not self._xlink_master:
            new_plot.plotItem.setXLink(self._xlink_master.plotItem)

        self._feature_plot = new_plot
        self._feature_type = new_type
        self.current_plot = new_plot
        self.current_plot_type = new_type

        self._splitter.setSizes(sizes)
        new_plot.set_x_range(mode="preserve", curr_xlim=prev_xlim)
        new_plot.update_time_marker(prev_marker)
        new_plot._apply_zoom_constraints(x_bounds_override=self._trial_bounds_tuple())

        self.plot_changed.emit(new_type)
        self.labels_redraw_needed.emit()

    # ------------------------------------------------------------------
    # Ephys panel show/hide
    # ------------------------------------------------------------------

    def set_neo_visible(self, visible: bool):
        self._set_panel_visible("neo", visible)

    def set_ephys_visible(self, visible: bool):
        if self._panel_visible["ephys"] == visible:
            return
        self._panel_visible["ephys"] = visible
        if not visible:
            self._panel_visible["raster"] = False
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

    def set_featureplot_visible(self, visible: bool):
        self._set_panel_visible("feature", visible)

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
        self._update_label_indicator(time_s)

    def _on_seek_time_requested(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        video = getattr(self.app_state, 'video', None)
        if video:
            frame = video.time_to_frame(time_s)
            video.blockSignals(True)
            video.seek_to_frame(frame)
            video.blockSignals(False)
            self.app_state.current_frame = frame

    def update_time_marker_and_window(self, frame_number):
        video = getattr(self.app_state, 'video', None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps
        for plot in self._visible_plots():
            plot.update_time_marker(current_time)
        self.time_slider.set_slider_time(current_time)
        self._update_label_indicator(current_time)

    def apply_y_range(self, ymin, ymax):
        return self._feature_plot.apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot.toggle_axes_lock(x_bounds_override=bounds)

    def _apply_all_zoom_constraints(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot._apply_zoom_constraints(x_bounds_override=bounds)

    def _trial_bounds_tuple(self):
        tr = self.app_state.trial_bounds
        return (tr.start_s, tr.end_s) if tr is not None else None

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
        """Refresh audio-driven panels (waveform + spectrogram) after mic change."""
        source = build_audio_source(self.app_state)
        self.spectrogram_plot.set_source(source)
        self.audio_trace_plot.set_source(source)

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

        if self._panel_visible["audiotrace"]:
            self.audio_trace_plot.update_plot(t0=t0, t1=t1)
            self.audio_trace_plot.vb.enableAutoRange(x=False, y=True)
            self.audio_trace_plot._apply_y_constraints()
        if self._panel_visible["spectrogram"]:
            self.spectrogram_plot.update_plot(t0=t0, t1=t1)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.update_time_range_from_data()

    # --- Time slider ---

    def _on_slider_time(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        center = getattr(self.app_state, 'center_playback', False)
        visible = TimeRange(*self.get_current_xlim())
        if center or not visible.contains(time_s):
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            master = self._xlink_master or self._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    def update_time_range_from_data(self):
        tr = self.app_state.trial_bounds
        if tr is not None and tr.duration > 0:
            self.time_slider.set_time_range(tr.start_s, tr.end_s)

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
        if self._panel_visible["audiotrace"] and self.audio_trace_plot.isVisible():
            return self.audio_trace_plot
        if self._panel_visible["ephys"] and self.ephys_trace_plot.isVisible():
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
            if self._panel_visible["audiotrace"] and self.audio_trace_plot.isVisible():
                return self.audio_trace_plot
            return None
        if target == "ephys":
            if not self._panel_visible["ephys"] or not self.ephys_trace_plot.isVisible():
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
        self.audio_trace_plot.set_source(None)
