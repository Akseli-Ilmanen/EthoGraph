"""Simple container widget for switching between different plot types."""

from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .app_constants import (
    PLOT_CONTAINER_SIZE_HINT_HEIGHT,
    SPECTROGRAM_OVERLAY_OPACITY,
    SPECTROGRAM_OVERLAY_DEBOUNCE_MS,
    SPECTROGRAM_OVERLAY_ZOOM_OUT_THRESHOLD,
    SPECTROGRAM_OVERLAY_ZOOM_IN_THRESHOLD,
    ENVELOPE_OVERLAY_DEBOUNCE_MS,
    ENVELOPE_OVERLAY_COLOR,
    ENVELOPE_OVERLAY_WIDTH,
    PREDICTION_LABELS_HEIGHT_RATIO,
    SPECTROGRAM_LABELS_HEIGHT_RATIO,
    PREDICTION_FALLBACK_Y_TOP,
    PREDICTION_FALLBACK_Y_HEIGHT,
    SPECTROGRAM_FALLBACK_Y_HEIGHT,
    CP_ZOOM_VERY_OUT_THRESHOLD,
    CP_ZOOM_MEDIUM_THRESHOLD,
    CP_LINE_WIDTH_THIN,
    CP_LINE_WIDTH_MEDIUM,
    CP_LINE_WIDTH_THICK,
    CP_COLOR_WAVEFORM,
    CP_COLOR_SPECTROGRAM,
    CP_METHOD_COLORS,
    CP_SCATTER_SIZE,
    CP_SCATTER_Y_POSITION_RATIO,
    Z_INDEX_LABELS,
    Z_INDEX_PREDICTIONS,
    Z_INDEX_CHANGEPOINTS,
)
from .plots_audiotrace import AudioTracePlot
from .plots_heatmap import HeatmapPlot
from .plots_lineplot import LinePlot
from .plots_spectrogram import SpectrogramPlot
from .widgets_plot import OverlayManager


class PlotContainer(QWidget):
    """Container that holds and switches between LinePlot, SpectrogramPlot, and AudioTracePlot.

    It just manages the widget switching and exposes the current_plot for direct access.
    """

    plot_changed = Signal(str)  # Emits 'lineplot', 'spectrogram', or 'audiotrace'
    labels_redraw_needed = Signal()  # Emits when labels need to be redrawn on new plot
    spectrogram_overlay_shown = Signal()  # Emits after spectrogram overlay is created

    def __init__(self, napari_viewer, app_state):
        super().__init__()
        self.viewer = napari_viewer
        self.app_state = app_state

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.line_plot = LinePlot(napari_viewer, app_state)
        self.spectrogram_plot = SpectrogramPlot(app_state)
        self.audio_trace_plot = AudioTracePlot(app_state)
        self.heatmap_plot = HeatmapPlot(app_state)

        self.current_plot = self.line_plot
        self.current_plot_type = 'lineplot'

        self.layout.addWidget(self.line_plot)
        self.spectrogram_plot.hide()
        self.audio_trace_plot.hide()
        self.heatmap_plot.hide()

        self.overlay_manager = OverlayManager()

        self.audio_overlay_type = None
        self.audio_overlay_item = None
        self.audio_overlay_vb = None
        self._original_line_pen = None

        self.label_mappings: Dict[int, Dict[str, Any]] = {}

        self.audio_cp_items: list = []
        self.dataset_cp_items: list = []

        self.amp_envelope_vb = None
        self.amp_envelope_item = None
        self.amp_threshold_lines: list = []
        self._amp_envelope_geometry_updater = None
        self._amp_envelope_host_plot = None

        # Envelope overlay x-range refresh (data loading/debounce)
        self._envelope_xrange_updater = None
        self._envelope_debounce = None

        # Connect sigYRangeChanged to rescale overlays
        self.line_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.line_plot)
        )
        self.audio_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.audio_trace_plot)
        )

        # Connect zoom events to update changepoint line styles
        self.spectrogram_plot.vb.sigRangeChanged.connect(self._on_audio_plot_zoom)
        self.audio_trace_plot.vb.sigRangeChanged.connect(self._on_audio_plot_zoom)
        self.heatmap_plot.vb.sigRangeChanged.connect(self._on_audio_plot_zoom)

    def _on_audio_plot_zoom(self):
        """Update changepoint line styles when zoom level changes."""
        self.update_audio_changepoint_styles()

    def sizeHint(self):
        return QSize(self.width(), PLOT_CONTAINER_SIZE_HINT_HEIGHT)

    def _switch_to_plot(self, target_type: str):
        """Generic method to switch between plot types."""
        if self.current_plot_type == target_type:
            return

        if self.current_plot_type in ('lineplot', 'audiotrace'):
            self.hide_audio_overlay()
            self.hide_envelope_overlay()
            self.overlay_manager.clear_plot(self.current_plot)

        plot_map = {
            'lineplot': self.line_plot,
            'spectrogram': self.spectrogram_plot,
            'audiotrace': self.audio_trace_plot,
            'heatmap': self.heatmap_plot,
        }

        prev_xlim = self.current_plot.get_current_xlim()
        prev_time_marker = self.current_plot.time_marker.value()

        self.current_plot.hide()
        self.layout.removeWidget(self.current_plot)

        target_plot = plot_map[target_type]
        self.layout.addWidget(target_plot)
        target_plot.show()

        self.current_plot = target_plot
        self.current_plot_type = target_type

        target_plot.set_x_range(mode='preserve', curr_xlim=prev_xlim)
        target_plot.update_time_marker(prev_time_marker)

        self.plot_changed.emit(target_type)
        self.labels_redraw_needed.emit()

    def switch_to_spectrogram(self):
        """Switch to spectrogram display."""
        self._switch_to_plot('spectrogram')

    def switch_to_lineplot(self):
        """Switch to line plot display."""
        self._switch_to_plot('lineplot')

    def switch_to_audiotrace(self):
        """Switch to audio trace display."""
        self._switch_to_plot('audiotrace')

    def switch_to_heatmap(self):
        """Switch to heatmap display."""
        self._switch_to_plot('heatmap')

    def show_confidence_plot(self, confidence_data):
        """Display confidence values as a scaled overlay via OverlayManager."""
        self.overlay_manager.remove_overlay('confidence')

        if confidence_data is None or len(confidence_data) == 0:
            return

        time = self.app_state.time.values
        item = pg.PlotCurveItem(
            pen=pg.mkPen(color='k', width=2, style=pg.QtCore.Qt.DashLine)
        )
        self.overlay_manager.add_scaled_overlay(
            'confidence',
            self.current_plot,
            item,
            time,
            np.asarray(confidence_data, dtype=np.float64),
            tick_format="{:.2f}",
        )

    def hide_confidence_plot(self):
        """Hide the confidence overlay."""
        self.overlay_manager.remove_overlay('confidence')

    def draw_amplitude_envelope(
        self,
        time: np.ndarray,
        envelope: np.ndarray,
        threshold: float | None = None,
        thresholds: list[tuple[float, Any]] | None = None,
    ):
        """Draw amplitude envelope and threshold line(s) as overlay on the active plot.

        Args:
            time: Time array for x-axis.
            envelope: Amplitude envelope values.
            threshold: Single threshold value (backward compat). Drawn as red dashed.
            thresholds: List of values for multiple threshold lines.
                        If both threshold and thresholds are given, thresholds wins.
        """
        self.clear_amplitude_envelope()

        if self.is_lineplot():
            host = self.line_plot
        elif self.is_audiotrace():
            host = self.audio_trace_plot
        else:
            return

        self._amp_envelope_host_plot = host

        if thresholds is None and threshold is not None:
            default_pen = pg.mkPen(color=(255, 50, 50, 200), width=2, style=Qt.DashLine)
            if isinstance(threshold, (tuple, list)):
                thresholds = [(v, default_pen) for v in threshold]
            else:
                thresholds = [(threshold, default_pen)]

        self.amp_envelope_vb = pg.ViewBox()
        self.amp_envelope_vb.setZValue(1000)
        self.amp_envelope_vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.amp_envelope_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        host.plot_item.scene().addItem(self.amp_envelope_vb)
        self.amp_envelope_vb.setXLink(host.plot_item.vb)

        self.amp_envelope_item = pg.PlotDataItem(
            time, envelope,
            pen=pg.mkPen(color=(255, 165, 0, 100), width=2),
            downsample=10, downsampleMethod='peak',
        )
        self.amp_envelope_vb.addItem(self.amp_envelope_item)

        max_thresh = 0.0
        if thresholds:
            for value, pen in thresholds:
                line = pg.InfiniteLine(pos=value, angle=0, pen=pen)
                self.amp_envelope_vb.addItem(line)
                self.amp_threshold_lines.append(line)
                max_thresh = max(max_thresh, float(value))

        env_max = max(float(envelope.max()), max_thresh * 1.5) if max_thresh > 0 else float(envelope.max())
        self.amp_envelope_vb.setYRange(0, env_max, padding=0.05)

        t0, t1 = host.get_current_xlim()
        self.amp_envelope_vb.setXRange(t0, t1, padding=0)

        def update_geometry():
            if self.amp_envelope_vb is not None:
                rect = host.plot_item.vb.sceneBoundingRect()
                if rect.width() > 0 and rect.height() > 0:
                    self.amp_envelope_vb.setGeometry(rect)

        QTimer.singleShot(0, update_geometry)
        QTimer.singleShot(100, update_geometry)
        host.plot_item.vb.sigResized.connect(update_geometry)
        self._amp_envelope_geometry_updater = update_geometry

    def clear_amplitude_envelope(self):
        """Remove amplitude envelope overlay from the plot it was drawn on."""
        host = self._amp_envelope_host_plot or self.line_plot
        if self._amp_envelope_geometry_updater:
            try:
                host.plot_item.vb.sigResized.disconnect(self._amp_envelope_geometry_updater)
            except (RuntimeError, TypeError):
                pass
            self._amp_envelope_geometry_updater = None

        if self.amp_envelope_vb is not None:
            try:
                if self.amp_envelope_item:
                    self.amp_envelope_vb.removeItem(self.amp_envelope_item)
                for line in self.amp_threshold_lines:
                    self.amp_envelope_vb.removeItem(line)
                host.plot_item.scene().removeItem(self.amp_envelope_vb)
            except (RuntimeError, AttributeError, ValueError):
                pass
            self.amp_envelope_vb = None
            self.amp_envelope_item = None
            self.amp_threshold_lines.clear()

        self._amp_envelope_host_plot = None

    def show_audio_overlay(self, overlay_type: str):
        """Show audio waveform or spectrogram as overlay behind line plot.

        Args:
            overlay_type: 'waveform' or 'spectrogram'
        """
        if not self.is_lineplot():
            return

        self.hide_audio_overlay()

        self.audio_overlay_type = overlay_type

        if overlay_type == 'waveform':
            self._show_waveform_overlay()
        elif overlay_type == 'spectrogram':
            self._show_spectrogram_overlay()

    def _show_waveform_overlay(self):
        """Add waveform trace as overlay on line plot with separate Y-axis."""
        t0, t1 = self.line_plot.get_current_xlim()
        result = self.audio_trace_plot.buffer.get_trace_data(
            self.app_state.audio_path, t0, t1
        )
        if result is None:
            return

        times, amplitudes, step = result

        self.audio_overlay_vb = pg.ViewBox()
        self.line_plot.plot_item.scene().addItem(self.audio_overlay_vb)
        self.audio_overlay_vb.setXLink(self.line_plot.plot_item.vb)

        amp_min, amp_max = amplitudes.min(), amplitudes.max()
        if amp_max - amp_min < 1e-10:
            amp_min, amp_max = -1, 1

        self.audio_overlay_item = pg.PlotDataItem(
            times, amplitudes,
            pen=pg.mkPen(color=(100, 100, 100, 150), width=1),
        )
        self.audio_overlay_vb.addItem(self.audio_overlay_item)
        self.audio_overlay_vb.setYRange(amp_min * 1.1, amp_max * 1.1, padding=0)
        self.audio_overlay_vb.setXRange(t0, t1, padding=0)

        def update_overlay_geometry():
            self.audio_overlay_vb.setGeometry(self.line_plot.plot_item.vb.sceneBoundingRect())

        update_overlay_geometry()
        self.line_plot.plot_item.vb.sigResized.connect(update_overlay_geometry)
        self._overlay_geometry_updater = update_overlay_geometry

        def on_x_range_changed():
            if self.audio_overlay_type == 'waveform':
                self._refresh_waveform_data()

        self.line_plot.plot_item.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._overlay_xrange_updater = on_x_range_changed

    def _refresh_waveform_data(self):
        """Refresh waveform data for current X range."""
        if self.audio_overlay_item is None or self.audio_overlay_vb is None:
            return

        t0, t1 = self.line_plot.get_current_xlim()
        result = self.audio_trace_plot.buffer.get_trace_data(
            self.app_state.audio_path, t0, t1
        )
        if result is None:
            return

        times, amplitudes, step = result
        self.audio_overlay_item.setData(times, amplitudes)

        amp_min, amp_max = amplitudes.min(), amplitudes.max()
        if amp_max - amp_min > 1e-10:
            self.audio_overlay_vb.setYRange(amp_min * 1.1, amp_max * 1.1, padding=0)

    def _show_spectrogram_overlay(self):
        """Add spectrogram as overlay on line plot with separate frequency Y-axis."""
        from .spectrogram_sources import build_audio_source

        source = build_audio_source(self.app_state)
        if source is None:
            return

        t0, t1 = self.line_plot.get_current_xlim()
        result = self.spectrogram_plot.buffer.get_spectrogram(source, t0, t1)
        if result is None:
            return

        Sxx_db, spec_rect = result
        buf_t0, buf_f0, buf_width, buf_fmax = spec_rect

        spec_ymin = self.app_state.get_with_default('spec_ymin')
        spec_ymax = self.app_state.get_with_default('spec_ymax')

        self.audio_overlay_vb = pg.ViewBox()
        self.line_plot.plot_item.scene().addItem(self.audio_overlay_vb)
        self.line_plot.plot_item.getAxis('right').linkToView(self.audio_overlay_vb)
        self.audio_overlay_vb.setXLink(self.line_plot.plot_item.vb)

        right_axis = self.line_plot.plot_item.getAxis('right')
        right_axis.setLabel('Frequency', units='Hz')
        right_axis.setStyle(showValues=True)
        right_axis.show()

        self.audio_overlay_item = pg.ImageItem()
        self.audio_overlay_item.setImage(Sxx_db.T, autoLevels=False)

        vmin = self.app_state.get_with_default('vmin_db')
        vmax = self.app_state.get_with_default('vmax_db')
        self.audio_overlay_item.setLevels([vmin, vmax])

        colormap_name = self.app_state.get_with_default('spec_colormap')
        try:
            cmap = pg.colormap.get(colormap_name)
            self.audio_overlay_item.setColorMap(cmap)
        except (KeyError, ValueError, TypeError):
            pass  # Use default colormap if specified one is unavailable

        self.audio_overlay_item.setOpacity(SPECTROGRAM_OVERLAY_OPACITY)
        self.audio_overlay_vb.addItem(self.audio_overlay_item)

        self.audio_overlay_item.setRect(pg.QtCore.QRectF(buf_t0, 0, buf_width, buf_fmax))
        self.audio_overlay_vb.setYRange(0, buf_fmax, padding=0)
        self.audio_overlay_vb.setLimits(yMin=0, yMax=buf_fmax)

        def update_overlay_geometry():
            self.audio_overlay_vb.setGeometry(self.line_plot.plot_item.vb.sceneBoundingRect())

        update_overlay_geometry()
        self.line_plot.plot_item.vb.sigResized.connect(update_overlay_geometry)
        self._overlay_geometry_updater = update_overlay_geometry

        self._spec_overlay_last_range = (t0, t1)
        self._spec_overlay_pending_refresh = False

        from qtpy.QtCore import QTimer
        self._spec_overlay_debounce = QTimer()
        self._spec_overlay_debounce.setSingleShot(True)
        self._spec_overlay_debounce.setInterval(SPECTROGRAM_OVERLAY_DEBOUNCE_MS)

        def do_refresh():
            if self._spec_overlay_pending_refresh:
                self._spec_overlay_pending_refresh = False
                self._refresh_spectrogram_data()

        self._spec_overlay_debounce.timeout.connect(do_refresh)

        def on_x_range_changed():
            if self.audio_overlay_type == 'spectrogram':
                new_t0, new_t1 = self.line_plot.get_current_xlim()
                old_t0, old_t1 = self._spec_overlay_last_range
                old_width = old_t1 - old_t0
                new_width = new_t1 - new_t0

                if new_width < old_width * SPECTROGRAM_OVERLAY_ZOOM_OUT_THRESHOLD or new_width > old_width * SPECTROGRAM_OVERLAY_ZOOM_IN_THRESHOLD:
                    self.spectrogram_plot.buffer._clear_buffer()
                    self._spec_overlay_last_range = (new_t0, new_t1)

                self._spec_overlay_pending_refresh = True
                self._spec_overlay_debounce.start()

        self.line_plot.plot_item.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._overlay_xrange_updater = on_x_range_changed

        if hasattr(self.line_plot, 'plot_items') and self.line_plot.plot_items:
            for item in self.line_plot.plot_items:
                if hasattr(item, 'opts') and 'pen' in item.opts:
                    self._original_line_pen = item.opts['pen']
                    item.setPen(pg.mkPen(color='yellow', width=2))
                    break

        self.spectrogram_overlay_shown.emit()

    def _refresh_spectrogram_data(self):
        """Refresh spectrogram data for current X range."""
        from .spectrogram_sources import build_audio_source

        if self.audio_overlay_item is None or self.audio_overlay_vb is None:
            return

        source = build_audio_source(self.app_state)
        if source is None:
            return

        t0, t1 = self.line_plot.get_current_xlim()
        result = self.spectrogram_plot.buffer.get_spectrogram(source, t0, t1)
        if result is None:
            return

        Sxx_db, spec_rect = result
        buf_t0, buf_f0, buf_width, buf_fmax = spec_rect

        self.audio_overlay_item.setImage(Sxx_db.T, autoLevels=False)

        vmin = self.app_state.get_with_default('vmin_db')
        vmax = self.app_state.get_with_default('vmax_db')
        self.audio_overlay_item.setLevels([vmin, vmax])

        self.audio_overlay_item.setRect(pg.QtCore.QRectF(buf_t0, 0, buf_width, buf_fmax))
        self.audio_overlay_vb.setYRange(0, buf_fmax, padding=0)

    def hide_audio_overlay(self):
        """Remove audio overlay from line plot."""
        if hasattr(self, '_overlay_geometry_updater') and self._overlay_geometry_updater:
            try:
                self.line_plot.plot_item.vb.sigResized.disconnect(self._overlay_geometry_updater)
            except (RuntimeError, TypeError):
                pass  # Signal was already disconnected or never connected
            self._overlay_geometry_updater = None

        if hasattr(self, '_overlay_xrange_updater') and self._overlay_xrange_updater:
            try:
                self.line_plot.plot_item.vb.sigXRangeChanged.disconnect(self._overlay_xrange_updater)
            except (RuntimeError, TypeError):
                pass  # Signal was already disconnected or never connected
            self._overlay_xrange_updater = None

        if hasattr(self, '_spec_overlay_debounce') and self._spec_overlay_debounce:
            self._spec_overlay_debounce.stop()
            self._spec_overlay_debounce = None

        if self.audio_overlay_vb is not None:
            try:
                if self.audio_overlay_item is not None:
                    self.audio_overlay_vb.removeItem(self.audio_overlay_item)
                self.line_plot.plot_item.scene().removeItem(self.audio_overlay_vb)
                right_axis = self.line_plot.plot_item.getAxis('right')
                right_axis.hide()
            except (RuntimeError, AttributeError, ValueError):
                pass  # Item already removed or scene not available
            self.audio_overlay_vb = None
            self.audio_overlay_item = None
        elif self.audio_overlay_item is not None:
            try:
                self.line_plot.plot_item.removeItem(self.audio_overlay_item)
            except (RuntimeError, AttributeError, ValueError):
                pass  # Item already removed
            self.audio_overlay_item = None

        if self._original_line_pen is not None and hasattr(self.line_plot, 'plot_items'):
            for item in self.line_plot.plot_items:
                if hasattr(item, 'setPen'):
                    item.setPen(self._original_line_pen)
                    break
            self._original_line_pen = None

        self.audio_overlay_type = None

    def update_audio_overlay(self):
        """Refresh the audio overlay with current view range."""
        if self.audio_overlay_type:
            self.show_audio_overlay(self.audio_overlay_type)

    def get_current_plot(self):
        """Get the currently active plot widget."""
        return self.current_plot

    def is_spectrogram(self):
        """Check if currently showing spectrogram."""
        return self.current_plot_type == 'spectrogram'

    def is_audiotrace(self):
        """Check if currently showing audio trace."""
        return self.current_plot_type == 'audiotrace'

    def is_lineplot(self):
        """Check if currently showing line plot."""
        return self.current_plot_type == 'lineplot'

    def is_heatmap(self):
        """Check if currently showing heatmap."""
        return self.current_plot_type == 'heatmap'

    def has_spectrogram_overlay(self) -> bool:
        return self.audio_overlay_type == 'spectrogram' and self.audio_overlay_item is not None

    def apply_overlay_levels(self, vmin: float, vmax: float):
        if self.audio_overlay_item is not None and self.audio_overlay_type == 'spectrogram':
            self.audio_overlay_item.setLevels([vmin, vmax])

    def apply_overlay_colormap(self, colormap_name: str):
        if self.audio_overlay_item is not None and self.audio_overlay_type == 'spectrogram':
            try:
                cmap = pg.colormap.get(colormap_name)
                self.audio_overlay_item.setColorMap(cmap)
            except (KeyError, ValueError, TypeError):
                pass

    def get_current_xlim(self):
        """Get current x-axis limits from active plot."""
        return self.current_plot.get_current_xlim()

    def set_x_range(self, mode='default', curr_xlim=None, center_on_frame=None):
        """Set x-axis range on active plot."""
        return self.current_plot.set_x_range(mode=mode, curr_xlim=curr_xlim, center_on_frame=center_on_frame)

    def update_time_marker_and_window(self, frame_number):
        """Update time marker on active plot."""
        return self.current_plot.update_time_marker_and_window(frame_number)

    def apply_y_range(self, ymin, ymax):
        """Apply y-axis range to active plot."""
        return self.current_plot.apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        """Toggle axes lock on active plot."""
        return self.current_plot.toggle_axes_lock()

    def set_label_mappings(self, mappings: Dict[int, Dict[str, Any]]):
        """Set the label color/name mappings for label drawing."""
        self.label_mappings = mappings

    def draw_all_labels(self, intervals_df, predictions_df=None, show_predictions=False):
        """Draw labels on ALL plots to ensure synchronization.

        Args:
            intervals_df: DataFrame with onset_s, offset_s, labels, individual columns
            predictions_df: Optional prediction intervals DataFrame
            show_predictions: Whether to show prediction rectangles
        """
        if intervals_df is None or not self.label_mappings:
            return

        all_plots = [self.line_plot, self.spectrogram_plot, self.audio_trace_plot, self.heatmap_plot]

        for plot in all_plots:
            if plot is None:
                continue

            self._clear_labels_on_plot(plot)
            self._draw_intervals_on_plot(plot, intervals_df, is_main=True)

            if predictions_df is not None and show_predictions:
                self._draw_intervals_on_plot(plot, predictions_df, is_main=False)

    def _clear_labels_on_plot(self, plot):
        """Clear all label items from a specific plot."""
        if not hasattr(plot, 'label_items'):
            plot.label_items = []
            return

        for item in plot.label_items:
            try:
                plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass  # Item already removed from plot
        plot.label_items.clear()

    def _draw_intervals_on_plot(self, plot, intervals_df, is_main=True):
        """Draw interval-based label segments on a specific plot.

        Args:
            plot: The plot widget to draw on
            intervals_df: DataFrame with onset_s, offset_s, labels, individual
            is_main: If True, draw full-height; if False, draw prediction strip at top
        """
        if not hasattr(plot, 'label_items'):
            plot.label_items = []

        if intervals_df is None or intervals_df.empty:
            return

        for _, row in intervals_df.iterrows():
            labels = int(row["labels"])
            if labels == 0:
                continue
            self._draw_single_label(plot, row["onset_s"], row["offset_s"], labels, is_main)

    def _draw_single_label(self, plot, start_time, end_time, labels, is_main=True):
        """Draw a single label rectangle on a plot with appropriate style.

        Args:
            plot: The plot widget to draw on
            start_time: Start time of the label
            end_time: End time of the label
            labels: ID of the label for color mapping
            is_main: If True, draw full-height; if False, draw prediction strip
        """
        if labels not in self.label_mappings:
            return

        color = self.label_mappings[labels]["color"]
        color_rgb = tuple(int(c * 255) for c in color)

        # Use bottom strip style for spectrogram or when spectrogram overlay is active
        use_bottom_strip = (
            plot == self.spectrogram_plot or
            plot == self.heatmap_plot or
            (plot == self.line_plot and self.audio_overlay_type == 'spectrogram')
        )

        if is_main:
            if use_bottom_strip:
                self._draw_spectrogram_style_label(plot, start_time, end_time, color_rgb)
            else:
                self._draw_standard_label(plot, start_time, end_time, color_rgb)
        else:
            self._draw_prediction_label(plot, start_time, end_time, color_rgb)

    def _draw_standard_label(self, plot, start_time, end_time, color_rgb):
        """Draw standard semi-transparent rectangle for lineplot/audiotrace."""
        rect = pg.LinearRegionItem(
            values=(start_time, end_time),
            orientation="vertical",
            brush=(*color_rgb, 180),
            movable=False,
        )
        rect.setZValue(Z_INDEX_LABELS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_spectrogram_style_label(self, plot, start_time, end_time, color_rgb):
        """Draw label as colored strip at bottom of spectrogram."""
        self._draw_label_strip_bottom(plot, start_time, end_time, color_rgb)

    def _draw_prediction_label(self, plot, start_time, end_time, color_rgb):
        """Draw small rectangle at top for prediction data."""
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_top = y_range[1]
        y_height = (y_range[1] - y_range[0]) * PREDICTION_LABELS_HEIGHT_RATIO

        if y_top <= y_range[0]:
            y_top = PREDICTION_FALLBACK_Y_TOP
            y_height = PREDICTION_FALLBACK_Y_HEIGHT

        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_top, y_top, y_top - y_height, y_top - y_height, y_top]

        rect = pg.PlotDataItem(
            x_coords, y_coords,
            fillLevel=y_top - y_height,
            brush=(*color_rgb, 200),
            pen=None
        )
        rect.setZValue(Z_INDEX_PREDICTIONS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_label_strip_bottom(self, plot, start_time, end_time, color_rgb):
        """Draw small rectangle at bottom for labels on spectrogram."""
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_bottom = y_range[0]
        y_height = (y_range[1] - y_range[0]) * SPECTROGRAM_LABELS_HEIGHT_RATIO

        if y_range[1] <= y_bottom:
            y_bottom = 0
            y_height = SPECTROGRAM_FALLBACK_Y_HEIGHT

        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_bottom, y_bottom, y_bottom + y_height, y_bottom + y_height, y_bottom]

        rect = pg.PlotDataItem(
            x_coords, y_coords,
            fillLevel=y_bottom,
            brush=(*color_rgb, 220),
            pen=None
        )
        rect.setZValue(Z_INDEX_PREDICTIONS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def redraw_current_plot_labels(self, intervals_df, predictions_df=None, show_predictions=False):
        """Redraw labels only on the current plot (for view range changes)."""
        if intervals_df is None or not self.label_mappings:
            return

        plot = self.current_plot
        self._clear_labels_on_plot(plot)
        self._draw_intervals_on_plot(plot, intervals_df, is_main=True)

        if predictions_df is not None and show_predictions:
            self._draw_intervals_on_plot(plot, predictions_df, is_main=False)

    @property
    def vb(self):
        """Get ViewBox from active plot for direct access."""
        return self.current_plot.vb

    def draw_audio_changepoints(self, onsets: np.ndarray, offsets: np.ndarray):
        """Draw audio changepoints as vertical lines on audio plots.

        Only draws on spectrogram and audiotrace plots. White lines that
        adapt style based on zoom level (dotted when zoomed out, solid when zoomed in).
        Times are rounded to nearest label sample for alignment with label grid.

        Args:
            onsets: Array of onset times in seconds
            offsets: Array of offset times in seconds
        """
        self.clear_audio_changepoints()

        plots_to_draw = [self.spectrogram_plot, self.audio_trace_plot]

        line_style = self._get_changepoint_line_style()

        for plot in plots_to_draw:
            if plot is None:
                continue

            # Black for waveform, white for spectrogram
            color = CP_COLOR_WAVEFORM if plot == self.audio_trace_plot else CP_COLOR_SPECTROGRAM

            for onset_t in onsets:
                line = pg.InfiniteLine(
                    pos=onset_t,
                    angle=90,
                    pen=pg.mkPen(color=color, width=line_style['width'], style=line_style['style']),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, 'onset'))

            for offset_t in offsets:
                line = pg.InfiniteLine(
                    pos=offset_t,
                    angle=90,
                    pen=pg.mkPen(color=color, width=line_style['width'], style=line_style['style']),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, 'offset'))

    def _get_changepoint_line_style(self):
        """Get line style based on current zoom level.

        Returns dotted lines when zoomed out (>2s visible), solid thin lines when zoomed in.
        """
        try:
            xmin, xmax = self.current_plot.get_current_xlim()
            visible_range = xmax - xmin

            if visible_range > CP_ZOOM_VERY_OUT_THRESHOLD:
                # Very zoomed out: thin dotted lines
                return {'style': pg.QtCore.Qt.DotLine, 'width': CP_LINE_WIDTH_THIN}
            elif visible_range > CP_ZOOM_MEDIUM_THRESHOLD:
                # Medium zoom: dashed lines
                return {'style': pg.QtCore.Qt.DashLine, 'width': CP_LINE_WIDTH_MEDIUM}
            else:
                # Zoomed in: solid lines for precision
                return {'style': pg.QtCore.Qt.SolidLine, 'width': CP_LINE_WIDTH_THICK}
        except (AttributeError, TypeError, ValueError):
            return {'style': pg.QtCore.Qt.DashLine, 'width': CP_LINE_WIDTH_MEDIUM}

    def update_audio_changepoint_styles(self):
        """Update changepoint line styles based on current zoom level."""
        if not self.audio_cp_items:
            return

        line_style = self._get_changepoint_line_style()

        for item in self.audio_cp_items:
            plot, line, _ = item
            # Black for waveform, white for spectrogram
            color = CP_COLOR_WAVEFORM if plot == self.audio_trace_plot else CP_COLOR_SPECTROGRAM
            line.setPen(pg.mkPen(color=color, width=line_style['width'], style=line_style['style']))

    def clear_audio_changepoints(self):
        """Remove all audio changepoint lines from plots."""
        for item in self.audio_cp_items:
            plot, line = item[0], item[1]
            try:
                plot.plot_item.removeItem(line)
            except (RuntimeError, AttributeError, ValueError):
                pass  # Item already removed from plot
        self.audio_cp_items.clear()

    def draw_dataset_changepoints(self, time_array: np.ndarray, cp_by_method: dict):
        """Draw dataset changepoints as circles on lineplot.

        Args:
            time_array: Time coordinate array
            cp_by_method: Dict mapping method name to array of indices
        """
        self.clear_dataset_changepoints()

        if not self.is_lineplot():
            return

        y_range = self.line_plot.plot_item.getViewBox().viewRange()[1]
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * CP_SCATTER_Y_POSITION_RATIO

        for method_name, indices in cp_by_method.items():
            if len(indices) == 0:
                continue

            times = time_array[indices]
            y_values = np.full_like(times, y_pos)

            color = CP_METHOD_COLORS.get(method_name, CP_METHOD_COLORS['default'])

            scatter = pg.ScatterPlotItem(
                x=times,
                y=y_values,
                size=CP_SCATTER_SIZE,
                pen=pg.mkPen(color=color, width=1),
                brush=pg.mkBrush(color=color),
                symbol='o',
                name=method_name,
            )
            scatter.setZValue(Z_INDEX_CHANGEPOINTS)
            self.line_plot.plot_item.addItem(scatter)
            self.dataset_cp_items.append(scatter)

    def clear_dataset_changepoints(self):
        """Remove all dataset changepoint markers from lineplot."""
        for item in self.dataset_cp_items:
            try:
                self.line_plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass  # Item already removed from plot
        self.dataset_cp_items.clear()

    def clear_audio_cache(self):
        """Clear audio caches when noise reduction setting changes."""
        from .plots_spectrogram import SharedAudioCache
        SharedAudioCache.clear_cache()

        if hasattr(self.spectrogram_plot, 'buffer'):
            self.spectrogram_plot.buffer._clear_buffer()

        if hasattr(self.audio_trace_plot, 'buffer'):
            self.audio_trace_plot.buffer.audio_loader = None
            self.audio_trace_plot.buffer.current_path = None

    # --- Envelope overlay ---

    def _get_envelope_host_plot(self):
        """Return the plot widget to host the envelope overlay, or None."""
        if self.is_audiotrace():
            return self.audio_trace_plot
        if self.is_lineplot():
            return self.line_plot
        return None


    def _load_envelope_feature(self, t0, t1):
        """Load the currently selected feature's 1D data for the envelope.

        Returns (data_1d, time_array, sample_rate) or (None, None, None).
        """
        from ethograph.utils.data_utils import sel_valid, get_time_coord

        feature_sel = getattr(self.app_state, 'features_sel', None)
        ds = getattr(self.app_state, 'ds', None)
        if not feature_sel or ds is None or feature_sel not in ds:
            return None, None, None

        da = ds[feature_sel]
        time_coord = get_time_coord(da)
        if time_coord is None:
            return None, None, None

        time_vals = time_coord.values
        ds_kwargs = self.app_state.get_ds_kwargs()
        data, _ = sel_valid(da, ds_kwargs)

        if data.ndim > 1:
            data = data[:, 0]

        mask = (time_vals >= t0) & (time_vals <= t1)
        if not np.any(mask):
            return None, None, None

        dt = np.median(np.diff(time_vals[:min(1000, len(time_vals))]))
        fs = 1.0 / dt if dt > 0 else 1.0

        return data[mask], time_vals[mask], fs

    def _load_envelope_data(self, host, t0, t1):
        """Load signal data for envelope computation.

        For audiotrace (Audio Waveform): loads from SharedAudioCache.
        For lineplot: loads from the current dataset feature.
        Returns (signal_1d, sample_rate, buf_t0) or (None, None, None).
        buf_t0 is the start time of the (possibly padded) buffer.
        """
        if self.is_audiotrace():
            from .plots_spectrogram import SharedAudioCache

            audio_path = getattr(self.app_state, 'audio_path', None)
            if not audio_path:
                return None, None, None
            loader = SharedAudioCache.get_loader(audio_path)
            if loader is None:
                return None, None, None
            fs = loader.rate
            _, channel_idx = self.app_state.get_audio_source()
            audio_data, buf_t0 = self._load_envelope_audio(loader, fs, channel_idx, t0, t1)
            return audio_data, fs, buf_t0

        feature_data, _, fs = self._load_envelope_feature(t0, t1)
        return feature_data, fs, t0

    def show_envelope_overlay(self):
        """Compute and draw an energy envelope overlay via OverlayManager.

        The overlay is added with ``ignoreBounds=True`` so autoscale only
        considers the primary data.  The OverlayManager rescales the overlay
        automatically when the y-range changes.
        """
        host = self._get_envelope_host_plot()
        if host is None:
            return

        self.hide_envelope_overlay()

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        metric = self.app_state.get_with_default('energy_metric')
        env_time, env_data = self._compute_envelope(signal_data, fs, metric, buf_t0, t0, t1)

        if env_data is None or len(env_data) == 0:
            return

        item = pg.PlotCurveItem(
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=ENVELOPE_OVERLAY_WIDTH),
        )
        self.overlay_manager.add_scaled_overlay(
            'envelope', host, item, env_time, env_data,
        )

        self._envelope_debounce = QTimer()
        self._envelope_debounce.setSingleShot(True)
        self._envelope_debounce.setInterval(ENVELOPE_OVERLAY_DEBOUNCE_MS)
        self._envelope_debounce.timeout.connect(self._refresh_envelope_data)

        def on_x_range_changed():
            if self.overlay_manager.has_overlay('envelope'):
                self._envelope_debounce.start()

        host.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._envelope_xrange_updater = on_x_range_changed

    def hide_envelope_overlay(self):
        """Remove envelope overlay from whichever plot hosts it."""
        if self._envelope_xrange_updater:
            for plot in (self.line_plot, self.audio_trace_plot):
                try:
                    plot.vb.sigXRangeChanged.disconnect(self._envelope_xrange_updater)
                except (RuntimeError, TypeError):
                    pass
            self._envelope_xrange_updater = None

        if self._envelope_debounce:
            self._envelope_debounce.stop()
            self._envelope_debounce = None

        self.overlay_manager.remove_overlay('envelope')

    def _refresh_envelope_data(self):
        """Recompute envelope for the current visible range and update the overlay."""
        if not self.overlay_manager.has_overlay('envelope'):
            return

        host = self._get_envelope_host_plot()
        if host is None:
            return

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        metric = self.app_state.get_with_default('energy_metric')
        env_time, env_data = self._compute_envelope(signal_data, fs, metric, buf_t0, t0, t1)

        if env_data is None or len(env_data) == 0:
            return

        self.overlay_manager.update_overlay_data('envelope', env_time, env_data)

    def _load_envelope_audio(self, loader, fs, channel_idx, t0, t1):
        """Load audio with padding to avoid edge artifacts from lowpass filtering.

        Returns (audio_1d, actual_t0) where actual_t0 is the start time of the
        padded buffer.  The caller is responsible for trimming the result.
        """
        cutoff = self.app_state.get_with_default('env_cutoff')
        pad_s = max(0.5, 3.0 / cutoff) if cutoff > 0 else 0.5
        buf_t0 = max(0.0, t0 - pad_s)
        buf_t1 = min(len(loader) / fs, t1 + pad_s)

        start_idx = max(0, int(buf_t0 * fs))
        stop_idx = min(len(loader), int(buf_t1 * fs))
        if stop_idx <= start_idx:
            return None, buf_t0

        audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if audio_data.ndim > 1:
            ch = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, ch]
        return audio_data, buf_t0

    def _compute_envelope(
        self, signal_1d: np.ndarray, fs: float, metric: str,
        buf_t0: float, view_t0: float, view_t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute envelope on a signal buffer and trim to the visible window.

        Parameters
        ----------
        signal_1d : 1D signal samples.
        fs : sample rate of the signal.
        metric : ``"amplitude_envelope"`` or ``"meansquared"``.
        buf_t0 : start time of the buffer.
        view_t0, view_t1 : visible window boundaries used for trimming.
        """
        nyquist = fs / 2.0

        if metric == "meansquared":
            from ethograph.features.audio_features import compute_meansquared_envelope

            freq_min = min(self.app_state.get_with_default('freq_cutoffs_min'), nyquist * 0.9)
            freq_max = min(self.app_state.get_with_default('freq_cutoffs_max'), nyquist * 0.9)
            smooth = self.app_state.get_with_default('smooth_win')
            env_time, env_data = compute_meansquared_envelope(
                signal_1d, fs, freq_cutoffs=(freq_min, freq_max), smooth_win=smooth,
            )
        else:
            from ethograph.features.filter import envelope as amp_envelope

            env_rate = min(self.app_state.get_with_default('env_rate'), fs)
            cutoff = min(self.app_state.get_with_default('env_cutoff'), nyquist * 0.9)
            env_data = np.squeeze(amp_envelope(signal_1d, rate=fs, cutoff=cutoff, env_rate=env_rate))
            env_time = np.linspace(0, len(signal_1d) / fs, len(env_data))

        env_time = env_time + buf_t0

        mask = (env_time >= view_t0) & (env_time <= view_t1)
        return env_time[mask], env_data[mask]