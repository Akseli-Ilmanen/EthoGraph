"""Simple container widget for switching between different plot types."""

from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .app_constants import (
    PLOT_CONTAINER_SIZE_HINT_HEIGHT,
    SPECTROGRAM_OVERLAY_OPACITY,
    SPECTROGRAM_OVERLAY_DEBOUNCE_MS,
    SPECTROGRAM_OVERLAY_ZOOM_OUT_THRESHOLD,
    SPECTROGRAM_OVERLAY_ZOOM_IN_THRESHOLD,
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

        self.confidence_item = None

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

        if self.current_plot_type == 'lineplot':
            self.hide_audio_overlay()

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
        """Display confidence values on a secondary y-axis."""
        
    
        if self.confidence_item is not None:
            self.current_plot.plot_item.removeItem(self.confidence_item)
            self.confidence_item = None

        if confidence_data is None or len(confidence_data) == 0:
            return

        time = self.app_state.time.values
        
        right_axis = self.current_plot.plot_item.getAxis('right')
        right_axis.setStyle(showValues=True)
        right_axis.show()
        
        main_range = self.current_plot.plot_item.viewRange()[1]
        conf_min, conf_max = np.min(confidence_data), np.max(confidence_data)
        conf_range = conf_max - conf_min if conf_max > conf_min else 1.0
        
        scaled_confidence = ((confidence_data - conf_min) / conf_range) * (main_range[1] - main_range[0]) + main_range[0]
        
        self.confidence_item = pg.PlotCurveItem(
            time,
            scaled_confidence,
            pen=pg.mkPen(color='k', width=2, style=pg.QtCore.Qt.DashLine)
        )
        self.current_plot.plot_item.addItem(self.confidence_item)
        
        ticks = []
        for conf_val in np.linspace(conf_min, conf_max, 5):
            main_val = ((conf_val - conf_min) / conf_range) * (main_range[1] - main_range[0]) + main_range[0]
            ticks.append((main_val, f'{conf_val:.2f}'))
        
        right_axis.setTicks([ticks])
    
    def hide_confidence_plot(self):
        """Hide the confidence plot if it exists."""
        if self.confidence_item is not None:
            self.current_plot.plot_item.removeItem(self.confidence_item)
            right_axis = self.current_plot.plot_item.getAxis('right')
            right_axis.hide()
            self.confidence_item = None

    def draw_amplitude_envelope(
        self,
        time: np.ndarray,
        envelope: np.ndarray,
        threshold: float | None = None,
        thresholds: list[tuple[float, Any]] | None = None,
    ):
        """Draw amplitude envelope and threshold line(s) as overlay on the line plot.

        Args:
            time: Time array for x-axis.
            envelope: Amplitude envelope values.
            threshold: Single threshold value (backward compat). Drawn as red dashed.
            thresholds: List of values for multiple threshold lines.
                        If both threshold and thresholds are given, thresholds wins.
        """
        self.clear_amplitude_envelope()

        if not self.is_lineplot():
            return

        if thresholds is None and threshold is not None:
            thresholds = [
                (threshold, pg.mkPen(color=(255, 50, 50, 200), width=2, style=Qt.DashLine))
            ]

        self.amp_envelope_vb = pg.ViewBox()
        self.line_plot.plot_item.scene().addItem(self.amp_envelope_vb)
        self.amp_envelope_vb.setXLink(self.line_plot.plot_item.vb)

        self.amp_envelope_item = pg.PlotDataItem(
            time, envelope,
            pen=pg.mkPen(color=(255, 165, 0, 100), width=2),
            downsample=10, downsampleMethod='peak',
        )
        self.amp_envelope_vb.addItem(self.amp_envelope_item)

        max_thresh = 0.0
        if thresholds:
            for value in thresholds:
                line = pg.InfiniteLine(pos=value, angle=0)
                self.amp_envelope_vb.addItem(line)
                self.amp_threshold_lines.append(line)
                max_thresh = max(max_thresh, value)

        env_max = max(float(envelope.max()), max_thresh * 1.5) if max_thresh > 0 else float(envelope.max())
        self.amp_envelope_vb.setYRange(0, env_max, padding=0.05)

        t0, t1 = self.line_plot.get_current_xlim()
        self.amp_envelope_vb.setXRange(t0, t1, padding=0)

        def update_geometry():
            self.amp_envelope_vb.setGeometry(self.line_plot.plot_item.vb.sceneBoundingRect())

        update_geometry()
        self.line_plot.plot_item.vb.sigResized.connect(update_geometry)
        self._amp_envelope_geometry_updater = update_geometry

    def clear_amplitude_envelope(self):
        """Remove amplitude envelope overlay from the line plot."""
        if self._amp_envelope_geometry_updater:
            try:
                self.line_plot.plot_item.vb.sigResized.disconnect(self._amp_envelope_geometry_updater)
            except (RuntimeError, TypeError):
                pass
            self._amp_envelope_geometry_updater = None

        if self.amp_envelope_vb is not None:
            try:
                if self.amp_envelope_item:
                    self.amp_envelope_vb.removeItem(self.amp_envelope_item)
                for line in self.amp_threshold_lines:
                    self.amp_envelope_vb.removeItem(line)
                self.line_plot.plot_item.scene().removeItem(self.amp_envelope_vb)
            except (RuntimeError, AttributeError, ValueError):
                pass
            self.amp_envelope_vb = None
            self.amp_envelope_item = None
            self.amp_threshold_lines.clear()

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

    def draw_all_labels(self, time_data, labels, predictions=None, show_predictions=False):
        """Draw labels on ALL plots to ensure synchronization.

        This is the central entry point for label drawing. It draws labels on
        every plot type, so switching plots always shows correct labels.

        Args:
            time_data: Time array for x-axis
            labels: Primary label data array
            predictions: Optional prediction data array
            show_predictions: Whether to show prediction rectangles
        """
        if labels is None or not self.label_mappings:
            return

        all_plots = [self.line_plot, self.spectrogram_plot, self.audio_trace_plot, self.heatmap_plot]

        for plot in all_plots:
            if plot is None:
                continue

            self._clear_labels_on_plot(plot)
            self._draw_labels_on_plot(plot, time_data, labels, is_main=True)

            if predictions is not None and show_predictions:
                self._draw_labels_on_plot(plot, time_data, predictions, is_main=False)

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

    def _draw_labels_on_plot(self, plot, time_data, data, is_main=True):
        """Draw label segments on a specific plot.

        Args:
            plot: The plot widget to draw on
            time_data: Time array for x-axis
            data: Label data array
            is_main: If True, draw full-height; if False, draw prediction strip at top
        """
        if not hasattr(plot, 'label_items'):
            plot.label_items = []

        data = np.asarray(data)
        if len(data) == 0:
            return

        # Vectorized: find indices where value changes (including start/end boundaries)
        # Pad with different sentinel values to detect first and last segments
        padded = np.concatenate([[-1], data, [-1]])
        change_mask = padded[:-1] != padded[1:]
        change_indices = np.nonzero(change_mask)[0]

        # Each consecutive pair of change_indices defines a segment
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1] - 1

            if start_idx >= len(data):
                continue
            end_idx = min(end_idx, len(data) - 1)

            label_id = int(data[start_idx])
            if label_id == 0:
                continue

            start_time = time_data[start_idx]
            end_time = time_data[end_idx]
            self._draw_single_label(plot, start_time, end_time, label_id, is_main)

    def _draw_single_label(self, plot, start_time, end_time, label_id, is_main=True):
        """Draw a single label rectangle on a plot with appropriate style.

        Args:
            plot: The plot widget to draw on
            start_time: Start time of the label
            end_time: End time of the label
            label_id: ID of the label for color mapping
            is_main: If True, draw full-height; if False, draw prediction strip
        """
        if label_id not in self.label_mappings:
            return

        color = self.label_mappings[label_id]["color"]
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

    def redraw_current_plot_labels(self, time_data, labels, predictions=None, show_predictions=False):
        """Redraw labels only on the current plot (for view range changes)."""
        if labels is None or not self.label_mappings:
            return

        plot = self.current_plot
        self._clear_labels_on_plot(plot)
        self._draw_labels_on_plot(plot, time_data, labels, is_main=True)

        if predictions is not None and show_predictions:
            self._draw_labels_on_plot(plot, time_data, predictions, is_main=False)

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

        # Determine line style based on current zoom level
        line_style = self._get_changepoint_line_style()

        # Round times to nearest label sample for alignment
        label_sr = getattr(self.app_state, 'label_sr', None)
        if label_sr and label_sr > 0:
            onsets = np.round(onsets * label_sr) / label_sr
            offsets = np.round(offsets * label_sr) / label_sr

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