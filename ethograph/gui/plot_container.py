"""Simple container widget for switching between different plot types."""

from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Signal, QSize
from .plots_lineplot import LinePlot
from .plots_spectrogram import SpectrogramPlot
from .plots_audiotrace import AudioTracePlot
import pyqtgraph as pg
import numpy as np
from typing import Optional, Dict, Any


class PlotContainer(QWidget):
    """Container that holds and switches between LinePlot, SpectrogramPlot, and AudioTracePlot.

    It just manages the widget switching and exposes the current_plot for direct access.
    """

    plot_changed = Signal(str)  # Emits 'lineplot', 'spectrogram', or 'audiotrace'
    labels_redraw_needed = Signal()  # Emits when labels need to be redrawn on new plot

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

        self.current_plot = self.line_plot
        self.current_plot_type = 'lineplot'

        self.layout.addWidget(self.line_plot)
        self.spectrogram_plot.hide()
        self.audio_trace_plot.hide()

        self.confidence_item = None

        self.audio_overlay_type = None
        self.audio_overlay_item = None
        self.audio_overlay_vb = None
        self._original_line_pen = None

        self.motif_mappings: Dict[int, Dict[str, Any]] = {}

    def sizeHint(self):
        return QSize(self.width(), 300)

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

    def show_confidence_plot(self, confidence_data):
        """Display confidence values on a secondary y-axis."""
        
    
        if self.confidence_item is not None:
            self.current_plot.plot_item.removeItem(self.confidence_item)
            self.confidence_item = None

        if confidence_data is None or len(confidence_data) == 0:
            return

        time = self.app_state.ds.time.values
        
        right_axis = self.current_plot.plot_item.getAxis('right')
        right_axis.setLabel('Confidence', color='m')
        right_axis.setStyle(showValues=True)
        right_axis.show()
        
        main_range = self.current_plot.plot_item.viewRange()[1]
        conf_min, conf_max = np.min(confidence_data), np.max(confidence_data)
        conf_range = conf_max - conf_min if conf_max > conf_min else 1.0
        
        scaled_confidence = ((confidence_data - conf_min) / conf_range) * (main_range[1] - main_range[0]) + main_range[0]
        
        self.confidence_item = pg.PlotCurveItem(
            time,
            scaled_confidence,
            pen=pg.mkPen(color='k', width=2, style=pg.QtCore.Qt.DashLine),
            name='Confidence'
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
        t0, t1 = self.line_plot.get_current_xlim()
        result = self.spectrogram_plot.buffer.get_spectrogram(
            self.app_state.audio_path, t0, t1
        )
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
        except Exception:
            pass

        self.audio_overlay_item.setOpacity(0.6)
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
        self._spec_overlay_debounce.setInterval(100)

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

                if new_width < old_width * 0.5 or new_width > old_width * 2.0:
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

    def _refresh_spectrogram_data(self):
        """Refresh spectrogram data for current X range."""
        if self.audio_overlay_item is None or self.audio_overlay_vb is None:
            return

        t0, t1 = self.line_plot.get_current_xlim()
        result = self.spectrogram_plot.buffer.get_spectrogram(
            self.app_state.audio_path, t0, t1
        )
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
            except Exception:
                pass
            self._overlay_geometry_updater = None

        if hasattr(self, '_overlay_xrange_updater') and self._overlay_xrange_updater:
            try:
                self.line_plot.plot_item.vb.sigXRangeChanged.disconnect(self._overlay_xrange_updater)
            except Exception:
                pass
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
            except Exception:
                pass
            self.audio_overlay_vb = None
            self.audio_overlay_item = None
        elif self.audio_overlay_item is not None:
            try:
                self.line_plot.plot_item.removeItem(self.audio_overlay_item)
            except Exception:
                pass
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

    def set_motif_mappings(self, mappings: Dict[int, Dict[str, Any]]):
        """Set the motif color/name mappings for label drawing."""
        self.motif_mappings = mappings

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
        if labels is None or not self.motif_mappings:
            return

        all_plots = [self.line_plot, self.spectrogram_plot, self.audio_trace_plot]

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
            except Exception:
                pass
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

            motif_id = int(data[start_idx])
            if motif_id == 0:
                continue

            start_time = time_data[start_idx]
            end_time = time_data[end_idx]
            self._draw_single_label(plot, start_time, end_time, motif_id, is_main)

    def _draw_single_label(self, plot, start_time, end_time, motif_id, is_main=True):
        """Draw a single label rectangle on a plot with appropriate style.

        Args:
            plot: The plot widget to draw on
            start_time: Start time of the motif
            end_time: End time of the motif
            motif_id: ID of the motif for color mapping
            is_main: If True, draw full-height; if False, draw prediction strip
        """
        if motif_id not in self.motif_mappings:
            return

        color = self.motif_mappings[motif_id]["color"]
        color_rgb = tuple(int(c * 255) for c in color)

        use_edge_style = (plot == self.spectrogram_plot)

        if is_main:
            if use_edge_style:
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
        rect.setZValue(-10)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_spectrogram_style_label(self, plot, start_time, end_time, color_rgb):
        """Draw transparent fill with thick colored edges for spectrogram."""
        spec_ymin = self.app_state.get_with_default('spec_ymin')
        spec_ymax = self.app_state.get_with_default('spec_ymax')

        if spec_ymin is not None and spec_ymax is not None and spec_ymax > spec_ymin:
            y_min, y_max = spec_ymin, spec_ymax
        else:
            y_range = plot.plot_item.getViewBox().viewRange()[1]
            y_min, y_max = y_range[0], y_range[1]
            if y_max <= y_min:
                y_min, y_max = 0, 20000

        rect = pg.LinearRegionItem(
            values=(start_time, end_time),
            orientation="vertical",
            brush=(*color_rgb, 40),
            movable=False,
        )
        rect.setZValue(-10)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

        edge_pen = pg.mkPen(color=(*color_rgb, 255), width=3)

        left_edge = pg.PlotDataItem([start_time, start_time], [y_min, y_max], pen=edge_pen)
        right_edge = pg.PlotDataItem([end_time, end_time], [y_min, y_max], pen=edge_pen)
        top_edge = pg.PlotDataItem([start_time, end_time], [y_max, y_max], pen=edge_pen)
        bottom_edge = pg.PlotDataItem([start_time, end_time], [y_min, y_min], pen=edge_pen)

        for edge in [left_edge, right_edge, top_edge, bottom_edge]:
            edge.setZValue(-5)
            plot.plot_item.addItem(edge)
            plot.label_items.append(edge)

    def _draw_prediction_label(self, plot, start_time, end_time, color_rgb):
        """Draw small rectangle at top for prediction data."""
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_top = y_range[1]
        y_height = (y_range[1] - y_range[0]) * 0.10

        if y_top <= y_range[0]:
            y_top = 20000
            y_height = 2000

        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_top, y_top, y_top - y_height, y_top - y_height, y_top]

        rect = pg.PlotDataItem(
            x_coords, y_coords,
            fillLevel=y_top - y_height,
            brush=(*color_rgb, 200),
            pen=None
        )
        rect.setZValue(10)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def redraw_current_plot_labels(self, time_data, labels, predictions=None, show_predictions=False):
        """Redraw labels only on the current plot (for view range changes)."""
        if labels is None or not self.motif_mappings:
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