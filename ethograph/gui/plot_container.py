"""Simple container widget for switching between different plot types."""

from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Signal, QSize
from .plots_lineplot import LinePlot
from .plots_spectrogram import SpectrogramPlot
import pyqtgraph as pg
import numpy as np

class PlotContainer(QWidget):
    """Container that holds and switches between LinePlot and SpectrogramPlot.

    It just manages the widget switching and exposes the current_plot for direct access.
    """

    plot_changed = Signal(str)  # Emits 'lineplot' or 'spectrogram'
    labels_redraw_needed = Signal()  # Emits when labels need to be redrawn on new plot

    def __init__(self, napari_viewer, app_state):
        super().__init__()
        self.viewer = napari_viewer
        self.app_state = app_state

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create both plot types
        self.line_plot = LinePlot(napari_viewer, app_state)
        self.spectrogram_plot = SpectrogramPlot(app_state)

        # Current active plot
        self.current_plot = self.line_plot
        self.current_plot_type = 'lineplot'

        # Add line plot by default
        self.layout.addWidget(self.line_plot)
        self.spectrogram_plot.hide()
        
        self.confidence_item = None

    def sizeHint(self):
        return QSize(self.width(), 300)

    def switch_to_spectrogram(self):
        """Switch to spectrogram display."""
        if self.current_plot_type == 'spectrogram':
            return

        prev_xlim = self.line_plot.get_current_xlim()
        prev_time_marker = self.line_plot.time_marker.value()

        self.line_plot.hide()
        self.layout.removeWidget(self.line_plot)

        self.layout.addWidget(self.spectrogram_plot)
        self.spectrogram_plot.show()

        self.current_plot = self.spectrogram_plot
        self.current_plot_type = 'spectrogram'

        self.spectrogram_plot.set_x_range(mode='preserve', curr_xlim=prev_xlim)
        self.spectrogram_plot.update_time_marker(prev_time_marker)

        self.plot_changed.emit('spectrogram')
        self.labels_redraw_needed.emit()

    def switch_to_lineplot(self):
        """Switch to line plot display."""
        if self.current_plot_type == 'lineplot':
            return

        prev_xlim = self.spectrogram_plot.get_current_xlim()
        prev_time_marker = self.spectrogram_plot.time_marker.value()

        self.spectrogram_plot.hide()
        self.layout.removeWidget(self.spectrogram_plot)

        self.layout.addWidget(self.line_plot)
        self.line_plot.show()

        self.current_plot = self.line_plot
        self.current_plot_type = 'lineplot'

        self.line_plot.set_x_range(mode='preserve', curr_xlim=prev_xlim)
        self.line_plot.update_time_marker(prev_time_marker)

        self.plot_changed.emit('lineplot')
        self.labels_redraw_needed.emit()

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
            
            
    def get_current_plot(self):
        """Get the currently active plot widget."""
        return self.current_plot

    def is_spectrogram(self):
        """Check if currently showing spectrogram."""
        return self.current_plot_type == 'spectrogram'

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



    @property
    def vb(self):
        """Get ViewBox from active plot for direct access."""
        return self.current_plot.vb