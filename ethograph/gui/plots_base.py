"""Shared base class for plot widgets with sync and marker functionality."""

import pyqtgraph as pg
import numpy as np
from qtpy.QtCore import Signal
from typing import Optional, Tuple


class BasePlot(pg.PlotWidget):
    """Base class for plot widgets with shared sync and marker functionality.

    Handles:
    - Time marker for video sync
    - Stream/label mode switching
    - Axes locking
    - X-axis range management
    - Common plot interactions

    Subclasses must implement the display-specific methods.
    """

    plot_clicked = Signal(object)

    def __init__(self, app_state, parent=None, **kwargs):
        super().__init__(parent, background='white', **kwargs)
        self.app_state = app_state

        # Common setup
        self.setLabel('bottom', 'Time', units='s') 

        # Time marker with enhanced styling
        self.time_marker = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('r', width=2),
            movable=False
        )
        self.addItem(self.time_marker)
        self.time_marker.setZValue(1000)

        # Setup viewbox and interaction
        self.plot_item = self.plotItem
        self.vb = self.plot_item.vb
        self.vb.setMenuEnabled(False)

        # Store interaction state
        self._interaction_enabled = True

        # Data bounds for smart zoom constraints
        self._data_time_min = None
        self._data_time_max = None
        self._min_time_range = 0.001


        # Connect click handler
        self.scene().sigMouseClicked.connect(self._handle_click)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the specific plot content (line plot, spectrogram, etc.).

        Subclasses should override this method.
        """
        raise NotImplementedError("Subclasses must implement update_plot_content")

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply y-axis range specific to the plot type.

        Subclasses should override this method.
        """
        raise NotImplementedError("Subclasses must implement apply_y_range")

    def update_plot(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update plot with current data and time window."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return


        self.update_plot_content(t0, t1)


        if t0 is not None and t1 is not None:
            self.set_x_range(mode='preserve', curr_xlim=(t0, t1))
        else:
            self.set_x_range(mode='default')

        # Only apply axis lock after setting the desired range
        self.toggle_axes_lock(preserve_default_range=(t0 is None and t1 is None))

    def update_time_marker(self, time_position: float):
        """Update time marker position for video sync."""
        self.time_marker.setValue(time_position)
        self.time_marker.show()

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker position and window for video sync."""
        if not hasattr(self.app_state, 'current_frame') or not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        current_time = frame_number / self.app_state.ds.fps
        self.update_time_marker(current_time)

        
        

    def set_x_range(self, mode='default', curr_xlim=None, center_on_frame=None):
        """Set plot x-range with different behaviors."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        time = self.app_state.ds.time.values

        if mode == 'center':
            if not hasattr(self.app_state, 'window_size'):
                return

            if center_on_frame is not None:
                current_time = center_on_frame / self.app_state.ds.fps
            else:
                current_time = self.app_state.current_frame / self.app_state.ds.fps

            window_size = self.app_state.get_with_default('window_size')
            half_window = window_size / 2.0
            t0 = current_time - half_window
            t1 = current_time + half_window

        elif mode == 'preserve' and curr_xlim:
            t0 = curr_xlim[0]
            t1 = curr_xlim[1]

            data_tmin = float(time[0])
            data_tmax = float(time[-1])
            if t0 < data_tmin:
                t0 = data_tmin
            elif t1 > data_tmax:
                t1 = data_tmax

        else:  # mode == 'default'
            window_size = self.app_state.get_with_default("window_size")
            t0 = float(time[0])
            t1 = min(t0 + float(window_size), float(time[-1]))

        self.vb.setXRange(t0, t1, padding=0)

    def get_current_xlim(self) -> Tuple[float, float]:
        """Get current x-axis limits."""
        return self.vb.viewRange()[0]

    def toggle_axes_lock(self, preserve_default_range=False):
        """Enable or disable axes locking to prevent zoom but allow panning."""
        locked = self.app_state.lock_axes

        if locked:
            current_xlim = self.vb.viewRange()[0]
            current_ylim = self.vb.viewRange()[1]
            x_range = current_xlim[1] - current_xlim[0]

            if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
                time = self.app_state.ds.time.values
                data_xmin = float(time[0])
                data_xmax = float(time[-1])

                if preserve_default_range:
                    # For default range (trial changes), use window_size as the base range
                    # but allow some flexibility for panning
                    window_size = self.app_state.get_with_default("window_size")
                    min_range = window_size * 0.8  # Allow slightly smaller
                    max_range = window_size * 1.5  # Allow zooming out a bit
                else:
                    # For preserve mode, lock exactly to current range
                    min_range = x_range
                    max_range = x_range

                self.vb.setLimits(
                    xMin=data_xmin - 3,
                    xMax=data_xmax + 3,
                    minXRange=min_range,
                    maxXRange=max_range,
                    yMin=current_ylim[0],
                    yMax=current_ylim[1]
                )

            self.vb.setMouseEnabled(x=True, y=False)
        else:
            self._apply_zoom_constraints()
            self.vb.setMouseEnabled(x=True, y=True)

    def _apply_zoom_constraints(self):
        """Apply data-aware zoom constraints to the plot viewbox."""
        # Reset limits first
        self.vb.setLimits(
            xMin=None, xMax=None, yMin=None, yMax=None,
            minXRange=None, maxXRange=None,
            minYRange=None, maxYRange=None
        )

        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        time = self.app_state.ds.time.values
        xMin = time[0]
        xMax = time[-1]
        xRange = xMax - xMin

        self.vb.setLimits(
            xMin=xMin - 3,
            xMax=xMax + 3,
            minXRange=None,
            maxXRange=xRange + 1,
        )

        # Apply plot-specific y constraints
        self._apply_y_constraints()

    def _apply_y_constraints(self):
        """Apply y-axis constraints specific to the plot type.

        Subclasses should override this method.
        """
        pass  # Default implementation does nothing

    def _handle_click(self, event):
        """Handle mouse clicks on plot."""
        if not self._interaction_enabled:
            return

        pos = self.plot_item.vb.mapSceneToView(event.scenePos())

        click_info = {
            'x': pos.x(),
            'button': event.button()
        }
        self.plot_clicked.emit(click_info)



    def update_yrange(self, ymin: Optional[float], ymax: Optional[float]):
        """Update y-axis range."""
        if self.app_state.sync_state == "pyav_stream_mode":
            return

        self.apply_y_range(ymin, ymax)