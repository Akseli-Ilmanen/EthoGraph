"""Shared base class for plot widgets with sync and marker functionality."""

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QTimer, QRunnable, QThreadPool, QObject, Signal, Qt


from .app_constants import (
    LOCKED_RANGE_MIN_FACTOR,
    LOCKED_RANGE_MAX_FACTOR,
    AXIS_LIMIT_PADDING_RATIO,
    Z_INDEX_TIME_MARKER,
)

# -------------------------------
# Worker helper
# -------------------------------
class WorkerSignals(QObject):
    """Signals for worker completion."""
    finished = Signal(object)  # emit computation result

class Worker(QRunnable):
    """Run a function in a background thread."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        self.signals.finished.emit(result)


# -------------------------------
# ThrottleDebounce (GUI-thread safe)
# -------------------------------
class ThrottleDebounce(QObject):
    """Throttle + debounce helper for rate-limiting expensive plot updates.

    All callbacks are invoked on the GUI (main) thread — safe to call any
    Qt operation inside them.

    For callbacks that do heavy computation (numpy/IO), split them into a
    pure-compute function and a render function, then use ``run_async``
    inside the callback so that only the Qt rendering touches the main thread.

    Usage::

        self._td = ThrottleDebounce(
            throttle_ms=16,
            debounce_ms=40,
            throttle_cb=self._on_throttle,   # called on main thread
            debounce_cb=self._on_debounce,   # called on main thread
        )
        self._td.trigger()   # call from any GUI event
    """

    def __init__(self,
                 throttle_ms: int = 16,
                 debounce_ms: int = 40,
                 throttle_cb=None,
                 debounce_cb=None):
        super().__init__()
        self._throttle_cb = throttle_cb
        self._debounce_cb = debounce_cb

        # Timers live in the main thread (created here, which is the main thread).
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setInterval(max(throttle_ms, 1))
        self._throttle_timer.timeout.connect(self._run_throttle)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._on_debounce)

    def trigger(self):
        """Call from a GUI-thread event (e.g. sigRangeChanged handler)."""
        if not self._throttle_timer.isActive():
            self._throttle_timer.start()
        self._debounce_timer.start()

    def _run_throttle(self):
        # Runs on main thread via QTimer — safe for any Qt operation.
        if self._throttle_cb is not None:
            self._throttle_cb()

    def _on_debounce(self):
        # Drag stopped: stop throttle, fire final update.
        self._throttle_timer.stop()
        if self._debounce_cb is not None:
            self._debounce_cb()

    def stop(self):
        self._throttle_timer.stop()
        self._debounce_timer.stop()


def run_async(compute_fn, render_fn):
    """Run ``compute_fn`` in a background thread; deliver its return value to
    ``render_fn`` on the main (GUI) thread.

    ``compute_fn`` must not touch any Qt objects.
    ``render_fn(result)`` may freely update the GUI.

    Example inside a plot callback::

        def _do_range_update(self):
            if self._busy:
                return
            self._busy = True
            t0, t1 = self.get_current_xlim()
            run_async(
                lambda: self._compute_data(t0, t1),   # background thread
                lambda data: self._render(data),      # main thread
            )
    """
    worker = Worker(compute_fn)
    # WorkerSignals was created in the main thread, so this connection is
    # automatically a Qt.QueuedConnection → render_fn runs on the main thread.
    worker.signals.finished.connect(render_fn, Qt.QueuedConnection)
    QThreadPool.globalInstance().start(worker)
        
        



class TimeAxisItem(pg.AxisItem):
    """Custom axis that displays time in min:sec format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """Convert seconds to min:sec format."""
        strings = []
        for v in values:
            total_seconds = abs(v)
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            sign = '-' if v < 0 else ''

            if minutes > 0:
                if seconds == int(seconds):
                    strings.append(f'{sign}{minutes}:{int(seconds):02d}')
                else:
                    strings.append(f'{sign}{minutes}:{seconds:05.2f}')
            else:
                if seconds == int(seconds):
                    strings.append(f'{sign}{int(seconds)}s')
                else:
                    strings.append(f'{sign}{seconds:.2f}s')
        return strings


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
        time_axis = TimeAxisItem(orientation='bottom')
        super().__init__(parent, background='white', axisItems={'bottom': time_axis}, **kwargs)
        self.app_state = app_state

        self.setLabel('bottom', 'Time') 

        # Time marker with enhanced styling
        self.time_marker = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('r', width=2),
            movable=False
        )
        self.addItem(self.time_marker)
        self.time_marker.setZValue(Z_INDEX_TIME_MARKER)

        # Setup viewbox and interaction
        self.plot_item = self.plotItem
        self.vb = self.plot_item.vb
        self.vb.setMenuEnabled(False)

        # Store interaction state
        self._interaction_enabled = True

        # Last-rendered time range; subclasses update this to know when to re-render
        self.current_range: tuple[float, float] | None = None

        # Connect click handler
        self.scene().sigMouseClicked.connect(self._handle_click)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the specific plot content (line plot, spectrogram, etc.).

        Subclasses should override this method.
        """
        print(f"[plots_base] update_plot_content called in {self.__class__.__name__} (id={id(self)}) t0={t0}, t1={t1}")
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
        is_new_trial = t0 is None and t1 is None
        self.toggle_axes_lock(preserve_default_range=is_new_trial)

    def update_time_marker(self, time_position: float):
        """Update time marker position for video sync."""
        self.time_marker.setValue(time_position)
        self.time_marker.show()

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker position and window for video sync."""
        video = getattr(self.app_state, 'video', None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps
        self.update_time_marker(current_time)

        if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
            if getattr(self.app_state, 'center_playback', False):
                self.set_x_range(mode='center', center_on_frame=frame_number)
                t0, t1 = self.get_current_xlim()
                self.update_plot_content(t0, t1)
            else:
                t0, t1 = self.get_current_xlim()
                self.update_plot_content(t0, t1)



    def _get_time_bounds(self) -> Optional[Tuple[float, float]]:
        """Return (t_min, t_max) for this plot's time domain.

        Subclasses override to provide their own time source.
        Default uses app_state.time (current feature's time coord).
        """
        bounds = self.app_state.trial_bounds
        if bounds is None:
            return None
        return bounds.start_s, bounds.end_s


    def set_x_range(self, mode='default', curr_xlim=None, center_on_frame=None):
        """Set plot x-range with different behaviors."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        bounds = self._get_time_bounds()
        if bounds is None:
            if mode == 'preserve' and curr_xlim:
                self.vb.setXRange(curr_xlim[0], curr_xlim[1], padding=0)
            return
        data_tmin, data_tmax = bounds

        if mode == 'center':
            video = getattr(self.app_state, 'video', None)
            if center_on_frame is not None:
                current_time = video.frame_to_time(center_on_frame) if video else center_on_frame / self.app_state.video_fps
            else:
                current_time = video.frame_to_time(self.app_state.current_frame) if video else self.app_state.current_frame / self.app_state.video_fps

            xlim = self.get_current_xlim()
            half_window = (xlim[1] - xlim[0]) / 2.0
            t0 = current_time - half_window
            t1 = current_time + half_window

        elif mode == 'preserve' and curr_xlim:
            t0 = curr_xlim[0]
            t1 = curr_xlim[1]

            if t0 < data_tmin:
                t0 = data_tmin
            elif t1 > data_tmax:
                t1 = data_tmax

        else:  # mode == 'default'
            window_size = self.app_state.get_with_default("window_size")
            t0 = data_tmin
            t1 = min(t0 + float(window_size), data_tmax)

        self.vb.setXRange(t0, t1, padding=0)

    def get_current_xlim(self) -> Tuple[float, float]:
        """Get current x-axis limits."""
        return self.vb.viewRange()[0]

    def toggle_axes_lock(self, preserve_default_range=False, x_bounds_override=None):
        """Enable or disable axes locking to prevent zoom but allow panning."""
        locked = self.app_state.lock_axes

        if locked:
            current_xlim = self.vb.viewRange()[0]
            current_ylim = self.vb.viewRange()[1]
            x_range = current_xlim[1] - current_xlim[0]

            bounds = x_bounds_override or self._get_time_bounds()
            if hasattr(self.app_state, 'ds') and self.app_state.ds is not None and bounds is not None:
                data_xmin, data_xmax = bounds
                data_range = data_xmax - data_xmin
                padding = min(data_range * AXIS_LIMIT_PADDING_RATIO, 5)

                if preserve_default_range:
                    window_size = self.app_state.get_with_default("window_size")
                    min_range = window_size * LOCKED_RANGE_MIN_FACTOR
                    max_range = window_size * LOCKED_RANGE_MAX_FACTOR
                else:
                    min_range = x_range
                    max_range = x_range

                self.vb.setLimits(
                    xMin=data_xmin - padding,
                    xMax=data_xmax + padding,
                    minXRange=min_range,
                    maxXRange=max_range,
                    yMin=current_ylim[0],
                    yMax=current_ylim[1]
                )

            self.vb.setMouseEnabled(x=True, y=False)
        else:
            self._apply_zoom_constraints(x_bounds_override=x_bounds_override)
            self.vb.setMouseEnabled(x=True, y=True)

    def _apply_zoom_constraints(self, x_bounds_override=None):
        """Apply data-aware zoom constraints to the plot viewbox.

        Parameters
        ----------
        x_bounds_override
            Optional ``(xMin, xMax)`` to use instead of this plot's own
            ``_get_time_bounds()``.  The container passes the tightest
            bounds across all visible panels so that no panel scrolls past
            another's data.
        """
        self.vb.setLimits(
            xMin=None, xMax=None, yMin=None, yMax=None,
            minXRange=None, maxXRange=None,
            minYRange=None, maxYRange=None
        )

        if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
            bounds = x_bounds_override or self._get_time_bounds()
            if bounds is not None:
                xMin, xMax = bounds
                xRange = xMax - xMin
                padding = xRange * AXIS_LIMIT_PADDING_RATIO

                self.vb.setLimits(
                    xMin=xMin - padding,
                    xMax=xMax + padding,
                    minXRange=None,
                    maxXRange=xRange * (1 + AXIS_LIMIT_PADDING_RATIO),
                )

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

        from qtpy.QtCore import Qt

        if event.double() and event.button() == Qt.LeftButton:
            self.autoscale()
            return

        pos = self.plot_item.vb.mapSceneToView(event.scenePos())

        click_info = {
            'x': pos.x(),
            'button': event.button()
        }
        self.plot_clicked.emit(click_info)

    def autoscale(self):
        """Reset Y-axis to auto-fit visible data."""
        self.vb.enableAutoRange(x=False, y=True)

