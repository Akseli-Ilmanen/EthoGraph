"""Space plot widget for displaying arbitrary 2D/3D scatter trajectories.

Users pick which feature (and sub-dimension) to plot on each axis via
combo boxes embedded in the dock widget itself.  Data is fetched through
the DataLoader so xarray, pynapple, and NWB sources all work.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import yaml
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ethograph.features.preprocessing import interpolate_nans
from ethograph.io.catalog import DataLoader
from ethograph.gui.plots_lineplot import MultiColoredLineItem


logger = logging.getLogger(__name__)

SEPARATOR = " · "


# ---------------------------------------------------------------------------
# Axis item helpers
# ---------------------------------------------------------------------------

def _build_axis_items(store: DataLoader) -> list[str]:
    """Build combo items from store features + their sub-dimensions.

    Each item is either ``"feature"`` (1-D) or ``"feature · column"`` (2-D+).
    Multi-dimensional features (e.g. position with space×keypoints×individuals)
    expand only the *first* non-time dimension into separate axis items.
    The remaining dimensions are controlled by the main GUI selections.
    """
    items: list[str] = []

    for feat in store.features:
        feat_dims = store.feature_dims(feat)
        if feat_dims:
            first_dim_values = next(iter(feat_dims.values()))
            for val in first_dim_values:
                items.append(f"{feat}{SEPARATOR}{val}")
        else:
            items.append(feat)

    return items


def _parse_axis_item(item: str) -> tuple[str, str | None]:
    """Parse ``"feature · column"`` → ``(feature, column)``."""
    if SEPARATOR in item:
        feat, col = item.split(SEPARATOR, 1)
        return feat, col
    return item, None


def _select_axis(store: DataLoader, item: str, selections: dict,
                 t0: float | None = None, t1: float | None = None):
    """Fetch 1-D numpy array + time for a single axis item.

    Returns ``(time, data)`` or ``(None, None)`` on failure.
    """
    feat, col = _parse_axis_item(item)

    # Build selections: start from app-level selections, then add
    # or override with the column picked for this axis.
    sel = dict(selections)
    if col is not None:
        feat_dims = store.feature_dims(feat)
        for dim_name, dim_vals in feat_dims.items():
            if col in dim_vals:
                sel[dim_name] = col
                break

    # Ensure every non-time dim of this feature has a selection so
    # sel_valid gets a 1-D or 2-D array (never 3-D+).
    feat_dims = store.feature_dims(feat)
    for dim_name, dim_vals in feat_dims.items():
        if dim_name not in sel and dim_vals:
            sel[dim_name] = dim_vals[0]

    pd = store.select(feat, sel, t0=t0, t1=t1)
    if pd is None:
        return None, None

    data = pd.data
    if data.ndim == 2:
        data = data[:, 0]

    return pd.time, data.astype(np.float64)


# ---------------------------------------------------------------------------
# Reference geometry: vertices + edges
# ---------------------------------------------------------------------------

@dataclass
class ReferenceGeometry:
    """A set of vertices connected by indexed edges."""
    name: str
    vertices: np.ndarray   # (N, 2) or (N, 3)
    edges: list[tuple[int, int]]
    color: str = "black"


def load_space_config(config_path: Path) -> Optional[dict]:
    """Load space config from YAML. Returns None if file is absent."""
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_references(cfg: dict) -> list[ReferenceGeometry]:
    """Parse reference geometry from space.yaml config.

    Supports new format (``references`` list with vertices + edges)
    and old format (``arena.xy_polygon`` + ``z_bot``/``z_top``).
    """
    refs: list[ReferenceGeometry] = []

    # New format: list of {name, vertices, edges, color}
    if "references" in cfg:
        for entry in cfg["references"]:
            verts = np.array(entry["vertices"], dtype=np.float64)
            edges = [tuple(e) for e in entry["edges"]]
            refs.append(ReferenceGeometry(
                name=entry.get("name", "ref"),
                vertices=verts,
                edges=edges,
                color=entry.get("color", "black"),
            ))
        return refs

    # Old format: arena with xy_polygon + z_bot/z_top → auto-convert
    arena = cfg.get("arena", cfg)
    if "xy_polygon" not in arena:
        return refs

    xy = np.array(arena["xy_polygon"], dtype=np.float64)
    z_bot = arena.get("z_bot")
    z_top = arena.get("z_top")

    if z_bot is not None and z_top is not None:
        # Build 3D box wireframe from 2D polygon floor + ceiling
        n = len(xy)
        floor = np.column_stack([xy, np.full(n, z_bot)])
        ceil = np.column_stack([xy, np.full(n, z_top)])
        verts = np.vstack([floor, ceil])

        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))          # floor edges
            edges.append((n + i, n + i + 1))  # ceiling edges
            edges.append((i, n + i))          # verticals
        # Close floor/ceiling if not already closed
        if not np.allclose(xy[0], xy[-1]):
            edges.append((n - 1, 0))
            edges.append((2 * n - 1, n))
        edges.append((n - 1, 2 * n - 1))     # last vertical

        refs.append(ReferenceGeometry("arena", verts, edges))
    else:
        # 2D polygon only
        n = len(xy)
        edges = [(i, i + 1) for i in range(n - 1)]
        if not np.allclose(xy[0], xy[-1]):
            edges.append((n - 1, 0))
        refs.append(ReferenceGeometry("arena", xy, edges))

    return refs


def _color_to_rgba(color_str: str) -> tuple:
    """Convert color name/hex to (r, g, b, a) float tuple for GL."""
    try:
        from pyqtgraph.functions import colorStr
        qc = pg.mkColor(color_str)
        return (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)


def _render_reference_2d(plot_item, ref: ReferenceGeometry):
    """Draw a ReferenceGeometry on a 2D PlotWidget."""
    verts = ref.vertices
    for i0, i1 in ref.edges:
        line = pg.PlotCurveItem(
            x=np.array([verts[i0, 0], verts[i1, 0]]),
            y=np.array([verts[i0, 1], verts[i1, 1]]),
            pen=pg.mkPen(color=ref.color, width=2),
        )
        plot_item.addItem(line)


def _render_reference_3d(gl_widget, ref: ReferenceGeometry):
    """Draw a ReferenceGeometry on a 3D GLViewWidget."""
    verts = ref.vertices
    if verts.shape[1] < 3:
        verts = np.column_stack([verts, np.zeros(len(verts))])

    segments = []
    for i0, i1 in ref.edges:
        segments.extend([verts[i0], verts[i1], [np.nan, np.nan, np.nan]])
    if not segments:
        return
    segments = segments[:-1]  # drop trailing NaN separator

    color = _color_to_rgba(ref.color)
    wireframe = gl.GLLinePlotItem(
        pos=np.array(segments, dtype=np.float32),
        color=color, width=2, antialias=True,
    )
    gl_widget.addItem(wireframe)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_2d(plot_widget, X, Y, color_data=None):
    """Plot 2D trajectory on a PlotWidget. Returns the line item."""
    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] >= 3:
        line = MultiColoredLineItem(x=X, y=Y, colors=color_data, width=3)
    else:
        line = pg.PlotCurveItem(x=X, y=Y, pen=pg.mkPen(color='b', width=3))
    line._is_trajectory = True
    plot_widget.addItem(line)
    return line


def _render_3d(gl_widget, X, Y, Z, color_data=None):
    """Plot 3D trajectory on a GLViewWidget. Returns the line item."""
    xyz = np.column_stack([X, Y, Z]).astype(np.float32)
    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] >= 3:
        if color_data.shape[1] == 3:
            alpha = np.ones((color_data.shape[0], 1), dtype=color_data.dtype)
            color_data = np.concatenate([color_data, alpha], axis=1)
        if color_data.max() > 1.0:
            color_data = color_data / 255.0
        line = gl.GLLinePlotItem(pos=xyz, color=color_data, width=3, antialias=True)
    else:
        line = gl.GLLinePlotItem(pos=xyz, color=(0, 0, 1, 1), width=3, antialias=True)
    line._is_trajectory = True
    gl_widget.addItem(line)
    return line


def _auto_camera_3d(gl_widget, X, Y, Z):
    """Set a reasonable default camera for 3D data."""
    cx, cy, cz = float(np.nanmean(X)), float(np.nanmean(Y)), float(np.nanmean(Z))
    extent = float(max(np.nanmax(X) - np.nanmin(X), np.nanmax(Y) - np.nanmin(Y), np.nanmax(Z) - np.nanmin(Z))) * 1.5
    gl_widget.setCameraPosition(
        pos=pg.Vector(cx, cy, cz),
        distance=max(extent, 1.0),
        elevation=30,
        azimuth=200,
    )


# ---------------------------------------------------------------------------
# SpacePlot widget
# ---------------------------------------------------------------------------

class SpacePlot(QWidget):
    """Dock widget for displaying spatial plots with user-selectable axes."""

    def __init__(self, viewer, app_state):
        super().__init__()
        self.viewer = viewer
        self.app_state = app_state
        self.dock_widget = None

        self._store: DataLoader | None = None

        # --- Layout ---
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 0)
        root.setSpacing(4)
        self.setLayout(root)

        # Row 1: axis combos
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        toolbar.addWidget(QLabel("X"))
        self.x_combo = QComboBox()
        self.x_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.x_combo)

        toolbar.addWidget(QLabel("Y"))
        self.y_combo = QComboBox()
        self.y_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.y_combo)

        self.z_label = QLabel("Z")
        toolbar.addWidget(self.z_label)
        self.z_combo = QComboBox()
        self.z_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.z_combo)

        root.addLayout(toolbar)

        # Row 2: 3D checkbox + keypoint filter
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)

        self.cb_3d = QCheckBox("3D")
        row2.addWidget(self.cb_3d)

        self.cb_hide_zeros = QCheckBox("Hide zeros")
        self.cb_hide_zeros.setToolTip("Hide points where all dimensions are exactly zero")
        row2.addWidget(self.cb_hide_zeros)

        self.keypoint_label = QLabel("Keypoint")
        row2.addWidget(self.keypoint_label)
        self.keypoint_combo = QComboBox()
        self.keypoint_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row2.addWidget(self.keypoint_combo)

        root.addLayout(row2)

        # Plot area (created on first update)
        self.space_widget = None
        self.is_3d = False
        self._plot_container = None
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._update_plot)

        # Trajectory state for highlight / time marker
        self._trajectory_pos: tuple | None = None
        self._trajectory_times: np.ndarray | None = None
        self._time_marker_item = None
        self._locked_ranges: dict | None = None  # saved axis ranges when lock is on

        # Connect combo/checkbox signals
        self.x_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.y_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.z_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.cb_3d.toggled.connect(self._on_3d_toggled)
        self.keypoint_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.cb_hide_zeros.toggled.connect(self._on_axis_changed)

        # Listen for settings changes via app_state
        app_state.space_percentile_xyzlim_changed.connect(self._on_settings_changed)
        app_state.space_limit_to_window_changed.connect(self._on_settings_changed)

        self._set_3d_visible(False)
        super().hide()

    # --- Public API --------------------------------------------------------

    def set_plot_container(self, plot_container):
        """Wire up the main plot container for x-range queries."""
        self._plot_container = plot_container

    def set_store(self, store: DataLoader | None):
        """Set the feature store and repopulate axis combos."""
        self._store = store
        self._populate_combos()


    def show(self):
        if not self.dock_widget:
            self.dock_widget = self.viewer.window.add_dock_widget(
                self, area="left", name="Space Plot"
            )
            main_window = self.viewer.window._qt_window
            desired_width = int(main_window.width() * 0.2)
            self.setMinimumSize(120, 120)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.dock_widget.resize(desired_width, self.dock_widget.height())
        else:
            self.dock_widget.setVisible(True)
        super().show()

    def hide(self):
        if self.dock_widget:
            self.dock_widget.setVisible(False)
        super().hide()

    def refresh(self):
        """Re-render with current axis selections."""
        self._update_plot()

    # --- Combo population --------------------------------------------------

    def _populate_combos(self):
        """Fill axis combos from the current store."""
        all_combos = (self.x_combo, self.y_combo, self.z_combo, self.keypoint_combo)
        for combo in all_combos:
            combo.blockSignals(True)
            combo.clear()

        if self._store is None:
            for combo in all_combos:
                combo.blockSignals(False)
            return

        items = _build_axis_items(self._store)
        for combo in (self.x_combo, self.y_combo, self.z_combo):
            combo.addItems(items)

        # Smart defaults: pick position · x/y/z if available, else first two
        self._set_default_axes(items)

        # Populate keypoint combo from non-first (non-axis-expanded) dimensions
        self._populate_keypoint_combo()

        for combo in all_combos:
            combo.blockSignals(False)

    def _set_default_axes(self, items: list[str]):
        """Pick sensible defaults for X/Y/Z combos."""
        def find(suffix: str) -> int:
            for i, item in enumerate(items):
                if item.endswith(f"{SEPARATOR}{suffix}"):
                    return i
            return -1

        ix = find("x")
        iy = find("y")
        iz = find("z")

        if ix >= 0 and iy >= 0:
            self.x_combo.setCurrentIndex(ix)
            self.y_combo.setCurrentIndex(iy)
            if iz >= 0:
                self.z_combo.setCurrentIndex(iz)
        else:
            # Fallback: first two items
            if len(items) >= 2:
                self.x_combo.setCurrentIndex(0)
                self.y_combo.setCurrentIndex(1)
            if len(items) >= 3:
                self.z_combo.setCurrentIndex(2)

    _KEYPOINT_DIM_NAMES = {"keypoint", "keypoints"}

    def _populate_keypoint_combo(self):
        """Populate keypoint combo from keypoint dimensions of store features."""
        if self._store is None:
            return

        keypoint_vals: list[str] = []
        self._keypoint_dim_name = None
        for feat in self._store.features:
            for dim_name, dim_vals in self._store.feature_dims(feat).items():
                if dim_name.lower() not in self._KEYPOINT_DIM_NAMES or not dim_vals:
                    continue
                self._keypoint_dim_name = dim_name
                for v in dim_vals:
                    if v not in keypoint_vals:
                        keypoint_vals.append(v)

        has_keypoints = bool(keypoint_vals)
        self.keypoint_combo.setVisible(has_keypoints)
        self.keypoint_label.setVisible(has_keypoints)
        if has_keypoints:
            self.keypoint_combo.addItems(keypoint_vals)

    def _get_keypoint_selection(self) -> dict[str, str]:
        """Return selection dict override from the keypoint combo."""
        text = self.keypoint_combo.currentText()
        if not text:
            return {}
        if SEPARATOR in text:
            dim_name, val = text.split(SEPARATOR, 1)
            return {dim_name: val}
        dim_name = getattr(self, '_keypoint_dim_name', None)
        if dim_name:
            return {dim_name: text}
        return {}

    # --- Axis change handlers ----------------------------------------------

    def _on_axis_changed(self, *_args):
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    def _on_3d_toggled(self, checked: bool):
        self._set_3d_visible(checked)
        self._save_to_app_state()
        if self._store is not None:
            self._update_plot()

    def _on_settings_changed(self, *_args):
        """Re-render when a plot-settings value changes (debounced)."""
        if self._store is not None and self.isVisible():
            self._debounce_timer.start()

    def on_xrange_changed(self):
        """Called by DataWidget when the lineplot x-range changes."""
        if getattr(self.app_state, 'space_limit_to_window', False) and self.isVisible():
            self._debounce_timer.start()

    def _save_to_app_state(self):
        self.app_state.space_x_axis = self.x_combo.currentText() or None
        self.app_state.space_y_axis = self.y_combo.currentText() or None
        self.app_state.space_z_axis = self.z_combo.currentText() or None
        self.app_state.space_3d = self.cb_3d.isChecked()



    def _set_3d_visible(self, visible: bool):
        self.z_label.setVisible(visible)
        self.z_combo.setVisible(visible)

    # --- Core plot logic ---------------------------------------------------

    def _get_window_time_range(self) -> tuple[float | None, float | None]:
        """Return (t0, t1) from the lineplot x-range if limit-to-window is on."""
        if not getattr(self.app_state, 'space_limit_to_window', False):
            return None, None
        if self._plot_container is None:
            return None, None
        try:
            return self._plot_container.get_current_xlim()
        except Exception:
            return None, None

    def _update_plot(self):
        """Fetch data for selected axes and render."""
        store = self._store
        if store is None:
            return

        x_item = self.x_combo.currentText()
        y_item = self.y_combo.currentText()
        if not x_item or not y_item:
            return

        view_3d = self.cb_3d.isChecked()
        z_item = self.z_combo.currentText() if view_3d else None

        selections = self.app_state.get_selections()
        selections.update(self._get_keypoint_selection())
        t0, t1 = self._get_window_time_range()

        time_x, data_x = _select_axis(store, x_item, selections, t0=t0, t1=t1)
        time_y, data_y = _select_axis(store, y_item, selections, t0=t0, t1=t1)
        if time_x is None or time_y is None:
            return

        n = min(len(data_x), len(data_y))
        data_x, data_y = data_x[:n], data_y[:n]
        times = time_x[:n]

        data_z = None
        if view_3d and z_item:
            _, dz = _select_axis(store, z_item, selections, t0=t0, t1=t1)
            if dz is not None:
                data_z = dz[:n]

        # Mask points where all dimensions are exactly zero
        if self.cb_hide_zeros.isChecked():
            zero_mask = (data_x == 0) & (data_y == 0)
            if data_z is not None:
                zero_mask &= (data_z == 0)
            data_x = np.where(zero_mask, np.nan, data_x)
            data_y = np.where(zero_mask, np.nan, data_y)
            if data_z is not None:
                data_z = np.where(zero_mask, np.nan, data_z)

        color_data = self._get_color_data(store, selections, n)

        use_3d = view_3d and data_z is not None
        locked = getattr(self.app_state, 'space_lock_axes', False)

        # Save current ranges before rebuilding the widget
        saved_ranges = self._capture_ranges() if locked else None

        self._rebuild_plot_widget(use_3d)

        if use_3d:
            _render_3d(self.space_widget, data_x, data_y, data_z, color_data)
            _auto_camera_3d(self.space_widget, data_x, data_y, data_z)
        else:
            _render_2d(self.space_widget, data_x, data_y, color_data)
            plot_item = self.space_widget.getPlotItem()
            plot_item.setLabel('bottom', x_item)
            plot_item.setLabel('left', y_item)

        if locked and saved_ranges:
            self._restore_ranges(saved_ranges)
        else:
            self._apply_percentile_limits(data_x, data_y, data_z)
        self._draw_references()

        self._trajectory_pos = (data_x, data_y, data_z)
        self._trajectory_times = times
        self._time_marker_item = None
        self.is_3d = use_3d

        # Place marker at current time
        current_frame = getattr(self.app_state, 'current_frame', 0)
        video = getattr(self.app_state, 'video', None)
        if video:
            t = video.frame_to_time(current_frame)
        else:
            fps = getattr(self.app_state, 'video_fps', 30)
            t = current_frame / fps if fps else 0.0
        self.update_time_marker(t)


    def _get_color_data(self, store: DataLoader, selections: dict, n: int):
        """Fetch color data if a color variable is selected."""
        color_var = None
        if hasattr(self.app_state, 'colors_sel') and self.app_state.colors_sel not in (None, "None", ""):
            color_var = self.app_state.colors_sel

        if not color_var or color_var not in store.features:
            return None

        pd = store.select(color_var, selections)
        if pd is None:
            return None

        cd = pd.data
        if cd.ndim == 2 and cd.shape[1] >= 3:
            if cd.max() > 1.0:
                cd = cd / 255.0
            return cd[:n]
        return None

    def _rebuild_plot_widget(self, view_3d: bool):
        """Remove old widget and create the right type."""
        if self.space_widget is not None:
            self.layout().removeWidget(self.space_widget)
            self.space_widget.deleteLater()

        if view_3d:
            self.space_widget = gl.GLViewWidget()
            self.space_widget.setBackgroundColor('w')
        else:
            self.space_widget = pg.PlotWidget()
            self.space_widget.setBackground('w')

        self.layout().addWidget(self.space_widget)

    def _load_references(self) -> list[ReferenceGeometry]:
        """Load reference geometry from space.yaml or arena.yaml."""
        from ethograph.utils.paths import find_config

        nc_path = getattr(self.app_state, 'nc_file_path', None)
        if not nc_path:
            nc_path = getattr(self.app_state, 'nap_path', None)
        data_dir = Path(nc_path).parent if nc_path else None

        for name in ("space.yaml", "arena.yaml"):
            cfg_path = find_config(name, data_dir)
            if cfg_path is None:
                continue
            cfg = load_space_config(cfg_path)
            if cfg is None:
                continue
            refs = _parse_references(cfg)
            if refs:
                return refs

        logger.debug("No reference geometry found (searched space.yaml, arena.yaml)")
        return []

    def _draw_references(self):
        """Draw all reference geometry items."""
        refs = self._load_references()
        if not refs:
            return

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)
        for ref in refs:
            if is_gl:
                _render_reference_3d(self.space_widget, ref)
            else:
                plot_item = self.space_widget.getPlotItem()
                _render_reference_2d(plot_item, ref)

    # --- Percentile axis limits (zoom constraints) --------------------------

    def _apply_percentile_limits(self, data_x, data_y, data_z=None):
        """Constrain zoom to per-axis percentile range using vb.setLimits()."""
        percentile = getattr(self.app_state, 'space_percentile_xyzlim', 100.0)
        if percentile >= 100.0 or self.space_widget is None:
            return

        lo = (100 - percentile) / 2
        hi = 100 - lo

        x_lo, x_hi = np.nanpercentile(data_x, [lo, hi])
        y_lo, y_hi = np.nanpercentile(data_y, [lo, hi])

        x_range = x_hi - x_lo
        y_range = y_hi - y_lo
        if x_range <= 0 or y_range <= 0:
            return

        x_buf = x_range * 0.2
        y_buf = y_range * 0.2

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)
        if is_gl:
            cx = float((x_lo + x_hi) / 2)
            cy = float((y_lo + y_hi) / 2)
            extent = max(x_range, y_range)
            if data_z is not None:
                z_lo, z_hi = np.nanpercentile(data_z, [lo, hi])
                cz = float((z_lo + z_hi) / 2)
                extent = max(extent, z_hi - z_lo)
            else:
                cz = 0.0
            self.space_widget.setCameraPosition(
                pos=pg.Vector(cx, cy, cz),
                distance=max(float(extent) * 1.5, 1.0),
                elevation=30,
                azimuth=200,
            )
        else:
            vb = self.space_widget.getPlotItem().vb
            vb.setLimits(
                xMin=x_lo - x_buf, xMax=x_hi + x_buf,
                yMin=y_lo - y_buf, yMax=y_hi + y_buf,
                minXRange=x_range * 0.1, maxXRange=x_range + x_buf,
                minYRange=y_range * 0.1, maxYRange=y_range + y_buf,
            )
            vb.setRange(xRange=(x_lo, x_hi), yRange=(y_lo, y_hi), padding=0.05)

    def _capture_ranges(self) -> dict | None:
        """Snapshot the current axis ranges (2D) or camera position (3D)."""
        if self.space_widget is None:
            return None
        if isinstance(self.space_widget, gl.GLViewWidget):
            opts = self.space_widget.cameraParams()
            return {"mode": "3d", "camera": opts}
        vb = self.space_widget.getPlotItem().vb
        xr, yr = vb.viewRange()
        return {"mode": "2d", "x": tuple(xr), "y": tuple(yr)}

    def _restore_ranges(self, ranges: dict):
        """Restore previously captured axis ranges."""
        if self.space_widget is None:
            return
        if ranges["mode"] == "3d" and isinstance(self.space_widget, gl.GLViewWidget):
            cam = ranges["camera"]
            self.space_widget.setCameraPosition(
                pos=cam.get("center"),
                distance=cam.get("distance"),
                elevation=cam.get("elevation"),
                azimuth=cam.get("azimuth"),
            )
        elif ranges["mode"] == "2d" and not isinstance(self.space_widget, gl.GLViewWidget):
            vb = self.space_widget.getPlotItem().vb
            vb.setRange(xRange=ranges["x"], yRange=ranges["y"], padding=0)

    # --- Highlight / time marker -------------------------------------------

    def highlight_time_segment(self, start_time: float, end_time: float,
                               color=(255, 102, 0)):
        """Highlight a time segment of the trajectory."""
        if not self.space_widget or self._trajectory_pos is None or self._trajectory_times is None:
            return

        X, Y, Z = self._trajectory_pos
        times = self._trajectory_times

        i0 = int(np.searchsorted(times, start_time))
        i1 = int(np.searchsorted(times, end_time))
        if i1 <= i0:
            return

        # Normalize color to 0-255 int tuple regardless of input format
        c = np.asarray(color, dtype=np.float64).ravel()[:3]
        if c.max() <= 1.0:
            c = c * 255
        r8, g8, b8 = int(c[0]), int(c[1]), int(c[2])
        rf, gf, bf = r8 / 255.0, g8 / 255.0, b8 / 255.0

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)

        if is_gl:
            for item in list(self.space_widget.items):
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False):
                    self.space_widget.removeItem(item)

            z_arr = Z if Z is not None else np.zeros_like(X)
            xyz = np.column_stack([X, Y, z_arr]).astype(np.float32)
            bg = gl.GLLinePlotItem(pos=xyz, color=(0.7, 0.7, 0.7, 0.5), width=2, antialias=True)
            bg._is_trajectory = True
            self.space_widget.addItem(bg)

            seg = xyz[i0:i1 + 1]
            if len(seg) > 1:
                hl = gl.GLLinePlotItem(pos=seg, color=(rf, gf, bf, 1), width=5, antialias=True)
                hl._is_highlight = True
                self.space_widget.addItem(hl)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False):
                    plot_item.removeItem(item)

            bg = pg.PlotCurveItem(x=X, y=Y, pen=pg.mkPen(color=(180, 180, 180, 128), width=2))
            bg._is_trajectory = True
            plot_item.addItem(bg)

            x_seg, y_seg = X[i0:i1 + 1], Y[i0:i1 + 1]
            if len(x_seg) > 1:
                hl = pg.PlotCurveItem(
                    x=x_seg, y=y_seg,
                    pen=pg.mkPen(color=(r8, g8, b8), width=4),
                )
                hl._is_highlight = True
                plot_item.addItem(hl)

    def update_time_marker(self, time_position: float):
        """Show a red circle at the current time position on the trajectory."""
        if not getattr(self.app_state, 'space_marker_visible', True):
            self._remove_time_marker()
            return
        if not self.space_widget or self._trajectory_pos is None or self._trajectory_times is None:
            return

        times = self._trajectory_times
        idx = int(np.searchsorted(times, time_position, side='right') - 1)
        idx = np.clip(idx, 0, len(times) - 1)

        X, Y, Z = self._trajectory_pos
        x, y = float(X[idx]), float(Y[idx])

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)

        if is_gl:
            z = float(Z[idx]) if Z is not None else 0.0
            pos_arr = np.array([[x, y, z]], dtype=np.float32)
            color_arr = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            if self._time_marker_item is not None:
                self._time_marker_item.setData(pos=pos_arr, color=color_arr)
            else:
                self._time_marker_item = gl.GLScatterPlotItem(
                    pos=pos_arr, color=color_arr, size=20,
                    pxMode=True, glOptions='translucent',
                )
                self.space_widget.addItem(self._time_marker_item)
        else:
            if self._time_marker_item is not None:
                self._time_marker_item.setData([x], [y])
            else:
                self._time_marker_item = pg.ScatterPlotItem(
                    [x], [y],
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(255, 0, 0),
                    size=12,
                    symbol='o',
                )
                self._time_marker_item.setZValue(1000)
                plot_item = self.space_widget.getPlotItem()
                plot_item.addItem(self._time_marker_item)

    def _remove_time_marker(self):
        if self._time_marker_item is not None and self.space_widget is not None:
            is_gl = isinstance(self.space_widget, gl.GLViewWidget)
            if is_gl:
                self.space_widget.removeItem(self._time_marker_item)
            else:
                self.space_widget.getPlotItem().removeItem(self._time_marker_item)
            self._time_marker_item = None
