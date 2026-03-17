"""Space plot widget for displaying box topview and centroid trajectory plots."""

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import xarray as xr
import yaml
from qtpy.QtWidgets import QVBoxLayout, QWidget, QSizePolicy

from ethograph.features.preprocessing import interpolate_nans
from ethograph.gui.plots_lineplot import MultiColoredLineItem
import ethograph as eto


def load_arena_config(config_path: Path) -> Optional[dict]:
    """Load arena geometry from YAML. Returns None if file is absent."""
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def _add_3d_reference(widget, x_min, x_max, y_min, y_max, z_floor):
    """Add semi-transparent floor grid and XYZ axis indicator for spatial orientation."""
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    sx, sy = x_max - x_min, y_max - y_min
    spacing = max(sx, sy) / 10

    grid = gl.GLGridItem(glOptions='translucent')
    grid.setSize(sx, sy)
    grid.setSpacing(spacing, spacing, spacing)
    grid.translate(cx, cy, z_floor)
    widget.addItem(grid)

    axis_len = min(sx, sy) * 0.18
    axis = gl.GLAxisItem()
    axis.setSize(axis_len, axis_len, axis_len)
    axis.translate(x_min, y_min, z_floor)
    widget.addItem(axis)


def _add_pca_axes(widget, center, extent):
    """Add XYZ axis indicator centered on PCA data for orientation."""
    axis_len = extent * 0.15
    axis = gl.GLAxisItem()
    axis.setSize(axis_len, axis_len, axis_len)
    axis.translate(center[0] - axis_len / 2, center[1] - axis_len / 2, center[2] - axis_len / 2)
    widget.addItem(axis)


def space_plot_pyqt(
    space_widget,
    ds: xr.Dataset,
    color_variable: Optional[str] = None,
    view_3d: bool = False,
    arena: Optional[dict] = None,
    **ds_kwargs
) -> tuple:
    """Plot trajectory from top view (2D) or 3D view using PyQtGraph.

    Returns:
        Tuple of (X, Y, Z) position arrays. Z is None for 2D plots.
    """

    space_widget.clear()

    spaces = ['x', 'y', 'z'] if view_3d else ['x', 'y']

    pos, _ = eto.sel_valid(ds.sel(space=spaces).position, ds_kwargs)
    pos = interpolate_nans(pos)

    X, Y = pos[:, 0], pos[:, 1]
    Z = pos[:, 2] if view_3d else None

    box_xy_base = np.array(arena["xy_polygon"]) if arena else None
    z_bot = arena.get("z_bot", 0.0) if arena else None
    z_top = arena.get("z_top", 1.0) if arena else None

    color_data = None
    if color_variable and color_variable in ds.data_vars:
        color_data, _ = eto.sel_valid(ds[color_variable], ds_kwargs)
        if color_data.max() > 1.0:
            color_data = color_data / 255.0
        color_data = np.concatenate([color_data, np.ones((color_data.shape[0], 1))], axis=1)

    if view_3d:
        XYZ = np.column_stack([X, Y, Z]).astype(np.float32)

        if color_data is not None:
            line = gl.GLLinePlotItem(pos=XYZ, color=color_data, width=3, antialias=True)
        else:
            line = gl.GLLinePlotItem(pos=XYZ, color=(0, 0, 1, 1), width=3, antialias=True)
        line._is_trajectory = True
        space_widget.addItem(line)

        if box_xy_base is not None:
            x_min, x_max = box_xy_base[:, 0].min(), box_xy_base[:, 0].max()
            y_min, y_max = box_xy_base[:, 1].min(), box_xy_base[:, 1].max()

            vertices = np.array([
                [x_min, y_min, z_bot], [x_max, y_min, z_bot],
                [x_max, y_max, z_bot], [x_min, y_max, z_bot],
                [x_min, y_min, z_top], [x_max, y_min, z_top],
                [x_max, y_max, z_top], [x_min, y_max, z_top]
            ])
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ]
            segments = []
            for v1, v2 in edges:
                segments.extend([vertices[v1], vertices[v2], [np.nan, np.nan, np.nan]])
            box_wireframe = gl.GLLinePlotItem(
                pos=np.array(segments[:-1]),
                color=(0, 0, 0, 1),
                width=2,
                antialias=True,
            )
            space_widget.addItem(box_wireframe)
            _add_3d_reference(space_widget, x_min, x_max, y_min, y_max, z_bot)

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_bot + z_top) / 2
        else:
            center_x, center_y, center_z = X.mean(), Y.mean(), Z.mean() if Z is not None else 0.0

        space_widget.setCameraPosition(
            pos=pg.Vector(center_x, center_y, center_z),
            distance=25,
            elevation=30,
            azimuth=200
        )

    else:
        if color_data is not None:
            line = MultiColoredLineItem(x=X, y=Y, colors=color_data, width=3)
        else:
            line = pg.PlotCurveItem(
                x=X, y=Y,
                pen=pg.mkPen(color='b', width=3)
            )
        line._is_trajectory = True
        space_widget.addItem(line)

        if box_xy_base is not None:
            box_line = pg.PlotCurveItem(
                x=box_xy_base[:, 0],
                y=box_xy_base[:, 1],
                pen=pg.mkPen(color='k', width=2)
            )
            space_widget.addItem(box_line)

    return X, Y, Z


def _time_gradient_colors(n: int) -> np.ndarray:
    """Blue-to-red gradient normalized over n points. Returns (n, 4) RGBA float array."""
    t = np.linspace(0.0, 1.0, n)
    colors = np.zeros((n, 4), dtype=np.float32)
    colors[:, 0] = t        # R increases
    colors[:, 2] = 1.0 - t  # B decreases
    colors[:, 3] = 1.0      # A
    return colors


def pca_plot_pyqt(
    space_widget,
    pca_da: xr.DataArray,
    view_3d: bool = False,
) -> tuple:
    """Plot PCA trajectory with blue-to-red time gradient.

    Returns (PC1, PC2, PC3_or_None) arrays for highlight support.
    """
    space_widget.clear()

    pc1 = pca_da.sel(pc="PC1").values
    pc2 = pca_da.sel(pc="PC2").values
    pc3 = pca_da.sel(pc="PC3").values if view_3d else None

    n = len(pc1)
    colors = _time_gradient_colors(n)

    if view_3d:
        xyz = np.column_stack([pc1, pc2, pc3]).astype(np.float32)
        line = gl.GLLinePlotItem(pos=xyz, color=colors, width=3, antialias=True)
        line._is_trajectory = True
        space_widget.addItem(line)

        center = xyz.mean(axis=0)
        extent = max(xyz.ptp(axis=0)) * 1.5
        _add_pca_axes(space_widget, center, extent)
        space_widget.setCameraPosition(
            pos=pg.Vector(*center),
            distance=extent,
            elevation=30,
            azimuth=200,
        )
    else:
        line = MultiColoredLineItem(x=pc1, y=pc2, colors=colors, width=3)
        line._is_trajectory = True
        space_widget.addItem(line)

        plot_item = space_widget.getPlotItem()
        plot_item.setLabel('bottom', 'PC1')
        plot_item.setLabel('left', 'PC2')

    return pc1, pc2, pc3


class SpacePlot(QWidget):
    """Widget for displaying spatial plots in napari dock area."""

    def __init__(self, viewer, app_state):
        super().__init__()
        self.viewer = viewer
        self.app_state = app_state
        self.dock_widget = None

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.setLayout(self.layout)

        self.space_widget = None
        self.is_3d = False
        self.is_pca = False
        self.ds_kwargs = {}
        self._trajectory_pos = None
        self._pca_times = None
        self.hide()





    def show(self):
        """Show the space plot by replacing the layer controls area."""

        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        if not self.dock_widget:
            # Add space plot at the left side
            self.dock_widget = self.viewer.window.add_dock_widget(
                self, area="left", name="Space Plot"
            )

            # Set the dock widget to take up 20% of the window width
            main_window = self.viewer.window._qt_window
            total_width = main_window.width()
            desired_width = int(total_width * 0.2)

            # Keep this dock highly shrinkable so it does not inflate the app min size.
            self.setMinimumSize(120, 120)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.dock_widget.resize(desired_width, self.dock_widget.height())
        else:
            # Dock widget exists but might be hidden - make sure it's visible
            self.dock_widget.setVisible(True)

        super().show()

    def hide(self):
        """Hide the space plot dock widget and show layer controls."""
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(True)
        if self.dock_widget:
            self.dock_widget.setVisible(False)

        super().hide()

    def update_plot(self, individual: str = None, keypoints: str = None, color_variable: str = None, view_3d: bool = False):
        if not self.app_state.ds:
            return

        if not hasattr(self.app_state.ds, 'position') or 'x' not in self.app_state.ds.coords["space"] or 'y' not in self.app_state.ds.coords["space"]:
            raise ValueError("Dataset must have 'position' variable with 'x' and 'y' coordinates for space plots")

        if self.space_widget:
            self.layout.removeWidget(self.space_widget)
            self.space_widget.deleteLater()

        if view_3d:
            self.space_widget = gl.GLViewWidget()
            self.space_widget.setBackgroundColor('w')
            self.is_3d = True
        else:
            self.space_widget = pg.PlotWidget()
            self.space_widget.setBackground('w')
            self.is_3d = False

        self.is_pca = False
        self._pca_times = None
        self.layout.addWidget(self.space_widget)

        ds_kwargs = {}
        if individual and individual != "None":
            ds_kwargs["individuals"] = individual
        if keypoints and keypoints != "None":
            ds_kwargs["keypoints"] = keypoints

        self.ds_kwargs = ds_kwargs

        arena = load_arena_config(eto.get_project_root() / "configs" / "arena.yaml")
        X, Y, Z = space_plot_pyqt(
            self.space_widget, self.app_state.ds, color_variable, view_3d, arena=arena, **ds_kwargs
        )
        self._trajectory_pos = (X, Y, Z)


    def update_pca_plot(self, view_3d: bool = False):
        ds = self.app_state.ds
        if ds is None or "pca" not in ds.data_vars:
            return

        if self.space_widget:
            self.layout.removeWidget(self.space_widget)
            self.space_widget.deleteLater()

        if view_3d:
            self.space_widget = gl.GLViewWidget()
            self.space_widget.setBackgroundColor('w')
            self.is_3d = True
        else:
            self.space_widget = pg.PlotWidget()
            self.space_widget.setBackground('w')
            self.is_3d = False

        self.is_pca = True
        self.layout.addWidget(self.space_widget)

        pca_da = ds["pca"]
        pc1, pc2, pc3 = pca_plot_pyqt(self.space_widget, pca_da, view_3d)
        self._trajectory_pos = (pc1, pc2, pc3)
        self._pca_times = pca_da.coords["time_fr"].values

    def highlight_pca(self, start_time: float, end_time: float, color: tuple):
        """Highlight a time segment of the PCA trajectory."""
        if not self.space_widget or self._trajectory_pos is None or self._pca_times is None:
            return

        pc1, pc2, pc3 = self._trajectory_pos
        times = self._pca_times

        i0 = int(np.searchsorted(times, start_time))
        i1 = int(np.searchsorted(times, end_time))
        if i1 <= i0:
            return

        gray = (0.7, 0.7, 0.7, 0.5)
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0

        if self.is_3d:
            for item in list(self.space_widget.items):
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False):
                    self.space_widget.removeItem(item)

            xyz = np.column_stack([pc1, pc2, pc3]).astype(np.float32)
            bg_line = gl.GLLinePlotItem(pos=xyz, color=gray, width=2, antialias=True)
            bg_line._is_trajectory = True
            self.space_widget.addItem(bg_line)

            seg = xyz[i0:i1 + 1]
            if len(seg) > 1:
                hl = gl.GLLinePlotItem(pos=seg, color=(r, g, b, 1), width=5, antialias=True)
                hl._is_highlight = True
                self.space_widget.addItem(hl)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False):
                    plot_item.removeItem(item)

            bg = pg.PlotCurveItem(x=pc1, y=pc2, pen=pg.mkPen(color=(180, 180, 180, 128), width=2))
            bg._is_trajectory = True
            plot_item.addItem(bg)

            x_seg, y_seg = pc1[i0:i1 + 1], pc2[i0:i1 + 1]
            if len(x_seg) > 1:
                hl = pg.PlotCurveItem(
                    x=x_seg, y=y_seg,
                    pen=pg.mkPen(color=(int(color[0]), int(color[1]), int(color[2])), width=4),
                )
                hl._is_highlight = True
                plot_item.addItem(hl)

    def highlight_positions(self, start_frame: int, end_frame: int):
        """Highlight positions: full trajectory in green, selected portion in orange."""
        if not self.space_widget or self._trajectory_pos is None:
            return

        X, Y, Z = self._trajectory_pos

        if self.is_3d:
            for item in list(self.space_widget.items):
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False):
                    self.space_widget.removeItem(item)

            full_pos = np.column_stack([X, Y, Z]).astype(np.float32)
            green_line = gl.GLLinePlotItem(
                pos=full_pos, color=(0.2, 0.8, 0.2, 1), width=3, antialias=True
            )
            green_line._is_trajectory = True
            self.space_widget.addItem(green_line)

            highlight_pos = full_pos[start_frame:end_frame + 1]
            if len(highlight_pos) > 1:
                orange_line = gl.GLLinePlotItem(
                    pos=highlight_pos, color=(1, 0.4, 0, 1), width=5, antialias=True
                )
                orange_line._is_highlight = True
                self.space_widget.addItem(orange_line)
        else:
            plot_item = self.space_widget.getPlotItem()
            items_to_remove = [
                item for item in plot_item.items
                if getattr(item, '_is_trajectory', False) or getattr(item, '_is_highlight', False)
            ]
            for item in items_to_remove:
                plot_item.removeItem(item)

            green_line = pg.PlotCurveItem(
                x=X, y=Y, pen=pg.mkPen(color=(50, 200, 50), width=3)
            )
            green_line._is_trajectory = True
            plot_item.addItem(green_line)

            x_highlight = X[start_frame:end_frame + 1]
            y_highlight = Y[start_frame:end_frame + 1]
            if len(x_highlight) > 1:
                orange_line = pg.PlotCurveItem(
                    x=x_highlight, y=y_highlight,
                    pen=pg.mkPen(color=(255, 102, 0), width=4)
                )
                orange_line._is_highlight = True
                plot_item.addItem(orange_line)