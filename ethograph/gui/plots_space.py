"""Space plot widget for displaying box topview and centroid trajectory plots."""

from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import xarray as xr
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ethograph.features.preprocessing import interpolate_nans
from ethograph.plots.lineplot_qtgraph import MultiColoredLineItem
from ethograph.utils.data_utils import sel_valid


def space_plot_pyqt(
    space_widget,
    ds: xr.Dataset,
    color_variable: Optional[str] = None,
    view_3d: bool = False,
    **ds_kwargs
) -> tuple:
    """Plot trajectory from top view (2D) or 3D view using PyQtGraph.

    Returns:
        Tuple of (X, Y, Z) position arrays. Z is None for 2D plots.
    """

    space_widget.clear()

    spaces = ['x', 'y', 'z'] if view_3d else ['x', 'y']

    pos, _ = sel_valid(ds.sel(space=spaces).position, ds_kwargs)
    pos = interpolate_nans(pos)
    
    
    X, Y = pos[:, 0], pos[:, 1]
    Z = pos[:, 2] if view_3d else None


    box_xy_base = np.array([
        [-7.00,  0.00],
        [-7.00,  9.80],
        [ 6.80,  9.80],
        [ 6.80,  0.00],
        [-7.00,  0.00] 
    ])
    z_bot, z_top = 0.65, 2.75


    color_data = None
    if color_variable and color_variable in ds.data_vars:
        color_data, _ = sel_valid(ds[color_variable], ds_kwargs)
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

        x_min, x_max = box_xy_base[:, 0].min(), box_xy_base[:, 0].max()
        y_min, y_max = box_xy_base[:, 1].min(), box_xy_base[:, 1].max()
        

        vertices = np.array([
            [x_min, y_min, z_bot], [x_max, y_min, z_bot],
            [x_max, y_max, z_bot], [x_min, y_max, z_bot],
            [x_min, y_min, z_top], [x_max, y_min, z_top],
            [x_max, y_max, z_top], [x_min, y_max, z_top]
        ])
        

        edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom
            [4,5],[5,6],[6,7],[7,4],  # top
            [0,4],[1,5],[2,6],[3,7]   # vertical
        ]
        
   
        segments = []
        for v1, v2 in edges:
            segments.extend([vertices[v1], vertices[v2], [np.nan, np.nan, np.nan]])
        
        # Create wireframe
        box_wireframe = gl.GLLinePlotItem(
            pos=np.array(segments[:-1]),  # Remove trailing NaN
            color=(0, 0, 0, 1),
            width=2,
            antialias=True,
        )
        space_widget.addItem(box_wireframe)
        

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_bot + z_top) / 2
        
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

        box_line = pg.PlotCurveItem(
            x=box_xy_base[:, 0],
            y=box_xy_base[:, 1],
            pen=pg.mkPen(color='k', width=2)
        )
        space_widget.addItem(box_line)

    return X, Y, Z


class SpacePlot(QWidget):
    """Widget for displaying spatial plots in napari dock area."""

    def __init__(self, viewer, app_state):
        super().__init__()
        self.viewer = viewer
        self.app_state = app_state
        self.dock_widget = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Plot widgets - will be set based on view type
        self.space_widget = None  # Can be either PlotWidget (2D) or GLViewWidget (3D)
        self.is_3d = False
        self.ds_kwargs = {}
        self._trajectory_pos = None
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

            # Set reasonable minimum size but let napari handle max
            self.setMinimumHeight(300)
            self.setMinimumWidth(300)
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
        """Update the plot based on the selected type and parameters."""
        if not self.app_state.ds:
            return

        if not hasattr(self.app_state.ds, 'position') or 'x' not in self.app_state.ds.coords["space"] or 'y' not in self.app_state.ds.coords["space"]:
            raise ValueError("Dataset must have 'position' variable with 'x' and 'y' coordinates for space plots")

        # Remove existing plot widget if it exists
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
            
        
        self.layout.addWidget(self.space_widget)

        ds_kwargs = {}
        if individual and individual != "None":
            ds_kwargs["individuals"] = individual
        if keypoints and keypoints != "None":
            ds_kwargs["keypoints"] = keypoints

        self.ds_kwargs = ds_kwargs

        X, Y, Z = space_plot_pyqt(
            self.space_widget, self.app_state.ds, color_variable, view_3d, **ds_kwargs
        )
        self._trajectory_pos = (X, Y, Z)


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