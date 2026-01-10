"""Space plot widget for displaying box topview and centroid trajectory plots."""

import numpy as np
from pyparsing import line
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy.QtWidgets import QWidget, QVBoxLayout
from typing import Optional
import xarray as xr
from moveseg.utils.data_utils import sel_valid
from movement.plots import plot_centroid_trajectory
from movement.filtering import rolling_filter
from moveseg.plots.plot_qtgraph import MultiColoredLineItem
from moveseg.features.preprocessing import interpolate_nans



def space_plot_pyqt(
    space_widget,
    ds: xr.Dataset,
    color_variable: Optional[str] = None,
    view_3d: bool = False,
    **ds_kwargs
) -> None:
    """Plot trajectory from top view (2D) or 3D view using PyQtGraph.

    Args:
        x_limits: Tuple of (min, max) for X axis
        y_limits: Tuple of (min, max) for Y axis
        z_limits: Tuple of (min, max) for Z axis (3D only)
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
            
        space_widget.addItem(line)

        box_line = pg.PlotCurveItem(
            x=box_xy_base[:, 0],
            y=box_xy_base[:, 1],
            pen=pg.mkPen(color='k', width=2)
        )
        
        
        space_widget.addItem(box_line)




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

        # Use space_plot_pyqt for both 2D and 3D
        space_plot_pyqt(self.space_widget, self.app_state.ds,
                        color_variable, view_3d,
                        **ds_kwargs)


    def highlight_positions(self, start_frame: int, end_frame: int):
        """Highlight positions in the plot based on current frame."""
        if not self.space_widget:
            return

        if self.is_3d:
       
            for item in list(self.space_widget.items):
                if hasattr(item, '_is_highlight') and item._is_highlight:
                    self.space_widget.removeItem(item)


            spaces = ['x', 'y', 'z']
            if ('space' in self.app_state.ds.coords and
                all(s in self.app_state.ds.coords['space'] for s in spaces)):

                highlighted_points = self.app_state.ds.sel(space=spaces, **self.ds_kwargs).isel(time=slice(start_frame, end_frame + 1)).position
                
                x = highlighted_points.sel(space='x').values
                y = highlighted_points.sel(space='y').values
                z = highlighted_points.sel(space='z').values

                pos = np.column_stack([x, y, z])
                scatter = gl.GLScatterPlotItem(
                    pos=pos, size=1, color=(0, 0, 0, 1), pxMode=False
                )
                scatter.setGLOptions('translucent')
                scatter._is_highlight = True
                self.space_widget.addItem(scatter)
        else:
   
            plot_item = self.space_widget.getPlotItem()
            items_to_remove = []
            for item in plot_item.items:
                if hasattr(item, '_is_highlight') and item._is_highlight:
                    items_to_remove.append(item)

            for item in items_to_remove:
                plot_item.removeItem(item)

            # Add 2D highlights
            if ('space' in self.app_state.ds.coords and
                'x' in self.app_state.ds.coords['space'] and
                'y' in self.app_state.ds.coords['space']):

                highlighted_points = self.app_state.ds.sel(space=["x", "y"], **self.ds_kwargs).isel(time=slice(start_frame, end_frame + 1)).position

       
                x = highlighted_points.sel(space='x').values
                y = highlighted_points.sel(space='y').values

                scatter = pg.ScatterPlotItem(
                    x=x, y=y,
                    pen=pg.mkPen(color='k', width=2),
                    brush=pg.mkBrush(color='k'),
                    symbol='o',
                    size=5
                )
                scatter._is_highlight = True
                plot_item.addItem(scatter)