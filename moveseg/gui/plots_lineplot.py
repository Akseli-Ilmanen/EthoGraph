"""Enhanced line plot inheriting from BasePlot."""

import numpy as np
from typing import Optional
from moveseg.utils.data_utils import sel_valid
from moveseg.plots.plot_qtgraph import plot_ds_variable, clear_plot_items
from .plots_base import BasePlot
import pyqtgraph as pg

class LinePlot(BasePlot):
    """Line plot with shared sync and marker functionality from BasePlot."""

    def __init__(self, napari_viewer, app_state, parent=None):
        super().__init__(app_state, parent)
        self.viewer = napari_viewer

        # Line plot specific setup
        self.setLabel('left', 'Value')

        # Plot items storage
        self.plot_items = []
        self.label_items = []

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the line plot with current data."""
        # Clear existing plot items
        clear_plot_items(self.plot_item, self.plot_items)

        ds_kwargs = self.app_state.get_ds_kwargs()

        color_var = None
        if (hasattr(self.app_state, 'colors_sel') and
            self.app_state.colors_sel != "None"):
            color_var = self.app_state.colors_sel

        self.plot_items = plot_ds_variable(
            self.plot_item,
            self.app_state.ds,
            ds_kwargs,
            self.app_state.features_sel,
            color_variable=color_var
        )

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply y-axis range for line plot."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        """Apply y-axis constraints based on current feature data."""
        if not hasattr(self.app_state, 'features_sel'):
            return

        feature_sel = self.app_state.features_sel
        ds_kwargs = self.app_state.get_ds_kwargs()

        try:
            data, _ = sel_valid(self.app_state.ds[feature_sel], ds_kwargs)

            percentile_ylim = self.app_state.get_with_default("percentile_ylim")
            y_min = np.nanpercentile(data, 100 - percentile_ylim)
            y_max = np.nanpercentile(data, percentile_ylim)
            y_range = y_max - y_min
            y_buffer = y_range * 0.2

            if y_range > 0:
                self.vb.setLimits(
                    yMin=y_min - y_buffer,
                    yMax=y_max + y_buffer,
                    minYRange=y_range * 0.1,
                    maxYRange=y_range + y_buffer
                )
        except (KeyError, AttributeError, ValueError):
            pass

