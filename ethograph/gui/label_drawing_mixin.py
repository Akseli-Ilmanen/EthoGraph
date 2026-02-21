"""Mixin providing label and changepoint drawing methods for plot containers."""

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg

from .app_constants import (
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


class LabelDrawingMixin:
    """Mixin that provides label and changepoint drawing on plot widgets.

    Requires the host class to have:
      - label_mappings: Dict[int, Dict[str, Any]]
      - audio_overlay_type: str | None
      - audio_cp_items: list
      - dataset_cp_items: list
      - spectrogram_plot, line_plot, audio_trace_plot, heatmap_plot, ephys_trace_plot
      - current_plot (property or attribute)
    """

    def set_label_mappings(self, mappings: Dict[int, Dict[str, Any]]):
        self.label_mappings = mappings

    def _get_all_plots(self) -> list:
        """Return all plot widgets that exist on this container."""
        candidates = []
        for attr in ("line_plot", "spectrogram_plot", "audio_trace_plot",
                      "heatmap_plot", "ephys_trace_plot"):
            plot = getattr(self, attr, None)
            if plot is not None:
                candidates.append(plot)
        return candidates

    def draw_all_labels(self, intervals_df, predictions_df=None, show_predictions=False):
        if intervals_df is None or not self.label_mappings:
            return

        for plot in self._get_all_plots():
            self._clear_labels_on_plot(plot)
            self._draw_intervals_on_plot(plot, intervals_df, is_main=True)
            if predictions_df is not None and show_predictions:
                self._draw_intervals_on_plot(plot, predictions_df, is_main=False)

    def _clear_labels_on_plot(self, plot):
        if not hasattr(plot, "label_items"):
            plot.label_items = []
            return
        for item in plot.label_items:
            try:
                plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass
        plot.label_items.clear()

    def _draw_intervals_on_plot(self, plot, intervals_df, is_main=True):
        if not hasattr(plot, "label_items"):
            plot.label_items = []
        if intervals_df is None or intervals_df.empty:
            return
        for _, row in intervals_df.iterrows():
            labels = int(row["labels"])
            if labels == 0:
                continue
            self._draw_single_label(plot, row["onset_s"], row["offset_s"], labels, is_main)

    def _is_bottom_strip_plot(self, plot) -> bool:
        """Whether this plot should use bottom-strip style labels."""
        return (
            plot is getattr(self, "spectrogram_plot", None)
            or plot is getattr(self, "heatmap_plot", None)
            or plot is getattr(self, "ephys_trace_plot", None)
            or (
                plot is getattr(self, "line_plot", None)
                and getattr(self, "audio_overlay_type", None) == "spectrogram"
            )
        )

    def _is_inverted_y_plot(self, plot) -> bool:
        return plot is getattr(self, "heatmap_plot", None)

    def _draw_single_label(self, plot, start_time, end_time, labels, is_main=True):
        if labels not in self.label_mappings:
            return
        color = self.label_mappings[labels]["color"]
        color_rgb = tuple(int(c * 255) for c in color)

        use_bottom_strip = self._is_bottom_strip_plot(plot)
        inverted_y = self._is_inverted_y_plot(plot)

        if is_main:
            if use_bottom_strip:
                if inverted_y:
                    self._draw_label_strip_top(plot, start_time, end_time, color_rgb)
                else:
                    self._draw_spectrogram_style_label(plot, start_time, end_time, color_rgb)
            else:
                self._draw_standard_label(plot, start_time, end_time, color_rgb)
        else:
            if inverted_y:
                self._draw_label_strip_bottom(plot, start_time, end_time, color_rgb)
            else:
                self._draw_prediction_label(plot, start_time, end_time, color_rgb)

    def _draw_standard_label(self, plot, start_time, end_time, color_rgb):
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
        self._draw_label_strip_bottom(plot, start_time, end_time, color_rgb)

    def _draw_prediction_label(self, plot, start_time, end_time, color_rgb):
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_top = y_range[1]
        y_height = (y_range[1] - y_range[0]) * PREDICTION_LABELS_HEIGHT_RATIO
        if y_top <= y_range[0]:
            y_top = PREDICTION_FALLBACK_Y_TOP
            y_height = PREDICTION_FALLBACK_Y_HEIGHT
        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_top, y_top, y_top - y_height, y_top - y_height, y_top]
        rect = pg.PlotDataItem(
            x_coords, y_coords, fillLevel=y_top - y_height,
            brush=(*color_rgb, 200), pen=None,
        )
        rect.setZValue(Z_INDEX_PREDICTIONS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_label_strip_bottom(self, plot, start_time, end_time, color_rgb):
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_bottom = y_range[0]
        y_height = (y_range[1] - y_range[0]) * SPECTROGRAM_LABELS_HEIGHT_RATIO
        if y_range[1] <= y_bottom:
            y_bottom = 0
            y_height = SPECTROGRAM_FALLBACK_Y_HEIGHT
        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_bottom, y_bottom, y_bottom + y_height, y_bottom + y_height, y_bottom]
        rect = pg.PlotDataItem(
            x_coords, y_coords, fillLevel=y_bottom,
            brush=(*color_rgb, 220), pen=None,
        )
        rect.setZValue(Z_INDEX_PREDICTIONS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_label_strip_top(self, plot, start_time, end_time, color_rgb):
        y_range = plot.plot_item.getViewBox().viewRange()[1]
        y_top = y_range[1]
        y_height = (y_range[1] - y_range[0]) * SPECTROGRAM_LABELS_HEIGHT_RATIO
        if y_top <= y_range[0]:
            y_top = PREDICTION_FALLBACK_Y_TOP
            y_height = SPECTROGRAM_FALLBACK_Y_HEIGHT
        x_coords = [start_time, end_time, end_time, start_time, start_time]
        y_coords = [y_top, y_top, y_top - y_height, y_top - y_height, y_top]
        rect = pg.PlotDataItem(
            x_coords, y_coords, fillLevel=y_top,
            brush=(*color_rgb, 220), pen=None,
        )
        rect.setZValue(Z_INDEX_LABELS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def redraw_current_plot_labels(self, intervals_df, predictions_df=None, show_predictions=False):
        if intervals_df is None or not self.label_mappings:
            return
        plot = self.current_plot
        self._clear_labels_on_plot(plot)
        self._draw_intervals_on_plot(plot, intervals_df, is_main=True)
        if predictions_df is not None and show_predictions:
            self._draw_intervals_on_plot(plot, predictions_df, is_main=False)

    # --- Audio changepoints ---

    def draw_audio_changepoints(self, onsets: np.ndarray, offsets: np.ndarray):
        self.clear_audio_changepoints()
        plots_to_draw = [
            getattr(self, "spectrogram_plot", None),
            getattr(self, "audio_trace_plot", None),
        ]
        line_style = self._get_changepoint_line_style()
        for plot in plots_to_draw:
            if plot is None:
                continue
            color = CP_COLOR_WAVEFORM if plot is getattr(self, "audio_trace_plot", None) else CP_COLOR_SPECTROGRAM
            for onset_t in onsets:
                line = pg.InfiniteLine(
                    pos=onset_t, angle=90,
                    pen=pg.mkPen(color=color, width=line_style["width"], style=line_style["style"]),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, "onset"))
            for offset_t in offsets:
                line = pg.InfiniteLine(
                    pos=offset_t, angle=90,
                    pen=pg.mkPen(color=color, width=line_style["width"], style=line_style["style"]),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, "offset"))

    def _get_changepoint_line_style(self):
        try:
            xmin, xmax = self.current_plot.get_current_xlim()
            visible_range = xmax - xmin
            if visible_range > CP_ZOOM_VERY_OUT_THRESHOLD:
                return {"style": pg.QtCore.Qt.DotLine, "width": CP_LINE_WIDTH_THIN}
            elif visible_range > CP_ZOOM_MEDIUM_THRESHOLD:
                return {"style": pg.QtCore.Qt.DashLine, "width": CP_LINE_WIDTH_MEDIUM}
            else:
                return {"style": pg.QtCore.Qt.SolidLine, "width": CP_LINE_WIDTH_THICK}
        except (AttributeError, TypeError, ValueError):
            return {"style": pg.QtCore.Qt.DashLine, "width": CP_LINE_WIDTH_MEDIUM}

    def update_audio_changepoint_styles(self):
        if not self.audio_cp_items:
            return
        line_style = self._get_changepoint_line_style()
        for item in self.audio_cp_items:
            plot, line, _ = item
            color = CP_COLOR_WAVEFORM if plot is getattr(self, "audio_trace_plot", None) else CP_COLOR_SPECTROGRAM
            line.setPen(pg.mkPen(color=color, width=line_style["width"], style=line_style["style"]))

    def clear_audio_changepoints(self):
        for item in self.audio_cp_items:
            plot, line = item[0], item[1]
            try:
                plot.plot_item.removeItem(line)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self.audio_cp_items.clear()

    def draw_dataset_changepoints(self, time_array: np.ndarray, cp_by_method: dict):
        self.clear_dataset_changepoints()
        line_plot = getattr(self, "line_plot", None)
        if line_plot is None or not self.is_lineplot():
            return
        y_range = line_plot.plot_item.getViewBox().viewRange()[1]
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * CP_SCATTER_Y_POSITION_RATIO
        for method_name, indices in cp_by_method.items():
            if len(indices) == 0:
                continue
            times = time_array[indices]
            y_values = np.full_like(times, y_pos)
            color = CP_METHOD_COLORS.get(method_name, CP_METHOD_COLORS["default"])
            scatter = pg.ScatterPlotItem(
                x=times, y=y_values, size=CP_SCATTER_SIZE,
                pen=pg.mkPen(color=color, width=1),
                brush=pg.mkBrush(color=color),
                symbol="o", name=method_name,
            )
            scatter.setZValue(Z_INDEX_CHANGEPOINTS)
            line_plot.plot_item.addItem(scatter)
            self.dataset_cp_items.append(scatter)

    def clear_dataset_changepoints(self):
        line_plot = getattr(self, "line_plot", None)
        for item in self.dataset_cp_items:
            try:
                if line_plot is not None:
                    line_plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self.dataset_cp_items.clear()
