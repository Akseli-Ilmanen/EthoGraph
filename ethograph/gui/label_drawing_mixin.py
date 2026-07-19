"""Mixin providing label and changepoint drawing methods for plot containers."""

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg

from ethograph.labels.intervals import EVENT_TYPE_POINT

from .app_constants import (
    CP_COLOR_OSC_EVENT,
    DEFAULT_LABEL_OVERLAY_MODES,
    LABEL_OVERLAY_MODE_FULL,
    LABEL_OVERLAY_MODE_NONE,
    CP_COLOR_SPECTROGRAM,
    CP_COLOR_WAVEFORM,
    CP_LINE_WIDTH_MEDIUM,
    CP_LINE_WIDTH_THICK,
    CP_LINE_WIDTH_THIN,
    CP_METHOD_COLORS,
    CP_SCATTER_SIZE,
    CP_SCATTER_Y_POSITION_RATIO,
    CP_ZOOM_MEDIUM_THRESHOLD,
    CP_ZOOM_VERY_OUT_THRESHOLD,
    PREDICTION_FALLBACK_Y_HEIGHT,
    PREDICTION_FALLBACK_Y_TOP,
    PREDICTION_LABELS_HEIGHT_RATIO,
    SPECTROGRAM_FALLBACK_Y_HEIGHT,
    SPECTROGRAM_LABELS_HEIGHT_RATIO,
    Z_INDEX_CHANGEPOINTS,
    Z_INDEX_LABELS,
    Z_INDEX_PREDICTIONS,
)

# Point events render as a vertical line in the label class's color.  Thicker
# than CP lines (so they read as user-coded data, not reference markers) and
# drawn above the state-event rectangles but below the changepoint markers.
_POINT_EVENT_LINE_WIDTH = 4.0
_POINT_EVENT_Z_INDEX = Z_INDEX_CHANGEPOINTS - 1


class LabelDrawingMixin:
    """Mixin that provides label and changepoint drawing on plot widgets.

    Requires the host class to have:
      - app_state (for label_overlay_modes)
      - label_mappings: Dict[int, Dict[str, Any]]
      - audio_cp_items: list
      - osc_event_items: list
      - dataset_cp_items: list
      - spectrogram_plots, audio_trace_plots, heatmap_plots, neo_trace_plots
        (instance lists), ephys_trace_plot
      - current_plot (property or attribute)
    """

    # Fixed-panel attribute -> plot-type key in label_overlay_modes.
    # Dynamic panels are instance lists; any other plot is a line-plot instance.
    _PLOT_TYPE_ATTRS = {
        "ephys_trace_plot": "ephys",
    }

    def set_label_mappings(self, mappings: Dict[int, Dict[str, Any]]):
        self.label_mappings = mappings

    def set_active_label_ids(self, ids: set[int]):
        self._active_label_ids = ids

    def _get_all_plots(self) -> list:
        """Return all plot widgets that exist on this container."""
        candidates = list(getattr(self, "spectrogram_plots", ()) or ())
        candidates += list(getattr(self, "audio_trace_plots", ()) or ())
        candidates += list(getattr(self, "heatmap_plots", ()) or ())
        candidates += list(getattr(self, "neo_trace_plots", ()) or ())
        plot = getattr(self, "ephys_trace_plot", None)
        if plot is not None:
            candidates.append(plot)
        return candidates

    def _plot_type_key(self, plot) -> str:
        if plot in (getattr(self, "spectrogram_plots", ()) or ()):
            return "spectrogram"
        if plot in (getattr(self, "audio_trace_plots", ()) or ()):
            return "audio"
        if plot in (getattr(self, "heatmap_plots", ()) or ()):
            return "heatmap"
        if plot in (getattr(self, "neo_trace_plots", ()) or ()):
            return "neo"
        for attr, type_key in self._PLOT_TYPE_ATTRS.items():
            if plot is getattr(self, attr, None):
                return type_key
        return "lineplot"

    def _label_overlay_mode(self, plot) -> str:
        """Rendering mode ("full" | "bottom" | "none") for this plot's type."""
        type_key = self._plot_type_key(plot)
        modes = getattr(self.app_state, "label_overlay_modes", None) or {}
        return modes.get(type_key, DEFAULT_LABEL_OVERLAY_MODES[type_key])

    def draw_all_labels(self, slots):
        """Render label slots on every plot whose type's overlay mode isn't "none".

        slots: list of dicts ``{"df", "label_ids", "position"}``.
          - ``df``: DataFrame with onset_s, offset_s, labels (and optional event_type).
          - ``label_ids``: filter set; if None, every non-zero label is drawn (used
            for the predictions slot, since it isn't gated by branch membership).
          - ``position``: ``"main"``, ``"top1"`` or ``"top2"``.
        """
        if not self.label_mappings:
            return

        for plot in self._get_all_plots():
            self._clear_labels_on_plot(plot)
            mode = self._label_overlay_mode(plot)
            if mode == LABEL_OVERLAY_MODE_NONE:
                continue
            for slot in slots or []:
                self._draw_intervals_on_plot(
                    plot,
                    slot["df"],
                    label_ids=slot.get("label_ids"),
                    position=slot["position"],
                    mode=mode,
                )

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

    def _draw_intervals_on_plot(self, plot, intervals_df, label_ids=None, position="main", mode=LABEL_OVERLAY_MODE_FULL):
        if not hasattr(plot, "label_items"):
            plot.label_items = []
        if intervals_df is None or intervals_df.empty:
            return
        has_event_type = "event_type" in intervals_df.columns
        for _, row in intervals_df.iterrows():
            labels = int(row["labels"])
            if labels == 0:
                continue
            if label_ids is not None and labels not in label_ids:
                continue
            is_point = has_event_type and row["event_type"] == EVENT_TYPE_POINT
            if is_point:
                self._draw_single_point(plot, row["onset_s"], labels)
            else:
                self._draw_single_label(plot, row["onset_s"], row["offset_s"], labels, position, mode)

    def _draw_single_point(self, plot, time_s, labels):
        """Draw a point event as a thick vertical line in the label's color."""
        if labels not in self.label_mappings:
            return
        color_rgb = tuple(int(c * 255) for c in self.label_mappings[labels]["color"])
        line = pg.InfiniteLine(
            pos=time_s,
            angle=90,
            pen=pg.mkPen(
                color=(*color_rgb, 230),
                width=_POINT_EVENT_LINE_WIDTH,
                style=pg.QtCore.Qt.SolidLine,
            ),
            movable=False,
        )
        line.setZValue(_POINT_EVENT_Z_INDEX)
        plot.plot_item.addItem(line)
        plot.label_items.append(line)

    def _is_inverted_y_plot(self, plot) -> bool:
        return plot in (getattr(self, "heatmap_plots", ()) or ())

    def _draw_single_label(self, plot, start_time, end_time, labels, position="main", mode=LABEL_OVERLAY_MODE_FULL):
        """Draw a single label rectangle.

        position: ``"main"`` -> standard full-plot rectangle, or — when the
        plot type's overlay mode is ``"bottom"`` — a bottom strip (top strip
        on the inverted-Y heatmap). ``"top1"``/``"top2"`` -> stacked thin top
        strips, drawn over the main rectangles. Top2 sits directly under Top1
        so two prediction-like sources can co-exist visibly.
        """
        if labels not in self.label_mappings:
            return
        color_rgb = tuple(int(c * 255) for c in self.label_mappings[labels]["color"])

        is_main = position == "main"

        if is_main and mode == LABEL_OVERLAY_MODE_FULL:
            self._draw_standard_label(plot, start_time, end_time, color_rgb)
            return

        inverted_y = self._is_inverted_y_plot(plot)
        y_lo, y_hi = plot.plot_item.getViewBox().viewRange()[1]
        degenerate = y_hi <= y_lo

        if is_main:
            # Main in "bottom" mode: bottom strip (or top strip when y is inverted)
            height = SPECTROGRAM_FALLBACK_Y_HEIGHT if degenerate else (y_hi - y_lo) * SPECTROGRAM_LABELS_HEIGHT_RATIO
            if inverted_y:
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y0, y1 = y_top - height, y_top
            else:
                y_bottom = 0 if degenerate else y_lo
                y0, y1 = y_bottom, y_bottom + height
            z, alpha = Z_INDEX_LABELS, 220
        else:
            # Top1 / Top2: stacked thin strips at the y_hi side. On heatmaps
            # (inverted_y) the strips visually appear at the bottom of the
            # screen, mirroring how the existing main bar already works there.
            height = PREDICTION_FALLBACK_Y_HEIGHT if degenerate else (y_hi - y_lo) * PREDICTION_LABELS_HEIGHT_RATIO
            slot_idx = 0 if position == "top1" else 1
            if inverted_y:
                y_bottom = 0 if degenerate else y_lo
                y0 = y_bottom + slot_idx * height
                y1 = y0 + height
            else:
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y1 = y_top - slot_idx * height
                y0 = y1 - height
            # Top2 a touch dimmer than Top1 so they're distinguishable when
            # they show overlapping classes.
            alpha = 200 if slot_idx == 0 else 170
            z = Z_INDEX_PREDICTIONS + slot_idx

        self._draw_label_region(plot, start_time, end_time, color_rgb, y0, y1, z, alpha)

    def _draw_standard_label(self, plot, start_time, end_time, color_rgb):
        rect = pg.LinearRegionItem(
            values=(start_time, end_time),
            orientation="vertical",
            brush=(*color_rgb, 180),
            pen=pg.mkPen(None),
            movable=False,
        )
        sep_pen = pg.mkPen(color=(255, 255, 255, 180), width=1)
        for line in rect.lines:
            line.setPen(sep_pen)
        rect.setZValue(Z_INDEX_LABELS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    def _draw_label_region(self, plot, start_time, end_time, color_rgb, y0, y1, z_value, alpha=220):
        sep_pen = pg.mkPen(color=(255, 255, 255, 180), width=0)
        rect = pg.PlotDataItem(
            [start_time, end_time, end_time, start_time, start_time],
            [y0, y0, y1, y1, y0],
            fillLevel=y0,
            brush=(*color_rgb, alpha),
            pen=sep_pen,
        )
        rect.setZValue(z_value)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)

    # --- Audio changepoints ---

    def draw_audio_changepoints(self, onsets: np.ndarray, offsets: np.ndarray):
        self.clear_audio_changepoints()
        audio_traces = list(getattr(self, "audio_trace_plots", ()) or ())
        plots_to_draw = list(getattr(self, "spectrogram_plots", ()) or ()) + audio_traces
        line_style = self._get_changepoint_line_style()
        for plot in plots_to_draw:
            color = CP_COLOR_WAVEFORM if plot in audio_traces else CP_COLOR_SPECTROGRAM
            for onset_t in onsets:
                line = pg.InfiniteLine(
                    pos=onset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, "onset"))
            for offset_t in offsets:
                line = pg.InfiniteLine(
                    pos=offset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
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
        audio_traces = getattr(self, "audio_trace_plots", ()) or ()
        for item in self.audio_cp_items:
            plot, line, _ = item
            color = CP_COLOR_WAVEFORM if plot in audio_traces else CP_COLOR_SPECTROGRAM
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
        line_plots = getattr(self, "line_plots", None)
        if not line_plots or not self.is_lineplot():
            return
        line_plot = self.get_current_plot()
        y_range = line_plot.plot_item.getViewBox().viewRange()[1]
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * CP_SCATTER_Y_POSITION_RATIO
        for method_name, indices in cp_by_method.items():
            if len(indices) == 0:
                continue
            times = time_array[indices]
            y_values = np.full_like(times, y_pos)
            color = CP_METHOD_COLORS.get(method_name, CP_METHOD_COLORS["default"])
            scatter = pg.ScatterPlotItem(
                x=times,
                y=y_values,
                size=CP_SCATTER_SIZE,
                pen=pg.mkPen(color=color, width=1),
                brush=pg.mkBrush(color=color),
                symbol="o",
                name=method_name,
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

    # --- Oscillatory events ---

    def draw_oscillatory_events(self, onsets: np.ndarray, offsets: np.ndarray):
        self.clear_oscillatory_events()
        all_plots = self._get_all_plots()
        line_style = self._get_changepoint_line_style()
        for plot in all_plots:
            for onset_t in onsets:
                line = pg.InfiniteLine(
                    pos=onset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=CP_COLOR_OSC_EVENT,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.osc_event_items.append((plot, line, "onset"))
            for offset_t in offsets:
                line = pg.InfiniteLine(
                    pos=offset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=CP_COLOR_OSC_EVENT,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.osc_event_items.append((plot, line, "offset"))

    def update_oscillatory_event_styles(self):
        if not self.osc_event_items:
            return
        line_style = self._get_changepoint_line_style()
        for plot, line, _ in self.osc_event_items:
            line.setPen(
                pg.mkPen(
                    color=CP_COLOR_OSC_EVENT,
                    width=line_style["width"],
                    style=line_style["style"],
                )
            )

    def clear_oscillatory_events(self):
        for item in self.osc_event_items:
            plot, line = item[0], item[1]
            try:
                plot.plot_item.removeItem(line)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self.osc_event_items.clear()
