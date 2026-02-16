"""Combined plot settings widget with LinePlot / Spectrogram / HeatMap tabs."""

from __future__ import annotations

from typing import Optional

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .app_state import AppStateSpec

HEATMAP_COLORMAPS = [
    "RdBu_r",
    "viridis",
    "inferno",
    "coolwarm",
    "plasma",
    "magma",
    "cividis",
]

_NORM_DISPLAY_TO_KEY = {
    "Per-channel z-normalization": "per_channel",
    "Global z-normalization": "global",
}
_NORM_KEY_TO_DISPLAY = {v: k for k, v in _NORM_DISPLAY_TO_KEY.items()}


class PlotSettingsWidget(QWidget):
    """Combined plot settings with toggle-button tabs: LinePlot | Spectrogram | HeatMap."""

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None
        self._needs_auto_levels = True

        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_lineplot_panel(main_layout)
        self._create_spectrogram_panel(main_layout)
        self._create_heatmap_panel(main_layout)

        self._restore_lineplot_defaults()
        self._restore_spectrogram_defaults()
        self._restore_heatmap_defaults()

        self._show_panel("lineplot")

    # ------------------------------------------------------------------
    # Toggle buttons
    # ------------------------------------------------------------------

    def _create_toggle_buttons(self, main_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("lineplot_toggle", "LinePlot", self._toggle_lineplot),
            ("spectrogram_toggle", "Spectrogram", self._toggle_spectrogram),
            ("heatmap_toggle", "HeatMap", self._toggle_heatmap),
        ]
        for attr, label, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "lineplot": (self.lineplot_panel, self.lineplot_toggle),
            "spectrogram": (self.spectrogram_panel, self.spectrogram_toggle),
            "heatmap": (self.heatmap_panel, self.heatmap_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_lineplot(self):
        self._show_panel("lineplot" if self.lineplot_toggle.isChecked() else "spectrogram")

    def _toggle_spectrogram(self):
        self._show_panel("spectrogram" if self.spectrogram_toggle.isChecked() else "lineplot")

    def _toggle_heatmap(self):
        self._show_panel("heatmap" if self.heatmap_toggle.isChecked() else "lineplot")

    def _refresh_layout(self):
        if self.meta_widget and hasattr(self.meta_widget, "collapsible_widgets"):
            for collapsible in self.meta_widget.collapsible_widgets:
                if hasattr(collapsible, "content_widget"):
                    content = collapsible.content_widget
                    if content and self in content.findChildren(QWidget):
                        collapsible.collapse()
                        QApplication.processEvents()
                        collapsible.expand()

    # ------------------------------------------------------------------
    # LinePlot panel (from AxesWidget)
    # ------------------------------------------------------------------

    def _create_lineplot_panel(self, main_layout):
        self.lineplot_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lineplot_panel.setLayout(layout)

        group_box = QGroupBox("Axes Controls")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        self.ymin_edit = QLineEdit()
        self.ymax_edit = QLineEdit()

        self.percentile_ylim_edit = QLineEdit()
        validator = QDoubleValidator(95.0, 100, 2, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.percentile_ylim_edit.setValidator(validator)

        self.window_s_edit = QLineEdit()

        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")

        self.autoscale_checkbox = QCheckBox("Autoscale Y")
        self.lock_axes_checkbox = QCheckBox("Lock Axes")

        row = 0
        group_layout.addWidget(QLabel("Y min:"), row, 0)
        group_layout.addWidget(self.ymin_edit, row, 1)
        group_layout.addWidget(QLabel("Y max:"), row, 2)
        group_layout.addWidget(self.ymax_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("Percentile Y-lim:"), row, 0)
        group_layout.addWidget(self.percentile_ylim_edit, row, 1)
        group_layout.addWidget(QLabel("Window (s):"), row, 2)
        group_layout.addWidget(self.window_s_edit, row, 3)

        row += 1
        group_layout.addWidget(self.autoscale_checkbox, row, 0)
        group_layout.addWidget(self.lock_axes_checkbox, row, 1)
        group_layout.addWidget(self.apply_button, row, 2)
        group_layout.addWidget(self.reset_button, row, 3)

        self.ymin_edit.editingFinished.connect(self._on_axes_edited)
        self.ymax_edit.editingFinished.connect(self._on_axes_edited)
        self.percentile_ylim_edit.editingFinished.connect(self._on_axes_edited)
        self.window_s_edit.editingFinished.connect(self._on_axes_edited)

        self.apply_button.clicked.connect(self._on_axes_edited)
        self.reset_button.clicked.connect(self._reset_axes_to_defaults)
        self.autoscale_checkbox.toggled.connect(self._autoscale_y_toggle)
        self.lock_axes_checkbox.toggled.connect(self._on_lock_axes_toggled)

        main_layout.addWidget(self.lineplot_panel)

    def _restore_lineplot_defaults(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))

        lock_axes = self.app_state.get_with_default("lock_axes")
        self.lock_axes_checkbox.setChecked(lock_axes)

    def _autoscale_y_toggle(self, checked: bool):
        if not self.plot_container:
            return
        if checked:
            self.plot_container.vb.enableAutoRange(x=False, y=True)
            self.lock_axes_checkbox.setChecked(False)
        else:
            self.plot_container.vb.disableAutoRange()

    def _on_lock_axes_toggled(self, checked: bool):
        self.app_state.lock_axes = checked
        if self.plot_container:
            self.plot_container.toggle_axes_lock()
        if checked:
            self.autoscale_checkbox.setChecked(False)

    def _on_axes_edited(self):
        if not self.plot_container:
            return

        edits = {
            "ymin": self.ymin_edit,
            "ymax": self.ymax_edit,
            "percentile_ylim": self.percentile_ylim_edit,
            "window_size": self.window_s_edit,
        }

        values = {}
        for attr, edit in edits.items():
            val = self._parse_float(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        user_set_yrange = (self._parse_float(self.ymin_edit.text()) is not None or
                           self._parse_float(self.ymax_edit.text()) is not None)

        if not self.plot_container.is_spectrogram() and not self.autoscale_checkbox.isChecked():
            if user_set_yrange:
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'vb'):
                    current_plot.vb.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
            self.plot_container.apply_y_range(values["ymin"], values["ymax"])

        if not user_set_yrange and not self.autoscale_checkbox.isChecked() and "percentile_ylim" in values:
            current_plot = self.plot_container.get_current_plot()
            if hasattr(current_plot, '_apply_zoom_constraints'):
                current_plot._apply_zoom_constraints()

        new_xmin, new_xmax = self._calculate_new_window_size()
        if new_xmin is not None and new_xmax is not None:
            self.plot_container.set_x_range(mode='preserve', curr_xlim=(new_xmin, new_xmax))

    def _calculate_new_window_size(self):
        if not self.plot_container:
            return None, None
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return None, None
        current_time = self.app_state.current_frame / self.app_state.ds.fps
        window_size = self.app_state.get_with_default("window_size")
        half_window = window_size / 2
        return current_time - half_window, current_time + half_window

    def _reset_axes_to_defaults(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
        ]:
            value = self.app_state.get_with_default(attr)
            edit.setText("" if value is None else str(value))
            setattr(self.app_state, attr, value)

        self.lock_axes_checkbox.setChecked(False)
        self.app_state.lock_axes = False
        self._on_axes_edited()

    # ------------------------------------------------------------------
    # Spectrogram panel (from SpectrogramWidget)
    # ------------------------------------------------------------------

    def _create_spectrogram_panel(self, main_layout):
        self.spectrogram_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spectrogram_panel.setLayout(layout)

        group_box = QGroupBox("Spectrogram Controls")
        group_layout = QGridLayout()
        group_layout.setVerticalSpacing(6)
        group_layout.setContentsMargins(6, 10, 6, 6)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        self.spec_ymin_edit = QLineEdit()
        self.spec_ymax_edit = QLineEdit()
        self.vmin_db_edit = QLineEdit()
        self.vmax_db_edit = QLineEdit()
        self.nfft_edit = QLineEdit()
        self.hop_frac_edit = QLineEdit()

        self.colormap_combo = QComboBox()
        self.colormap_display = {
            'CET-R4': 'jet',
            'CET-L8': 'blue-pink-yellow',
            'CET-L16': 'black-blue-green-white',
            'CET-CBL2': 'black-blue-yellow-white',
            'CET-L1': 'black-white',
            'CET-L3': 'inferno',
        }
        self.colormaps = list(self.colormap_display.keys())
        self.colormap_combo.addItems(self.colormap_display.values())

        self.levels_mode_combo = QComboBox()
        self.levels_mode_combo.addItems(["Always auto dB levels", "Remember dB levels"])

        self.auto_levels_button = QPushButton("Auto dB levels")
        self.spec_apply_button = QPushButton("Apply settings")

        row = 0
        group_layout.addWidget(QLabel("Freq min (kHz):"), row, 0)
        group_layout.addWidget(self.spec_ymin_edit, row, 1)
        group_layout.addWidget(QLabel("Freq max (kHz):"), row, 2)
        group_layout.addWidget(self.spec_ymax_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("NFFT:"), row, 0)
        group_layout.addWidget(self.nfft_edit, row, 1)
        group_layout.addWidget(QLabel("Hop fraction:"), row, 2)
        group_layout.addWidget(self.hop_frac_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("dB min:"), row, 0)
        group_layout.addWidget(self.vmin_db_edit, row, 1)
        group_layout.addWidget(QLabel("dB max:"), row, 2)
        group_layout.addWidget(self.vmax_db_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("Colormap:"), row, 0)
        group_layout.addWidget(self.colormap_combo, row, 1)
        group_layout.addWidget(QLabel("Levels:"), row, 2)
        group_layout.addWidget(self.levels_mode_combo, row, 3)

        row += 1
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.auto_levels_button)
        button_layout.addWidget(self.spec_apply_button)
        button_widget.setLayout(button_layout)
        group_layout.addWidget(button_widget, row, 0, 1, 4)

        self.spec_ymin_edit.editingFinished.connect(self._on_spec_edited)
        self.spec_ymax_edit.editingFinished.connect(self._on_spec_edited)
        self.vmin_db_edit.editingFinished.connect(self._on_spec_edited)
        self.vmax_db_edit.editingFinished.connect(self._on_spec_edited)
        self.nfft_edit.editingFinished.connect(self._on_spec_edited)
        self.hop_frac_edit.editingFinished.connect(self._on_spec_edited)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.levels_mode_combo.currentIndexChanged.connect(self._on_levels_mode_changed)
        self.auto_levels_button.clicked.connect(self._auto_levels)
        self.spec_apply_button.clicked.connect(self._on_spec_edited)

        main_layout.addWidget(self.spectrogram_panel)

    def _restore_spectrogram_defaults(self):
        for attr, edit in [
            ("vmin_db", self.vmin_db_edit),
            ("vmax_db", self.vmax_db_edit),
            ("nfft", self.nfft_edit),
            ("hop_frac", self.hop_frac_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))

        default_vmin = AppStateSpec.get_default("vmin_db")
        default_vmax = AppStateSpec.get_default("vmax_db")
        if (getattr(self.app_state, "vmin_db", default_vmin) != default_vmin or
                getattr(self.app_state, "vmax_db", default_vmax) != default_vmax):
            self._needs_auto_levels = False

        for attr, edit in [
            ("spec_ymin", self.spec_ymin_edit),
            ("spec_ymax", self.spec_ymax_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            display_val = value / 1000 if value is not None else None
            edit.setText("" if display_val is None else str(display_val))

        colormap = self.app_state.get_with_default("spec_colormap")
        if colormap in self.colormap_display:
            self.colormap_combo.setCurrentText(self.colormap_display[colormap])

        levels_mode = getattr(self.app_state, 'spec_levels_mode', None)
        if levels_mode is None:
            levels_mode = self.app_state.get_with_default('spec_levels_mode')
            self.app_state.spec_levels_mode = levels_mode
        self.levels_mode_combo.setCurrentIndex(0 if levels_mode == 'auto' else 1)
        if levels_mode == 'remember':
            self._needs_auto_levels = False

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container
        plot_container.plot_changed.connect(self._on_plot_changed)
        plot_container.spectrogram_overlay_shown.connect(self._on_overlay_shown)
        if plot_container.spectrogram_plot:
            plot_container.spectrogram_plot.bufferUpdated.connect(self._on_buffer_updated)

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_enabled_state(self, has_audio: bool = False):
        self.setEnabled(True)

    def _is_auto_levels_mode(self) -> bool:
        return getattr(self.app_state, 'spec_levels_mode', 'auto') == 'auto'

    def _on_plot_changed(self, plot_type: str):
        if plot_type == 'spectrogram' and self._needs_auto_levels:
            QTimer.singleShot(500, self._try_initial_auto_levels)

    def _try_initial_auto_levels(self):
        if not self._needs_auto_levels:
            return
        if not self.plot_container or not self.plot_container.is_spectrogram():
            return
        current_plot = self.plot_container.get_current_plot()
        if not hasattr(current_plot, 'buffer') or current_plot.buffer.Sxx_db is None:
            return
        self._needs_auto_levels = False
        self._auto_levels()

    def _on_buffer_updated(self):
        if self._is_auto_levels_mode():
            self._auto_levels()

    def _on_overlay_shown(self):
        colormap_name = self.app_state.get_with_default('spec_colormap')
        self.plot_container.apply_overlay_colormap(colormap_name)
        if self._is_auto_levels_mode():
            QTimer.singleShot(200, self._auto_levels)
        else:
            self._apply_remembered_levels()

    def _on_levels_mode_changed(self, index: int):
        mode = "auto" if index == 0 else "remember"
        self.app_state.spec_levels_mode = mode
        if mode == "auto":
            self._auto_levels()
        else:
            self._apply_remembered_levels()

    def _on_colormap_changed(self, display_name: str):
        display_to_internal = {v: k for k, v in self.colormap_display.items()}
        colormap_name = display_to_internal.get(display_name, display_name)
        self.app_state.spec_colormap = colormap_name
        if self.plot_container:
            if self.plot_container.is_spectrogram():
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'update_colormap'):
                    current_plot.update_colormap(colormap_name)
            if self.plot_container.has_spectrogram_overlay():
                self.plot_container.apply_overlay_colormap(colormap_name)

    def _auto_levels(self):
        if not self.plot_container:
            return

        is_spec = self.plot_container.is_spectrogram()
        has_overlay = self.plot_container.has_spectrogram_overlay()

        if not is_spec and not has_overlay:
            return

        spec_plot = self.plot_container.spectrogram_plot
        if not hasattr(spec_plot, 'buffer') or spec_plot.buffer.Sxx_db is None:
            return

        Sxx_db = spec_plot.buffer.Sxx_db
        if Sxx_db.size == 0:
            return

        nf = max(1, Sxx_db.shape[0] // 16)

        with np.errstate(all='ignore'):
            zmin = np.percentile(Sxx_db[-nf:, :], 95)
            zmax = np.max(Sxx_db)

        if not np.isfinite(zmin) or not np.isfinite(zmax):
            return

        zmax = zmin + 0.95 * (zmax - zmin)

        if zmax - zmin < 20:
            zmax = zmin + 20
        if zmax - zmin > 80:
            zmin = zmax - 80

        zmin = round(zmin, 1)
        zmax = round(zmax, 1)

        self.vmin_db_edit.setText(str(zmin))
        self.vmax_db_edit.setText(str(zmax))

        self.app_state.vmin_db = zmin
        self.app_state.vmax_db = zmax

        if is_spec and hasattr(spec_plot, 'update_levels'):
            spec_plot.update_levels(zmin, zmax)

        if has_overlay:
            self.plot_container.apply_overlay_levels(zmin, zmax)

    def _apply_remembered_levels(self):
        vmin = self._parse_float(self.vmin_db_edit.text())
        vmax = self._parse_float(self.vmax_db_edit.text())
        if vmin is None:
            vmin = self.app_state.get_with_default("vmin_db")
        if vmax is None:
            vmax = self.app_state.get_with_default("vmax_db")
        self.app_state.vmin_db = vmin
        self.app_state.vmax_db = vmax
        if self.plot_container:
            if self.plot_container.is_spectrogram():
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'update_levels'):
                    current_plot.update_levels(vmin, vmax)
            if self.plot_container.has_spectrogram_overlay():
                self.plot_container.apply_overlay_levels(vmin, vmax)

    def _on_spec_edited(self):
        if not self.plot_container:
            return

        float_edits = {
            "vmin_db": self.vmin_db_edit,
            "vmax_db": self.vmax_db_edit,
            "hop_frac": self.hop_frac_edit,
        }

        khz_edits = {
            "spec_ymin": self.spec_ymin_edit,
            "spec_ymax": self.spec_ymax_edit,
        }

        int_edits = {
            "nfft": self.nfft_edit,
        }

        values = {}
        for attr, edit in float_edits.items():
            val = self._parse_float(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        for attr, edit in khz_edits.items():
            val = self._parse_float(edit.text())
            if val is not None:
                val = val * 1000
            else:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        for attr, edit in int_edits.items():
            val = self._parse_int(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        if self.plot_container.is_spectrogram():
            current_plot = self.plot_container.get_current_plot()

            if hasattr(current_plot, 'update_buffer_settings'):
                current_plot.update_buffer_settings()

            if hasattr(current_plot, 'update_levels'):
                current_plot.update_levels(values["vmin_db"], values["vmax_db"])

            self.plot_container.apply_y_range(values["spec_ymin"], values["spec_ymax"])

            if hasattr(current_plot, 'update_plot_content'):
                current_plot.update_plot_content()

        if self.plot_container.has_spectrogram_overlay():
            self.plot_container.apply_overlay_levels(values["vmin_db"], values["vmax_db"])
            spec_plot = self.plot_container.spectrogram_plot
            if hasattr(spec_plot, 'buffer') and hasattr(spec_plot.buffer, '_clear_buffer'):
                spec_plot.buffer._clear_buffer()
            self.plot_container.update_audio_overlay()

    # ------------------------------------------------------------------
    # HeatMap panel (from AxesWidget heatmap controls)
    # ------------------------------------------------------------------

    def _create_heatmap_panel(self, main_layout):
        self.heatmap_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_panel.setLayout(layout)

        hm_group = QGroupBox("Heatmap Display")
        hm_layout = QGridLayout()
        hm_group.setLayout(hm_layout)
        layout.addWidget(hm_group)

        hm_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.addItems(HEATMAP_COLORMAPS)
        self.heatmap_colormap_combo.currentTextChanged.connect(self._on_heatmap_colormap_changed)
        hm_layout.addWidget(self.heatmap_colormap_combo, 0, 1)

        hm_layout.addWidget(QLabel("Excl. percentile:"), 0, 2)
        self.heatmap_percentile_spin = QDoubleSpinBox()
        self.heatmap_percentile_spin.setRange(50.0, 100.0)
        self.heatmap_percentile_spin.setSingleStep(1.0)
        self.heatmap_percentile_spin.setDecimals(1)
        self.heatmap_percentile_spin.setToolTip("Percentile of abs(z-scores) for symmetric color range")
        self.heatmap_percentile_spin.valueChanged.connect(self._on_heatmap_percentile_changed)
        hm_layout.addWidget(self.heatmap_percentile_spin, 0, 3)

        hm_layout.addWidget(QLabel("Normalization:"), 1, 0)
        self.heatmap_norm_combo = QComboBox()
        self.heatmap_norm_combo.addItems(list(_NORM_DISPLAY_TO_KEY.keys()))
        self.heatmap_norm_combo.currentTextChanged.connect(self._on_heatmap_normalization_changed)
        hm_layout.addWidget(self.heatmap_norm_combo, 1, 1)

        main_layout.addWidget(self.heatmap_panel)

    def _restore_heatmap_defaults(self):
        cmap = self.app_state.get_with_default("heatmap_colormap")
        if cmap in HEATMAP_COLORMAPS:
            self.heatmap_colormap_combo.setCurrentText(cmap)

        self.heatmap_percentile_spin.setValue(
            self.app_state.get_with_default("heatmap_exclusion_percentile")
        )

        norm_key = self.app_state.get_with_default("heatmap_normalization")
        display = _NORM_KEY_TO_DISPLAY.get(norm_key, "Per-channel")
        self.heatmap_norm_combo.setCurrentText(display)

    def _on_heatmap_colormap_changed(self, colormap_name: str):
        self.app_state.heatmap_colormap = colormap_name
        if self.plot_container:
            heatmap = self.plot_container.heatmap_plot
            heatmap.update_colormap(colormap_name)
            if self.plot_container.is_heatmap():
                heatmap._clear_buffer()
                heatmap.update_plot_content()

    def _on_heatmap_percentile_changed(self, value: float):
        self.app_state.heatmap_exclusion_percentile = value
        if self.plot_container and self.plot_container.is_heatmap():
            heatmap = self.plot_container.heatmap_plot
            heatmap._clear_buffer()
            heatmap.update_plot_content()

    def _on_heatmap_normalization_changed(self, display_name: str):
        self.app_state.heatmap_normalization = _NORM_DISPLAY_TO_KEY.get(display_name, "per_channel")
        if self.plot_container and self.plot_container.is_heatmap():
            heatmap = self.plot_container.heatmap_plot
            heatmap._clear_buffer()
            heatmap.update_plot_content()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _parse_float(self, text: str) -> Optional[float]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _parse_int(self, text: str) -> Optional[int]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
