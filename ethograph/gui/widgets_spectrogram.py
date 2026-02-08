"""Spectrogram controls widget - spectrogram settings and noise reduction."""

from typing import Optional

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
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


class SpectrogramWidget(QWidget):
    """Spectrogram controls - spectrogram settings and noise reduction.

    Keys used in gui_settings.yaml (via app_state):
      - spec_ymin, spec_ymax (frequency range in Hz, displayed as kHz)
      - vmin_db, vmax_db (dB levels)
      - nfft
      - hop_frac
      - spec_colormap
      - noise_reduce_enabled
      - noise_reduce_prop_decrease
    """

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None

        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_spectrogram_controls(main_layout)
        self._create_noise_reduction_controls(main_layout)

        self._restore_or_set_default_selections()
        self.setEnabled(False)

    def _create_spectrogram_controls(self, main_layout):
        """Create spectrogram parameter controls."""
        group_box = QGroupBox("Spectrogram Controls")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

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
            'viridis': 'viridis',
        }
        self.colormaps = list(self.colormap_display.keys())
        self.colormap_combo.addItems(self.colormap_display.values())

        self.auto_levels_button = QPushButton("Auto levels")
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")

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
        group_layout.addWidget(self.auto_levels_button, row, 2, 1, 2)
        group_layout.addWidget(self.apply_button, row, 2)
        group_layout.addWidget(self.reset_button, row, 3)

        self.spec_ymin_edit.editingFinished.connect(self._on_spec_edited)
        self.spec_ymax_edit.editingFinished.connect(self._on_spec_edited)
        self.vmin_db_edit.editingFinished.connect(self._on_spec_edited)
        self.vmax_db_edit.editingFinished.connect(self._on_spec_edited)
        self.nfft_edit.editingFinished.connect(self._on_spec_edited)
        self.hop_frac_edit.editingFinished.connect(self._on_spec_edited)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.auto_levels_button.clicked.connect(self._auto_levels)
        self.apply_button.clicked.connect(self._on_spec_edited)
        self.reset_button.clicked.connect(self._reset_spec_to_defaults)

    def _create_noise_reduction_controls(self, main_layout):
        """Create noise reduction controls with reference."""
        group_box = QGroupBox("Remove stationary noise")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

        self.noise_reduce_checkbox = QCheckBox("Enable")
        self.noise_reduce_checkbox.setToolTip(
            "Apply spectral gating noise reduction to audio.\n"
            "Affects spectrogram and waveform display."
        )
        self.noise_reduce_checkbox.stateChanged.connect(self._on_noise_reduce_changed)

        self.prop_decrease_spin = QDoubleSpinBox()
        self.prop_decrease_spin.setRange(0.0, 1.0)
        self.prop_decrease_spin.setSingleStep(0.1)
        self.prop_decrease_spin.setValue(1.0)
        self.prop_decrease_spin.setDecimals(2)
        self.prop_decrease_spin.setToolTip(
            "Proportion to reduce noise by (0.0-1.0).\n"
            "1.0 = 100% reduction (default)"
        )
        self.prop_decrease_spin.valueChanged.connect(self._on_noise_reduce_changed)

        ref_label = QLabel('<a href="https://github.com/timsainb/noisereduce" style="color: #87CEEB; text-decoration: none;">noisereduce (Sainburg, 2020)</a>')
        ref_label.setOpenExternalLinks(True)

        row = 0
        group_layout.addWidget(self.noise_reduce_checkbox, row, 0)
        group_layout.addWidget(QLabel("Reduction:"), row, 1)
        group_layout.addWidget(self.prop_decrease_spin, row, 2)
        group_layout.addWidget(ref_label, row, 3)

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_enabled_state(self, has_audio: bool):
        self.setEnabled(has_audio)

    def _restore_or_set_default_selections(self):
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

        noise_reduce = getattr(self.app_state, 'noise_reduce_enabled', None)
        if noise_reduce is None:
            noise_reduce = self.app_state.get_with_default('noise_reduce_enabled')
            self.app_state.noise_reduce_enabled = noise_reduce
        self.noise_reduce_checkbox.setChecked(noise_reduce)

        prop_decrease = getattr(self.app_state, 'noise_reduce_prop_decrease', None)
        if prop_decrease is None:
            prop_decrease = self.app_state.get_with_default('noise_reduce_prop_decrease')
            self.app_state.noise_reduce_prop_decrease = prop_decrease
        self.prop_decrease_spin.setValue(prop_decrease)

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

    def _on_colormap_changed(self, display_name: str):
        display_to_internal = {v: k for k, v in self.colormap_display.items()}
        colormap_name = display_to_internal.get(display_name, display_name)
        self.app_state.spec_colormap = colormap_name
        if self.plot_container and self.plot_container.is_spectrogram():
            current_plot = self.plot_container.get_current_plot()
            if hasattr(current_plot, 'update_colormap'):
                current_plot.update_colormap(colormap_name)

    def _auto_levels(self):
        """Estimate optimal dB levels from current spectrogram data."""
        if not self.plot_container or not self.plot_container.is_spectrogram():
            return

        current_plot = self.plot_container.get_current_plot()
        if not hasattr(current_plot, 'buffer') or current_plot.buffer.Sxx_db is None:
            return

        Sxx_db = current_plot.buffer.Sxx_db

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

        if hasattr(current_plot, 'update_levels'):
            current_plot.update_levels(zmin, zmax)

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

    def _reset_spec_to_defaults(self):
        for attr, edit in [
            ("nfft", self.nfft_edit),
            ("hop_frac", self.hop_frac_edit),
        ]:
            value = self.app_state.get_with_default(attr)
            edit.setText("" if value is None else str(value))
            setattr(self.app_state, attr, value)

        nyquist_freq = 22050.0
        if self.plot_container and self.plot_container.spectrogram_plot:
            buffer = self.plot_container.spectrogram_plot.buffer
            if buffer.fs is not None:
                nyquist_freq = float(buffer.fs / 2)

        default_ymin = self.app_state.get_with_default("spec_ymin")
        default_ymax = self.app_state.get_with_default("spec_ymax")

        if default_ymin is None:
            default_ymin = 0.0
        if default_ymax is None:
            default_ymax = nyquist_freq

        self.app_state.spec_ymin = default_ymin
        self.app_state.spec_ymax = default_ymax
        self.spec_ymin_edit.setText(str(default_ymin / 1000))
        self.spec_ymax_edit.setText(str(default_ymax / 1000))

        default_colormap = self.app_state.get_with_default("spec_colormap")
        if default_colormap in self.colormap_display:
            self.colormap_combo.setCurrentText(self.colormap_display[default_colormap])

        self._on_spec_edited()

    def _on_noise_reduce_changed(self, state=None):
        """Handle noise reduction checkbox or prop_decrease change."""
        self.app_state.noise_reduce_enabled = self.noise_reduce_checkbox.isChecked()

        prop_decrease = self.prop_decrease_spin.value()
        if prop_decrease is not None:
            prop_decrease = max(0.0, min(1.0, prop_decrease))
            self.app_state.noise_reduce_prop_decrease = prop_decrease

        if self.plot_container:
            self.plot_container.clear_audio_cache()

            if self.plot_container.is_spectrogram():
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'buffer') and hasattr(current_plot.buffer, '_clear_buffer'):
                    current_plot.buffer._clear_buffer()
                if hasattr(current_plot, 'update_plot_content'):
                    current_plot.update_plot_content()
            elif self.plot_container.is_audiotrace():
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'buffer'):
                    current_plot.buffer.audio_loader = None
                    current_plot.buffer.current_path = None
                if hasattr(current_plot, 'update_plot_content'):
                    current_plot.update_plot_content()

    def get_audio_changepoints_for_snap(self) -> Optional[np.ndarray]:
        """Get audio changepoints for snap-to-changepoint functionality.

        Returns combined onset/offset times if changepoints are computed
        and display is enabled, otherwise None.
        """
        if not getattr(self.app_state, 'show_changepoints', False):
            return None

        onsets = getattr(self.app_state, 'audio_changepoint_onsets', None)
        offsets = getattr(self.app_state, 'audio_changepoint_offsets', None)

        if onsets is None or offsets is None:
            return None

        all_cp = np.concatenate([onsets, offsets])
        return np.unique(np.sort(all_cp))
