"""Data transformation widget — energy envelopes and noise removal."""

from __future__ import annotations

from napari.viewer import Viewer
from qtpy.QtCore import Qt
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


ENERGY_METRICS = {
    "amplitude_envelope": "Amplitude envelope",
    "meansquared": "Meansquared energy",
    "band_envelope": "Band envelope",
}


class TransformWidget(QWidget):
    """Data transformation with toggle-button tabs: Energy envelopes | Noise removal."""

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_energy_panel(main_layout)
        self._create_noise_panel(main_layout)

        self._restore_energy_selections()
        self._restore_noise_selections()

        self._show_panel("energy")
        self.setEnabled(False)

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
            ("energy_toggle", "Energy envelopes", self._toggle_energy),
            ("noise_toggle", "Noise removal", self._toggle_noise),
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
            "energy": (self.energy_panel, self.energy_toggle),
            "noise": (self.noise_panel, self.noise_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_energy(self):
        self._show_panel("energy" if self.energy_toggle.isChecked() else "noise")

    def _toggle_noise(self):
        self._show_panel("noise" if self.noise_toggle.isChecked() else "energy")

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
    # Energy envelopes panel (from EnergyWidget)
    # ------------------------------------------------------------------

    def _create_energy_panel(self, main_layout):
        self.energy_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.energy_panel.setLayout(layout)

        group = QGroupBox("Create energy envelope")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)

        grid.addWidget(QLabel("Energy metric:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(ENERGY_METRICS.values())
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        grid.addWidget(self.metric_combo, 0, 1, 1, 3)

        # Amplitude envelope params
        self.amp_label_rate = QLabel("Env rate (Hz):")
        grid.addWidget(self.amp_label_rate, 1, 0)
        self.env_rate_spin = QDoubleSpinBox()
        self.env_rate_spin.setRange(100.0, 44100.0)
        self.env_rate_spin.setSingleStep(100.0)
        self.env_rate_spin.setDecimals(0)
        self.env_rate_spin.setToolTip("Output sample rate for amplitude envelope")
        grid.addWidget(self.env_rate_spin, 1, 1)

        self.amp_label_cutoff = QLabel("Cutoff (Hz):")
        grid.addWidget(self.amp_label_cutoff, 1, 2)
        self.env_cutoff_spin = QDoubleSpinBox()
        self.env_cutoff_spin.setRange(10.0, 22050.0)
        self.env_cutoff_spin.setSingleStep(50.0)
        self.env_cutoff_spin.setDecimals(0)
        self.env_cutoff_spin.setToolTip("Lowpass cutoff for amplitude envelope")
        grid.addWidget(self.env_cutoff_spin, 1, 3)

        # Meansquared params
        self.ms_label_fmin = QLabel("Freq min (Hz):")
        grid.addWidget(self.ms_label_fmin, 2, 0)
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 44100.0)
        self.freq_min_spin.setSingleStep(100.0)
        self.freq_min_spin.setDecimals(0)
        grid.addWidget(self.freq_min_spin, 2, 1)

        self.ms_label_fmax = QLabel("Freq max (Hz):")
        grid.addWidget(self.ms_label_fmax, 2, 2)
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 44100.0)
        self.freq_max_spin.setSingleStep(100.0)
        self.freq_max_spin.setDecimals(0)
        grid.addWidget(self.freq_max_spin, 2, 3)

        self.ms_label_smooth = QLabel("Smooth win:")
        grid.addWidget(self.ms_label_smooth, 3, 0)
        self.smooth_win_spin = QDoubleSpinBox()
        self.smooth_win_spin.setRange(0.1, 100.0)
        self.smooth_win_spin.setSingleStep(0.5)
        self.smooth_win_spin.setDecimals(1)
        self.smooth_win_spin.setToolTip("Smoothing window for meansquared energy")
        grid.addWidget(self.smooth_win_spin, 3, 1)

        # Band envelope params
        self.be_label_fmin = QLabel("Band min (Hz):")
        grid.addWidget(self.be_label_fmin, 4, 0)
        self.band_env_min_spin = QDoubleSpinBox()
        self.band_env_min_spin.setRange(0.1, 22050.0)
        self.band_env_min_spin.setSingleStep(50.0)
        self.band_env_min_spin.setDecimals(0)
        self.band_env_min_spin.setToolTip("Lower frequency of the bandpass filter")
        grid.addWidget(self.band_env_min_spin, 4, 1)

        self.be_label_fmax = QLabel("Band max (Hz):")
        grid.addWidget(self.be_label_fmax, 4, 2)
        self.band_env_max_spin = QDoubleSpinBox()
        self.band_env_max_spin.setRange(0.1, 22050.0)
        self.band_env_max_spin.setSingleStep(50.0)
        self.band_env_max_spin.setDecimals(0)
        self.band_env_max_spin.setToolTip("Upper frequency of the bandpass filter")
        grid.addWidget(self.band_env_max_spin, 4, 3)

        self.be_label_rate = QLabel("Env rate (Hz):")
        grid.addWidget(self.be_label_rate, 5, 0)
        self.band_env_rate_spin = QDoubleSpinBox()
        self.band_env_rate_spin.setRange(10.0, 44100.0)
        self.band_env_rate_spin.setSingleStep(100.0)
        self.band_env_rate_spin.setDecimals(0)
        self.band_env_rate_spin.setToolTip("Output sample rate for band envelope")
        grid.addWidget(self.band_env_rate_spin, 5, 1)

        self.energy_apply_button = QPushButton("Apply")
        self.energy_apply_button.clicked.connect(self._on_energy_apply)
        grid.addWidget(self.energy_apply_button, 5, 2, 1, 2)

        self._amp_widgets = [self.amp_label_rate, self.env_rate_spin, self.amp_label_cutoff, self.env_cutoff_spin]
        self._ms_widgets = [self.ms_label_fmin, self.freq_min_spin, self.ms_label_fmax, self.freq_max_spin,
                            self.ms_label_smooth, self.smooth_win_spin]
        self._be_widgets = [self.be_label_fmin, self.band_env_min_spin, self.be_label_fmax, self.band_env_max_spin,
                            self.be_label_rate, self.band_env_rate_spin]

        main_layout.addWidget(self.energy_panel)

    def _restore_energy_selections(self):
        metric = self.app_state.get_with_default("energy_metric")
        display = ENERGY_METRICS.get(metric, "Amplitude envelope")
        self.metric_combo.setCurrentText(display)

        self.env_rate_spin.setValue(self.app_state.get_with_default("env_rate"))
        self.env_cutoff_spin.setValue(self.app_state.get_with_default("env_cutoff"))
        self.freq_min_spin.setValue(self.app_state.get_with_default("freq_cutoffs_min"))
        self.freq_max_spin.setValue(self.app_state.get_with_default("freq_cutoffs_max"))
        self.smooth_win_spin.setValue(self.app_state.get_with_default("smooth_win"))
        self.band_env_min_spin.setValue(self.app_state.get_with_default("band_env_min"))
        self.band_env_max_spin.setValue(self.app_state.get_with_default("band_env_max"))
        self.band_env_rate_spin.setValue(self.app_state.get_with_default("band_env_rate"))

        self._update_conditional_visibility()

    def _on_metric_changed(self, _index):
        self._update_conditional_visibility()

    def _update_conditional_visibility(self):
        current = self.metric_combo.currentText()
        is_amplitude = current == "Amplitude envelope"
        is_meansquared = current == "Meansquared energy"
        is_band = current == "Band envelope"
        for w in self._amp_widgets:
            w.setVisible(is_amplitude)
        for w in self._ms_widgets:
            w.setVisible(is_meansquared)
        for w in self._be_widgets:
            w.setVisible(is_band)

    def _display_to_internal(self, display_text: str) -> str:
        for key, val in ENERGY_METRICS.items():
            if val == display_text:
                return key
        return "amplitude_envelope"

    def _on_energy_apply(self):
        self.app_state.energy_metric = self._display_to_internal(self.metric_combo.currentText())
        self.app_state.env_rate = self.env_rate_spin.value()
        self.app_state.env_cutoff = self.env_cutoff_spin.value()
        self.app_state.freq_cutoffs_min = self.freq_min_spin.value()
        self.app_state.freq_cutoffs_max = self.freq_max_spin.value()
        self.app_state.smooth_win = self.smooth_win_spin.value()
        self.app_state.band_env_min = self.band_env_min_spin.value()
        self.app_state.band_env_max = self.band_env_max_spin.value()
        self.app_state.band_env_rate = self.band_env_rate_spin.value()

        if self.plot_container:
            heatmap = self.plot_container.heatmap_plot
            heatmap._clear_buffer()
            if self.plot_container.is_heatmap():
                heatmap.update_plot_content()

            if self.plot_container.overlay_manager.has_overlay('envelope'):
                self.plot_container._refresh_envelope_data()

    # ------------------------------------------------------------------
    # Noise removal panel (noisereduce + ephys preprocessing)
    # ------------------------------------------------------------------

    def _create_noise_panel(self, main_layout):
        self.noise_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.noise_panel.setLayout(layout)

        # --- noisereduce group ---
        nr_group = QGroupBox("noisereduce")
        nr_layout = QGridLayout()
        nr_group.setLayout(nr_layout)
        layout.addWidget(nr_group)

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

        ref_label = QLabel(
            '<a href="https://github.com/timsainb/noisereduce" '
            'style="color: #87CEEB; text-decoration: none;">noisereduce (Sainburg, 2020)</a>'
        )
        ref_label.setOpenExternalLinks(True)

        nr_layout.addWidget(self.noise_reduce_checkbox, 0, 0)
        nr_layout.addWidget(QLabel("Reduction:"), 0, 1)
        nr_layout.addWidget(self.prop_decrease_spin, 0, 2)
        nr_layout.addWidget(ref_label, 0, 3)

        # --- Ephys pre-processing group ---
        ephys_group = QGroupBox("Ephys pre-processing")
        ephys_layout = QVBoxLayout()
        ephys_group.setLayout(ephys_layout)
        layout.addWidget(ephys_group)

        self.ephys_subtract_mean_cb = QCheckBox("1. Subtract channel mean")
        self.ephys_subtract_mean_cb.setToolTip("Remove DC offset from each channel")
        self.ephys_car_cb = QCheckBox("2. Common average reference (CAR)")
        self.ephys_car_cb.setToolTip("Subtract median across channels at each time point")
        self.ephys_temporal_filter_cb = QCheckBox("3. Temporal filtering")
        self.ephys_temporal_filter_cb.setToolTip("3rd-order Butterworth highpass filter")
        self.ephys_hp_cutoff_edit = QLineEdit("300")
        self.ephys_hp_cutoff_edit.setFixedWidth(50)
        self.ephys_hp_cutoff_edit.setToolTip("Highpass cutoff frequency in Hz")
        self.ephys_hp_cutoff_label = QLabel("Hz highpass")
        self.ephys_whitening_cb = QCheckBox("4. (Global) channel whitening")
        self.ephys_whitening_cb.setToolTip("Decorrelate channels via SVD-based whitening")

        self._ephys_checkboxes = [
            self.ephys_subtract_mean_cb,
            self.ephys_car_cb,
            self.ephys_temporal_filter_cb,
            self.ephys_whitening_cb,
        ]

        ephys_layout.addWidget(self.ephys_subtract_mean_cb)
        ephys_layout.addWidget(self.ephys_car_cb)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(self.ephys_temporal_filter_cb)
        filter_row.addWidget(self.ephys_hp_cutoff_edit)
        filter_row.addWidget(self.ephys_hp_cutoff_label)
        filter_row.addStretch()
        filter_row_widget = QWidget()
        filter_row_widget.setLayout(filter_row)
        ephys_layout.addWidget(filter_row_widget)

        ephys_layout.addWidget(self.ephys_whitening_cb)

        for cb in self._ephys_checkboxes:
            cb.toggled.connect(self._on_ephys_checkbox_toggled)
        self.ephys_hp_cutoff_edit.editingFinished.connect(self._on_ephys_checkbox_toggled)

        ref_label_ks = QLabel(
            '<a href="https://www.nature.com/articles/s41592-024-02232-7#Sec10" '
            'style="color: #87CEEB; text-decoration: none;">Adapted from Kilosort4 methods</a>'
        )
        ref_label_ks.setOpenExternalLinks(True)
        ephys_layout.addWidget(ref_label_ks)

        self._enforce_ephys_sequential()

        main_layout.addWidget(self.noise_panel)

    def _restore_noise_selections(self):
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

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_enabled_state(self, has_audio: bool = False):
        self.setEnabled(True)
        self.noise_reduce_checkbox.setEnabled(has_audio)
        self.prop_decrease_spin.setEnabled(has_audio)

    def _on_noise_reduce_changed(self, state=None):
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

    # --- Ephys preprocessing ---

    def _on_ephys_checkbox_toggled(self, _checked=None):
        self._enforce_ephys_sequential()
        self._apply_ephys_preprocessing()

    def _enforce_ephys_sequential(self):
        for i, cb in enumerate(self._ephys_checkboxes):
            if i == 0:
                cb.setEnabled(True)
                continue
            prev_checked = self._ephys_checkboxes[i - 1].isChecked()
            cb.setEnabled(prev_checked)
            if not prev_checked and cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)

    def _parse_hp_cutoff(self) -> float:
        try:
            return max(1.0, float(self.ephys_hp_cutoff_edit.text()))
        except (ValueError, TypeError):
            return 300.0

    def get_ephys_preprocessing_flags(self) -> dict:
        return {
            "subtract_mean": self.ephys_subtract_mean_cb.isChecked(),
            "car": self.ephys_car_cb.isChecked(),
            "temporal_filter": self.ephys_temporal_filter_cb.isChecked(),
            "hp_cutoff": self._parse_hp_cutoff(),
            "whitening": self.ephys_whitening_cb.isChecked(),
        }

    def _apply_ephys_preprocessing(self):
        if not self.plot_container:
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        if ephys_plot is None:
            return
        flags = self.get_ephys_preprocessing_flags()
        ephys_plot.buffer.set_preprocessing(flags)
        if ephys_plot.current_range:
            ephys_plot.update_plot_content(*ephys_plot.current_range)
