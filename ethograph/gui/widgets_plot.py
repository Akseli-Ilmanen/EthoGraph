"""Axes control widget for plot settings."""

from qtpy.QtWidgets import (
    QGridLayout, QLineEdit, QWidget, QPushButton,
    QVBoxLayout, QLabel, QCheckBox, QGroupBox,
)
from napari.viewer import Viewer
from typing import Optional
from qtpy.QtGui import QDoubleValidator


class AxesWidget(QWidget):
    """Axes controls for line plots and general plot settings.

    Keys used in gui_settings.yaml (via app_state):
      - ymin, ymax
      - percentile_ylim
      - window_size
      - lock_axes
    """

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        group_box = QGroupBox("Axes Controls")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

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

        self.ymin_edit.editingFinished.connect(self._on_edited)
        self.ymax_edit.editingFinished.connect(self._on_edited)
        self.percentile_ylim_edit.editingFinished.connect(self._on_edited)
        self.window_s_edit.editingFinished.connect(self._on_edited)

        self.apply_button.clicked.connect(self._on_edited)
        self.reset_button.clicked.connect(self._reset_to_defaults)
        self.autoscale_checkbox.toggled.connect(self._autoscale_y_toggle)
        self.lock_axes_checkbox.toggled.connect(self._on_lock_axes_toggled)

        self._restore_or_set_default_selections()

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def _restore_or_set_default_selections(self):
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

    def _parse_float(self, text: str) -> Optional[float]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

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

    def _on_edited(self):
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

        if not self.plot_container.is_spectrogram():
            self.plot_container.apply_y_range(values["ymin"], values["ymax"])

        if "percentile_ylim" in values:
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

        new_xmin = current_time - half_window
        new_xmax = current_time + half_window
        return new_xmin, new_xmax

    def _reset_to_defaults(self):
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

        self._on_edited()


# Backward compatibility alias
PlotsWidget = AxesWidget
