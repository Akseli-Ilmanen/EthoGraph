"""Widget container for other collapsible widgets."""

import logging

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from ethograph.utils.paths import default_config_dir
from ethograph.utils.qt import (
    apply_compact_widget_style,
    normalize_child_layouts,
)

from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    PLOT_CONTAINER_MIN_HEIGHT,
    SIDEBAR_DEFAULT_WIDTH_RATIO,
    SIDEBAR_MIN_WIDTH_PX,
)
from .app_state import ObservableAppState
from .grid_section_container import GridSectionContainer
from .make_pretty import LayoutManager
from .plots_container import UnifiedPanelContainer
from .shortcuts import bind_global_shortcuts
from .widget_trials import TrialsWidget
from .widgets_changepoints import ChangepointsWidget
from .widgets_data import DataPanel, DataWidget
from .widgets_ephys import EphysWidget
from .widgets_help import HelpWidget
from .widgets_io import IOWidget
from .widgets_labels import LabelsWidget
from .widgets_navigation import NavigationWidget
from .widgets_plot_settings import PlotSettingsWidget

logger = logging.getLogger(__name__)


class MetaWidget(GridSectionContainer):
    def __init__(self, shell):
        """Initialize the meta-widget.

        Parameters
        ----------
        shell : EthographMainWindow
            The main application window hosting video, plots and this sidebar.
        """
        super().__init__()

        self.shell = shell

        # Set smaller font for this widget and all children
        self._set_compact_font()

        # Create centralized app_state with YAML persistence
        global_settings = default_config_dir() / "gui_settings.yaml"
        logger.info("Settings file: %s", global_settings)

        self.app_state = ObservableAppState(yaml_path=str(global_settings))

        # Try to load previous settings
        self.app_state.load_from_yaml()

        # Initialize all widgets with app_state
        self._create_widgets()

        self.collapsible_widgets[1].expand()  # Expand I/O by default

        self._connect_collapsible_layout_refresh()

        self._bind_global_shortcuts(self.labels_widget, self.data_widget)

        # Set sidebar to 30% of the window by default (user can resize freely)
        self._set_sidebar_default_width()

    def _create_widgets(self):
        """Create all widgets with app_state passed to each one."""

        # Unified container replaces both PlotContainer and MultiPanelContainer
        self.plot_container = UnifiedPanelContainer(self.app_state)

        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_container.setMinimumHeight(PLOT_CONTAINER_MIN_HEIGHT)

        self.layout_mgr = LayoutManager(self.shell, self.plot_container)

        # Create all widgets with app_state
        self.help_widget = HelpWidget(self.app_state)
        self.plot_settings_widget = PlotSettingsWidget(self.shell, self.app_state)
        self.changepoints_widget = ChangepointsWidget(self.shell, self.app_state)
        self.labels_widget = LabelsWidget(self.shell, self.app_state)
        self.navigation_widget = NavigationWidget(self.shell, self.app_state)
        self.trials_widget = TrialsWidget(self.app_state)
        self.ephys_widget = EphysWidget(self.shell, self.app_state)

        # Create I/O widget first, then pass it to data widget
        self.io_widget = IOWidget(self.app_state, None, self.labels_widget)
        self.data_panel = DataPanel(self.app_state)
        self.data_widget = DataWidget(self.shell, self.app_state, self, self.io_widget)
        self.data_widget.set_data_panel(self.data_panel)

        # Now set the data_widget reference in io_widget
        self.io_widget.data_widget = self.data_widget
        self.io_widget.changepoints_widget = self.changepoints_widget
        self.io_widget.meta_widget = self

        # Set up cross-references between widgets
        self.labels_widget.set_plot_container(self.plot_container)
        self.labels_widget.set_meta_widget(self)
        self.labels_widget.set_data_widget(self.data_widget)

        self.labels_widget.changepoints_widget = self.changepoints_widget
        self.labels_widget.io_widget = self.io_widget
        self.plot_settings_widget.set_plot_container(self.plot_container)
        self.plot_settings_widget.set_meta_widget(self)
        self.changepoints_widget.set_plot_container(self.plot_container)
        self.changepoints_widget.set_meta_widget(self)
        self.changepoints_widget.data_widget = self.data_widget
        self.changepoints_widget.set_motif_mappings(self.labels_widget._mappings)
        self.navigation_widget.set_mappings(self.labels_widget._mappings)
        self.navigation_widget._labels_widget = self.labels_widget
        self.navigation_widget._data_widget = self.data_widget
        self.navigation_widget.set_plot_container(self.plot_container)
        self.ephys_widget.set_plot_container(self.plot_container)
        self.ephys_widget.set_meta_widget(self)
        self.ephys_widget.set_data_widget(self.data_widget)
        self.ephys_widget.io_widget = self.io_widget

        # Wire IOWidget signals to LabelsWidget and EphysWidget methods
        self.io_widget.wire_label_signals()
        self.io_widget.wire_ephys_signals(self.ephys_widget)

        # Signal connections for decoupled communication
        self.plot_container.labels_redraw_needed.connect(self._on_labels_redraw_needed)
        self.app_state.trial_changed.connect(self.data_widget.on_trial_changed)
        self.app_state.trial_changed.connect(self.changepoints_widget._update_cp_status)
        self.app_state.trial_changed.connect(self.update_labels_widget_title)
        self.app_state.trial_changed.connect(self.io_widget._update_human_verified_status)
        self.app_state.trial_changed.connect(self.io_widget._update_correct_offsets_status)
        self.app_state.trial_changed.connect(self.io_widget._update_purge_small_labels_status)
        self.changepoints_widget.changepoint_correction_checkbox.stateChanged.connect(
            self.update_changepoints_widget_title
        )

        # The one widget to rule them all (loading data, updating plots, managing sync)
        self.data_widget.set_references(
            self.plot_container,
            self.labels_widget,
            self.plot_settings_widget,
            self.navigation_widget,
            self.changepoints_widget,
            ephys_widget=self.ephys_widget,
            layout_mgr=self.layout_mgr,
            trials_widget=self.trials_widget,
        )

        self.plot_settings_widget.reset_layout_button.clicked.connect(self._on_reset_layout)

        for widget in [
            self.help_widget,
            self.io_widget,
            self.data_panel,
            self.labels_widget,
            self.changepoints_widget,
            self.ephys_widget,
            self.plot_settings_widget,
            self.navigation_widget,
            self.trials_widget,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add widgets to the collapsible container (index order matters for titles)
        self.add_widget(self.help_widget, collapsible=True, widget_title="Help and Tutorials")
        self.add_widget(self.io_widget, collapsible=True, widget_title="I/O")
        self.add_widget(self.data_panel, collapsible=True, widget_title="Data")
        self.add_widget(self.ephys_widget, collapsible=True, widget_title="Phy TraceView")
        self.add_widget(self.labels_widget, collapsible=True, widget_title="Labelling")
        self.add_widget(self.changepoints_widget, collapsible=True, widget_title="Changepoints (CPs)")
        self.add_widget(self.plot_settings_widget, collapsible=True, widget_title="Plot settings")
        self.add_widget(self.trials_widget, collapsible=True, widget_title="Trials")
        self.add_widget(self.navigation_widget, collapsible=True, widget_title="Navigation")

        normalize_child_layouts(
            self,
            spacing=DEFAULT_LAYOUT_SPACING,
            margin=DEFAULT_LAYOUT_MARGIN,
        )

        self.update_changepoints_widget_title()
        self.update_labels_widget_title()

    def _connect_collapsible_layout_refresh(self):
        from qtpy.QtCore import QEvent

        self._layout_refresh_timer = QTimer(self)
        self._layout_refresh_timer.setSingleShot(True)
        self._layout_refresh_timer.setInterval(50)
        self._layout_refresh_timer.timeout.connect(self._recalc_collapsible_heights)
        self._watched_events = {QEvent.Type.LayoutRequest, QEvent.Type.Resize}
        for collapsible in self.collapsible_widgets:
            collapsible.toggled.connect(self._schedule_layout_refresh)
            content = collapsible.content()
            if content:
                content.installEventFilter(self)

    def eventFilter(self, obj, event):
        if hasattr(self, "_watched_events") and event.type() in self._watched_events:
            self._schedule_layout_refresh()
        return False

    def _schedule_layout_refresh(self, *_args):
        self._layout_refresh_timer.start()

    def _recalc_collapsible_heights(self):
        from qtpy.QtCore import QPropertyAnimation

        for collapsible in self.collapsible_widgets:
            if not collapsible.isExpanded():
                continue
            content = collapsible.content()
            if content is None:
                continue
            content.updateGeometry()
            layout = content.layout()
            if layout:
                layout.invalidate()
                layout.activate()
            collapsible._expand_collapse(
                QPropertyAnimation.Direction.Forward,
                animate=False,
                emit=False,
            )

    def refresh_widget_layout(self, widget: QWidget):
        self._schedule_layout_refresh()
        self.notify_content_resized()

    def _cycle_channel(self, direction: int):
        if not self.app_state.ready:
            return
        if self.plot_container and self.plot_container.is_ephystrace():
            spin = self.ephys_widget.ephys_channel_spin
            if spin.isVisible():
                new_val = spin.value() + direction
                new_val = max(spin.minimum(), min(new_val, spin.maximum()))
                spin.setValue(new_val)

    def _on_labels_redraw_needed(self):
        if not self.app_state.ready:
            return
        ds_kwargs = self.app_state.get_ds_kwargs()
        self.data_widget.update_label_plot(ds_kwargs)
        self.data_widget.update_trials_combo()

    def update_labels_widget_title(self):
        if hasattr(self, "collapsible_widgets") and len(self.collapsible_widgets) > 4:
            labels_collapsible = self.collapsible_widgets[4]
            verification_emoji = "❌"
            if hasattr(self.app_state, "trials_sel") and self.app_state.trials_sel is not None:
                trial_meta = self.app_state.get_trial_meta(self.app_state.trials_sel)
                if trial_meta.get("human_verified", 0):
                    verification_emoji = "✅"
            self._set_collapsible_title(labels_collapsible, f"Label controls {verification_emoji}")

    def update_changepoints_widget_title(self):
        if hasattr(self, "collapsible_widgets") and len(self.collapsible_widgets) > 5:
            cp_collapsible = self.collapsible_widgets[5]
            correction_enabled = self.changepoints_widget.changepoint_correction_checkbox.isChecked()
            indicator = "\U0001f3af" if correction_enabled else "⭕"
            self._set_collapsible_title(cp_collapsible, f"Changepoints (CPs) {indicator}")

    @staticmethod
    def _set_collapsible_title(collapsible, new_title: str):
        if hasattr(collapsible, "setText"):
            collapsible.setText(new_title)
        elif hasattr(collapsible, "setTitle"):
            collapsible.setTitle(new_title)
        elif hasattr(collapsible, "_title_widget") and hasattr(collapsible._title_widget, "setText"):
            collapsible._title_widget.setText(new_title)

    def _check_unsaved_changes(self, event):
        """Check for unsaved changes and prompt. Returns True if OK to close."""
        if not self.app_state.changes_saved:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Unsaved Changes")
            msg_box.setText("You have unsaved changes to your labels.")
            msg_box.setInformativeText("Would you like to save your changes before closing?")
            msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Save)
            response = msg_box.exec_()
            if response == QMessageBox.Save:
                try:
                    self.app_state.save_labels()
                    return True
                except (OSError, PermissionError) as e:
                    error_msg = QMessageBox()
                    error_msg.setWindowTitle("Save Error")
                    error_msg.setText(f"Failed to save changes: {str(e)}")
                    error_msg.exec_()
                    event.ignore()
                    return False
            elif response == QMessageBox.Cancel:
                event.ignore()
                return False
        return True

    def reapply_shortcuts(self):
        bind_global_shortcuts(self)

    def _bind_global_shortcuts(self, labels_widget, data_widget):
        bind_global_shortcuts(self)

    def _set_compact_font(self, font_size: int = 8):
        apply_compact_widget_style(self, font_size=font_size)

    def _set_sidebar_default_width(self):
        from qtpy.QtCore import Qt

        self.setMinimumWidth(SIDEBAR_MIN_WIDTH_PX)

        def _apply():
            dock = getattr(self.shell, "_sidebar_dock", None)
            if dock is None:
                return
            total_w = self.shell.width()
            if total_w <= 0:
                return
            ratio = max(0.15, min(0.6, SIDEBAR_DEFAULT_WIDTH_RATIO))
            target_w = max(SIDEBAR_MIN_WIDTH_PX, int(total_w * ratio))
            self.shell.resizeDocks([dock], [target_w], Qt.Horizontal)

        QTimer.singleShot(0, _apply)

    def configure_layout_for_data(self):
        """Configure panel visibility and layout after a dataset load."""
        self.plot_container.configure_panels()

        if self.app_state.ephys_path:
            self.data_widget._configure_neo_panel()
            neo_cb = getattr(self.data_widget, "neo_viewer_checkbox", None)
            self.plot_container.set_neo_visible(bool(neo_cb and neo_cb.isChecked()))

        if self.app_state.has_neurons and self.app_state.ephys_visible:
            self.data_widget._configure_ephys_trace_plot()

        self.layout_mgr.register_docks()

        if not self.app_state.video_viewer_visible:
            self.layout_mgr.set_video_viewer_visible(False)

        space_type = getattr(self.app_state, "space_plot_type", "Layers")
        if space_type == "Space Plot":
            self.data_widget.update_space_plot()

    def _on_reset_layout(self):
        space_type = getattr(self.app_state, "space_plot_type", "Layers")
        self.layout_mgr.reset_layout(
            show_layers=space_type == "Layers",
            show_space=space_type == "Space Plot",
        )

