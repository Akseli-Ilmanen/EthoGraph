"""Widget container for other collapsible widgets."""

from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ethograph.utils.paths import gui_default_settings_path

from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    DOCK_WIDGET_BOTTOM_MARGIN,
    PLOT_CONTAINER_MIN_HEIGHT,
    SIDEBAR_DEFAULT_WIDTH_RATIO,
    SIDEBAR_MIN_WIDTH_PX,
)
from .app_state import ObservableAppState
from .makepretty import LayoutManager, apply_compact_widget_style, normalize_child_layouts
from .shortcuts import bind_global_shortcuts
from .plots_container import UnifiedPanelContainer
from .widgets_changepoints import ChangepointsWidget
from .widgets_data import DataPanel, DataWidget
from .widgets_io import IOWidget
from .widgets_labels import LabelsWidget
from .widgets_navigation import NavigationWidget
from .widgets_plot_settings import PlotSettingsWidget
from .widgets_transform import TransformWidget
from .widgets_ephys import EphysWidget



class MetaWidget(CollapsibleWidgetContainer):

    def __init__(self, napari_viewer: Viewer):
        """Initialize the meta-widget."""
        super().__init__()

        # Store the napari viewer reference
        self.viewer = napari_viewer

        # Set smaller font for this widget and all children
        self._set_compact_font()

        # Create centralized app_state with YAML persistence
        yaml_path = gui_default_settings_path()
        print(f"Settings file: {yaml_path}")

        self.app_state = ObservableAppState(yaml_path=str(yaml_path))

        # Try to load previous settings
        self.app_state.load_from_yaml()

        # Initialize all widgets with app_state
        self._create_widgets()

        self.collapsible_widgets[0].expand()

        self._connect_collapsible_layout_refresh()

        self._bind_global_shortcuts(self.labels_widget, self.data_widget)

        # Set sidebar to 30% of the napari window by default (user can resize freely)
        self._set_sidebar_default_width()

        # Connect to napari window close event to check for unsaved changes
        if hasattr(self.viewer, 'window') and hasattr(self.viewer.window, '_qt_window'):
            self._original_close_event = self.viewer.window._qt_window.closeEvent
            def napari_close_event(event):
                if not self._check_unsaved_changes(event):
                    return
                try:
                    self._original_close_event(event)
                except RuntimeError:
                    event.accept()
            self.viewer.window._qt_window.closeEvent = napari_close_event

    def _create_widgets(self):
        """Create all widgets with app_state passed to each one."""

        # Unified container replaces both PlotContainer and MultiPanelContainer
        self.plot_container = UnifiedPanelContainer(self.viewer, self.app_state)

        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_container.setMinimumHeight(PLOT_CONTAINER_MIN_HEIGHT)

        # Corner ownership:
        # - Bottom-right → right dock area: sidebar extends full height
        # - Bottom-left → bottom dock area: plots extend under secondary camera
        qt_window = self.viewer.window._qt_window
        qt_window.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        qt_window.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)

        # Add dock widget with margins to prevent covering notifications
        dock_widget = self.viewer.window.add_dock_widget(self.plot_container, area="bottom")

        # Try to set margins on the dock widget to leave space for notifications
        try:
            if hasattr(dock_widget, 'setContentsMargins'):
                dock_widget.setContentsMargins(0, 0, 0, DOCK_WIDGET_BOTTOM_MARGIN)
        except (AttributeError, TypeError):
            pass  # Dock widget doesn't support margin setting

        # Ensure napari notifications are positioned correctly
        self._configure_notifications()

        self.layout_mgr = LayoutManager(qt_window, self.plot_container)

        # Create all widgets with app_state
        self.plot_settings_widget = PlotSettingsWidget(self.viewer, self.app_state)
        self.transform_widget = TransformWidget(self.viewer, self.app_state)
        self.changepoints_widget = ChangepointsWidget(self.viewer, self.app_state)
        self.labels_widget = LabelsWidget(self.viewer, self.app_state)
        self.navigation_widget = NavigationWidget(self.viewer, self.app_state)
        self.ephys_widget = EphysWidget(self.viewer, self.app_state)

        # Create I/O widget first, then pass it to data widget
        self.io_widget = IOWidget(self.app_state, None, self.labels_widget)
        self.data_panel = DataPanel(self.app_state)
        self.data_widget = DataWidget(self.viewer, self.app_state, self, self.io_widget)
        self.data_widget.set_data_panel(self.data_panel)

        # Now set the data_widget reference in io_widget
        self.io_widget.data_widget = self.data_widget
        self.io_widget.changepoints_widget = self.changepoints_widget

        # Set up cross-references between widgets
        self.labels_widget.set_plot_container(self.plot_container)
        self.labels_widget.set_meta_widget(self)
        self.labels_widget.set_data_widget(self.data_widget)
        
        self.labels_widget.changepoints_widget = self.changepoints_widget
        self.labels_widget.io_widget = self.io_widget
        self.plot_settings_widget.set_plot_container(self.plot_container)
        self.plot_settings_widget.set_meta_widget(self)
        self.transform_widget.set_plot_container(self.plot_container)
        self.transform_widget.set_meta_widget(self)
        self.changepoints_widget.set_plot_container(self.plot_container)
        self.changepoints_widget.set_meta_widget(self)
        self.changepoints_widget.set_motif_mappings(self.labels_widget._mappings)
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
        self.changepoints_widget.changepoint_correction_checkbox.stateChanged.connect(
            self.update_changepoints_widget_title
        )

        # The one widget to rule them all (loading data, updating plots, managing sync)
        self.data_widget.set_references(
            self.plot_container, self.labels_widget, self.plot_settings_widget,
            self.navigation_widget, self.transform_widget, self.changepoints_widget,
            ephys_widget=self.ephys_widget,
            layout_mgr=self.layout_mgr,
        )

        for widget in [
            self.io_widget,
            self.data_panel,
            self.labels_widget,
            self.changepoints_widget,
            self.ephys_widget,
            self.plot_settings_widget,
            self.transform_widget,
            self.navigation_widget,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add widgets to collapsible container
        self.add_widget(
            self.io_widget,
            collapsible=True,
            widget_title="I/O",
        )

        self.add_widget(
            self.data_panel,
            collapsible=True,
            widget_title="Data",
        )


        self.add_widget(
            self.ephys_widget,
            collapsible=True,
            widget_title="Phy extension",
        )

        self.add_widget(
            self.labels_widget,
            collapsible=True,
            widget_title="Labelling",
        )

        self.add_widget(
            self.changepoints_widget,
            collapsible=True,
            widget_title="Changepoints (CPs)",
        )
        
        self.add_widget(
            self.transform_widget,
            collapsible=True,
            widget_title="Energy envelopes",
        )
        
        self.add_widget(
            self.plot_settings_widget,
            collapsible=True,
            widget_title="Plot settings",
        )

        self.add_widget(
            self.navigation_widget,
            collapsible=True,
            widget_title="Navigation / Help",
        )





        normalize_child_layouts(
            self,
            spacing=DEFAULT_LAYOUT_SPACING,
            margin=DEFAULT_LAYOUT_MARGIN,
        )

        self.update_changepoints_widget_title()

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
        from qtpy.QtCore import QEvent

        if hasattr(self, '_watched_events') and event.type() in self._watched_events:
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
                QPropertyAnimation.Direction.Forward, animate=False, emit=False,
            )

    def refresh_widget_layout(self, widget: QWidget):
        self._schedule_layout_refresh()

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
        """Handle label redraw request when switching between plots."""
        if not self.app_state.ready:
            return
        ds_kwargs = self.app_state.get_ds_kwargs()
        self.data_widget.update_label_plot(ds_kwargs)


        self.data_widget.update_trials_combo()

    def update_labels_widget_title(self):
        """Update the Label controls title with verification status emoji."""
        if hasattr(self, 'collapsible_widgets') and len(self.collapsible_widgets) > 3:
            # Labels widget is at index 4 (0: I/O, 1: Data, 2: Ephys, 3: Labelling)
            labels_collapsible = self.collapsible_widgets[3]

            # Get verification status
            verification_emoji = "❌"  # Default to not verified
            if (hasattr(self.app_state, 'label_dt') and self.app_state.label_dt is not None and
                hasattr(self.app_state, 'trials_sel') and self.app_state.trials_sel is not None):
                try:
                    attrs = self.app_state.label_dt.trial(self.app_state.trials_sel).attrs
                    if attrs.get('human_verified', None) == True:
                        verification_emoji = "✅"
                except (KeyError, AttributeError):
                    pass

            # Update the title
            new_title = f"Label controls {verification_emoji}"

            # Try to access the title/header of the collapsible widget
            if hasattr(labels_collapsible, 'setText'):
                labels_collapsible.setText(new_title)
            elif hasattr(labels_collapsible, 'setTitle'):
                labels_collapsible.setTitle(new_title)
            elif hasattr(labels_collapsible, '_title_widget') and hasattr(labels_collapsible._title_widget, 'setText'):
                labels_collapsible._title_widget.setText(new_title)

    def update_changepoints_widget_title(self):
        """Update the Changepoints title with correction mode indicator."""
        if hasattr(self, 'collapsible_widgets') and len(self.collapsible_widgets) > 4:
            # Changepoints widget is at index 5 (0: I/O, 1: Data, 2: Ephys,  3: Labelling, 4: Changepoints)
            cp_collapsible = self.collapsible_widgets[4]

            correction_enabled = self.changepoints_widget.changepoint_correction_checkbox.isChecked()
            indicator = "🎯" if correction_enabled else "⭕"

            new_title = f"Changepoints (CPs) {indicator}"

            if hasattr(cp_collapsible, 'setText'):
                cp_collapsible.setText(new_title)
            elif hasattr(cp_collapsible, 'setTitle'):
                cp_collapsible.setTitle(new_title)
            elif hasattr(cp_collapsible, '_title_widget') and hasattr(cp_collapsible._title_widget, 'setText'):
                cp_collapsible._title_widget.setText(new_title)

    def _check_unsaved_changes(self, event):
        """Check for unsaved changes and prompt user. Returns True if OK to close, False if not."""
        # Check for unsaved changes in labels widget
        if not self.app_state.changes_saved:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Unsaved Changes")
            msg_box.setText("You have unsaved changes to your labels.")
            msg_box.setInformativeText("Would you like to save your changes to labels.nc file before closing?")
            msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Save)

            response = msg_box.exec_()


            if response == QMessageBox.Save:
                # Attempt to save
                try:
                    self.app_state.save_labels()
                    # If save was successful, changes_saved will be True now
                    return True  # OK to close
                except Exception as e:
                    error_msg = QMessageBox()
                    error_msg.setWindowTitle("Save Error")
                    error_msg.setText(f"Failed to save changes: {str(e)}")
                    error_msg.exec_()
                    event.ignore()  # Prevent closing
                    return False  # Don't close
            elif response == QMessageBox.Cancel:
                event.ignore()  # Prevent closing
                return False  # Don't close
            # If Discard was selected, continue with closing

        return True  # OK to close

    def closeEvent(self, event):
        """Handle close event by stopping auto-save and saving state one final time."""
        if hasattr(self, "app_state"):
            if hasattr(self.app_state, "stop_auto_save"):
                self.app_state.stop_auto_save()
        super().closeEvent(event)


    def reapply_shortcuts(self):
        bind_global_shortcuts(self)

    def _bind_global_shortcuts(self, labels_widget, data_widget):
        bind_global_shortcuts(self)

    def _set_compact_font(self, font_size: int = 8):
        """Apply compact font to this widget and all children."""
        apply_compact_widget_style(self, font_size=font_size)

    def _set_sidebar_default_width(self):
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH_PX)
        if not hasattr(self.viewer, 'window') or not hasattr(self.viewer.window, '_qt_window'):
            return
        QTimer.singleShot(
            0, lambda: self.layout_mgr.set_sidebar_default_width(self, SIDEBAR_DEFAULT_WIDTH_RATIO),
        )

    def _configure_notifications(self):
        """Configure napari notifications to be visible above docked widgets."""
        try:
            # Access napari's notification manager
            if hasattr(self.viewer.window, '_qt_viewer'):
                qt_viewer = self.viewer.window._qt_viewer

                # Try to access the notification overlay
                if hasattr(qt_viewer, '_overlays'):
                    for overlay in qt_viewer._overlays.values():
                        if hasattr(overlay, 'setContentsMargins'):
                            # Add bottom margin to keep notifications above docked widgets
                            overlay.setContentsMargins(0, 0, 0, 60)

                        # Try to adjust positioning
                        if hasattr(overlay, 'resize') and hasattr(overlay, 'parent'):
                            parent = overlay.parent()
                            if parent:
                                parent_rect = parent.geometry()
                                # Position overlay to leave space at bottom
                                overlay.resize(parent_rect.width(), parent_rect.height() - 80)

        except (AttributeError, KeyError, TypeError) as e:
            # Silently handle any issues with notification configuration
            print(f"Notification configuration warning: {e}")

    def configure_layout_for_data(self):
        """Configure panel visibility and napari canvas after data load."""
        self.plot_container.configure_panels()

        self.plot_container.set_audiotrace_visible(self.app_state.audiotrace_visible)
        self.plot_container.set_spectrogram_visible(self.app_state.spectrogram_visible)
        self.plot_container.set_featureplot_visible(self.app_state.featureplot_visible)
        # Neo-Viewer: configure if ephys file is available
        if self.app_state.ephys_path:
            self.data_widget._configure_neo_panel()
            neo_visible = getattr(self.data_widget, 'neo_viewer_checkbox', None)
            if neo_visible and neo_visible.isChecked():
                self.plot_container.set_neo_visible(True)

        # Phy-Viewer: configure if ephys/kilosort is available
        if self.app_state.has_kilosort and self.app_state.ephys_visible:
            self.data_widget._configure_ephys_trace_plot()
            self.plot_container.set_ephys_visible(True)
        elif self.app_state.has_kilosort and not self.app_state.ephys_visible:
            self.plot_container.set_ephys_visible(False)

        self.layout_mgr.register_docks()

        if not self.app_state.video_viewer_visible:
            self.layout_mgr.set_video_viewer_visible(False)

        if self.app_state.has_video:
            slot1_text = getattr(self.app_state, 'space_plot_type', 'Layers')
            show_layers = slot1_text == "Layers"

            if show_layers:
                self.layout_mgr.show_layer_docks()
                self.layout_mgr.cap_layer_width()
            else:
                self.layout_mgr.hide_layer_docks()

            self.layout_mgr.set_vertical_ratio()

        if not self.app_state.has_video:
            self.layout_mgr.configure_no_video(self.navigation_widget)


