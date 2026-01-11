"""Widget container for other collapsible widgets."""

from pathlib import Path

from ethograph.utils.paths import gui_default_settings_path
from napari.layers import Image
from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer
from qtpy.QtWidgets import QApplication, QSizePolicy, QMessageBox, QAction, QMenu, QPushButton, QWidget, QHBoxLayout
from qtpy.QtGui import QFont


from .app_state import ObservableAppState
from .widgets_data import DataWidget
from .plot_container import PlotContainer
from .widgets_labels import LabelsWidget
from .widgets_navigation import NavigationWidget
from .widgets_io import IOWidget
from .widgets_plot import PlotsWidget
from .widgets_documentation import InteractiveDocsDialog


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

        self.collapsible_widgets[1].expand()

        self._bind_global_shortcuts(self.labels_widget, self.data_widget)
        
        # Connect to napari window close event to check for unsaved changes
        if hasattr(self.viewer, 'window') and hasattr(self.viewer.window, '_qt_window'):
            original_close_event = self.viewer.window._qt_window.closeEvent
            def napari_close_event(event):
                if not self._check_unsaved_changes(event):
                    return  
                original_close_event(event)
            self.viewer.window._qt_window.closeEvent = napari_close_event

    def _create_widgets(self):
        """Create all widgets with app_state passed to each one."""

        # PlotContainer widget docked at the bottom with 1/3 height from bottom
        # This container manages switching between LinePlot and SpectrogramPlot
        self.plot_container = PlotContainer(self.viewer, self.app_state)


        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_container.setMinimumHeight(200)

        # Add dock widget with margins to prevent covering notifications
        dock_widget = self.viewer.window.add_dock_widget(self.plot_container, area="bottom")
        
        # Try to set margins on the dock widget to leave space for notifications
        try:
            if hasattr(dock_widget, 'setContentsMargins'):
                dock_widget.setContentsMargins(0, 0, 0, 50)  # Leave 50px at bottom for notifications
        except:
            pass
        
        # Ensure napari notifications are positioned correctly
        self._configure_notifications()

        # Create all widgets with app_state
        self.plots_widget = PlotsWidget(self.viewer, self.app_state)
        self.labels_widget = LabelsWidget(self.viewer, self.app_state)
        self.help_widget = self._create_help_widget()
        self.navigation_widget = NavigationWidget(self.viewer, self.app_state)
        
        # Create I/O widget first, then pass it to data widget
        self.io_widget = IOWidget(self.app_state, None, self.labels_widget)  # Will set data_widget reference after creation
        self.data_widget = DataWidget(self.viewer, self.app_state, self, self.io_widget)
        
        # Now set the data_widget reference in io_widget
        self.io_widget.data_widget = self.data_widget

        # Set up cross-references between widgets
        # Plot widgets need plot_container for unified plot access
        self.labels_widget.set_plot_container(self.plot_container)
        self.plots_widget.set_plot_container(self.plot_container)

        # Signal connections for decoupled communication
        self.plot_container.labels_redraw_needed.connect(self._on_labels_redraw_needed)
        self.app_state.labels_modified.connect(self._on_labels_modified)
        self.app_state.verification_changed.connect(self._on_verification_changed)
        self.app_state.verification_changed.connect(self.labels_widget._update_human_verified_status)
        self.app_state.trial_changed.connect(self.data_widget.on_trial_changed)
 

        # The one widget to rule them all (loading data, updating plots, managing sync)
        self.data_widget.set_references(self.plot_container, self.labels_widget, self.plots_widget, self.navigation_widget)

        for widget in [
            self.help_widget,
            self.io_widget,
            self.data_widget,
            self.labels_widget,
            self.plots_widget,
            self.navigation_widget,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add widgets to collapsible container
        self.add_widget(
            self.help_widget,
            collapsible=True,
            widget_title="Documentation",
        )

        self.add_widget(
            self.io_widget,
            collapsible=True,
            widget_title="I/O",
        )

        self.add_widget(
            self.data_widget,
            collapsible=True,
            widget_title="Data controls",
        )

        self.add_widget(
            self.labels_widget,
            collapsible=True,
            widget_title="Label controls",
        )

        self.add_widget(
            self.plots_widget,
            collapsible=True,
            widget_title="Plotting controls",
        )

        self.add_widget(
            self.navigation_widget,
            collapsible=True,
            widget_title="Navigation controls",
        )


        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def _on_labels_redraw_needed(self):
        """Handle label redraw request when switching between plots."""
        if not self.app_state.ready:
            return
        ds_kwargs = self.app_state.get_ds_kwargs()
        self.data_widget.update_motif_plot(ds_kwargs)

    def _on_labels_modified(self):
        """Handle label modification - update plots with current view range."""
        if not self.app_state.ready:
            return
        xmin, xmax = self.plot_container.get_current_xlim()
        self.data_widget.update_main_plot(t0=xmin, t1=xmax)

    def _on_verification_changed(self):
        """Handle verification status change - update UI indicators."""
        self.update_labels_widget_title()
        self.data_widget.update_trials_combo()

    def update_labels_widget_title(self):
        """Update the Label controls title with verification status emoji."""
        if hasattr(self, 'collapsible_widgets') and len(self.collapsible_widgets) > 3:
            # Labels widget is at index 3 (0: Documentation, 1: I/O, 2: Data controls, 3: Label controls)
            labels_collapsible = self.collapsible_widgets[3]

            # Get verification status
            verification_emoji = "❌"  # Default to not verified
            if (hasattr(self.app_state, 'label_dt') and self.app_state.label_dt is not None and
                hasattr(self.app_state, 'trials_sel') and self.app_state.trials_sel is not None):
                try:
                    attrs = self.app_state.label_dt.sel(trials=self.app_state.trials_sel).attrs
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
        # This method is now mainly for the dock widget itself, not the main napari window
        if hasattr(self, "app_state") and hasattr(self.app_state, "stop_auto_save"):
            self.app_state.stop_auto_save()
        super().closeEvent(event)

    def _create_help_widget(self) -> QWidget:
        """Create the help widget with documentation button."""
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)

        # Documentation button
        self.docs_button = QPushButton("📚 Tutorials & Shortcuts")
        self.docs_button.clicked.connect(self._show_documentation)
        layout.addWidget(self.docs_button)

        # GitHub button
        self.github_button = QPushButton("🔗 GitHub Issues")
        self.github_button.clicked.connect(self._open_github)
        layout.addWidget(self.github_button)

        self.docs_dialog = None
        return widget

    def _show_documentation(self):
        """Show the interactive documentation dialog."""
        if self.docs_dialog is None:
            self.docs_dialog = InteractiveDocsDialog(parent=self)
        self.docs_dialog.show()
        self.docs_dialog.raise_()
        self.docs_dialog.activateWindow()

    def _open_github(self):
        """Open GitHub issues page."""
        import webbrowser
        webbrowser.open("https://github.com/akseli-ilmanen/ethograph/issues")

    def _override_napari_shortcuts(self):
        """Aggressively unbind napari shortcuts at all levels."""
        
        
        number_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        qwerty_row = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p']
        home_row = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';']
        control_row = ['e', 'd', 'f', 'i', 'k', 'c', 'm', 't', 'n', 'p']
        other = ['y', 'space', 'Up', 'Down', 'v', 'x']

        combos = ['Ctrl-a', 'Ctrl-s', 'Ctrl+x', 'Ctrl-v']

        all_keys = number_keys + qwerty_row + home_row + control_row + other + combos
        
        from napari.layers import Labels, Points, Shapes, Surface, Tracks, Image
        
        layer_types = [Image, Points, Shapes, Labels, Tracks, Surface]
        

        for layer_type in layer_types:
            for key in all_keys:
                try:
                    if hasattr(layer_type, 'bind_key'):
                        layer_type.bind_key(key, None)
                except Exception as e:
                    print(f"Could not unbind {key} from {layer_type.__name__}: {e}")

        for key in all_keys:
            if hasattr(self.viewer, "keymap") and key in self.viewer.keymap:
                del self.viewer.keymap[key]
            
            if hasattr(self.viewer, "_keymap") and key in self.viewer._keymap:
                del self.viewer._keymap[key]
                    


        if self.viewer.layers.selection.active:
            active_layer = self.viewer.layers.selection.active
            for key in all_keys:
                if hasattr(active_layer, 'keymap') and key in active_layer.keymap:
                    del active_layer.keymap[key]

        # Handle application-level shortcuts (File menu, etc.)
        if hasattr(self.viewer, 'window') and self.viewer.window:
            window = self.viewer.window
            
    
            if hasattr(window, 'file_menu'):
                for action in window.file_menu.actions():
                    if 'save' in action.text().lower():
                        action.setShortcut('')  # Remove shortcut
                        print(f"Removed shortcut from: {action.text()}")
            
            if hasattr(window, 'qt_viewer'):
                qt_window = window.qt_viewer
                for action in qt_window.window().findChildren(QAction):
                    shortcut = action.shortcut().toString()
                    if shortcut in ['Ctrl+S', 'Ctrl+A', 'Ctrl+Shift+S']:
                        action.setShortcut('')
                        print(f"Removed {shortcut} from {action.text()}")
            

            if hasattr(window, '_qt_window'):
                menubar = window._qt_window.menuBar()
                for menu in menubar.findChildren(QMenu):
                    for action in menu.actions():
                        if action.shortcut().toString() in ['Ctrl+S', 'Ctrl+A']:
                            action.setShortcut('')



    def _bind_global_shortcuts(self, labels_widget, data_widget):
        """Bind all global shortcuts using napari's @viewer.bind_key syntax."""

        # Manually unbind previous keys.
        self._override_napari_shortcuts()
        

        # TO ADD documentation for inbuild pyqgt graph shortcuts
        # Right click hold - pull left/right to adjust xlim, up/down to adjust ylim

        
        
        # Pause/play video/audio
        viewer = self.viewer
        


        
        @viewer.bind_key("ctrl+s", overwrite=True)
        def save_labels(v):
            self.app_state.save_labels()  
        
        @viewer.bind_key("space", overwrite=True)
        def toggle_pause_resume(v):
            self.data_widget.toggle_pause_resume()
            
        @viewer.bind_key("v", overwrite=True)
        def play_segment(v):
            self.labels_widget._play_segment() 
        
        # In napari video, can user left, right arrow keys to go back/forward one frame
        
        # Navigation shortcuts (avoiding conflicts with motif labeling)
        @viewer.bind_key("Down", overwrite=True) 
        def next_trial(v):
            self.navigation_widget.next_trial()

        @viewer.bind_key("Up", overwrite=True)
        def prev_trial(v):
            self.navigation_widget.prev_trial()
            
        @viewer.bind_key("ctrl+p", overwrite=True)
        def toggle_sync(v):
            current_index = self.navigation_widget.sync_toggle_btn.currentIndex()
            total_options = self.navigation_widget.sync_toggle_btn.count()
            next_index = (current_index + 1) % total_options
            self.navigation_widget.sync_toggle_btn.setCurrentIndex(next_index)
            
        @viewer.bind_key("ctrl+x", overwrite=True)
        def toggle_label_pred(v):
            status = self.labels_widget.pred_show_predictions.isChecked()
            self.labels_widget.pred_show_predictions.setChecked(not status)
            self.labels_widget._on_pred_show_predictions_changed()

        # Override napari's built-in Ctrl+A
        if 'Ctrl-A' in viewer.keymap:
            del viewer.keymap['Ctrl-A']

        @viewer.bind_key("ctrl+a", overwrite=True)
        def toggle_autoscale(v):
            autoscale_status = self.plots_widget.autoscale_checkbox.isChecked()
            self.plots_widget.autoscale_checkbox.setChecked(not autoscale_status)
            
        @viewer.bind_key("ctrl+l", overwrite=True)
        def toggle_lock(v):
            lock_status = self.plots_widget.lock_axes_checkbox.isChecked()
            self.plots_widget.lock_axes_checkbox.setChecked(not lock_status)
            
    
        @viewer.bind_key("ctrl+enter", overwrite=True)
        def apply_plot_settings(v):
            self.plots_widget.apply_button.click()
            
        @viewer.bind_key("ctrl+v", overwrite=True)
        def human_verified(v):
            self.labels_widget._human_verification_true(mode="single_trial")


        def setup_keybindings_grid_layout(viewer, labels_widget):
            """Setup using grid layout for motif activation"""
            
            # Row 1: 1-0 (Motifs 1-10)
            number_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
            
            # Row 2: Q-P (Motifs 11-20)
            qwerty_row = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p']
            
            # Row 3: A-; (Motifs 21-30)
            home_row = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
            
            # Bind number keys for motifs 1-10
            for i, key in enumerate(number_keys):
                motif_id = i + 1 if key != '0' else 10
                viewer.bind_key(key, lambda v, mk=motif_id: labels_widget.activate_motif(mk), overwrite=True)
            
            # Bind qwerty row for motifs 11-20
            for i, key in enumerate(qwerty_row):
                viewer.bind_key(key, lambda v, mk=i+11: labels_widget.activate_motif(mk), overwrite=True)
            
            # Bind home row for motifs 21-30
            for i, key in enumerate(home_row):
                viewer.bind_key(key, lambda v, mk=i+21: labels_widget.activate_motif(mk), overwrite=True)
            

        # Call the setup function
        setup_keybindings_grid_layout(viewer, labels_widget)

        # Left click to label a motif (Press shortcut, then left-click, left-click)
        # Right click on a motif to play it

        @viewer.bind_key("ctrl+e", overwrite=True)  
        def edit_motif(v):
            labels_widget._edit_motif()

        # Delete motif (Ctrl+D)
        @viewer.bind_key("ctrl+d", overwrite=True)  
        def delete_motif(v):
            labels_widget._delete_motif()

        # Toggle features selection (Ctrl+F)
        @viewer.bind_key("ctrl+f", overwrite=True)  
        def toggle_features(v):
            self.app_state.toggle_key_sel("features", self.data_widget)

        # Toggle individuals selection (Ctrl+I)
        @viewer.bind_key("ctrl+i", overwrite=True) 
        def toggle_individuals(v):
            self.app_state.toggle_key_sel("individuals", self.data_widget)

        # Toggle keypoints selection (Ctrl+K)
        @viewer.bind_key("ctrl+k", overwrite=True)  
        def toggle_keypoints(v):
            self.app_state.toggle_key_sel("keypoints", self.data_widget)

        # Toggle cameras selection (Ctrl+C)
        @viewer.bind_key("ctrl+c", overwrite=True)  
        def toggle_cameras(v):
            self.app_state.toggle_key_sel("cameras", self.data_widget)

        # Toggle mics selection (Ctrl+M)
        @viewer.bind_key("ctrl+m", overwrite=True) 
        def toggle_mics(v):
            self.app_state.toggle_key_sel("mics", self.data_widget)

        # Toggle tracking selection (Ctrl+T)
        @viewer.bind_key("ctrl+t", overwrite=True)  
        def toggle_tracking(v):
            self.app_state.toggle_key_sel("tracking", self.data_widget)

        

    def _set_compact_font(self, font_size: int = 8):
        """Apply compact font to this widget and all children."""
        font = QFont()
        font.setPointSize(font_size)
        self.setFont(font)

        self.setStyleSheet(f"""
            * {{
                font-size: {font_size}pt;
                padding: 1px;
                margin: 0px;
            }}
            QLabel {{
                font-size: {font_size}pt;
                padding: 1px;
            }}
            QPushButton {{
                font-size: {font_size}pt;
                padding: 2px 4px;
            }}
            QComboBox {{
                font-size: {font_size}pt;
                padding: 1px;
            }}
            QSpinBox, QDoubleSpinBox {{
                font-size: {font_size}pt;
                padding: 1px;
            }}
            QLineEdit {{
                font-size: {font_size}pt;
                padding: 1px;
            }}
        """)
    
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
                                
        except Exception as e:
            # Silently handle any issues with notification configuration
            print(f"Notification configuration warning: {e}")