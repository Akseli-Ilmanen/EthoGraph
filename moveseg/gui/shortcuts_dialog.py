"""Widget displaying all keyboard and mouse shortcuts."""

from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtGui import QGuiApplication
import yaml
from pathlib import Path


class ShortcutsWidget(QWidget):
    """Clickable widget that opens the shortcuts dialog when clicked."""
    
    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.shortcuts_dialog = None
        self.docs_widget = None
        self.napari_viewer = None  # Will be set by meta_widget

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Add a button to open the shortcuts dialog
        self.open_button = QPushButton("⌨️ Keyboard Shortcuts")
        self.open_button.clicked.connect(self.show_shortcuts_dialog)
        layout.addWidget(self.open_button)

        # Add documentation button
        self.docs_button = QPushButton("📚 Documentation")
        self.docs_button.clicked.connect(self.show_documentation)
        layout.addWidget(self.docs_button)
    
    def show_shortcuts_dialog(self):
        """Show the shortcuts dialog."""
        if self.shortcuts_dialog is None:
            self.shortcuts_dialog = ShortcutsDialog(self.app_state, self)
        self.shortcuts_dialog.show()
        self.shortcuts_dialog.raise_()
        self.shortcuts_dialog.activateWindow()

    def show_documentation(self):
        """Show the interactive documentation dialog."""
        from .documentation_widget import InteractiveDocsDialog

        if self.docs_widget is None:
            self.docs_widget = InteractiveDocsDialog(parent=self)

        self.docs_widget.show()
        self.docs_widget.raise_()
        self.docs_widget.activateWindow()


class ShortcutsDialog(QDialog):
    """Dialog displaying all available keyboard and mouse shortcuts in a table."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.setWindowTitle("Keyboard & Mouse Shortcuts")
        self.setModal(False)  # Allow interaction with napari while open
        # Size dialog within available screen geometry
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(800, available.width())
            height = min(600, available.height())
            self.resize(width, height)
        self._setup_ui()
        self._populate_shortcuts_table()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create shortcuts table
        self.shortcuts_table = QTableWidget()
        self.shortcuts_table.setColumnCount(3)
        self.shortcuts_table.setHorizontalHeaderLabels(["Shortcut", "Category", "Description"])

        # Hide row numbers (left column)
        self.shortcuts_table.verticalHeader().setVisible(False)

        # Set column widths - make description column much wider
        header = self.shortcuts_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Shortcut column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)  # Category column - fixed width
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Description column - stretches

        # Set specific widths - description column will be ~10x wider due to stretch
        self.shortcuts_table.setColumnWidth(0, 80)  # Shortcut column
        self.shortcuts_table.setColumnWidth(1, 100)  # Category column
        # Description column will automatically use remaining space (~620px with stretch)

        # Make table read-only
        self.shortcuts_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.shortcuts_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Enable word wrap for better text display
        self.shortcuts_table.setWordWrap(True)

        layout.addWidget(self.shortcuts_table)

        # Add buttons
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def _restore_to_defaults(self):
        """Restore YAML settings to defaults, keeping only audio_folder and video_folder."""
        try:
            # Get current YAML path from app state
            yaml_path = self.app_state._yaml_path
            
            # Load current YAML to extract audio_folder and video_folder
            current_settings = {}
            if Path(yaml_path).exists():
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    current_settings = yaml.safe_load(f) or {}
            
            # Keep only audio_folder and video_folder
            default_settings = {}
            if 'audio_folder' in current_settings:
                default_settings['audio_folder'] = current_settings['audio_folder']
            if 'video_folder' in current_settings:
                default_settings['video_folder'] = current_settings['video_folder']
            
            # Save minimal YAML
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_settings, f, default_flow_style=False, sort_keys=False)
            
            # Reload app state from YAML
            self.app_state.load_from_yaml(yaml_path)
            
            # Trigger plot update by emitting data_updated signal
            self.app_state.data_updated.emit()
            
        except Exception as e:
            print(f"Error restoring to defaults: {e}")

    def _populate_shortcuts_table(self):
        """Populate the shortcuts table with all available shortcuts."""
        shortcuts_data = [
            # Navigation
            ("Ctrl + Alt + P", "Navigation", "Play/pause video (napari default)"),
            ("M", "Navigation", "Go to next trial"),
            ("N", "Navigation", "Go to previous trial"),
            ("", "", ""),  # Empty row for spacing
            # Plot Navigation
            ("↑", "Plot Navigation", "Shift Y-axis range up by 5% of current range"),
            ("↓", "Plot Navigation", "Shift Y-axis range down by 5% of current range"),
            ("Shift+↑", "Plot Navigation", "Increase Y-axis upper and lower limits by 5% of current range (zoom out vertically)"),
            ("Shift+↓", "Plot Navigation", "Decrease Y-axis upper and lower limits by 5% of current range (zoom in vertically)"),
            ("Shift+←", "Plot Navigation", "Make window size 20% smaller (zoom in horizontally)"),
            ("Shift+→", "Plot Navigation", "Make window size 20% larger (zoom out horizontally)"),
            ("", "", ""),  # Empty row for spacing
            # Motif Labeling
            (
                "1",
                "Motif Label",
                "Motif 1 - Press key, then click twice on plot to define start/end",
            ),
            (
                "2",
                "Motif Label",
                "Motif 2 - Press key, then click twice on plot to define start/end",
            ),
            (
                "3",
                "Motif Label",
                "Motif 3 - Press key, then click twice on plot to define start/end",
            ),
            (
                "4",
                "Motif Label",
                "Motif 4 - Press key, then click twice on plot to define start/end",
            ),
            (
                "5",
                "Motif Label",
                "Motif 5 - Press key, then click twice on plot to define start/end",
            ),
            (
                "6",
                "Motif Label",
                "Motif 6 - Press key, then click twice on plot to define start/end",
            ),
            (
                "7",
                "Motif Label",
                "Motif 7 - Press key, then click twice on plot to define start/end",
            ),
            (
                "8",
                "Motif Label",
                "Motif 8 - Press key, then click twice on plot to define start/end",
            ),
            (
                "9",
                "Motif Label",
                "Motif 9 - Press key, then click twice on plot to define start/end",
            ),
            (
                "0",
                "Motif Label",
                "Motif 10 - Press key, then click twice on plot to define start/end",
            ),
            (
                "Q",
                "Motif Label",
                "Motif 11 - Press key, then click twice on plot to define start/end",
            ),
            (
                "W",
                "Motif Label",
                "Motif 12 - Press key, then click twice on plot to define start/end",
            ),
            (
                "R",
                "Motif Label",
                "Motif 13 - Press key, then click twice on plot to define start/end",
            ),
            (
                "T",
                "Motif Label",
                "Motif 14 - Press key, then click twice on plot to define start/end",
            ),
            ("", "", ""),  # Empty row for spacing
            # Motif Operations
            (
                "E",
                "Motif Edit",
                "Edit selected motif boundaries - Click on motif to select, then press E, then click twice to redefine boundaries",
            ),
            ("D", "Motif Edit", "Delete selected motif - Click on motif to select, then press D"),
            ("", "", ""),  # Empty row for spacing
            # Mouse Controls
            ("Left Click", "Mouse", "Click on motif to select, then press E (Edit) or D (Delete)."),
            (
                "Left Click x2",
                "Mouse",
                "After pressing a motif shortcut key (1-9, 0, Q, W, R, T), click twice on the plot to define start and end of new motif region",
            ),
            (
                "Right Click",
                "Mouse",
                "Play motif segment at cursor position - if clicked on an existing motif, plays that segment",
            ),
        ]

        self.shortcuts_table.setRowCount(len(shortcuts_data))

        for row, (shortcut, category, description) in enumerate(shortcuts_data):
            # Shortcut column
            shortcut_item = QTableWidgetItem(shortcut)
            self.shortcuts_table.setItem(row, 0, shortcut_item)

            # Category column
            category_item = QTableWidgetItem(category)
            self.shortcuts_table.setItem(row, 1, category_item)

            # Description column
            description_item = QTableWidgetItem(description)
            self.shortcuts_table.setItem(row, 2, description_item)

        # Resize rows to fit content
        self.shortcuts_table.resizeRowsToContents()


