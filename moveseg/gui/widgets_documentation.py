"""Simple markdown documentation viewer for MovFormer napari plugin."""

from pathlib import Path
from typing import Optional
import re
import base64
import mimetypes

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QSplitter,
    QDialog, QApplication
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


class InteractiveDocsDialog(QDialog):
    """Simple documentation viewer as a popup dialog."""

    def __init__(
        self,
        docs_path: Optional[Path] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.docs_path = docs_path or Path(__file__).parent / "docs"

        # Configure dialog settings
        self.setWindowTitle("📚 Documentation")
        self.setModal(False)  # Allow interaction with napari while open

        # Remove the help button (?) from title bar
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Apply dark theme to the dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d30;
                color: #cccccc;
            }
            QPushButton {
                background-color: #0e639c;
                border: 1px solid #007acc;
                color: #ffffff;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1177bb;
                border-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #005a9e;
                border-color: #005a9e;
            }
        """)

        # Set dialog size to 80% of screen
        self._set_dialog_size()

        self._init_ui()
        self._load_documentation_files()

    def _set_dialog_size(self):
        """Set dialog to 80% of screen size and center it."""
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                width = int(available.width() * 0.8)
                height = int(available.height() * 0.85)
                self.resize(width, height)

                # Center the dialog
                x = available.x() + (available.width() - width) // 2
                y = available.y() + (available.height() - height) // 2
                self.move(x, y)
            else:
                self.resize(1400, 900)  # Fallback size
        except (AttributeError, RuntimeError):
            self.resize(1400, 900)  # Fallback size

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Main content area with navigation tree
        splitter = QSplitter(Qt.Horizontal)

        # Navigation tree for markdown files
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderLabel("Documentation Files")
        self.nav_tree.setMinimumWidth(250)
        self.nav_tree.setMaximumWidth(350)
        self.nav_tree.itemClicked.connect(self._on_tree_item_clicked)

        # Style the navigation tree with dark theme
        self.nav_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #252526;
                border: 1px solid #3c3c3c;
                color: #cccccc;
                font-size: 11pt;
                padding: 5px;
                selection-background-color: #094771;
                outline: none;
            }
            QTreeWidget::item {
                padding: 8px 5px;
                border: none;
                color: #cccccc;
            }
            QTreeWidget::item:hover {
                background-color: #2a2d2e;
                color: #ffffff;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QTreeWidget::item:selected:active {
                background-color: #0e639c;
                color: #ffffff;
            }
        """)

        # Document viewer with dark theme
        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setReadOnly(True)

        # Set default font for better readability
        font = QFont("Segoe UI", 11)
        self.viewer.setFont(font)

        # Style the viewer with dark theme
        self.viewer.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                color: #cccccc;
                selection-background-color: #264f78;
            }
        """)

        # Custom CSS for markdown rendering
        self.viewer.document().setDefaultStyleSheet(self._get_default_css())

        splitter.addWidget(self.nav_tree)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        # Set initial splitter sizes
        total_width = self.width() if self.width() > 0 else 1400
        splitter.setSizes([300, total_width - 300])

        layout.addWidget(splitter)

        # Add close button at bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setMinimumWidth(100)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

    def _get_default_css(self):
        """Get default CSS for markdown rendering with dark mode inspired by VS Code."""
        return """
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #cccccc;
            background-color: #1e1e1e;
            margin: 24px;
            font-size: 14px;
            font-weight: 400;
        }

        /* Headings with VS Code inspired styling */
        h1 {
            color: #ffffff;
            font-size: 2.25em;
            font-weight: 600;
            line-height: 1.25;
            margin-top: 24px;
            margin-bottom: 16px;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #30363d;
        }
        h2 {
            color: #f0f6fc;
            font-size: 1.75em;
            font-weight: 600;
            line-height: 1.25;
            margin-top: 24px;
            margin-bottom: 16px;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #21262d;
        }
        h3 {
            color: #e6edf3;
            font-size: 1.5em;
            font-weight: 600;
            line-height: 1.25;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h4 {
            color: #e6edf3;
            font-size: 1.25em;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h5, h6 {
            color: #e6edf3;
            font-size: 1em;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 16px;
        }

        /* Paragraphs and text */
        p {
            margin-top: 0;
            margin-bottom: 16px;
        }

        /* Code styling */
        pre {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.45;
            color: #e6edf3;
            margin: 16px 0;
        }
        code {
            background-color: rgba(110, 118, 129, 0.4);
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'Monaco', monospace;
            font-size: 85%;
            padding: 0.2em 0.4em;
            color: #ff8c42;
        }
        pre code {
            background-color: transparent;
            border-radius: 0;
            color: inherit;
            font-size: 100%;
            padding: 0;
        }

        /* Lists */
        ul, ol {
            margin-top: 0;
            margin-bottom: 16px;
            padding-left: 2em;
        }
        li {
            margin-bottom: 4px;
        }
        li > p {
            margin-top: 16px;
        }

        /* Blockquotes */
        blockquote {
            border-left: 0.25em solid #30363d;
            margin: 0 0 16px 0;
            padding: 0 1em;
            color: #8b949e;
            font-style: italic;
        }
        blockquote > :first-child {
            margin-top: 0;
        }
        blockquote > :last-child {
            margin-bottom: 0;
        }

        /* Tables */
        table {
            border-spacing: 0;
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            overflow: auto;
            font-size: 14px;
        }
        th, td {
            border: 1px solid #30363d;
            padding: 6px 13px;
            text-align: left;
        }
        th {
            background-color: #161b22;
            color: #f0f6fc;
            font-weight: 600;
        }
        tr {
            background-color: #0d1117;
            border-top: 1px solid #30363d;
        }
        tr:nth-child(2n) {
            background-color: #161b22;
        }

        /* Images */
        img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            margin: 16px 0;
        }

        /* Links */
        a {
            color: #58a6ff;
            text-decoration: none;
        }
        a:hover {
            color: #79c0ff;
            text-decoration: underline;
        }

        /* Horizontal rules */
        hr {
            height: 0.25em;
            padding: 0;
            margin: 24px 0;
            background-color: #30363d;
            border: 0;
            border-radius: 2px;
        }

        /* Callout boxes inspired by VS Code extensions */
        .tip, .note, .warning, .danger, .info {
            padding: 16px;
            margin: 16px 0;
            border-radius: 8px;
            border-left: 4px solid;
            font-size: 14px;
            position: relative;
            overflow: hidden;
        }
        .tip {
            background-color: rgba(46, 160, 67, 0.15);
            border-left-color: #2ea043;
            color: #4ac26b;
        }
        .tip::before {
            content: "💡 ";
            font-weight: 600;
        }
        .note, .info {
            background-color: rgba(31, 111, 235, 0.15);
            border-left-color: #1f6feb;
            color: #58a6ff;
        }
        .note::before, .info::before {
            content: "ℹ️ ";
            font-weight: 600;
        }
        .warning {
            background-color: rgba(187, 128, 9, 0.15);
            border-left-color: #bb8009;
            color: #f2cc60;
        }
        .warning::before {
            content: "⚠️ ";
            font-weight: 600;
        }
        .danger {
            background-color: rgba(218, 54, 51, 0.15);
            border-left-color: #da3633;
            color: #f85149;
        }
        .danger::before {
            content: "🚨 ";
            font-weight: 600;
        }

        /* Keyboard shortcuts styling */
        kbd {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            box-shadow: inset 0 -1px 0 #30363d;
            color: #f0f6fc;
            font-family: 'Segoe UI', system-ui, monospace;
            font-size: 11px;
            padding: 3px 5px;
            vertical-align: middle;
        }

        /* Task lists */
        .task-list-item {
            list-style-type: none;
            margin-left: -1.6em;
        }
        .task-list-item-checkbox {
            margin: 0 0.2em 0.25em -1.6em;
            vertical-align: middle;
        }

        /* Emphasis */
        strong {
            font-weight: 600;
            color: #ffffff;
        }
        em {
            font-style: italic;
            color: #f0f6fc;
        }

        /* Selection */
        ::selection {
            background-color: #264f78;
            color: #ffffff;
        }

        /* Scrollbars for webkit browsers */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        ::-webkit-scrollbar-thumb {
            background: #424242;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4f4f4f;
        }
        """

    def _load_documentation_files(self):
        """Load all markdown files from docs directory."""
        self.nav_tree.clear()

        # Ensure docs directory exists
        if not self.docs_path.exists():
            self.docs_path.mkdir(parents=True, exist_ok=True)

        # Load all markdown files
        md_files = sorted(self.docs_path.glob("*.md"))

        for md_file in md_files:
            # Format the filename nicely
            display_name = md_file.stem.replace("_", " ").title()
            item = QTreeWidgetItem(self.nav_tree, [display_name])
            item.setData(0, Qt.UserRole, md_file)

        # Select first item if available
        if self.nav_tree.topLevelItemCount() > 0:
            first_item = self.nav_tree.topLevelItem(0)
            self.nav_tree.setCurrentItem(first_item)
            self._load_file(first_item.data(0, Qt.UserRole))

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item click."""
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            self._load_file(file_path)

    def _load_file(self, file_path: Path):
        """Load and display a markdown file."""
        try:
            # Read markdown content
            content = file_path.read_text(encoding='utf-8')

            # Convert markdown to HTML
            html = self._markdown_to_html(content, file_path.parent)

            # Set content in viewer
            self.viewer.setHtml(html)

        except Exception as e:
            self.viewer.setHtml(f"<h2>Error Loading File</h2><p>Could not load {file_path.name}: {e}</p>")

    def _markdown_to_html(self, markdown_text: str, base_path: Path) -> str:
        """Convert markdown to HTML with image support."""
        html = markdown_text

        # Convert headers
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Convert bold and italic
        html = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', html)
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
        html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)

        # Convert inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Convert code blocks
        html = re.sub(r'```python\n(.*?)\n```', r'<pre>\1</pre>', html, flags=re.DOTALL)
        html = re.sub(r'```\n(.*?)\n```', r'<pre>\1</pre>', html, flags=re.DOTALL)

        # Convert images with support for local files and GIFs
        def replace_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)

            # Handle remote URLs
            if img_path.startswith(('http://', 'https://')):
                return f'<img src="{img_path}" alt="{alt_text}"/>'

            # Handle local files
            full_path = base_path / img_path
            if full_path.exists():
                try:
                    # Get MIME type for proper handling
                    mime_type, _ = mimetypes.guess_type(str(full_path))
                    if mime_type is None:
                        # Default to common image types
                        if full_path.suffix.lower() in ['.gif']:
                            mime_type = 'image/gif'
                        elif full_path.suffix.lower() in ['.png']:
                            mime_type = 'image/png'
                        elif full_path.suffix.lower() in ['.jpg', '.jpeg']:
                            mime_type = 'image/jpeg'
                        else:
                            mime_type = 'image/png'  # Default fallback

                    # Read and encode image as base64 data URL for better compatibility
                    with open(full_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        data_url = f'data:{mime_type};base64,{img_base64}'
                        return f'<img src="{data_url}" alt="{alt_text}"/>'

                except Exception:
                    # Fallback to file path if base64 encoding fails
                    file_url = full_path.as_uri()
                    return f'<img src="{file_url}" alt="{alt_text}"/>'

            # If file doesn't exist, return broken image placeholder
            return f'<img alt="{alt_text} (file not found: {img_path})"/>'

        html = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, html)

        # Convert links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

        # Convert lists
        html = re.sub(r'^\* (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>', html, flags=re.MULTILINE)

        # Convert numbered lists
        html = re.sub(r'^\d+\. (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Convert blockquotes
        html = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

        # Convert special blocks (tip, note, warning)
        html = re.sub(r'!!! tip(.*?)\n((?:    .*\n)*)',
                     r'<div class="tip"><b>Tip:</b>\1\n\2</div>', html, flags=re.MULTILINE)
        html = re.sub(r'!!! note(.*?)\n((?:    .*\n)*)',
                     r'<div class="note"><b>Note:</b>\1\n\2</div>', html, flags=re.MULTILINE)
        html = re.sub(r'!!! warning(.*?)\n((?:    .*\n)*)',
                     r'<div class="warning"><b>Warning:</b>\1\n\2</div>', html, flags=re.MULTILINE)

        # Convert tables
        lines = html.split('\n')
        table_html = []
        in_table = False

        for line in lines:
            if '|' in line and not in_table:
                in_table = True
                table_html.append('<table>')
                # Parse header row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                table_html.append('<tr>')
                for cell in cells:
                    table_html.append(f'<th>{cell}</th>')
                table_html.append('</tr>')
            elif '|' in line and in_table:
                if '---' in line:
                    continue  # Skip separator row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                table_html.append('<tr>')
                for cell in cells:
                    table_html.append(f'<td>{cell}</td>')
                table_html.append('</tr>')
            elif in_table:
                in_table = False
                table_html.append('</table>')
                table_html.append(line)
            else:
                table_html.append(line)

        if in_table:
            table_html.append('</table>')

        html = '\n'.join(table_html)

        # Convert line breaks
        html = html.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'

        return f'<html><body>{html}</body></html>'