"""Two-step wizard for importing DANDI NWB sessions as ethograph projects.

Step 1: Enter dandiset ID → browse and select NWB files.
Step 2: Choose output folder → download via ``dandi download``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.make_pretty import styled_link
from ethograph.gui.notify import notify_dialog

logger = logging.getLogger(__name__)


_EXAMPLE_DANDISETS = [
    ("IBL Brainwide Map", "000409"),
    ("Neuropixels + 3D pose (DANNCE)", "001771"),
]

_SORT_VALUE_ROLE = Qt.UserRole + 1


def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


class _NumericTableItem(QTableWidgetItem):
    def __lt__(self, other):
        my_val = self.data(_SORT_VALUE_ROLE)
        other_val = other.data(_SORT_VALUE_ROLE)
        if my_val is not None and other_val is not None:
            return my_val < other_val
        return super().__lt__(other)


# =====================================================================
# Page 0: Dandiset browser
# =====================================================================


class _DandisetBrowserPage(QWidget):
    """Enter a dandiset ID and browse its NWB files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 1 of 2 — Select NWB files from DANDI</b>"))
        layout.addSpacing(4)

        # Dandiset ID input
        input_row = QHBoxLayout()
        form = QFormLayout()
        self.dandiset_edit = QLineEdit()
        self.dandiset_edit.setPlaceholderText("e.g. 001771")
        self.dandiset_edit.returnPressed.connect(self._on_fetch)
        form.addRow("Dandiset ID:", self.dandiset_edit)
        input_row.addLayout(form, stretch=1)

        self._fetch_btn = QPushButton("Fetch")
        self._fetch_btn.clicked.connect(self._on_fetch)
        input_row.addWidget(self._fetch_btn)
        layout.addLayout(input_row)

        # Examples
        examples_row = QHBoxLayout()
        examples_row.addWidget(QLabel("Examples:"))
        for name, did in _EXAMPLE_DANDISETS:
            btn = QPushButton(f"{name} ({did})")
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("text-decoration: underline; color: #5599ff;")
            btn.clicked.connect(lambda _, d=did: self._fill_and_fetch(d))
            examples_row.addWidget(btn)
        examples_row.addStretch()
        layout.addLayout(examples_row)

        links = QLabel(
            "Browse datasets on "
            + styled_link("https://dandiarchive.org/", "DANDI Archive")
            + " · "
            + styled_link("https://neurosift.app/", "Neurosift")
        )
        links.setOpenExternalLinks(True)
        layout.addWidget(links)

        layout.addSpacing(4)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status)

        # Assets table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Path", "Session", "Size", "Asset ID"])
        self._table.setSelectionMode(QAbstractItemView.MultiSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.setColumnHidden(3, True)
        self._table.setSortingEnabled(True)
        layout.addWidget(self._table, stretch=1)

        # Select all / none
        sel_row = QHBoxLayout()
        sel_all = QPushButton("Select all")
        sel_all.clicked.connect(self._table.selectAll)
        sel_none = QPushButton("Clear selection")
        sel_none.clicked.connect(self._table.clearSelection)
        sel_row.addWidget(sel_all)
        sel_row.addWidget(sel_none)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        self._assets: list[dict] = []

    def _fill_and_fetch(self, dandiset_id: str):
        self.dandiset_edit.setText(dandiset_id)
        self._on_fetch()

    def _on_fetch(self):
        dandiset_id = self.dandiset_edit.text().strip()
        if not dandiset_id:
            return

        self._fetch_btn.setEnabled(False)
        self._status.setText(f"Fetching assets from dandiset {dandiset_id}...")
        self._table.setRowCount(0)
        self._assets.clear()

        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            from dandi.dandiapi import DandiAPIClient

            with DandiAPIClient() as client:
                dandiset = client.get_dandiset(dandiset_id, "draft")
                all_assets = list(dandiset.get_assets())

            nwb_assets = [a for a in all_assets if a.path.endswith(".nwb")]

            if not nwb_assets:
                self._status.setText(f"No NWB files found in dandiset {dandiset_id} ({len(all_assets)} total assets).")
                return

            self._populate_table(nwb_assets)
            self._status.setText(
                f"{len(nwb_assets)} NWB files in dandiset {dandiset_id} ({len(all_assets)} total assets)"
            )
        except Exception as exc:
            s = str(exc).lower()
            if any(
                kw in s
                for kw in (
                    "getaddrinfo failed",
                    "failed to resolve",
                    "max retries exceeded",
                    "name or service not known",
                )
            ):
                self._status.setText("No internet connection or DANDI unreachable.")
            else:
                self._status.setText(f"Error: {exc}")
                logger.exception("Failed to fetch dandiset %s", dandiset_id)
        finally:
            self._fetch_btn.setEnabled(True)

    def _populate_table(self, assets):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(assets))

        for row, asset in enumerate(assets):
            path = asset.path
            # Session = directory prefix (e.g. "sub-X/sub-X_ses-Y")
            parts = path.split("/")
            session = "/".join(parts[:-1]) if len(parts) > 1 else ""
            size = getattr(asset, "size", 0) or 0

            path_item = QTableWidgetItem(path)
            path_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            session_item = QTableWidgetItem(session)
            session_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            size_item = _NumericTableItem(format_file_size(size))
            size_item.setData(_SORT_VALUE_ROLE, float(size))
            size_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            id_item = QTableWidgetItem(asset.identifier)
            id_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            self._table.setItem(row, 0, path_item)
            self._table.setItem(row, 1, session_item)
            self._table.setItem(row, 2, size_item)
            self._table.setItem(row, 3, id_item)

            self._assets.append(
                {
                    "path": path,
                    "session": session,
                    "asset_id": asset.identifier,
                    "size": size,
                }
            )

        self._table.setSortingEnabled(True)

    def get_dandiset_id(self) -> str:
        return self.dandiset_edit.text().strip()

    def get_selected_paths(self) -> list[str]:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        return [self._table.item(r, 0).text() for r in rows]

    def validate(self) -> str | None:
        if not self.get_dandiset_id():
            return "Please enter a Dandiset ID and click Fetch."
        paths = self.get_selected_paths()
        if not paths:
            return "Please select at least one NWB file."
        return None


# =====================================================================
# Page 1: Output folder + download
# =====================================================================


class _DownloadPage(QWidget):
    """Choose output folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 2 of 2 — Download</b>"))
        layout.addSpacing(8)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        layout.addSpacing(8)

        # Output folder
        dir_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select download folder...")
        self.output_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(QLabel("Download to:"))
        dir_row.addWidget(self.output_edit, stretch=1)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        self._info = QLabel(
            "Files will be downloaded using <code>dandi download</code>. Existing files are skipped automatically."
        )
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #888; padding-top: 8px;")
        layout.addWidget(self._info)

        layout.addStretch()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            self.output_edit.setText(folder)

    def set_summary(self, dandiset_id: str, paths: list[str]):
        self._summary.setText(f"<b>{len(paths)}</b> NWB file(s) from dandiset <b>{dandiset_id}</b> will be downloaded.")
        default_dir = Path.home() / ".ethograph" / "dandi" / dandiset_id
        if not self.output_edit.text():
            self.output_edit.setText(str(default_dir))

    def validate(self) -> str | None:
        if not self.output_edit.text():
            return "Please select a download folder."
        return None


# =====================================================================
# Download worker thread
# =====================================================================


class _DownloadWorker(QThread):
    """Run ``dandi download`` in a background thread."""

    progress = Signal(str)
    finished_ok = Signal(list)
    finished_error = Signal(str)

    def __init__(self, dandiset_id: str, paths: list[str], output_dir: str):
        super().__init__()
        self.dandiset_id = dandiset_id
        self.paths = paths
        self.output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from dandi.download import DownloadExisting, download

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        urls = [f"dandi://dandi/{self.dandiset_id}@draft/{p}" for p in self.paths]
        self.progress.emit(f"Downloading {len(urls)} file(s) via dandi...")

        try:
            download(
                urls,
                output_dir,
                existing=DownloadExisting.SKIP,
                jobs=6,
            )
        except Exception as exc:
            self.finished_error.emit(f"Download failed:\n{exc}")
            return

        downloaded = [str(output_dir / Path(p).name) for p in self.paths]
        self.finished_ok.emit(downloaded)


# =====================================================================
# Main wizard dialog
# =====================================================================


class NWBImportDialog(QDialog):
    """Two-step wizard: browse DANDI → download NWB files."""

    def __init__(self, app_state, io_widget, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Import from DANDI")
        self.setMinimumWidth(750)
        self.setMinimumHeight(550)

        self._worker: _DownloadWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_browse = _DandisetBrowserPage()
        self._page_download = _DownloadPage()
        self._stack.addWidget(self._page_browse)
        self._stack.addWidget(self._page_download)
        layout.addWidget(self._stack)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        self._prev_btn.setEnabled(False)

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)

        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        nav.addWidget(cancel_btn)
        layout.addLayout(nav)

    def _on_previous(self):
        page = self._stack.currentIndex()
        if page > 0:
            self._stack.setCurrentIndex(page - 1)
            self._update_nav()

    def _on_next(self):
        page = self._stack.currentIndex()
        if page == 0:
            err = self._page_browse.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._page_download.set_summary(
                self._page_browse.get_dandiset_id(),
                self._page_browse.get_selected_paths(),
            )
            self._stack.setCurrentIndex(1)
            self._update_nav()
        elif page == 1:
            err = self._page_download.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._start_download()

    def _update_nav(self):
        page = self._stack.currentIndex()
        self._prev_btn.setEnabled(page > 0)
        self._next_btn.setText("Download" if page == 1 else "Next →")

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        self.reject()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _start_download(self):
        dandiset_id = self._page_browse.get_dandiset_id()
        paths = self._page_browse.get_selected_paths()
        output_dir = self._page_download.output_edit.text()

        self._next_btn.setEnabled(False)
        self._prev_btn.setEnabled(False)
        self._next_btn.setText("Downloading...")

        self._worker = _DownloadWorker(dandiset_id, paths, output_dir)
        self._worker.progress.connect(self._on_download_progress)
        self._worker.finished_ok.connect(self._on_download_ok)
        self._worker.finished_error.connect(self._on_download_error)
        self._worker.start()

    def _on_download_progress(self, msg: str):
        self._next_btn.setText(msg)

    def _on_download_ok(self, downloaded_paths: list[str]):
        self._worker = None
        if not downloaded_paths:
            notify_dialog("No files were downloaded.", "warning", "Download", self)
            self._update_nav()
            return

        # Load the first NWB file into the GUI
        first_path = downloaded_paths[0]
        output_dir = self._page_download.output_edit.text()

        self.app_state.nc_file_path = first_path
        self.io_widget.nc_file_path_edit.setText(first_path)
        self.app_state.video_folder = output_dir
        self.io_widget.video_folder_edit.setText(output_dir)

        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, self.io_widget._on_load_clicked)

        n = len(downloaded_paths)
        QMessageBox.information(
            self,
            "Download complete",
            f"Downloaded {n} file(s). Loading {Path(first_path).name}...",
        )
        self.accept()

    def _on_download_error(self, msg: str):
        self._worker = None
        notify_dialog(msg, "error", "Download error", self)
        self._update_nav()
        self._next_btn.setEnabled(True)
        self._prev_btn.setEnabled(True)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().closeEvent(event)
