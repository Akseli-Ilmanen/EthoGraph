"""Multi-step wizard for importing DANDI NWB sessions as ethograph projects."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import xarray as xr
from dandi.dandiapi import DandiAPIClient

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.make_pretty import styled_link
from ethograph.gui.notify import notify_dialog
from ethograph.gui.wizard_multi_timeline import draw_session_timeline
from ethograph.labels.converters import NWBLabelConverter, write_mapping_file
from ethograph.labels.tsv_store import init_empty_labels, save_labels_tsv
from ethograph.io.nwb_import import (
    probe_behavioral_series,
    probe_electrical_series,
    probe_label_sources,
    read_trials_table,
)
from ethograph.utils.dandi import (
    download_clip,
    find_video_assets,
    format_file_size,
)
from ethograph.utils.nwb_video import (
    NWBDANDIPoseEstimationWidget,
    probe_dandi_video_metadata,
    stream_video_in_browser,
)

logger = logging.getLogger(__name__)


def _network_error_message(error: Exception) -> str | None:
    s = str(error).lower()
    if any(kw in s for kw in ("getaddrinfo failed", "failed to resolve", "max retries exceeded", "nodename nor servname", "name or service not known")):
        return "No internet connection or the DANDI archive is unreachable.\n\nPlease check your network and try again."
    return None


_SORT_VALUE_ROLE = Qt.UserRole + 1


class _NumericTableItem(QTableWidgetItem):
    def __lt__(self, other):
        my_val = self.data(_SORT_VALUE_ROLE)
        other_val = other.data(_SORT_VALUE_ROLE)
        if my_val is not None and other_val is not None:
            return my_val < other_val
        return super().__lt__(other)


# =====================================================================
# Page 0: Source selection
# =====================================================================

class _SourcePage(QWidget):
    """Page 0: Enter DANDI dataset + session identifiers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 1 of 4 — DANDI Source</b>"))
        layout.addSpacing(8)

        dandi_info = QLabel(
            "Enter the Dandiset ID and the Session EID to find all NWB files "
            "and videos for that recording session.<br><br>"
            "<b>Dataset ID</b>: 6-digit number (e.g. 000409)<br>"
            "<b>Session EID</b>: UUID identifying the recording session<br>"
            "&nbsp;&nbsp;(e.g. 64e3fb86-928c-4079-865c-b364205b502e)"
        )
        dandi_info.setWordWrap(True)
        layout.addWidget(dandi_info)

        dandi_form = QFormLayout()

        self.dandiset_edit = QLineEdit()
        self.dandiset_edit.setPlaceholderText("e.g. 000409")
        dandi_form.addRow("Dataset ID:", self.dandiset_edit)

        self.session_eid_edit = QLineEdit()
        self.session_eid_edit.setPlaceholderText("e.g. 64e3fb86-928c-4079-865c-b364205b502e")
        dandi_form.addRow("Session EID:", self.session_eid_edit)

        example_btn = QPushButton("Use example (dandiset 000409)")
        example_btn.clicked.connect(self._fill_example)
        dandi_form.addRow(example_btn)

        layout.addLayout(dandi_form)

        links = QLabel(
            'Browse datasets on '
            + styled_link("https://dandiarchive.org/", "DANDI Archive")
            + ' · '
            + styled_link("https://neurosift.app/", "Neurosift")
        )
        links.setOpenExternalLinks(True)
        links.setAlignment(Qt.AlignCenter)
        layout.addWidget(links)

        layout.addStretch()

    def _fill_example(self):
        self.dandiset_edit.setText("000409")
        self.session_eid_edit.setText("64e3fb86-928c-4079-865c-b364205b502e")

    def get_source(self) -> dict:
        return {
            "dandiset_id": self.dandiset_edit.text().strip(),
            "session_eid": self.session_eid_edit.text().strip(),
        }

    def validate(self) -> str | None:
        s = self.get_source()
        if not s["dandiset_id"] or not s["session_eid"]:
            return "Please provide both a Dataset ID and a Session EID."
        return None


# =====================================================================
# Page 1: Video / Pose matching + data options
# =====================================================================

class _VideoPosePage(QWidget):
    """Page 1: Video/pose matching, behavioral series, labels, ephys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 2 of 4 — Video & Data Options</b>"))
        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)

        # --- Session NWB files (shown for DANDI sessions) ---
        self._nwb_files_group = QGroupBox("Session NWB files")
        nfg = QVBoxLayout(self._nwb_files_group)
        self._nwb_file_widgets: list[dict] = []
        self._dandi_video_info: dict[str, dict] = {}
        self._video_check_widgets: list[QCheckBox] = []
        self._video_table: QTableWidget | None = None
        self._video_section: QWidget | None = None
        self._nwb_files_layout = QVBoxLayout()
        nfg.addLayout(self._nwb_files_layout)
        self._nwb_files_group.hide()
        self._inner_layout.addWidget(self._nwb_files_group)

        # --- Video download (DANDI) ---
        self._video_group = QGroupBox("Video")
        vg = QVBoxLayout(self._video_group)
        self._video_checkbox = QCheckBox("Download video clips locally (recommended for fast navigation)")
        self._video_checkbox.setChecked(True)
        self._video_checkbox.toggled.connect(self._on_video_toggled)
        vg.addWidget(self._video_checkbox)
        self._stream_note = QLabel(
            "Unchecked = stream from DANDI. Playback works but seeking/jumping "
            "may be slow."
        )
        self._stream_note.setWordWrap(True)
        self._stream_note.setStyleSheet("color: #888; font-style: italic; margin-left: 20px;")
        self._stream_note.setVisible(False)
        vg.addWidget(self._stream_note)
        self._download_row = QWidget()
        dir_row = QHBoxLayout(self._download_row)
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(QLabel("Download folder:"))
        self.download_dir_edit = QLineEdit()
        self.download_dir_edit.setPlaceholderText("Select folder to save video clips...")
        self.download_dir_edit.setReadOnly(True)
        dir_btn = QPushButton("Browse")
        dir_btn.clicked.connect(self._browse_download_dir)
        dir_row.addWidget(self.download_dir_edit)
        dir_row.addWidget(dir_btn)
        vg.addWidget(self._download_row)
        self._video_group.hide()
        self._inner_layout.addWidget(self._video_group)

        # --- Pose estimation ---
        pose_group = QGroupBox("Pose estimation")
        pg_layout = QVBoxLayout(pose_group)
        self.pose_checkbox = QCheckBox("Include pose estimation data")
        self.pose_checkbox.setChecked(True)
        pg_layout.addWidget(self.pose_checkbox)
        self._pose_label = QLabel("")
        self._pose_label.setWordWrap(True)
        pg_layout.addWidget(self._pose_label)
        self._inner_layout.addWidget(pose_group)

        # --- Behavioral time series ---
        self._behavior_group = QGroupBox("Behavioral time series")
        bg = QVBoxLayout(self._behavior_group)
        bg.addWidget(QLabel("Select series to include as features:"))
        self._behavior_checkboxes: list[QCheckBox] = []
        self._behavior_cb_layout = QVBoxLayout()
        bg.addLayout(self._behavior_cb_layout)
        self._behavior_group.hide()
        self._inner_layout.addWidget(self._behavior_group)

        # --- Behavioral labels ---
        self._labels_group = QGroupBox("Behavioral labels")
        lg2 = QVBoxLayout(self._labels_group)
        lg2.addWidget(QLabel("Select label source to import (or none):"))
        self._label_radios: list[QRadioButton] = []
        self._label_radio_layout = QVBoxLayout()
        lg2.addLayout(self._label_radio_layout)
        self._labels_group.hide()
        self._inner_layout.addWidget(self._labels_group)

        # --- Electrophysiology ---
        self._ephys_group = QGroupBox("Electrophysiology")
        eg = QVBoxLayout(self._ephys_group)
        eg.addWidget(QLabel("Select ElectricalSeries to link for ephys viewing (or none):"))
        self._ephys_radios: list[QRadioButton] = []
        self._ephys_radio_layout = QVBoxLayout()
        eg.addLayout(self._ephys_radio_layout)
        self._ephys_group.hide()
        self._inner_layout.addWidget(self._ephys_group)

        # --- Info note ---
        self._info_note = QLabel(
            "Note: A lightweight project file (.nc) will be created with trial "
            "metadata only. Pose estimation and behavioral time series are "
            "loaded lazily from the NWB file at runtime (via pynapple). "
            "Videos and electrophysiology can be streamed or downloaded."
        )
        self._info_note.setWordWrap(True)
        self._info_note.hide()
        self._inner_layout.addWidget(self._info_note)

        self._inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

    # -- Callbacks --

    def _on_video_toggled(self, checked: bool):
        self._download_row.setVisible(checked)
        self._stream_note.setVisible(not checked)

    def _browse_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            self.download_dir_edit.setText(folder)

    # -- Populate --

    def populate(
        self,
        nwb,
        cameras_with_pose: list[str],
        video_info: dict[str, dict],
        behavioral_series: list[dict],
        label_sources: list[dict],
        electrical_series: list[dict] | None = None,
    ) -> None:
        if video_info:
            self._download_row.setVisible(self._video_checkbox.isChecked())
            self._stream_note.setVisible(not self._video_checkbox.isChecked())
            self._video_group.show()

        # Pose
        if cameras_with_pose:
            self._pose_label.setText(f"Detected cameras: {', '.join(cameras_with_pose)}")
        else:
            self.pose_checkbox.setEnabled(False)
            self.pose_checkbox.setChecked(False)
            self._pose_label.setText("No pose estimation interfaces found.")

        # Behavioral series checkboxes
        for cb in self._behavior_checkboxes:
            self._behavior_cb_layout.removeWidget(cb)
            cb.deleteLater()
        self._behavior_checkboxes.clear()

        if behavioral_series:
            for entry in behavioral_series:
                cb = QCheckBox(f"{entry['source']}  ({entry['n_samples']:,} samples)")
                cb._source = entry["source"]
                cb.setChecked(True)
                self._behavior_cb_layout.addWidget(cb)
                self._behavior_checkboxes.append(cb)
            self._behavior_group.show()
        else:
            self._behavior_group.hide()

        # Label source radio buttons
        for rb in self._label_radios:
            self._label_radio_layout.removeWidget(rb)
            rb.deleteLater()
        self._label_radios.clear()

        if label_sources:
            none_rb = QRadioButton("None")
            none_rb._source = None
            none_rb.setChecked(True)
            self._label_radio_layout.addWidget(none_rb)
            self._label_radios.append(none_rb)
            for entry in label_sources:
                rb = QRadioButton(entry["description"])
                rb._source = entry["source"]
                self._label_radio_layout.addWidget(rb)
                self._label_radios.append(rb)
            self._labels_group.show()
        else:
            self._labels_group.hide()

        # Electrophysiology
        for rb in self._ephys_radios:
            self._ephys_radio_layout.removeWidget(rb)
            rb.deleteLater()
        self._ephys_radios.clear()

        if electrical_series:
            none_rb = QRadioButton("None")
            none_rb._series_name = None
            none_rb.setChecked(True)
            self._ephys_radio_layout.addWidget(none_rb)
            self._ephys_radios.append(none_rb)
            for entry in electrical_series:
                rate_str = f"{entry['rate']:.0f} Hz" if entry["rate"] else "unknown rate"
                label = f"{entry['name']}  ({entry['n_channels']} ch, {rate_str}, {entry['n_samples']:,} samples)"
                rb = QRadioButton(label)
                rb._series_name = entry["name"]
                self._ephys_radio_layout.addWidget(rb)
                self._ephys_radios.append(rb)
            self._ephys_group.show()
        else:
            self._ephys_group.hide()

    # -- Session NWB overview (DANDI) --

    def populate_session_overview(self, session_assets: dict) -> None:
        for entry in self._nwb_file_widgets:
            entry["widget"].deleteLater()
        self._nwb_file_widgets.clear()

        nwb_assets = [a for a in (session_assets["raw"], session_assets["processed"]) if a is not None]

        if not nwb_assets:
            self._nwb_files_group.hide()
            self._info_note.hide()
            return

        for asset in nwb_assets:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)

            is_processed = "desc-processed" in asset.path
            is_raw = "desc-raw" in asset.path
            cb = QCheckBox()
            cb.setChecked(is_processed or len(nwb_assets) == 1)
            row_layout.addWidget(cb)

            filename = Path(asset.path).name
            size = format_file_size(asset.size)
            type_tag = "raw" if is_raw else ("processed" if is_processed else "")
            tag_str = f" ({type_tag}, {size})" if type_tag else f" ({size})"
            label = QLabel(f"<b>{filename}</b>{tag_str}")
            label.setWordWrap(True)
            row_layout.addWidget(label, stretch=1)

            rb_stream = QRadioButton("Stream")
            rb_download = QRadioButton("Download")
            rb_stream.setChecked(True)
            bg = QButtonGroup(row)
            bg.addButton(rb_stream)
            bg.addButton(rb_download)
            row_layout.addWidget(rb_stream)
            row_layout.addWidget(rb_download)

            self._nwb_files_layout.addWidget(row)
            self._nwb_file_widgets.append({
                "asset": asset,
                "widget": row,
                "checkbox": cb,
                "stream_radio": rb_stream,
                "download_radio": rb_download,
            })

        self._nwb_files_group.show()
        self._info_note.show()

    # -- Video assets display --

    def populate_videos(self, video_info: dict[str, dict]) -> None:
        if self._video_section is not None:
            self._video_section.deleteLater()
            self._video_section = None
        self._video_check_widgets.clear()
        self._video_table = None
        self._dandi_video_info = video_info
        if video_info:
            self._build_video_section(video_info)
            self._nwb_files_group.show()

    def _build_video_section(self, video_info: dict[str, dict]) -> None:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(QLabel(f"<b>Video files ({len(video_info)})</b>"))

        if len(video_info) < 5:
            for name, info in video_info.items():
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 2, 0, 2)

                cb = QCheckBox()
                cb.setChecked(True)
                cb._video_name = name
                row_layout.addWidget(cb)
                self._video_check_widgets.append(cb)

                parts = [f"<b>{name}</b>"]
                dur = self._format_duration(info)
                if dur is not None:
                    parts.append(f"{dur / 60:.1f} min" if dur > 60 else f"{dur:.1f} s")
                if "fps" in info:
                    parts.append(f"{info['fps']:.0f} fps")
                if "width" in info and "height" in info:
                    parts.append(f"{info['width']}\u00d7{info['height']}")
                size_bytes = info.get("size_bytes")
                if size_bytes is not None:
                    parts.append(format_file_size(size_bytes))

                label = QLabel(" \u00b7 ".join(parts))
                label.setWordWrap(True)
                row_layout.addWidget(label, stretch=1)

                url = info.get("url", "")
                if url:
                    stream_btn = QPushButton("Stream \u25b6")
                    stream_btn.setFixedWidth(90)
                    stream_btn.clicked.connect(lambda checked, u=url, n=name: stream_video_in_browser(u, n))
                    row_layout.addWidget(stream_btn)

                layout.addWidget(row)
        else:
            btn_row = QHBoxLayout()
            select_all_btn = QPushButton("Select all")
            unselect_all_btn = QPushButton("Unselect all")
            select_all_btn.clicked.connect(lambda: self._set_all_video_selected(True))
            unselect_all_btn.clicked.connect(lambda: self._set_all_video_selected(False))
            btn_row.addWidget(select_all_btn)
            btn_row.addWidget(unselect_all_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            table = QTableWidget(len(video_info), 6)
            table.setHorizontalHeaderLabels(["Name", "Duration", "FPS", "Resolution", "File size", ""])
            table.setSelectionMode(QAbstractItemView.MultiSelection)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.setSortingEnabled(False)

            for r, (name, info) in enumerate(video_info.items()):
                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.UserRole, r)
                name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, 0, name_item)

                dur = self._format_duration(info)
                dur_text = f"{dur / 60:.1f} min" if dur and dur > 60 else (f"{dur:.1f} s" if dur else "--")
                dur_item = _NumericTableItem(dur_text)
                if dur:
                    dur_item.setData(_SORT_VALUE_ROLE, dur)
                dur_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, 1, dur_item)

                fps = info.get("fps")
                fps_item = QTableWidgetItem(f"{fps:.0f}" if fps else "--")
                fps_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, 2, fps_item)

                w, h = info.get("width"), info.get("height")
                res_item = QTableWidgetItem(f"{w}\u00d7{h}" if w and h else "--")
                res_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, 3, res_item)

                size_bytes = info.get("size_bytes")
                size_text = format_file_size(size_bytes) if size_bytes else "--"
                size_item = _NumericTableItem(size_text)
                if size_bytes:
                    size_item.setData(_SORT_VALUE_ROLE, float(size_bytes))
                size_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, 4, size_item)

                url = info.get("url", "")
                if url:
                    stream_btn = QPushButton("Stream \u25b6")
                    stream_btn.clicked.connect(lambda checked, u=url, n=name: stream_video_in_browser(u, n))
                    table.setCellWidget(r, 5, stream_btn)

            table.setSortingEnabled(True)
            table.selectAll()
            self._video_table = table
            layout.addWidget(table)

        self._video_section = section
        self._nwb_files_layout.addWidget(section)
        self._nwb_file_widgets.append({"widget": section})

    @staticmethod
    def _format_duration(info: dict) -> float | None:
        dur = info.get("duration_s")
        if dur is None and "start" in info and "end" in info:
            dur = info["end"] - info["start"]
        return dur if dur and dur > 0 else None

    def _set_all_video_selected(self, selected: bool) -> None:
        if self._video_table is not None:
            if selected:
                self._video_table.selectAll()
            else:
                self._video_table.clearSelection()

    # -- Accessors --

    def needs_video_download(self) -> bool:
        return bool(self._dandi_video_info) and self._video_checkbox.isChecked()

    def get_selected_label_source(self) -> str | None:
        for rb in self._label_radios:
            if rb.isChecked():
                return rb._source
        return None

    def get_selected_ephys_series(self) -> str | None:
        for rb in self._ephys_radios:
            if rb.isChecked():
                return rb._series_name
        return None

    def get_selected_video_names(self) -> list[str]:
        if self._video_check_widgets:
            return [cb._video_name for cb in self._video_check_widgets if cb.isChecked()]
        if self._video_table is not None:
            names = list(self._dandi_video_info.keys())
            selected_rows = sorted({idx.row() for idx in self._video_table.selectedIndexes()})
            return [names[self._video_table.item(row, 0).data(Qt.UserRole)] for row in selected_rows]
        return list(self._dandi_video_info.keys())

    def get_raw_nwb_asset(self):
        for entry in self._nwb_file_widgets:
            if "checkbox" not in entry or not entry["checkbox"].isChecked():
                continue
            if "desc-raw" in entry["asset"].path:
                return entry["asset"]
        return None


# =====================================================================
# Page 2: Trial selection
# =====================================================================

class _NWBTrialsPage(QWidget):
    """Page 2: Select which NWB trials to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 3 of 4 — Trial Selection</b>"))
        layout.addSpacing(8)

        self._trial_group = QGroupBox("Trial selection")
        tg = QVBoxLayout(self._trial_group)

        self._rb_all = QRadioButton("All trials")
        self._rb_first_n = QRadioButton("First N trials:")
        self._rb_select = QRadioButton("Select specific trials from table below")
        self._rb_all.setChecked(True)
        rb_grp = QButtonGroup(self)
        for rb in (self._rb_all, self._rb_first_n, self._rb_select):
            rb_grp.addButton(rb)
            tg.addWidget(rb)

        n_row = QHBoxLayout()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 100000)
        self.n_spin.setValue(5)
        n_row.addWidget(QLabel("  N ="))
        n_row.addWidget(self.n_spin)
        n_row.addStretch()
        tg.addLayout(n_row)

        self.trials_table = QTableWidget(0, 0)
        self.trials_table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.trials_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trials_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tg.addWidget(self.trials_table)
        self._rb_select.toggled.connect(self._on_select_toggled)

        layout.addWidget(self._trial_group)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        layout.addWidget(self._summary_label)
        layout.addStretch()

    def _on_select_toggled(self, checked: bool):
        self.trials_table.setSelectionMode(
            QAbstractItemView.MultiSelection if checked else QAbstractItemView.NoSelection
        )

    def populate(self, nwb) -> None:
        total = len(nwb.trials) if nwb.trials is not None and len(nwb.trials) > 0 else 1
        self._trial_group.setVisible(total > 1)

        if nwb.trials is not None and len(nwb.trials) > 0:
            df = nwb.trials.to_dataframe()
            self.trials_table.setSortingEnabled(False)
            self.trials_table.setRowCount(len(df))
            self.trials_table.setColumnCount(len(df.columns))
            self.trials_table.setHorizontalHeaderLabels(list(df.columns))
            for r, (_, row) in enumerate(df.iterrows()):
                for c, val in enumerate(row):
                    if isinstance(val, (int, float)):
                        item = _NumericTableItem(f"{val:.3f}" if isinstance(val, float) else str(val))
                        item.setData(_SORT_VALUE_ROLE, float(val))
                    else:
                        item = QTableWidgetItem(str(val))
                    if c == 0:
                        item.setData(Qt.UserRole, r)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.trials_table.setItem(r, c, item)
            self.trials_table.setSortingEnabled(True)
            self.n_spin.setMaximum(len(df))
            self._summary_label.setText(f"{len(df)} trials available")
        else:
            self._summary_label.setText("No trials table found — single trial will be created.")

    def get_trial_indices(self, total: int) -> list[int]:
        if self._rb_all.isChecked():
            return list(range(total))
        if self._rb_first_n.isChecked():
            return list(range(min(self.n_spin.value(), total)))
        visual_rows = sorted({idx.row() for idx in self.trials_table.selectedIndexes()})
        if not visual_rows:
            return list(range(total))
        return sorted(
            self.trials_table.item(r, 0).data(Qt.UserRole) for r in visual_rows
        )

    def validate(self) -> str | None:
        return None


# =====================================================================
# Page 3: Timeline visualization + output path
# =====================================================================

class _NWBTimelinePage(QWidget):
    """Page 3: Timeline visualization and output path selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 4 of 4 — Timeline & Output</b>"))
        layout.addSpacing(4)

        layout.addWidget(QLabel(
            "Review how video, pose, and behavioral data align across trials. "
            "Colored bars show data time ranges; dotted lines mark trial boundaries."
        ))
        layout.addSpacing(4)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1a1d21")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.getAxis("left").setTicks([])
        layout.addWidget(self._plot, stretch=1)

        # --- Output ---
        out_group = QGroupBox("Output")
        outg = QHBoxLayout(out_group)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Project folder...")
        self.output_edit.setReadOnly(True)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output)
        outg.addWidget(self.output_edit)
        outg.addWidget(out_browse)
        layout.addWidget(out_group)

        self._items: list = []

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select project folder")
        if folder:
            self.output_edit.setText(folder)

    def populate(
        self,
        trials_df,
        video_info: dict[str, dict],
        cameras_with_pose: list[str],
        pose_containers: dict[str, Any] | None,
        behavioral_series: list[dict],
        selected_trial_indices: list[int],
        matching: list[tuple[str, str]] | None = None,
    ) -> None:
        self._clear()

        selected_df = trials_df.iloc[selected_trial_indices].reset_index(drop=True) if trials_df is not None else None

        # Build video→pose mapping; video names are canonical for `cameras` dim.
        pose_to_video = {}
        if matching:
            pose_to_video = {pose_cam: video_cam for video_cam, pose_cam in matching}

        # Build a temporary TrialTree with session data so
        # draw_session_timeline can render from a single source of truth.
        from ethograph import TrialTree
        dt = TrialTree()
        has_behavior = bool(behavioral_series)
        if selected_df is not None and not selected_df.empty:
            for _, row in selected_df.iterrows():
                tid = int(row["trial"]) if "trial" in row.index else row.name
                ds = xr.Dataset(attrs={"trial": tid})
                if has_behavior:
                    ds["_features_placeholder"] = xr.DataArray(0, attrs={"type": "features"})
                dt[str(tid)] = xr.DataTree(ds)

        session_vars: dict[str, Any] = {}
        # TODO. old system, change
        trial_ids = dt.trials if dt.children else []

        if selected_df is not None and "start_time" in selected_df.columns:
            session_vars["start_time"] = ("trial", selected_df["start_time"].astype(float).values)
        if selected_df is not None and "stop_time" in selected_df.columns:
            session_vars["stop_time"] = ("trial", selected_df["stop_time"].astype(float).values)

        camera_names = list(video_info.keys())
        if camera_names:
            session_vars["video"] = xr.DataArray(
                camera_names, dims=["cameras"], coords={"cameras": camera_names},
            )
            start_times = [video_info[c].get("start", 0.0) for c in camera_names]
            session_vars["start_time_video"] = xr.DataArray(
                np.array(start_times, dtype=np.float64),
                dims=["cameras"], coords={"cameras": camera_names},
            )
            fps_values = [video_info[c].get("fps", 0.0) for c in camera_names]
            if any(f > 0 for f in fps_values):
                session_vars["video_fps"] = xr.DataArray(
                    np.array(fps_values, dtype=np.float64),
                    dims=["cameras"], coords={"cameras": camera_names},
                )

        if cameras_with_pose:
            pose_starts = []
            canonical_names = []
            for cam in cameras_with_pose:
                canonical_names.append(pose_to_video.get(cam, cam))
                t = 0.0
                container = (pose_containers or {}).get(cam)
                if container:
                    first_series = next(iter(container.pose_estimation_series.values()), None)
                    if first_series and first_series.timestamps is not None and len(first_series.timestamps) > 0:
                        t = float(first_series.timestamps[0])
                pose_starts.append(t)
            session_vars["pose"] = xr.DataArray(
                canonical_names, dims=["cameras"], coords={"cameras": canonical_names},
            )
            session_vars["start_time_pose"] = xr.DataArray(
                np.array(pose_starts, dtype=np.float64),
                dims=["cameras"], coords={"cameras": canonical_names},
            )

        coords = {"trial": trial_ids} if trial_ids else {}
        sess_ds = xr.Dataset(session_vars, coords=coords)
        dt["session"] = xr.DataTree(sess_ds)

        draw_session_timeline(
            self._plot,
            None,
            items_out=self._items,
        )

    def _clear(self):
        for item in self._items:
            self._plot.removeItem(item)
        self._items.clear()

    def validate(self) -> str | None:
        if not self.output_edit.text():
            return "Please select a project folder."
        return None


# =====================================================================
# Main wizard dialog
# =====================================================================

class NWBImportDialog(QDialog):
    """4-step wizard: NWB source → video/data options → trials → timeline/output."""

    def __init__(self, app_state, io_widget, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Import DANDI session")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._nwb = None
        self._cameras_with_pose: list[str] = []
        self._video_info: dict[str, dict] = {}
        self._session_assets: dict | None = None
        self._output_path: str = ""
        self._behavioral_series: list[dict] = []
        self._trials_df = None
        self._dandi_asset_id: str | None = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_source = _SourcePage()
        self._page_video_pose = _VideoPosePage()
        self._page_trials = _NWBTrialsPage()
        self._page_timeline = _NWBTimelinePage()
        self._stack.addWidget(self._page_source)
        self._stack.addWidget(self._page_video_pose)
        self._stack.addWidget(self._page_trials)
        self._stack.addWidget(self._page_timeline)
        layout.addWidget(self._stack)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("\u2190 Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        self._prev_btn.setEnabled(False)

        self._next_btn = QPushButton("Connect & Preview \u2192")
        self._next_btn.clicked.connect(self._on_next)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        nav.addWidget(cancel_btn)
        layout.addLayout(nav)

    def _current_page(self) -> int:
        return self._stack.currentIndex()

    def _on_previous(self):
        page = self._current_page()
        if page > 0:
            self._stack.setCurrentIndex(page - 1)
            self._update_nav()

    def _on_next(self):
        page = self._current_page()
        if page == 0:
            err = self._page_source.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._connect_to_nwb()
        elif page == 1:
            self._populate_trials_page()
            self._stack.setCurrentIndex(2)
            self._update_nav()
        elif page == 2:
            err = self._page_trials.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._populate_timeline_page()
            self._stack.setCurrentIndex(3)
            self._update_nav()
        elif page == 3:
            err = self._page_timeline.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._load_all()

    def _update_nav(self):
        page = self._current_page()
        self._prev_btn.setEnabled(page > 0)
        if page == 0:
            self._next_btn.setText("Connect & Preview \u2192")
        elif page == 3:
            self._next_btn.setText("Create project")
        else:
            self._next_btn.setText("Next \u2192")

    # ------------------------------------------------------------------
    # Page transitions
    # ------------------------------------------------------------------

    def _connect_to_nwb(self):
        source = self._page_source.get_source()
        dandiset_id = source["dandiset_id"]
        session_eid = source["session_eid"]

        def _open():
            client = DandiAPIClient()
            dandiset = client.get_dandiset(dandiset_id, "draft")
            session_assets = [asset for asset in dandiset.get_assets() if session_eid in asset.path]

            logger.info(
                "\n%s\n  DANDI session assets for %s\n  Found %d file(s) in dandiset %s\n%s",
                "=" * 60, session_eid, len(session_assets), dandiset_id, "=" * 60,
            )
            for asset in session_assets:
                neurosift_url = (
                    f"https://neurosift.app/nwb?url=https://api.dandiarchive.org"
                    f"/api/assets/{asset.identifier}/download/"
                    f"&dandisetId={dandiset_id}&dandisetVersion=draft"
                )
                logger.info("  %s", asset.path)
                logger.info("    %s", neurosift_url)
            logger.info(
                "\nBrowse all data on Neurosift:\n  https://neurosift.app/?dandisetId=%s&dandisetVersion=draft\n",
                dandiset_id,
            )

            raw_asset = next((a for a in session_assets if "desc-raw" in a.path), None)
            processed_asset = next((a for a in session_assets if "desc-processed" in a.path), None)

            widget = NWBDANDIPoseEstimationWidget(
                processed_asset=processed_asset,
                raw_asset=raw_asset,
            )

            nwb = widget.nwbfile
            cameras_with_pose = widget.available_cameras
            video_info = widget.video_info
            dandi_asset_id = (processed_asset or raw_asset).identifier

            # Fallback: discover video assets from DANDI when NWB has
            # no ImageSeries or the URL resolution failed.
            has_usable_urls = any(info.get("url") for info in video_info.values())
            if not has_usable_urls:
                asset_id = processed_asset.identifier if processed_asset else None
                dandi_videos = find_video_assets(dandiset_id, nwb, asset_id=asset_id)
                if dandi_videos:
                    discovered = {stem: url for stem, url in dandi_videos}
                    for name, info in video_info.items():
                        for stem, url in discovered.items():
                            if name.lower() in stem.lower() or stem.lower() in name.lower():
                                info["url"] = url
                                discovered.pop(stem)
                                break
                    for stem, url in discovered.items():
                        video_info[stem] = {"url": url, "start": 0.0, "end": 0.0}

            if video_info:
                with ThreadPoolExecutor(max_workers=max(1, len(video_info))) as pool:
                    futures = {
                        name: pool.submit(probe_dandi_video_metadata, info["url"])
                        for name, info in video_info.items() if info.get("url")
                    }
                    for name, future in futures.items():
                        try:
                            video_info[name].update(future.result(timeout=30))
                        except Exception:
                            pass

            session_assets_dict = {"raw": raw_asset, "processed": processed_asset}
            behavioral = probe_behavioral_series(nwb)
            labels = probe_label_sources(nwb)
            ephys = probe_electrical_series(nwb)
            return nwb, cameras_with_pose, video_info, behavioral, labels, ephys, session_assets_dict, dandi_asset_id

        progress = BusyProgressDialog("Accessing NWB metadata...", parent=self)
        (result, error) = progress.execute(_open)

        if progress.was_cancelled or error:
            if error:
                msg = _network_error_message(error) or f"Failed to open NWB:\n{error}"
                notify_dialog(msg, "error", "Error", self)
            return

        (
            self._nwb, self._cameras_with_pose, self._video_info,
            behavioral, labels, ephys, self._session_assets, self._dandi_asset_id,
        ) = result
        self._behavioral_series = behavioral

        self._page_video_pose.populate(
            self._nwb, self._cameras_with_pose, self._video_info,
            behavioral, labels, ephys,
        )

        if self._session_assets:
            self._page_video_pose.populate_session_overview(self._session_assets)

        if self._video_info:
            self._page_video_pose.populate_videos(self._video_info)

        project_name = self._default_project_dirname()
        default_dir = self._default_download_dir(source)
        if self._video_info:
            self._page_video_pose.download_dir_edit.setText(str(default_dir))
        self._page_timeline.output_edit.setText(str(default_dir / project_name))

        self._stack.setCurrentIndex(1)
        self._update_nav()

    def _populate_trials_page(self):
        self._page_trials.populate(self._nwb)
        self._trials_df = read_trials_table(self._nwb)

    def _populate_timeline_page(self):
        total = len(self._nwb.trials) if self._nwb.trials is not None and len(self._nwb.trials) > 0 else 1
        selected_indices = self._page_trials.get_trial_indices(total)
        self._page_timeline.populate(
            self._trials_df,
            self._video_info,
            self._cameras_with_pose,
            None,
            self._behavioral_series,
            selected_indices,
        )

    # ------------------------------------------------------------------
    # Final load
    # ------------------------------------------------------------------

    def _load_all(self):
        output_path = self._page_timeline.output_edit.text()
        total_trials = len(self._nwb.trials) if self._nwb.trials is not None and len(self._nwb.trials) > 0 else 1
        trial_indices = self._page_trials.get_trial_indices(total_trials)
        include_pose = self._page_video_pose.pose_checkbox.isChecked()
        label_source = self._page_video_pose.get_selected_label_source()
        ephys_series = self._page_video_pose.get_selected_ephys_series()

        output_dir = None
        if self._page_video_pose.needs_video_download():
            download_dir = self._page_video_pose.download_dir_edit.text()
            if not download_dir:
                reply = QMessageBox.question(
                    self, "No download folder",
                    "No download folder selected. Videos will be streamed from DANDI.\n\n"
                    "Streaming works for playback, but seeking and jumping between "
                    "frames may be slow for long recordings or files without trials.\n\n"
                    "Recommendation: Download the video files locally for fast navigation.\n\n"
                    "Continue with streaming?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return
            else:
                output_dir = self._download_videos(download_dir, trial_indices)
                if output_dir is None:
                    return

        source_info = self._page_source.get_source()
        selected_videos = self._page_video_pose.get_selected_video_names()

        def _build():
            import json

            trials_df = read_trials_table(self._nwb)
            if trial_indices is not None:
                trials_df = trials_df.iloc[trial_indices].reset_index(drop=True)

            project_dir = Path(output_path)
            ethograph_dir = project_dir / ".ethograph"
            ethograph_dir.mkdir(parents=True, exist_ok=True)

            # Save dandi.json — source info for re-loading
            # No alignment.nwb needed: the remote NWB is the source of truth
            # Store video URLs for cameras that are separate DANDI assets
            # (not stored in NWB acquisition ImageSeries).
            # NOTE: one URL per camera for the entire session — does not
            # scale to per-trial video files.
            video_urls = {}
            for vname in selected_videos:
                info = self._video_info.get(vname, {})
                if info.get("url"):
                    video_urls[vname] = {
                        "url": info["url"],
                        "start": info.get("start", 0.0),
                        "end": info.get("end", 0.0),
                        "fps": info.get("fps"),
                    }

            config = {
                "nwb_dandiset_id": source_info["dandiset_id"],
                "nwb_asset_id": self._dandi_asset_id,
                "video_info": video_urls,
            }
            if include_pose and self._cameras_with_pose:
                config["nwb_pose_keys"] = list(self._cameras_with_pose)

            config["nwb_ephys_dandiset_id"] = source_info["dandiset_id"]
            config["nwb_ephys_asset_id"] = self._dandi_asset_id

            config_path = ethograph_dir / "dandi.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Labels
            if label_source:
                converter = NWBLabelConverter()
                all_labels_df = converter.from_nwb(self._nwb, trials_df)
                write_mapping_file(ethograph_dir / "mapping.txt", converter.label_map)
            else:
                trial_ids = [str(int(r["trial"])) for _, r in trials_df.iterrows()]
                all_labels_df = init_empty_labels(trial_ids)

            labels_path = project_dir / "labels.tsv"
            save_labels_tsv(labels_path, all_labels_df)

        progress = BusyProgressDialog("Setting up project...", parent=self)
        (_, error) = progress.execute(_build)

        if progress.was_cancelled or error:
            if error:
                notify_dialog(f"Failed to create project:\n{error}", "error", "Error", self)
            return

        # Directly load the project
        self._output_path = output_path
        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if output_dir:
            self.app_state.video_folder = str(output_dir)
            self.io_widget.video_folder_edit.setText(str(output_dir))

        # Trigger loading after the dialog closes
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, self.io_widget._on_load_clicked)
        self.accept()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_path_component(s: str) -> str:
        return re.sub(r'[<>:"/\\|?*\s]+', "_", s).strip("_") or "unknown"

    @staticmethod
    def _default_project_dirname() -> str:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return f"session_{stamp}"

    def _default_download_dir(self, source: dict) -> Path:
        lab = getattr(self._nwb, "lab", "") or "unknown_lab"
        subject_id = getattr(getattr(self._nwb, "subject", None), "subject_id", "") or "unknown_subject"
        session_eid = source.get("session_eid", "unknown_session")
        sanitize = self._sanitize_path_component
        return Path.home() / ".ethograph" / "dandi" / sanitize(lab) / sanitize(subject_id) / sanitize(session_eid)

    def _download_videos(self, download_dir: str, trial_indices: list[int]) -> Path | None:
        output_dir = Path(download_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_info = self._video_info
        selected = self._page_video_pose.get_selected_video_names()
        trials_df = read_trials_table(self._nwb)
        trials_df = trials_df.iloc[trial_indices].reset_index(drop=True)

        def _download():
            for _, row in trials_df.iterrows():
                trial_id = int(row["trial"])
                for video_name in selected:
                    url = video_info.get(video_name, {}).get("url", "")
                    if not url:
                        continue
                    clip_path = output_dir / f"{video_name}_trial_{trial_id}.mp4"
                    if clip_path.exists():
                        continue
                    cam_start = video_info.get(video_name, {}).get("start", 0.0)
                    t_start = float(row["start_time"]) - cam_start
                    t_stop = float(row["stop_time"]) - cam_start
                    download_clip(url, t_start, t_stop, clip_path)

        progress = BusyProgressDialog("Downloading video segments...", parent=self)
        (_, error) = progress.execute(_download)

        if progress.was_cancelled or error:
            if error:
                notify_dialog(f"Download failed:\n{error}", "error", "Error", self)
            return None

        return output_dir

    def closeEvent(self, event):
        super().closeEvent(event)
