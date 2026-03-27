"""Multi-step wizard for importing NWB files (local or DANDI) as trials.nc."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from dandi.dandiapi import DandiAPIClient

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QDesktopServices
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
from ethograph.gui.dialog_pose_video_matcher import PoseVideoMatcherWidget
from ethograph.utils.label_intervals import NWBLabelConverter
from ethograph.utils.nwb import (
    download_clip,
    format_file_size,
    load_nwb_session,
    open_nwb_dandi,
    open_nwb_local,
    probe_behavioral_series,
    probe_electrical_series,
    probe_label_sources,
    read_trials_table,
)
from ethograph.gui.makepretty import styled_link
from ethograph.utils.nwb_video import (
    NWBDANDIPoseEstimationWidget,
    probe_dandi_video_metadata,
    stream_video_in_browser,
)


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


class _SourcePage(QWidget):
    """Page 0: Select local file or DANDI source."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 1 of 2 — Select NWB Source</b>"))
        layout.addSpacing(8)

        source_group = QGroupBox("Source type")
        sg_layout = QVBoxLayout(source_group)
        self._rb_local = QRadioButton("Local .nwb file")
        self._rb_dandi = QRadioButton("DANDI archive (streaming)")
        self._rb_local.setChecked(True)
        self._rb_group = QButtonGroup(self)
        self._rb_group.addButton(self._rb_local)
        self._rb_group.addButton(self._rb_dandi)
        sg_layout.addWidget(self._rb_local)
        sg_layout.addWidget(self._rb_dandi)
        layout.addWidget(source_group)

        self._local_group = QGroupBox("Local NWB file")
        lg = QHBoxLayout(self._local_group)
        self.local_edit = QLineEdit()
        self.local_edit.setPlaceholderText("Path to .nwb file...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_local)
        lg.addWidget(self.local_edit)
        lg.addWidget(browse_btn)
        layout.addWidget(self._local_group)

        self._dandi_group = QGroupBox("DANDI archive")
        dg = QVBoxLayout(self._dandi_group)

        dandi_info = QLabel(
            "Enter the Dandiset ID and the Session EID to find all NWB files "
            "and videos for that recording session.<br><br>"
            "<b>Dataset ID</b>: 6-digit number (e.g. 000409)<br>"
            "<b>Session EID</b>: UUID identifying the recording session<br>"
            "&nbsp;&nbsp;(e.g. 64e3fb86-928c-4079-865c-b364205b502e)"
        )
        dandi_info.setWordWrap(True)
        dg.addWidget(dandi_info)

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

        dg.addLayout(dandi_form)
        self._dandi_group.hide()
        layout.addWidget(self._dandi_group)

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

        self._rb_local.toggled.connect(self._toggle_source)
        self._rb_dandi.toggled.connect(self._toggle_source)

    def _toggle_source(self):
        is_local = self._rb_local.isChecked()
        self._local_group.setVisible(is_local)
        self._dandi_group.setVisible(not is_local)

    def _browse_local(self):
        result = QFileDialog.getOpenFileName(self, "Select NWB file", "", "NWB files (*.nwb);;All files (*)")
        if result and result[0]:
            self.local_edit.setText(result[0])

    def _fill_example(self):
        self.dandiset_edit.setText("000409")
        self.session_eid_edit.setText("64e3fb86-928c-4079-865c-b364205b502e")

    def get_source(self) -> dict:
        if self._rb_local.isChecked():
            return {"type": "local", "path": self.local_edit.text().strip()}
        return {
            "type": "dandi",
            "dandiset_id": self.dandiset_edit.text().strip(),
            "session_eid": self.session_eid_edit.text().strip(),
        }

    def validate(self) -> str | None:
        s = self.get_source()
        if s["type"] == "local":
            if not s["path"]:
                return "Please select a local .nwb file."
            if not os.path.isfile(s["path"]):
                return f"File not found: {s['path']}"
        else:
            if not s["dandiset_id"] or not s["session_eid"]:
                return "Please provide both a Dataset ID and a Session EID."
        return None


class _SelectionPage(QWidget):
    """Page 1: Combined trial selection + video matching + data options + output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 2 of 2 — Configure Import</b>"))
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

        # --- Trial selection ---
        self._trial_group = QGroupBox("Trial selection")
        trial_group = self._trial_group
        tg = QVBoxLayout(trial_group)
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
        self.trials_table.hide()
        tg.addWidget(self.trials_table)
        self._rb_select.toggled.connect(lambda checked: self.trials_table.setVisible(checked))
        self._inner_layout.addWidget(trial_group)

        # --- Video matching (shown only if DANDI videos are detected) ---
        self._video_group = QGroupBox("Video")
        vg = QVBoxLayout(self._video_group)
        self._video_checkbox = QCheckBox("Download video clips (uses ffmpeg to extract trial clips)")
        self._video_checkbox.setChecked(True)
        self._video_checkbox.toggled.connect(self._on_video_toggled)
        vg.addWidget(self._video_checkbox)
        self._video_details = QWidget()
        vd = QVBoxLayout(self._video_details)
        vd.setContentsMargins(0, 0, 0, 0)
        vd.addWidget(QLabel("Match pose cameras (left) to video sources (right). Reorder if needed."))
        self._matcher = PoseVideoMatcherWidget()
        vd.addWidget(self._matcher)
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Download folder:"))
        self.download_dir_edit = QLineEdit()
        self.download_dir_edit.setPlaceholderText("Select folder to save video clips...")
        self.download_dir_edit.setReadOnly(True)
        dir_btn = QPushButton("Browse")
        dir_btn.clicked.connect(self._browse_download_dir)
        dir_row.addWidget(self.download_dir_edit)
        dir_row.addWidget(dir_btn)
        vd.addLayout(dir_row)
        vg.addWidget(self._video_details)
        self._video_group.hide()
        self._inner_layout.addWidget(self._video_group)

        # --- Pose estimation ---
        pose_group = QGroupBox("Pose estimation")
        pg = QVBoxLayout(pose_group)
        self.pose_checkbox = QCheckBox("Include pose estimation data")
        self.pose_checkbox.setChecked(True)
        pg.addWidget(self.pose_checkbox)
        self._pose_label = QLabel("")
        self._pose_label.setWordWrap(True)
        pg.addWidget(self._pose_label)
        self._inner_layout.addWidget(pose_group)

        # --- Behavioral time series (shown only if any detected) ---
        self._behavior_group = QGroupBox("Behavioral time series")
        bg = QVBoxLayout(self._behavior_group)
        bg.addWidget(QLabel("Select series to include as features:"))
        self._behavior_checkboxes: list[QCheckBox] = []
        self._behavior_cb_layout = QVBoxLayout()
        bg.addLayout(self._behavior_cb_layout)
        self._behavior_group.hide()
        self._inner_layout.addWidget(self._behavior_group)

        # --- Behavioral labels (shown only if any detected) ---
        self._labels_group = QGroupBox("Behavioral labels")
        lg2 = QVBoxLayout(self._labels_group)
        lg2.addWidget(QLabel("Select label source to import (or none):"))
        self._label_radios: list[QRadioButton] = []
        self._label_radio_layout = QVBoxLayout()
        lg2.addLayout(self._label_radio_layout)
        self._labels_group.hide()
        self._inner_layout.addWidget(self._labels_group)

        # --- Electrophysiology (shown only if ElectricalSeries detected) ---
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
            "Note: Trials, behavioral labels, and time series (including "
            "pose estimation) will be converted to a local .nc file for fast "
            "access and xarray indexing. Raw data (electrophysiology, video, "
            "audio) can be streamed from DANDI or downloaded locally."
        )
        self._info_note.setWordWrap(True)
        self._info_note.hide()
        self._inner_layout.addWidget(self._info_note)

        # --- Output ---
        out_group = QGroupBox("Output")
        outg = QHBoxLayout(out_group)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Save trials.nc to...")
        self.output_edit.setReadOnly(True)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output)
        outg.addWidget(self.output_edit)
        outg.addWidget(out_browse)
        self._inner_layout.addWidget(out_group)

        self._inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

    def _on_video_toggled(self, checked: bool):
        self._video_details.setVisible(checked)

    def _browse_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            self.download_dir_edit.setText(folder)

    def _browse_output(self):
        result = QFileDialog.getSaveFileName(self, "Save trials.nc", "", "NetCDF files (*.nc);;All files (*)")
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)
            if not self.download_dir_edit.text():
                self.download_dir_edit.setText(str(Path(path).parent))

    def populate(
        self,
        nwb,
        cameras_with_pose: list[str],
        video_urls: dict[str, str],
        behavioral_series: list[dict],
        label_sources: list[dict],
        electrical_series: list[dict] | None = None,
    ) -> None:
        # Trials table
        total_trials = len(nwb.trials) if nwb.trials is not None and len(nwb.trials) > 0 else 1
        self._trial_group.setVisible(total_trials > 1)

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

        # Video matching (only for DANDI sources with video URLs)
        video_items = [cam for cam in cameras_with_pose if cam in video_urls]
        if video_items:
            self._matcher.set_items_direct(video_items, cameras_with_pose)
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

    def get_selected_behavioral_sources(self) -> set[str] | None:
        if not self._behavior_checkboxes:
            return None
        selected = {cb._source for cb in self._behavior_checkboxes if cb.isChecked()}
        return selected if selected else None

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

    def has_videos(self) -> bool:
        return (
            self._video_group.isVisible()
            and self._video_checkbox.isChecked()
            and bool(self._matcher._video_list.get_items())
        )

    def get_video_matching(self) -> list[tuple[str, str]]:
        return self._matcher.get_mapping()

    def validate(self) -> str | None:
        if not self.output_edit.text():
            return "Please select an output path."
        return None

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

    def _set_all_video_selected(self, selected: bool) -> None:
        if self._video_table is not None:
            if selected:
                self._video_table.selectAll()
            else:
                self._video_table.clearSelection()

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


class NWBImportDialog(QDialog):
    """2-step wizard: NWB source → configure import (trials + video + data + output)."""

    def __init__(self, app_state, io_widget, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Import NWB file as trials.nc")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._nwb = None
        self._nwb_io = None
        self._nwb_h5 = None
        self._nwb_rf = None
        self._cameras_with_pose: list[str] = []
        self._pose_containers: dict[str, Any] | None = None
        self._video_info: dict[str, dict]  = {}
        self._session_assets: dict | None = None
        self._output_path: str = ""

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_source = _SourcePage()
        self._page_selection = _SelectionPage()
        self._stack.addWidget(self._page_source)
        self._stack.addWidget(self._page_selection)
        layout.addWidget(self._stack)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        self._prev_btn.setEnabled(False)

        self._next_btn = QPushButton("Connect & Preview →")
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
                QMessageBox.warning(self, "Input error", err)
                return
            self._connect_to_nwb()
        elif page == 1:
            err = self._page_selection.validate()
            if err:
                QMessageBox.warning(self, "Input error", err)
                return
            self._load_all()

    def _update_nav(self):
        page = self._current_page()
        self._prev_btn.setEnabled(page > 0)
        if page == 0:
            self._next_btn.setText("Connect & Preview →")
        else:
            self._next_btn.setText("Load data")

    def _connect_to_nwb(self):
        source = self._page_source.get_source()

        def _open():
            nwb = None
            

            if source["type"] == "local":          
                nwb, io, h5, rf = open_nwb_local(source["path"])
                raise NotImplementedError("Will be added later.")
                
            else:
                
                dandiset_id = source["dandiset_id"]
                session_eid = source["session_eid"]
                
                client = DandiAPIClient()
                dandiset = client.get_dandiset(dandiset_id, "draft")
                session_assets = [asset for asset in dandiset.get_assets() if session_eid in asset.path]

                raw_asset = next((asset for asset in session_assets if "desc-raw" in asset.path), None)
                processed_asset = next((asset for asset in session_assets if "desc-processed" in asset.path), None)
                

                widget = NWBDANDIPoseEstimationWidget(
                    processed_asset=processed_asset,
                    raw_asset=raw_asset,
                )

            nwb = widget.nwbfile
            cameras_with_pose = widget.available_cameras
            video_info = widget.video_info
            pose_containers = widget.pose_containers

            if video_info:
                with ThreadPoolExecutor(max_workers=len(video_info)) as pool:
                    futures = {
                        name: pool.submit(probe_dandi_video_metadata, info["url"])
                        for name, info in video_info.items() if info.get("url")
                    }
                    for name, future in futures.items():
                        try:
                            video_info[name].update(future.result(timeout=30))
                        except Exception:
                            pass

            session_assets_dict = {
                "raw": raw_asset,
                "processed": processed_asset,
            }

            behavioral = probe_behavioral_series(nwb)
            labels = probe_label_sources(nwb)
            ephys = probe_electrical_series(nwb)
            return nwb, cameras_with_pose, video_info, pose_containers, behavioral, labels, ephys, session_assets_dict

        progress = BusyProgressDialog("Accessing NWB metadata...", parent=self)
        (result, error) = progress.execute(_open)

        if progress.was_cancelled or error:
            if error:
                msg = _network_error_message(error) or f"Failed to open NWB:\n{error}"
                QMessageBox.critical(self, "Error", msg)
            return

        (
            self._nwb, self._cameras_with_pose, self._video_info, self._pose_containers,
            behavioral, labels, ephys, self._session_assets,
        ) = result

        self._page_selection.populate(
            self._nwb, self._cameras_with_pose, self._video_info,
            behavioral, labels, ephys,
        )

        if self._session_assets:
            self._page_selection.populate_session_overview(self._session_assets)

        if self._video_info:
            self._page_selection.populate_videos(self._video_info)

        if source["type"] == "dandi" and self._video_info:
            default_dir = self._default_download_dir(source)
            self._page_selection.download_dir_edit.setText(str(default_dir))

        self._stack.setCurrentIndex(1)
        self._update_nav()

    def _load_all(self):
        output_path = self._page_selection.output_edit.text()
        total_trials = len(self._nwb.trials) if self._nwb.trials is not None and len(self._nwb.trials) > 0 else 1
        trial_indices = self._page_selection.get_trial_indices(total_trials)
        include_pose = self._page_selection.pose_checkbox.isChecked()
        behavioral_sources = self._page_selection.get_selected_behavioral_sources()
        label_source = self._page_selection.get_selected_label_source()
        ephys_series = self._page_selection.get_selected_ephys_series()

        output_dir = None
        if self._page_selection.has_videos():
            download_dir = self._page_selection.download_dir_edit.text()
            if not download_dir:
                reply = QMessageBox.question(
                    self, "No download folder",
                    "No video download folder selected. Continue without downloading videos?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return
            else:
                output_dir = self._download_videos(download_dir, trial_indices)
                if output_dir is None:
                    return

        source_info = self._page_source.get_source()
        nwb_source = (
            source_info["path"] if source_info["type"] == "local"
            else f"dandiset-{source_info['dandiset_id']}_session-{source_info['session_eid']}"
        )
        matching = self._page_selection.get_video_matching() if self._page_selection.has_videos() else []

        def _build():
            dt, trials_df = load_nwb_session(
                self._nwb, self._pose_containers,
                cameras_with_pose=self._cameras_with_pose,
                trial_indices=trial_indices,
                include_pose=include_pose,
                behavioral_sources=behavioral_sources,
            )
            dt.attrs["nwb_source"] = nwb_source
            self._build_session_table(dt, trials_df)
            self._set_ephys_attrs(dt, ephys_series, source_info)
            self._set_raw_asset_attr(dt, source_info)
            label_dt = NWBLabelConverter().from_nwb(self._nwb, trials_df) if label_source else dt.get_label_dt(empty=True)
            self._set_video_files(dt, matching, output_dir, include_pose)
            dt.to_netcdf(output_path)
            return dt, label_dt

        progress = BusyProgressDialog("Loading NWB data. This may take a few minutes.", parent=self)
        (result, error) = progress.execute(_build)

        if progress.was_cancelled or error:
            if error:
                QMessageBox.critical(self, "Error", f"Failed to load data:\n{error}")
            return

        self._output_path = output_path
        video_folder = str(output_dir) if output_dir else None
        self._populate_io_fields(video_folder=video_folder)

        msg = f"Successfully created:\n{output_path}"
        if output_dir:
            msg += f"\n\nVideos saved to:\n{output_dir}"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
        QMessageBox.information(self, "Success", msg)
        self.accept()

    # ------------------------------------------------------------------
    # Build helpers (called from _build closure)
    # ------------------------------------------------------------------

    def _build_session_table(self, dt, trials_df):
        trial_ids = list(dt.trials)
        if trials_df is None or "start_time" not in trials_df.columns:
            return
        session_vars: dict = {
            "start_time": ("trial", trials_df["start_time"].astype(float).values),
            "stop_time": ("trial", trials_df["stop_time"].astype(float).values),
        }

        session_ds = xr.Dataset(session_vars, coords={"trial": trial_ids})
        session_ds.attrs["session_start_time"] = str(self._nwb.session_start_time)

        dt.set_session_table(session_ds)

    def _set_ephys_attrs(self, dt, ephys_series: str | None, source_info: dict):
        if ephys_series is None:
            return
        dt.attrs["nwb_ephys_series"] = ephys_series
        if source_info["type"] == "local":
            dt.attrs["nwb_ephys_path"] = source_info["path"]
        else:
            dt.attrs["nwb_ephys_dandiset_id"] = source_info["dandiset_id"]
            main = self._session_assets and (self._session_assets["processed"] or self._session_assets["raw"])
            if main:
                dt.attrs["nwb_ephys_asset_id"] = main.identifier

    def _set_raw_asset_attr(self, dt, source_info: dict):
        if source_info["type"] != "dandi":
            return
        raw_asset = self._page_selection.get_raw_nwb_asset()
        if raw_asset:
            dt.attrs["nwb_raw_asset_id"] = raw_asset.identifier

    def _get_video_urls(self) -> list[str]:
        return [info["url"] for info in self._video_info.values() if info.get("url")]

    def _set_video_files(self, dt, matching: list[tuple[str, str]], output_dir: Path | None, include_pose: bool):
        trials = dt.trials
        
        camera_stems = [video_cam for video_cam, _ in matching]
        if output_dir is not None:
            
            video_files = [
                [f"{cam}_trial_{trial_id}.mp4" for cam in camera_stems]
                for trial_id in trials
            ]
            dt.set_media(video=video_files, cameras=camera_stems, per_trial=True)
            return

        camera_urls = self._get_video_urls()
        if not camera_urls and include_pose:
            pose_keys = dt.attrs.get("nwb_pose_keys")
            if pose_keys:
                camera_urls = list(pose_keys)
        if not camera_urls:
            return
        
        video_offsets = {cam: self._video_info.get(cam).get("start", 0.0) for cam in camera_stems}
        dt.set_media(video=camera_urls, cameras=camera_stems, 
                           per_trial=False, 
                           video_start=video_offsets)

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_path_component(s: str) -> str:
        return re.sub(r'[<>:"/\\|?*\s]+', "_", s).strip("_") or "unknown"

    def _default_download_dir(self, source: dict) -> Path:
        lab = getattr(self._nwb, "lab", "") or "unknown_lab"
        subject_id = getattr(getattr(self._nwb, "subject", None), "subject_id", "") or "unknown_subject"
        session_eid = source.get("session_eid", "unknown_session")
        sanitize = self._sanitize_path_component
        return Path.home() / ".ethograph" / "dandi_videos" / sanitize(lab) / sanitize(subject_id) / sanitize(session_eid)

    def _download_videos(self, download_dir: str, trial_indices: list[int]) -> Path | None:
        output_dir = Path(download_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_info = self._video_info
        matching = self._page_selection.get_video_matching()
        trials_df = read_trials_table(self._nwb)
        trials_df = trials_df.iloc[trial_indices].reset_index(drop=True)

        def _download():
            for _, row in trials_df.iterrows():
                trial_id = int(row["trial"])
                for video_cam, _ in matching:
                    url = video_info.get(video_cam, {}).get("url", "")
                    if not url:
                        continue
                    clip_path = output_dir / f"{video_cam}_trial_{trial_id}.mp4"
                    if clip_path.exists():
                        continue
                    cam_start = video_info.get(video_cam, {}).get("start", 0.0)
                    t_start = float(row["start_time"]) - cam_start
                    t_stop = float(row["stop_time"]) - cam_start
                    download_clip(url, t_start, t_stop, clip_path)

        progress = BusyProgressDialog("Downloading video segments...", parent=self)
        (_, error) = progress.execute(_download)

        if progress.was_cancelled or error:
            if error:
                QMessageBox.critical(self, "Error", f"Download failed:\n{error}")
            return None

        return output_dir

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _populate_io_fields(self, video_folder: str | None):
        self.app_state.nc_file_path = self._output_path
        self.io_widget.nc_file_path_edit.setText(self._output_path)
        if video_folder:
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)

    def closeEvent(self, event):
        for closeable in (self._nwb_io, self._nwb_h5, self._nwb_rf):
            if closeable is not None:
                try:
                    closeable.close()
                except Exception:
                    pass
        super().closeEvent(event)
