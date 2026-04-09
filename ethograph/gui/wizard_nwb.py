"""Multi-step wizard for importing DANDI NWB sessions as ethograph projects."""

from __future__ import annotations

import logging
import re
import webbrowser
from datetime import datetime
from pathlib import Path

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
from ethograph.labels.converters import NWBLabelConverter, write_mapping_file
from ethograph.labels.tsv_store import init_empty_labels, save_labels_tsv
from ethograph.io.nwb_import import (
    probe_behavioral_series,
    probe_electrical_series,
    probe_label_sources,
    read_trials_table,
)
from ethograph.utils.dandi import download_clip, open_nwb_dandi

logger = logging.getLogger(__name__)


def _network_error_message(error: Exception) -> str | None:
    s = str(error).lower()
    if any(kw in s for kw in ("getaddrinfo failed", "failed to resolve", "max retries exceeded", "nodename nor servname", "name or service not known")):
        return "No internet connection or the DANDI archive is unreachable.\n\nPlease check your network and try again."
    return None


def _stream_in_browser(url: str, title: str = "DANDI Video") -> None:
    html = (
        "<!DOCTYPE html>\n"
        f"<html><head><title>{title}</title></head>"
        '<body style="margin:0;background:#000">\n'
        '<video controls autoplay style="width:100%;height:100vh">\n'
        f'<source src="{url}" type="video/mp4">\n'
        "</video>\n"
        "</body></html>"
    )
    path = Path.home() / ".ethograph" / "dandi_video.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    webbrowser.open(path.as_uri())


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

_EXAMPLE_DATASETS = [
    (
        "IBL Brainwide Map",
        "000409",
        "64e3fb86-928c-4079-865c-b364205b502e",
    ),
    (
        "Neuropixels + 3D pose (DANNCE)",
        "001771",
        "2026-02-12-1",
    ),
]


class _SourcePage(QWidget):
    """Enter DANDI dataset + session identifiers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 1 of 3 \u2014 DANDI Source</b>"))
        layout.addSpacing(8)

        dandi_info = QLabel(
            "Enter the Dandiset ID and the Session EID, or pick an "
            "example dataset below."
        )
        dandi_info.setWordWrap(True)
        layout.addWidget(dandi_info)

        dandi_form = QFormLayout()

        self.dandiset_edit = QLineEdit()
        self.dandiset_edit.setPlaceholderText("e.g. 000409")
        dandi_form.addRow("Dataset ID:", self.dandiset_edit)

        self.session_eid_edit = QLineEdit()
        self.session_eid_edit.setPlaceholderText(
            "e.g. 64e3fb86-928c-4079-865c-b364205b502e (part after _ses-...)"
        )
        dandi_form.addRow("Session EID:", self.session_eid_edit)

        layout.addLayout(dandi_form)

        # Example datasets list
        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Example datasets</b>"))
        for name, dandiset_id, session_eid in _EXAMPLE_DATASETS:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)

            btn = QPushButton(f"{name}  \u2014  {dandiset_id} / {session_eid}")
            btn.setStyleSheet(
                "text-align: left; padding: 6px 10px; font-size: 12px;"
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda checked, d=dandiset_id, s=session_eid: self._fill(d, s)
            )
            row_layout.addWidget(btn, stretch=1)

            dandi_url = f"https://dandiarchive.org/dandiset/{dandiset_id}"
            link = QLabel(styled_link(dandi_url, "DANDI link"))
            link.setOpenExternalLinks(True)
            row_layout.addWidget(link)

            layout.addWidget(row)

        links = QLabel(
            'Browse datasets on '
            + styled_link("https://dandiarchive.org/", "DANDI Archive")
            + ' \u00b7 '
            + styled_link("https://neurosift.app/", "Neurosift")
        )
        links.setOpenExternalLinks(True)
        links.setAlignment(Qt.AlignCenter)
        layout.addWidget(links)

        layout.addStretch()

    def _fill(self, dandiset_id: str, session_eid: str):
        self.dandiset_edit.setText(dandiset_id)
        self.session_eid_edit.setText(session_eid)

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
# Page 1: Data options
# =====================================================================

class _DataOptionsPage(QWidget):
    """Video, pose, behavioral series, labels, ephys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 2 of 3 \u2014 Data Options</b>"))
        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)

        # --- Video ---
        self._video_group = QGroupBox("Video")
        vg = QVBoxLayout(self._video_group)
        self._video_checkboxes: list[QCheckBox] = []
        self._video_cb_layout = QVBoxLayout()
        vg.addLayout(self._video_cb_layout)
        self._video_group.hide()
        self._inner_layout.addWidget(self._video_group)

        # --- Pose estimation ---
        self._pose_group = QGroupBox("Pose estimation")
        pg_layout = QVBoxLayout(self._pose_group)
        self.pose_checkbox = QCheckBox("Include pose estimation data")
        self.pose_checkbox.setChecked(True)
        pg_layout.addWidget(self.pose_checkbox)
        self._pose_label = QLabel("")
        self._pose_label.setWordWrap(True)
        pg_layout.addWidget(self._pose_label)
        self._inner_layout.addWidget(self._pose_group)

        # --- Behavioral time series ---
        self._behavior_group = QGroupBox("Behavioral time series")
        bg_layout = QVBoxLayout(self._behavior_group)
        bg_layout.addWidget(QLabel("Select series to include as features:"))
        self._behavior_checkboxes: list[QCheckBox] = []
        self._behavior_cb_layout = QVBoxLayout()
        bg_layout.addLayout(self._behavior_cb_layout)
        self._behavior_group.hide()
        self._inner_layout.addWidget(self._behavior_group)

        # --- Behavioral labels ---
        self._labels_group = QGroupBox("Behavioral labels")
        lg = QVBoxLayout(self._labels_group)
        lg.addWidget(QLabel("Select label sources to import:"))
        self._label_checkboxes: list[QCheckBox] = []
        self._label_cb_layout = QVBoxLayout()
        lg.addLayout(self._label_cb_layout)
        self._labels_group.hide()
        self._inner_layout.addWidget(self._labels_group)

        # --- Electrophysiology ---
        self._ephys_group = QGroupBox("Electrophysiology")
        eg = QVBoxLayout(self._ephys_group)
        eg.addWidget(QLabel("Select ElectricalSeries to link (or none):"))
        self._ephys_radios: list[QRadioButton] = []
        self._ephys_radio_layout = QVBoxLayout()
        eg.addLayout(self._ephys_radio_layout)
        self._ephys_group.hide()
        self._inner_layout.addWidget(self._ephys_group)

        self._inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

    # -- Populate --

    def populate(
        self,
        video_info: dict[str, dict],
        cameras_with_pose: list[str],
        behavioral_series: list[dict],
        label_sources: list[dict],
        electrical_series: list[dict],
    ) -> None:
        # Videos
        for cb in list(self._video_checkboxes):
            cb.parent().deleteLater()
        self._video_checkboxes.clear()

        if video_info:
            for name, info in video_info.items():
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 2, 0, 2)

                cb = QCheckBox()
                cb.setChecked(True)
                cb._video_name = name
                row_layout.addWidget(cb)
                self._video_checkboxes.append(cb)

                dur = info.get("end", 0) - info.get("start", 0)
                dur_str = f"{dur / 60:.1f} min" if dur > 60 else f"{dur:.1f} s"
                label = QLabel(f"<b>{name}</b> \u00b7 {dur_str}")
                label.setWordWrap(True)
                row_layout.addWidget(label, stretch=1)

                url = info.get("url", "")
                if url:
                    stream_btn = QPushButton("Preview stream \u25b6")
                    stream_btn.setFixedWidth(120)
                    stream_btn.clicked.connect(
                        lambda checked, u=url, n=name: _stream_in_browser(u, n)
                    )
                    row_layout.addWidget(stream_btn)

                self._video_cb_layout.addWidget(row)

            self._video_group.show()
        else:
            self._video_group.hide()

        # Pose
        if cameras_with_pose:
            self._pose_label.setText(
                f"Detected cameras: {', '.join(cameras_with_pose)}"
            )
        else:
            self.pose_checkbox.setEnabled(False)
            self.pose_checkbox.setChecked(False)
            self._pose_label.setText("No pose estimation interfaces found.")

        # Behavioral series
        for cb in self._behavior_checkboxes:
            self._behavior_cb_layout.removeWidget(cb)
            cb.deleteLater()
        self._behavior_checkboxes.clear()

        if behavioral_series:
            for entry in behavioral_series:
                cb = QCheckBox(
                    f"{entry['source']}  ({entry['n_samples']:,} samples)"
                )
                cb._source = entry["source"]
                cb.setChecked(True)
                self._behavior_cb_layout.addWidget(cb)
                self._behavior_checkboxes.append(cb)
            self._behavior_group.show()
        else:
            self._behavior_group.hide()

        # Labels
        for cb in self._label_checkboxes:
            self._label_cb_layout.removeWidget(cb)
            cb.deleteLater()
        self._label_checkboxes.clear()

        if label_sources:
            for entry in label_sources:
                cb = QCheckBox(entry["description"])
                cb._source = entry["source"]
                cb.setChecked(True)
                self._label_cb_layout.addWidget(cb)
                self._label_checkboxes.append(cb)
            self._labels_group.show()
        else:
            self._labels_group.hide()

        # Ephys
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
                rate_str = (
                    f"{entry['rate']:.0f} Hz" if entry["rate"] else "unknown rate"
                )
                label = (
                    f"{entry['name']}  ({entry['n_channels']} ch, "
                    f"{rate_str}, {entry['n_samples']:,} samples)"
                )
                rb = QRadioButton(label)
                rb._series_name = entry["name"]
                self._ephys_radio_layout.addWidget(rb)
                self._ephys_radios.append(rb)
            self._ephys_group.show()
        else:
            self._ephys_group.hide()

    # -- Accessors --

    def get_selected_videos(self) -> list[str]:
        return [cb._video_name for cb in self._video_checkboxes if cb.isChecked()]

    def get_selected_label_sources(self) -> list[str]:
        return [cb._source for cb in self._label_checkboxes if cb.isChecked()]

    def get_selected_ephys_series(self) -> str | None:
        for rb in self._ephys_radios:
            if rb.isChecked():
                return rb._series_name
        return None


# =====================================================================
# Page 2: Trial selection + download + output
# =====================================================================

class _TrialsPage(QWidget):
    """Select trials to download and set output path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 3 of 3 \u2014 Trials & Output</b>"))
        layout.addSpacing(4)

        # --- Trial selection ---
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
        self.trials_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        tg.addWidget(self.trials_table)
        self._rb_select.toggled.connect(self._on_select_toggled)

        layout.addWidget(self._trial_group)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 4px;"
        )
        layout.addWidget(self._summary_label)

        # --- Download folder ---
        dl_group = QGroupBox("Download folder")
        dlg = QHBoxLayout(dl_group)
        self.download_dir_edit = QLineEdit()
        self.download_dir_edit.setPlaceholderText("Select folder for video clips...")
        self.download_dir_edit.setReadOnly(True)
        dl_browse = QPushButton("Browse")
        dl_browse.clicked.connect(self._browse_download_dir)
        dlg.addWidget(self.download_dir_edit)
        dlg.addWidget(dl_browse)
        layout.addWidget(dl_group)

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

    def _on_select_toggled(self, checked: bool):
        self.trials_table.setSelectionMode(
            QAbstractItemView.MultiSelection if checked
            else QAbstractItemView.NoSelection
        )

    def _browse_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            self.download_dir_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select project folder")
        if folder:
            self.output_edit.setText(folder)

    def populate(self, nwb) -> None:
        total = (
            len(nwb.trials)
            if nwb.trials is not None and len(nwb.trials) > 0
            else 1
        )
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
                        item = _NumericTableItem(
                            f"{val:.3f}" if isinstance(val, float) else str(val)
                        )
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
            self._summary_label.setText(
                "No trials table found \u2014 single trial will be created."
            )

    def get_trial_indices(self, total: int) -> list[int]:
        if self._rb_all.isChecked():
            return list(range(total))
        if self._rb_first_n.isChecked():
            return list(range(min(self.n_spin.value(), total)))
        visual_rows = sorted(
            {idx.row() for idx in self.trials_table.selectedIndexes()}
        )
        if not visual_rows:
            return list(range(total))
        return sorted(
            self.trials_table.item(r, 0).data(Qt.UserRole) for r in visual_rows
        )

    def validate(self) -> str | None:
        if not self.output_edit.text():
            return "Please select a project folder."
        if not self.download_dir_edit.text():
            return "Please select a download folder for video clips."
        return None


# =====================================================================
# Main wizard dialog
# =====================================================================

class NWBImportDialog(QDialog):
    """3-step wizard: DANDI source \u2192 data options \u2192 trials & output."""

    def __init__(self, app_state, io_widget, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Import DANDI session")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._nwb = None
        self._nwb_io = None
        self._nwb_h5 = None
        self._cameras_with_pose: list[str] = []
        self._video_info: dict[str, dict] = {}
        self._dandi_asset_id: str | None = None
        self._trials_df = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_source = _SourcePage()
        self._page_options = _DataOptionsPage()
        self._page_trials = _TrialsPage()
        self._stack.addWidget(self._page_source)
        self._stack.addWidget(self._page_options)
        self._stack.addWidget(self._page_trials)
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

    def _on_previous(self):
        page = self._stack.currentIndex()
        if page > 0:
            self._stack.setCurrentIndex(page - 1)
            self._update_nav()

    def _on_next(self):
        page = self._stack.currentIndex()
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
            self._build_project()

    def _update_nav(self):
        page = self._stack.currentIndex()
        self._prev_btn.setEnabled(page > 0)
        if page == 0:
            self._next_btn.setText("Connect & Preview \u2192")
        elif page == 2:
            self._next_btn.setText("Create project")
        else:
            self._next_btn.setText("Next \u2192")

    # ------------------------------------------------------------------
    # Connect to DANDI and probe NWB metadata
    # ------------------------------------------------------------------

    def _connect_to_nwb(self):
        source = self._page_source.get_source()
        dandiset_id = source["dandiset_id"]
        session_eid = source["session_eid"]

        def _open():
            from nwb_video_widgets import get_dandi_video_info

            client = DandiAPIClient()
            dandiset = client.get_dandiset(dandiset_id, "draft")
            session_assets = [
                a for a in dandiset.get_assets() if session_eid in a.path
            ]

            if not session_assets:
                raise ValueError(
                    f"No assets found for session '{session_eid}' "
                    f"in dandiset {dandiset_id}."
                )

            logger.info(
                "\n%s\n  DANDI session: %d file(s) in dandiset %s\n%s",
                "=" * 60, len(session_assets), dandiset_id, "=" * 60,
            )
            for a in session_assets:
                neurosift_url = (
                    f"https://neurosift.app/nwb?url=https://api.dandiarchive.org"
                    f"/api/assets/{a.identifier}/download/"
                    f"&dandisetId={dandiset_id}&dandisetVersion=draft"
                )
                logger.info("  %s\n    %s", a.path, neurosift_url)

            nwb_assets = [a for a in session_assets if a.path.endswith(".nwb")]
            image_asset = next(
                (a for a in nwb_assets if "_image" in a.path), None
            )
            raw_asset = next(
                (a for a in nwb_assets if "desc-raw" in a.path), None
            )
            processed_asset = next(
                (a for a in nwb_assets if "desc-processed" in a.path), None
            )

            # Video info from the NWB that contains ImageSeries
            video_nwb_asset = image_asset or raw_asset
            video_info = {}
            if video_nwb_asset:
                try:
                    video_info = get_dandi_video_info(video_nwb_asset)
                except Exception as exc:
                    logger.warning("Could not get video info: %s", exc)

            # Open NWB for probing behavioral / label / ephys / pose data
            probe_asset = processed_asset or raw_asset or nwb_assets[0]
            nwb, nwb_io, nwb_h5, _rf = open_nwb_dandi(
                dandiset_id, probe_asset.identifier
            )

            behavioral = probe_behavioral_series(nwb)
            labels = probe_label_sources(nwb)
            ephys = probe_electrical_series(nwb)

            cameras_with_pose = [
                obj.name
                for obj in nwb.objects.values()
                if getattr(obj, "neurodata_type", None) == "PoseEstimation"
            ]

            return (
                nwb, nwb_io, nwb_h5,
                video_info, behavioral, labels, ephys,
                cameras_with_pose, probe_asset.identifier,
            )

        progress = BusyProgressDialog("Accessing NWB metadata...", parent=self)
        (result, error) = progress.execute(_open)

        if progress.was_cancelled or error:
            if error:
                msg = _network_error_message(error) or f"Failed to open NWB:\n{error}"
                notify_dialog(msg, "error", "Error", self)
            return

        (
            self._nwb, self._nwb_io, self._nwb_h5,
            self._video_info, behavioral, labels, ephys,
            self._cameras_with_pose, self._dandi_asset_id,
        ) = result

        self._page_options.populate(
            self._video_info, self._cameras_with_pose,
            behavioral, labels, ephys,
        )

        self._stack.setCurrentIndex(1)
        self._update_nav()

    # ------------------------------------------------------------------
    # Populate trials page
    # ------------------------------------------------------------------

    def _populate_trials_page(self):
        self._page_trials.populate(self._nwb)
        self._trials_df = read_trials_table(self._nwb)

        source = self._page_source.get_source()
        default_dir = self._default_download_dir(source)
        project_name = self._default_project_dirname()
        self._page_trials.download_dir_edit.setText(str(default_dir))
        self._page_trials.output_edit.setText(str(default_dir / project_name))

    # ------------------------------------------------------------------
    # Build project
    # ------------------------------------------------------------------

    def _build_project(self):
        output_path = self._page_trials.output_edit.text()
        download_dir = self._page_trials.download_dir_edit.text()
        selected_videos = self._page_options.get_selected_videos()
        include_pose = self._page_options.pose_checkbox.isChecked()
        label_sources = self._page_options.get_selected_label_sources()
        source_info = self._page_source.get_source()

        total_trials = (
            len(self._nwb.trials)
            if self._nwb.trials is not None and len(self._nwb.trials) > 0
            else 1
        )
        trial_indices = self._page_trials.get_trial_indices(total_trials)

        def _build():
            import av
            import numpy as np
            from ethograph.utils.nwb import create_alignment_from_streams

            trials_df = self._trials_df
            if trial_indices is not None:
                trials_df = trials_df.iloc[trial_indices].reset_index(drop=True)

            project_dir = Path(output_path)
            ethograph_dir = project_dir / ".ethograph"
            ethograph_dir.mkdir(parents=True, exist_ok=True)

            output_dir = Path(download_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download per-trial video clips
            for _, row in trials_df.iterrows():
                trial_id = int(row["trial"])
                for vname in selected_videos:
                    url = self._video_info.get(vname, {}).get("url", "")
                    if not url:
                        continue
                    clip_path = output_dir / f"{vname}_trial_{trial_id}.mp4"
                    if clip_path.exists():
                        continue
                    cam_start = self._video_info.get(vname, {}).get("start", 0.0)
                    t_start = float(row["start_time"]) - cam_start
                    t_stop = float(row["stop_time"]) - cam_start
                    logger.info("Downloading %s trial %d ...", vname, trial_id)
                    download_clip(url, t_start, t_stop, clip_path)

            # Build video streams for alignment.nwb
            streams = []
            for vname in selected_videos:
                info = self._video_info.get(vname, {})
                url = info.get("url", "")
                if not url:
                    continue

                rate = None
                try:
                    container = av.open(url)
                    stream = container.streams.video[0]
                    if stream.average_rate:
                        rate = float(stream.average_rate)
                    container.close()
                except Exception:
                    pass

                if not rate:
                    continue

                files = [
                    str(output_dir / f"{vname}_trial_{int(row['trial'])}.mp4")
                    for _, row in trials_df.iterrows()
                ]
                streams.append({
                    "name": f"video_{vname}",
                    "files": files,
                    "rate": rate,
                })

            # Pose streams
            if include_pose and self._cameras_with_pose:
                pose_containers = {
                    obj.name: obj
                    for obj in self._nwb.objects.values()
                    if getattr(obj, "neurodata_type", None) == "PoseEstimation"
                }
                for pose_key in self._cameras_with_pose:
                    container = pose_containers.get(pose_key)
                    if container is None:
                        continue
                    first_series = next(
                        iter(container.pose_estimation_series.values()), None
                    )
                    if first_series is None:
                        continue
                    ts = getattr(first_series, "timestamps", None)
                    pose_rate = getattr(first_series, "rate", None)
                    if ts is not None and len(ts) >= 2:
                        streams.append({
                            "name": f"pose_{pose_key}",
                            "files": [
                                f"nwb://processing/pose_estimation/{pose_key}"
                            ],
                            "timestamps": np.asarray(ts[:], dtype=np.float64),
                        })
                    elif pose_rate:
                        t0 = (
                            float(first_series.starting_time)
                            if first_series.starting_time is not None
                            else 0.0
                        )
                        streams.append({
                            "name": f"pose_{pose_key}",
                            "files": [
                                f"nwb://processing/pose_estimation/{pose_key}"
                            ],
                            "rate": float(pose_rate),
                            "starting_time": t0,
                        })

            provenance = {
                "nwb_dandiset_id": source_info["dandiset_id"],
                "nwb_asset_id": self._dandi_asset_id,
                "nwb_ephys_dandiset_id": source_info["dandiset_id"],
                "nwb_ephys_asset_id": self._dandi_asset_id,
            }
            if include_pose and self._cameras_with_pose:
                provenance["nwb_pose_keys"] = list(self._cameras_with_pose)

            alignment_path = ethograph_dir / "alignment.nwb"
            create_alignment_from_streams(
                trials_df[["trial", "start_time", "stop_time"]],
                streams,
                alignment_path,
                provenance=provenance,
            )

            # Labels
            if label_sources:
                converter = NWBLabelConverter()
                all_labels_df = converter.from_nwb(
                    self._nwb, trials_df, sources=label_sources
                )
                write_mapping_file(
                    ethograph_dir / "mapping.txt", converter.label_map
                )
            else:
                trial_ids = [
                    str(int(r["trial"])) for _, r in trials_df.iterrows()
                ]
                all_labels_df = init_empty_labels(trial_ids)

            save_labels_tsv(project_dir / "labels.tsv", all_labels_df)

        progress = BusyProgressDialog(
            "Downloading video clips & setting up project...", parent=self
        )
        (_, error) = progress.execute(_build)

        if progress.was_cancelled or error:
            if error:
                notify_dialog(
                    f"Failed to create project:\n{error}", "error", "Error", self
                )
            return

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        self.app_state.video_folder = download_dir
        self.io_widget.video_folder_edit.setText(download_dir)

        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, self.io_widget._on_load_clicked)
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
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
        subject_id = (
            getattr(getattr(self._nwb, "subject", None), "subject_id", "")
            or "unknown_subject"
        )
        session_eid = source.get("session_eid", "unknown_session")
        sanitize = self._sanitize_path_component
        return (
            Path.home()
            / ".ethograph"
            / "dandi"
            / sanitize(lab)
            / sanitize(subject_id)
            / sanitize(session_eid)
        )

    def closeEvent(self, event):
        self._nwb = None
        if self._nwb_io:
            try:
                self._nwb_io.close()
            except Exception:
                pass
        super().closeEvent(event)
