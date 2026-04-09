"""Multi-step wizard for importing DANDI NWB sessions as ethograph projects."""

from __future__ import annotations

import logging
import re
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

from dandi.dandiapi import DandiAPIClient

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
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
from ethograph.utils.dandi import open_nwb_dandi

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


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)


# =====================================================================
# Page 0: Source selection
# =====================================================================

class _SourcePage(QWidget):
    """Enter DANDI dataset + session identifiers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Step 1 \u2014 DANDI Source</b>"))
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
            + ' \u00b7 '
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
# Page 1: Data options + output
# =====================================================================

class _DataOptionsPage(QWidget):
    """Video, pose, behavioral series, labels, ephys, and output folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 2 \u2014 Data & Output</b>"))
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

        mode_row = QHBoxLayout()
        self._rb_stream = QRadioButton("Stream from DANDI")
        self._rb_download = QRadioButton("Download locally")
        self._rb_stream.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_stream)
        bg.addButton(self._rb_download)
        self._rb_download.toggled.connect(self._on_download_toggled)
        mode_row.addWidget(self._rb_stream)
        mode_row.addWidget(self._rb_download)
        mode_row.addStretch()
        vg.addLayout(mode_row)

        self._stream_note = QLabel(
            "Streaming works for playback but seeking/jumping may be slow."
        )
        self._stream_note.setWordWrap(True)
        self._stream_note.setStyleSheet("color: #888; font-style: italic; margin-left: 20px;")
        vg.addWidget(self._stream_note)

        self._download_row = QWidget()
        dir_row = QHBoxLayout(self._download_row)
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(QLabel("Download folder:"))
        self.download_dir_edit = QLineEdit()
        self.download_dir_edit.setPlaceholderText("Select folder...")
        self.download_dir_edit.setReadOnly(True)
        dir_btn = QPushButton("Browse")
        dir_btn.clicked.connect(self._browse_download_dir)
        dir_row.addWidget(self.download_dir_edit)
        dir_row.addWidget(dir_btn)
        self._download_row.setVisible(False)
        vg.addWidget(self._download_row)

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
        lg.addWidget(QLabel("Select label source to import (or none):"))
        self._label_radios: list[QRadioButton] = []
        self._label_radio_layout = QVBoxLayout()
        lg.addLayout(self._label_radio_layout)
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
        self._inner_layout.addWidget(out_group)

        self._inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

    # -- Callbacks --

    def _on_download_toggled(self, checked: bool):
        self._download_row.setVisible(checked)
        self._stream_note.setVisible(not checked)

    def _browse_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            self.download_dir_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select project folder")
        if folder:
            self.output_edit.setText(folder)

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
                    stream_btn = QPushButton("Stream \u25b6")
                    stream_btn.setFixedWidth(90)
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

    def wants_download(self) -> bool:
        return self._rb_download.isChecked()

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

    def validate(self) -> str | None:
        if not self.output_edit.text():
            return "Please select a project folder."
        if self.wants_download() and not self.download_dir_edit.text():
            return "Please select a download folder for videos."
        return None


# =====================================================================
# Main wizard dialog
# =====================================================================

class NWBImportDialog(QDialog):
    """2-step wizard: DANDI source \u2192 data options & output."""

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

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_source = _SourcePage()
        self._page_options = _DataOptionsPage()
        self._stack.addWidget(self._page_source)
        self._stack.addWidget(self._page_options)
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
        if self._stack.currentIndex() > 0:
            self._stack.setCurrentIndex(0)
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
            err = self._page_options.validate()
            if err:
                notify_dialog(err, "warning", "Input error", self)
                return
            self._build_project()

    def _update_nav(self):
        page = self._stack.currentIndex()
        self._prev_btn.setEnabled(page > 0)
        if page == 0:
            self._next_btn.setText("Connect & Preview \u2192")
        else:
            self._next_btn.setText("Create project")

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

        default_dir = self._default_download_dir(source)
        project_name = self._default_project_dirname()
        if self._video_info:
            self._page_options.download_dir_edit.setText(str(default_dir))
        self._page_options.output_edit.setText(str(default_dir / project_name))

        self._stack.setCurrentIndex(1)
        self._update_nav()

    # ------------------------------------------------------------------
    # Build project
    # ------------------------------------------------------------------

    def _build_project(self):
        output_path = self._page_options.output_edit.text()
        selected_videos = self._page_options.get_selected_videos()
        wants_download = self._page_options.wants_download()
        download_dir = (
            self._page_options.download_dir_edit.text() if wants_download else None
        )
        include_pose = self._page_options.pose_checkbox.isChecked()
        label_source = self._page_options.get_selected_label_source()
        source_info = self._page_source.get_source()

        def _build():
            import av
            import numpy as np
            from ethograph.utils.nwb import create_alignment_from_streams

            trials_df = read_trials_table(self._nwb)

            project_dir = Path(output_path)
            ethograph_dir = project_dir / ".ethograph"
            ethograph_dir.mkdir(parents=True, exist_ok=True)

            # Download whole video files if requested
            if wants_download and download_dir:
                dl_dir = Path(download_dir)
                dl_dir.mkdir(parents=True, exist_ok=True)
                for vname in selected_videos:
                    url = self._video_info.get(vname, {}).get("url", "")
                    if not url:
                        continue
                    dest = dl_dir / f"{vname}.mp4"
                    if not dest.exists():
                        logger.info("Downloading %s ...", vname)
                        _download_file(url, dest)

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

                if wants_download and download_dir:
                    file_path = str(Path(download_dir) / f"{vname}.mp4")
                else:
                    file_path = url

                streams.append({
                    "name": f"video_{vname}",
                    "files": [file_path],
                    "rate": rate,
                    "starting_time": info.get("start", 0.0),
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
            if label_source:
                converter = NWBLabelConverter()
                all_labels_df = converter.from_nwb(self._nwb, trials_df)
                write_mapping_file(
                    ethograph_dir / "mapping.txt", converter.label_map
                )
            else:
                trial_ids = [
                    str(int(r["trial"])) for _, r in trials_df.iterrows()
                ]
                all_labels_df = init_empty_labels(trial_ids)

            save_labels_tsv(project_dir / "labels.tsv", all_labels_df)

        progress = BusyProgressDialog("Setting up project...", parent=self)
        (_, error) = progress.execute(_build)

        if progress.was_cancelled or error:
            if error:
                notify_dialog(
                    f"Failed to create project:\n{error}", "error", "Error", self
                )
            return

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if wants_download and download_dir:
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
