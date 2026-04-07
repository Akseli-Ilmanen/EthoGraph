"""
nwb_pose_viewer.py — Standalone NWB pose viewer in napari.

Reads PoseEstimationSeries directly from NWB (local or remote via DANDI).
No xarray Dataset or movement library needed for pose loading — lazy HDF5
slicing only fetches the requested time window.

Interactive dock widget for keypoint visibility and confidence thresholding.

Usage:
    python nwb_pose_viewer.py path/to/local.nwb
    python nwb_pose_viewer.py https://api.dandiarchive.org/api/assets/<ID>/download/
    python nwb_pose_viewer.py --dandiset 000409 --asset 773516a9-bd20-4b46-adad-2a1d5772be5d
    python nwb_pose_viewer.py file.nwb --t-start 10.0 --t-stop 20.0
    python nwb_pose_viewer.py file.nwb --trial 5
"""

from __future__ import annotations

import argparse
import sys
from colorsys import hsv_to_rgb

import h5py
import napari
import numpy as np
import pynwb
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ── NWB I/O ──────────────────────────────────────────────────────────


def open_nwb(path_or_url: str) -> tuple[pynwb.NWBFile, pynwb.NWBHDF5IO]:
    """Open a local NWB file or a remote URL (DANDI asset link)."""
    if path_or_url.startswith("http"):
        import remfile

        rf = remfile.File(path_or_url)
        h5 = h5py.File(rf, "r")
        io = pynwb.NWBHDF5IO(file=h5, mode="r", load_namespaces=True)
    else:
        io = pynwb.NWBHDF5IO(path_or_url, mode="r", load_namespaces=True)
    return io.read(), io


# ── Pose discovery ───────────────────────────────────────────────────


def find_pose_containers(nwb: pynwb.NWBFile) -> dict[str, tuple[str, object]]:
    """Find all PoseEstimation containers.

    Returns ``{interface_name: (processing_module_key, container)}``.
    """
    found: dict[str, tuple[str, object]] = {}
    for mod_name, mod in nwb.processing.items():
        for iface_name, iface in mod.data_interfaces.items():
            if hasattr(iface, "pose_estimation_series"):
                found[iface_name] = (mod_name, iface)
    return found


def _get_timestamps(series) -> np.ndarray:
    if getattr(series, "timestamps", None) is not None:
        return np.asarray(series.timestamps[:], dtype=np.float64)
    n = series.data.shape[0]
    t0 = float(series.starting_time) if series.starting_time else 0.0
    return t0 + np.arange(n, dtype=np.float64) / float(series.rate)


def get_trial_bounds(nwb: pynwb.NWBFile, trial_idx: int) -> tuple[float, float]:
    trials = nwb.trials
    if trials is None:
        raise ValueError("NWB file has no trials table")
    if trial_idx < 0 or trial_idx >= len(trials):
        raise ValueError(f"Trial {trial_idx} out of range (0..{len(trials) - 1})")
    return float(trials["start_time"][trial_idx]), float(trials["stop_time"][trial_idx])


# ── Lazy pose loading ────────────────────────────────────────────────


def load_pose_slice(
    container,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> dict[str, dict]:
    """Lazy-load a time slice directly from each PoseEstimationSeries.

    Only the HDF5 rows inside ``[t_start, t_stop]`` are read into memory.
    Keypoint names come from the series keys (not from an xarray Dataset).

    Returns ``{keypoint: {"data": (N,2+), "timestamps": (N,), "confidence": (N,)|None}}``.
    """
    result: dict[str, dict] = {}
    for kp_name, series in container.pose_estimation_series.items():
        ts = _get_timestamps(series)

        if t_start is not None and t_stop is not None:
            idx = np.where((ts >= t_start) & (ts <= t_stop))[0]
            if len(idx) == 0:
                continue
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            i0, i1 = 0, series.data.shape[0]

        data = np.asarray(series.data[i0:i1], dtype=np.float64)
        timestamps = ts[i0:i1]

        confidence = None
        if hasattr(series, "confidence") and series.confidence is not None:
            try:
                confidence = np.asarray(series.confidence[i0:i1], dtype=np.float64)
            except Exception:
                pass

        result[kp_name] = {
            "data": data,
            "timestamps": timestamps,
            "confidence": confidence,
        }
    return result


# ── Napari viewer ─────────────────────────────────────────────────────


def _generate_colors(n: int) -> list[tuple[float, ...]]:
    return [hsv_to_rgb(i / max(n, 1), 0.85, 0.95) + (1.0,) for i in range(n)]


class NWBPoseViewer:
    """Napari viewer for NWB pose data with keypoint/confidence filtering.

    All keypoints share a single Points layer. Filtering only updates the
    ``shown`` mask — no data is reloaded or layer recreated.
    """

    def __init__(
        self,
        nwb: pynwb.NWBFile,
        container_key: str,
        container,
        t_start: float | None = None,
        t_stop: float | None = None,
    ):
        self._nwb = nwb
        self._container_key = container_key

        self.pose_data = load_pose_slice(container, t_start, t_stop)
        if not self.pose_data:
            raise ValueError(
                f"No pose data in '{container_key}' for the given time range"
            )

        self.keypoints = list(self.pose_data.keys())
        self.hidden: set[str] = set()
        self.conf_threshold = 0.0

        self._build_arrays()

        self.viewer = napari.Viewer(title=f"NWB Pose — {container_key}")
        self._points_layer = None
        self._add_controls()
        self._create_layer()

    # ── Data assembly (runs once) ─────────────────────────────────

    def _build_arrays(self):
        all_pts: list[np.ndarray] = []
        all_conf: list[np.ndarray] = []
        all_not_nan: list[np.ndarray] = []
        kp_labels: list[str] = []
        face_colors: list[tuple] = []

        colors = _generate_colors(len(self.keypoints))
        self._kp_colors = dict(zip(self.keypoints, colors))
        self._kp_ranges: dict[str, tuple[int, int]] = {}
        offset = 0

        for kp_name, kp in self.pose_data.items():
            n = len(kp["data"])
            frames = np.arange(n, dtype=np.float64)
            xy = kp["data"]  # NWB: (x, y [, z])

            # napari Points: [frame, row(y), col(x)]
            pts = np.column_stack([frames, xy[:, 1], xy[:, 0]])
            not_nan = ~np.any(np.isnan(xy[:, :2]), axis=1)

            conf = kp["confidence"]
            if conf is None:
                conf = np.ones(n, dtype=np.float64)

            self._kp_ranges[kp_name] = (offset, offset + n)
            offset += n

            all_pts.append(pts)
            all_conf.append(conf)
            all_not_nan.append(not_nan)
            kp_labels.extend([kp_name] * n)
            face_colors.extend([self._kp_colors[kp_name]] * n)

        self._points = np.vstack(all_pts)
        self._confidence = np.concatenate(all_conf)
        self._base_shown = np.concatenate(all_not_nan)
        self._face_colors = np.array(face_colors)
        self._properties = {"keypoint": kp_labels, "confidence": self._confidence}

    # ── Filtering ─────────────────────────────────────────────────

    def _compute_shown(self) -> np.ndarray:
        shown = self._base_shown.copy()
        if self.conf_threshold > 0:
            shown &= self._confidence >= self.conf_threshold
        for kp in self.hidden:
            if kp in self._kp_ranges:
                a, b = self._kp_ranges[kp]
                shown[a:b] = False
        return shown

    def _refresh_shown(self):
        if self._points_layer is not None:
            self._points_layer.shown = self._compute_shown()

    # ── Layer creation ────────────────────────────────────────────

    def _create_layer(self):
        self._points_layer = self.viewer.add_points(
            self._points,
            properties=self._properties,
            shown=self._compute_shown(),
            face_color=self._face_colors,
            size=5,
            name=f"pose: {self._container_key}",
        )
        self._points_layer.text = {"string": "{keypoint}", "visible": False, "size": 9}

    # ── Controls dock widget ──────────────────────────────────────

    def _add_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        n_frames = len(next(iter(self.pose_data.values()))["timestamps"])
        has_conf = any(
            kp["confidence"] is not None for kp in self.pose_data.values()
        )
        info = f"{len(self.keypoints)} keypoints, {n_frames} frames"
        if not has_conf:
            info += " (no confidence data)"
        layout.addWidget(QLabel(info))

        # Confidence threshold
        conf_row = QWidget()
        cl = QHBoxLayout(conf_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(QLabel("Confidence >="))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.0)
        self._conf_spin.setEnabled(has_conf)
        self._conf_spin.valueChanged.connect(self._on_conf)
        cl.addWidget(self._conf_spin)
        layout.addWidget(conf_row)

        # Point size
        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Point size"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 50)
        self._size_spin.setValue(5)
        self._size_spin.valueChanged.connect(self._on_size)
        sl.addWidget(self._size_spin)
        layout.addWidget(size_row)

        # Show text
        self._text_cb = QCheckBox("Show keypoint labels")
        self._text_cb.toggled.connect(self._on_text)
        layout.addWidget(self._text_cb)

        # Select all / none
        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(lambda: self._set_all_kp(True))
        btn_none.clicked.connect(lambda: self._set_all_kp(False))
        bl.addWidget(btn_all)
        bl.addWidget(btn_none)
        layout.addWidget(btn_row)

        # Keypoint checkboxes in scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        kp_widget = QWidget()
        kp_layout = QVBoxLayout(kp_widget)
        kp_layout.setSpacing(2)

        self._kp_cbs: dict[str, QCheckBox] = {}
        for kp in self.keypoints:
            cb = QCheckBox(kp)
            cb.setChecked(True)
            cb.toggled.connect(self._on_kp_toggle)
            kp_layout.addWidget(cb)
            self._kp_cbs[kp] = cb

        kp_layout.addStretch()
        scroll.setWidget(kp_widget)
        layout.addWidget(scroll, stretch=1)

        self.viewer.window.add_dock_widget(panel, name="Pose Filter", area="right")

    def _on_conf(self, val: float):
        self.conf_threshold = val
        self._refresh_shown()

    def _on_size(self, val: int):
        if self._points_layer:
            self._points_layer.size = val

    def _on_text(self, checked: bool):
        if self._points_layer:
            self._points_layer.text.visible = checked

    def _on_kp_toggle(self):
        self.hidden = {
            name for name, cb in self._kp_cbs.items() if not cb.isChecked()
        }
        self._refresh_shown()

    def _set_all_kp(self, checked: bool):
        for cb in self._kp_cbs.values():
            cb.setChecked(checked)


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="View NWB pose estimation in napari"
    )
    parser.add_argument(
        "source", nargs="?", help="Local .nwb file path or remote URL"
    )
    parser.add_argument("--dandiset", help="DANDI dandiset ID (e.g. 000409)")
    parser.add_argument("--asset", help="DANDI asset ID")
    parser.add_argument("--t-start", type=float, default=None, help="Slice start (s)")
    parser.add_argument("--t-stop", type=float, default=None, help="Slice stop (s)")
    parser.add_argument(
        "--trial", type=int, default=None, help="Trial index (reads bounds from nwb.trials)"
    )
    parser.add_argument(
        "--container",
        default="LeftCamera",
        help="PoseEstimation container name (default: LeftCamera)",
    )
    args = parser.parse_args()

    if args.dandiset and args.asset:
        url = f"https://api.dandiarchive.org/api/assets/{args.asset}/download/"
    elif args.source:
        url = args.source
    else:
        parser.error("Provide a file/URL or --dandiset + --asset")
        return

    print(f"Opening: {url}")
    nwb, io = open_nwb(url)

    containers = find_pose_containers(nwb)
    if not containers:
        print("No PoseEstimation found in this NWB file.")
        sys.exit(1)

    print("Pose containers:")
    for name, (_proc, cont) in containers.items():
        kps = list(cont.pose_estimation_series.keys())
        print(f"  {name}: {len(kps)} keypoints — {kps}")

    key = args.container
    if key not in containers:
        print(f"'{key}' not found. Available: {list(containers.keys())}")
        sys.exit(1)
    _, container = containers[key]

    t_start, t_stop = args.t_start, args.t_stop
    if args.trial is not None:
        t_start, t_stop = get_trial_bounds(nwb, args.trial)
        print(f"Trial {args.trial}: {t_start:.3f}s — {t_stop:.3f}s")

    print(f"Loading pose from '{key}'...")
    pv = NWBPoseViewer(nwb, key, container, t_start, t_stop)
    pv._io = io  # prevent GC from closing the NWB file
    napari.run()


if __name__ == "__main__":
    main()
