"""Which video (S3D) dimensions are stereotypic for which behaviour (Model menu).

S3D gives 1024 dimensions per frame and a segmentation config wants a handful.
This dialog puts a face on :func:`ethograph.video_features.select.rank_features`:
it extracts the chosen video feature and the curated labels for every trial the
trials table shows, ranks the dimensions by Cohen's d, and draws the result as
a class × dimension heatmap. The top-k indices come out as a YAML list ready to
paste into a segment config, and the whole ranking can be saved as ``.npz``.

Two conventions are inherited from the ranking itself and matter to the reader:
background (unlabelled) frames are the contrast, never a class, and a
dimension's score is averaged over trials, so one lucky trial cannot promote it.

Only ``manual`` and ``curated`` labels are used — ranking video dimensions
against another model's output would measure that model, not the behaviour.

**Responsiveness**: the run is synchronous under a busy cursor, with the status
label naming the trial being extracted and ``processEvents`` pumped between
trials. Extraction dominates the cost and is per-trial coarse, so a thread buys
little against the risk of touching loaders off the GUI thread (the alignment
NWB is not thread-safe). Re-rendering the heatmap for a new top-k never
recomputes.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyqtgraph as pg
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ethograph.features.columns import extract_features, sampling_rate
from ethograph.gui.dialog_onset_model import _base_loader, _iter_trial_windows
from ethograph.gui.file_dialogs import browse_save_file
from ethograph.gui.notify import notify
from ethograph.io.schema import KIND, VIDEO_FEATURE, kind_of
from ethograph.labels.intervals import LABELING_AUTOMATED
from ethograph.labels.ml import intervals_to_dense
from ethograph.video_features.select import FeatureRanking, rank_features

#: A dim this wide is a feature bank, not a keypoint list — the fallback used
#: to offer candidates in a session whose variables declare no ``kind``.
WIDE_DIM = 32


#: Most x tick labels to draw; more than this and the indices collide.
MAX_X_TICKS = 20

#: Viridis-like ramp, spelled out so no colormap package has to be present.
_COLORMAP = pg.ColorMap([0.0, 0.5, 1.0], [(30, 32, 70), (58, 148, 140), (250, 231, 92)])

NOTHING_TO_RANK = "Nothing to rank"


# ---------------------------------------------------------------------------
# Which variables are video features
# ---------------------------------------------------------------------------


def session_dataset(app_state) -> xr.Dataset | None:
    """The ``xr.Dataset`` whose attrs describe this session, if it has one."""
    ds = getattr(app_state, "ds", None)
    if isinstance(ds, xr.Dataset):
        return ds
    dt = getattr(app_state, "dt", None)
    trials = list(getattr(dt, "trials", None) or [])
    return dt.trial(trials[0]) if trials else None


def video_feature_choices(app_state) -> list[str]:
    """Catalog features that are video features.

    A session whose variables declare ``kind`` is taken at its word. One that
    declares nothing — every file written before ``io/schema.py`` — falls back
    to shape: a feature with a non-time dim of at least :data:`WIDE_DIM`
    values is a feature bank.
    """
    base = _base_loader(app_state)
    if base is None:
        return []
    features = base.catalog.feature_choices()
    ds = session_dataset(app_state)
    if ds is not None:
        declared = {str(name): kind_of(var) for name, var in ds.data_vars.items() if var.attrs.get(KIND)}
        if declared:
            return [f for f in features if declared.get(f) == VIDEO_FEATURE]
    return [f for f in features if _is_wide(base, f)]


def _is_wide(loader: Any, feature: str) -> bool:
    return any(len(values) >= WIDE_DIM for values in (loader.feature_dims(feature) or {}).values())


def all_values_selection(loader: Any, feature: str) -> dict[str, dict[str, list[str]]]:
    """The ``extract_features`` config taking every value of every dim."""
    return {feature: {dim: list(values) for dim, values in (loader.feature_dims(feature) or {}).items()}}


def wide_dim(dims: dict[str, list[str]]) -> str:
    """The dim the ranking indexes: the one with the most values (the feature bank)."""
    if not dims:
        raise ValueError("The feature has no dims to rank along")
    return max(dims, key=lambda d: len(dims[d]))


def curated_rows(df: pd.DataFrame | None, trial: Any) -> pd.DataFrame:
    """This trial's labels a ranking may learn from: never ``automated``."""
    if df is None or df.empty:
        return pd.DataFrame()
    rows = df[df["trial"] == trial]
    if "labeling_method" in rows.columns:
        rows = rows[rows["labeling_method"] != LABELING_AUTOMATED]
    return rows


def class_names(app_state, class_ids: Iterable[int]) -> list[str]:
    """Label names for *class_ids*, falling back to the integer id."""
    mappings = getattr(app_state, "_label_mappings", None) or {}
    return [str(mappings.get(int(cid), {}).get("name", cid)) for cid in class_ids]


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class VideoFeatureRankDialog(QDialog):
    """Rank a video feature's dimensions by Cohen's d, per behaviour class."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video features: rank by Cohen's d")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self._ranking: FeatureRanking | None = None
        #: ``(feature, wide dim)`` of the last run — what the YAML line names.
        self._ranked: tuple[str, str] | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.feature_combo = QComboBox()
        for feature in video_feature_choices(self.app_state):
            self.feature_combo.addItem(feature)
        self.feature_combo.setToolTip(
            "The video-feature bank to rank. Its dimensions are scored one at a\n"
            "time against each behaviour class — no model is fitted."
        )
        form.addRow("Feature:", self.feature_combo)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(20)
        self.topk_spin.setToolTip("How many of the best dimensions to show and to paste. Re-draws only.")
        self.topk_spin.valueChanged.connect(self._render)
        form.addRow("Top k:", self.topk_spin)
        layout.addLayout(form)

        # Which trials: the trials table's filters, and nothing else — the one
        # place trials are included or excluded for every operation.
        self.trials_note = QLabel("")
        self.trials_note.setWordWrap(True)
        self.trials_note.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.trials_note)

        self.run_btn = QPushButton("Run")
        self.run_btn.setAutoDefault(False)
        self.run_btn.setToolTip(
            "Extract this feature and the curated labels for every trial the trials\n"
            "table shows, then rank the dimensions by mean-over-trials Cohen's d."
        )
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setLabel("bottom", "Feature index")
        self.plot_item.invertY(True)
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(_COLORMAP)
        self.plot_item.addItem(self.image_item)
        self.colorbar = pg.ColorBarItem(
            values=(0.0, 1.0),
            colorMap=_COLORMAP,
            interactive=False,
            width=15,
            label="Cohen's d",
        )
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot_item)
        layout.addWidget(self.plot_widget, stretch=1)

        paste_row = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setReadOnly(True)
        self.yaml_edit.setPlaceholderText("feature: {dims: […]} — appears after a run")
        paste_row.addWidget(self.yaml_edit, stretch=1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy)
        paste_row.addWidget(self.copy_btn)
        layout.addLayout(paste_row)

        self.save_btn = QPushButton("Save ranking…")
        self.save_btn.setAutoDefault(False)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Write the full ranking (every dimension, every class) as an .npz.")
        self.save_btn.clicked.connect(self._save)
        layout.addWidget(self.save_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.resize(700, 560)
        self._refresh_trials_note()
        self.app_state.trials_changed.connect(self._refresh_trials_note)
        if self.feature_combo.count() == 0:
            self.run_btn.setEnabled(False)
            self.status_label.setText(
                "This session carries no video features — no variable declares "
                f'kind="{VIDEO_FEATURE}", and none has a dim of {WIDE_DIM} or more values.'
            )

    # ------------------------------------------------------------------

    def _refresh_trials_note(self, *_args):
        """Say which trials the run covers: the ones the trials table shows."""
        n = len(getattr(self.app_state, "trials", None) or [])
        self.trials_note.setText(
            f"Runs over the {n} trial(s) the trials table currently shows — filter there "
            "(Navigation section) to include or exclude trials."
        )

    # ------------------------------------------------------------------

    def _individual(self) -> str | None:
        """Whose labels define the classes: the sidebar's actor."""
        individual = self.app_state.selected_individual()
        if individual is not None:
            return individual
        names = self.app_state.label_individuals()
        return names[0] if names else None

    def _collect(self, feature: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
        """``(trials, n_unlabelled, n_background)`` for *feature*.

        A trial with no curated label, or whose labels are entirely
        background, carries no contrast and is left out.
        """
        df = getattr(self.app_state, "_all_labels_df", None)
        individual = self._individual()
        collected: list[tuple[np.ndarray, np.ndarray]] = []
        n_unlabelled = n_background = 0
        for tid, loader, t0, t1, shift in _iter_trial_windows(self.app_state):
            rows = curated_rows(df, tid)
            if rows.empty or individual is None:
                n_unlabelled += 1
                continue
            self.status_label.setText(f"Extracting trial {tid}…")
            QApplication.processEvents()
            selection = all_values_selection(loader, feature)
            self._ranked = (feature, wide_dim(selection[feature]))
            time, values = extract_features(loader, selection, t0, t1)
            trial_time = time - shift  # labels are stored on the trial clock
            dense = intervals_to_dense(rows, sampling_rate(trial_time), [individual], len(trial_time))
            labels = dense[:, 0].astype(np.int64)
            if not np.any(labels):
                n_background += 1
                continue
            collected.append((values, labels))
        return collected, n_unlabelled, n_background

    def _run(self):
        feature = self.feature_combo.currentText()
        if not feature:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            collected, n_unlabelled, n_background = self._collect(feature)
        except ValueError as e:
            self._failed(f"Extraction failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not collected:
            self._nothing_to_rank(n_unlabelled, n_background)
            return
        try:
            ranking = rank_features(collected)
        except ValueError as e:
            self._failed(f"{NOTHING_TO_RANK}: {e}")
            return

        self._ranking = ranking
        self.topk_spin.setMaximum(min(200, max(1, ranking.n_features)))
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._render()
        skipped = ""
        if n_unlabelled or n_background:
            skipped = f" Skipped {n_unlabelled} trial(s) with no curated labels and {n_background} all-background."
        self.status_label.setText(
            f"Ranked {ranking.n_features} dimensions of {feature!r} over {ranking.n_trials} trial(s) "
            f"and {len(ranking.class_ids)} class(es).{skipped}"
        )

    def _nothing_to_rank(self, n_unlabelled: int, n_background: int):
        self._failed(
            f"{NOTHING_TO_RANK}: no trial the trials table shows carries a curated behaviour "
            f"({n_unlabelled} without curated labels, {n_background} entirely background).",
        )

    def _failed(self, message: str):
        """Report why there is no ranking, and leave the previous one cleared."""
        self._ranking = None
        self._clear_results()
        self.status_label.setText(message)
        notify(message, severity="warning")

    def _clear_results(self):
        self.image_item.clear()
        self.yaml_edit.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    # ------------------------------------------------------------------

    def top_indices(self) -> list[int]:
        """The dimensions the paste string names, best first."""
        if self._ranking is None:
            return []
        return [int(i) for i in self._ranking.top(self.topk_spin.value())]

    def yaml_text(self) -> str:
        """The ``features.columns`` entry naming the top-k: ``feature: {dim: [...]}``.

        The dim is the ranked feature's own wide dim, read off the loader —
        ``timm_dims``, ``s3d_dims``, whatever the session calls it — so the
        line pastes into a config for any extractor.
        """
        if self._ranked is None:
            return ""
        feature, dim = self._ranked
        return f"{feature}: {{{dim}: [{', '.join(str(i) for i in self.top_indices())}]}}"

    def _render(self):
        """Draw the class × top-k heatmap. Never recomputes the ranking."""
        ranking = self._ranking
        if ranking is None:
            return
        indices = self.top_indices()
        matrix = np.asarray(ranking.per_class, dtype=float)[indices, :]
        vmax = float(matrix.max()) if matrix.size else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        self.image_item.setImage(matrix, autoLevels=False, levels=(0.0, vmax))
        self.colorbar.setLevels((0.0, vmax))

        names = class_names(self.app_state, ranking.class_ids)
        self.plot_item.getAxis("left").setTicks([[(i + 0.5, name) for i, name in enumerate(names)]])
        step = max(1, len(indices) // MAX_X_TICKS)
        self.plot_item.getAxis("bottom").setTicks(
            [[(i + 0.5, str(dim)) for i, dim in enumerate(indices) if i % step == 0]]
        )
        self.plot_item.setXRange(0, max(len(indices), 1), padding=0)
        self.plot_item.setYRange(0, max(len(names), 1), padding=0)
        self.yaml_edit.setText(self.yaml_text())

    # ------------------------------------------------------------------

    def _copy(self):
        QApplication.clipboard().setText(self.yaml_edit.text())
        notify("Copied the top dimensions to the clipboard.")

    def _save(self):
        if self._ranking is None:
            return
        feature = self.feature_combo.currentText() or "video_features"
        path = browse_save_file(
            self,
            self.app_state,
            "Save feature ranking",
            f"{feature}_cohens_d.npz",
            "NumPy archive (*.npz)",
            preferred_dir=getattr(self.app_state, "nc_file_path", None),
        )
        if not path:
            return
        written = self._ranking.save(path)
        self.status_label.setText(f"Saved the ranking to {written}.")
        notify(f"Saved the ranking to {written}.")
