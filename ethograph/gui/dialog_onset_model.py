"""LightGBM onset-model dialogs (Model menu).

Two non-modal dialogs around :mod:`ethograph.labels.onset_model`:

* **Train** — the user names a model, ticks one or more point-event classes
  to predict (state events are out of scope; the model assumes at most one
  event per class per trial), ticks features and, per dim (keypoints,
  individuals, space, …), which values to include. The current session's
  existing point events become training trials stored under
  ``~/.ethograph/models/{name}/train_data``; more sessions can be added by
  reopening the dialog there, and Train fits one classifier per class from
  everything collected so far.
* **Predict** — pick a trained model and apply it to the current session. All
  of the model's classes are predicted in one pass; a trial that already
  carries a class is never overridden for that class. Each predicted event
  carries the model's own confidence into the labels TSV.

Both dialogs run over **the trials the trials table shows** — its filters are
the one place trials are included or excluded for every operation (see
``docs/source/advanced/metadata.md``); neither dialog has filters of its own.

When the classes follow a stereotypic order, both dialogs can use the model's
sequence CRF: Train fits it, Predict decodes the whole trial at once so the
events come out in an order the training data actually showed.

Extraction runs per trial through the same ``DataLoader.select`` path the
plots use, but on throwaway loaders holding no display offset, so it never
disturbs the GUI's navigation state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ethograph.gui.dialog_label_gridview import LabelGridViewDialog
from ethograph.gui.notify import notify
from ethograph.io.catalog import PynappleLoader, XarrayLoader
from ethograph.labels import onset_model as om
from ethograph.labels.intervals import EVENT_TYPE_POINT, LABELING_AUTOMATED, NO_RECIPIENT, add_point
from ethograph.labels.tsv_store import get_trial_from_tsv

logger = logging.getLogger(__name__)

_CAVEAT = (
    "Only point events can be predicted, and the model assumes each trial "
    "contains each ticked event at most once. Trials that do not carry a "
    "ticked event simply contribute nothing to that event's classifier."
)


# ---------------------------------------------------------------------------
# Session access helpers
# ---------------------------------------------------------------------------


def _base_loader(app_state):
    """The real DataLoader behind the console's DerivedLoader wrapper."""
    loader = getattr(app_state, "data_loader", None)
    return getattr(loader, "base", loader)


def _iter_trial_windows(app_state) -> Iterator[tuple[int | str, object, float | None, float | None, float]]:
    """Yield ``(trial_id, loader, t0, t1, shift)`` per visible trial.

    ``loader.select(feature, sel, t0, t1)`` returns times on the loader's own
    clock; subtracting *shift* makes them trial-relative — the clock labels
    are stored in. Loaders are throwaway (no display-offset provider), so the
    GUI's navigation state is untouched.
    """
    base = _base_loader(app_state)
    if base is None:
        return
    if base.backend == "xarray":
        dt = app_state.dt
        for tid in app_state.trials:
            yield tid, XarrayLoader(dt.trial(tid)), None, None, 0.0
    else:
        sc = getattr(app_state, "source_collection", None)
        fresh = PynappleLoader(base.data, base.catalog)
        for tid in app_state.trials:
            idx = sc.trial_index(tid) if sc is not None else None
            if idx is None:
                continue
            trial_range = sc.trial_range(idx)
            yield tid, fresh, trial_range.start_s, trial_range.end_s, trial_range.start_s


def _point_rows(df: pd.DataFrame, label_id: int) -> pd.DataFrame:
    """Rows of *df* that are the target point event."""
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["labels"] == label_id
    if "event_type" in df.columns:
        mask &= df["event_type"] == EVENT_TYPE_POINT
    else:
        mask &= df["offset_s"].isna()
    return df[mask]


def _point_mappings(app_state) -> dict[int, str]:
    """Point-event label classes: ``{label_id: name}``."""
    mappings = getattr(app_state, "_label_mappings", None) or {}
    return {
        int(label_id): str(info.get("name", label_id))
        for label_id, info in sorted(mappings.items())
        if isinstance(label_id, int)
        and label_id != 0
        and info.get("event_type", "state") == EVENT_TYPE_POINT
    }


# ---------------------------------------------------------------------------
# Feature tree — feature ▸ dim ▸ value, all checkable
# ---------------------------------------------------------------------------


class FeatureTree(QTreeWidget):
    """Checkable tree of features, their dims, and each dim's values.

    Auto-tristate makes the feature row an "All" toggle and dim rows show the
    partial state — the checked values per dim are exactly the subset the
    model config stores (see ``onset_model.enumerate_columns``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)

    def populate_from_loader(self, app_state) -> None:
        """One top-level item per catalog feature (derived features excluded —
        console recipes live for one trial and cannot travel between sessions)."""
        self.clear()
        loader = getattr(app_state, "data_loader", None)
        base = _base_loader(app_state)
        if base is None:
            return
        derived = getattr(loader, "derived", None) or {}
        for feature in base.catalog.feature_choices():
            if feature in derived:
                continue
            item = QTreeWidgetItem(self, [feature])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(0, Qt.Unchecked)
            for dim, values in (base.feature_dims(feature) or {}).items():
                dim_item = QTreeWidgetItem(item, [dim])
                dim_item.setFlags(dim_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Unchecked)

    def populate_from_config(self, features: dict[str, dict[str, list[str]]]) -> None:
        """Read-only view of a frozen config (existing model)."""
        self.clear()
        for feature, dims in features.items():
            item = QTreeWidgetItem(self, [feature])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(0, Qt.Checked)
            for dim, values in dims.items():
                dim_item = QTreeWidgetItem(item, [dim])
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Checked)
        self.expandAll()

    def selected_features(self) -> dict[str, dict[str, list[str]]]:
        """The checked subset as a model config. Raises ``ValueError`` when a
        checked feature leaves one of its dims with no values."""
        out: dict[str, dict[str, list[str]]] = {}
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.checkState(0) == Qt.Unchecked:
                continue
            feature = item.text(0)
            dims: dict[str, list[str]] = {}
            for j in range(item.childCount()):
                dim_item = item.child(j)
                values = [
                    dim_item.child(k).text(0)
                    for k in range(dim_item.childCount())
                    if dim_item.child(k).checkState(0) == Qt.Checked
                ]
                if not values:
                    raise ValueError(f"Feature {feature!r}: no values ticked for dim {dim_item.text(0)!r}.")
                dims[dim_item.text(0)] = values
            out[feature] = dims
        if not out:
            raise ValueError("Tick at least one feature.")
        return out


# ---------------------------------------------------------------------------
# Train dialog
# ---------------------------------------------------------------------------


class TrainOnsetDialog(QDialog):
    """Create a LightGBM onset model, collect training data, and train it."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LightGBM: Train onset detector")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.app_state = meta.app_state
        self._config: om.OnsetModelConfig | None = None

        layout = QVBoxLayout(self)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("New model…")
        for name in om.list_models():
            self.model_combo.addItem(name)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, stretch=1)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("model name")
        model_row.addWidget(self.name_edit, stretch=1)
        layout.addLayout(model_row)

        target_group = QGroupBox("1 — Point events to predict")
        target_lay = QVBoxLayout(target_group)
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(110)
        for label_id, name in _point_mappings(self.app_state).items():
            item = QListWidgetItem(f"{name} ({label_id})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.target_list.addItem(item)
        target_lay.addWidget(self.target_list)
        caveat = QLabel(_CAVEAT)
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color: grey; font-size: 10px;")
        target_lay.addWidget(caveat)
        layout.addWidget(target_group)

        feature_group = QGroupBox("2 — Features (tick dims/values to include)")
        feature_lay = QVBoxLayout(feature_group)
        self.tree = FeatureTree()
        self.tree.populate_from_loader(self.app_state)
        feature_lay.addWidget(self.tree)
        layout.addWidget(feature_group)

        params_group = QGroupBox("3 — Parameters")
        params_lay = QFormLayout(params_group)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.05, 30.0)
        self.window_spin.setSingleStep(0.1)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(0.5)
        self.window_spin.setToolTip("Width of the feature window the classifier sees around each frame")
        params_lay.addRow("Window size:", self.window_spin)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.005, 5.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setSuffix(" s")
        self.tolerance_spin.setValue(0.05)
        self.tolerance_spin.setToolTip(
            "Frames within this distance of the labelled event count as positive;\n"
            "their sample weight is a Gaussian bump peaking at the event."
        )
        params_lay.addRow("Tolerance:", self.tolerance_spin)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 2000)
        self.max_iter_spin.setValue(200)
        params_lay.addRow("Boosting iterations:", self.max_iter_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.1)
        params_lay.addRow("Learning rate:", self.lr_spin)
        self.crf_check = QCheckBox("Model the order of the events (CRF)")
        self.crf_check.setToolTip(
            "Tick when the ticked classes follow a stereotypic order in a trial\n"
            "(A then B then C). A linear-chain CRF then decodes the whole trial at\n"
            "once, so the predicted events come out in an order the training trials\n"
            "actually showed. Needs at least 2 training trials."
        )
        params_lay.addRow("Sequence:", self.crf_check)
        layout.addWidget(params_group)

        data_group = QGroupBox("4 — Training data")
        data_lay = QVBoxLayout(data_group)
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(90)
        data_lay.addWidget(self.session_list)
        self.add_btn = QPushButton("Add current session's events")
        self.add_btn.setAutoDefault(False)
        self.add_btn.setToolTip("Only trials visible in the trials table contribute.")
        self.add_btn.clicked.connect(self._add_session)
        data_lay.addWidget(self.add_btn)
        layout.addWidget(data_group)

        self.train_btn = QPushButton("Train")
        self.train_btn.setAutoDefault(False)
        self.train_btn.clicked.connect(self._train)
        layout.addWidget(self.train_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.resize(460, 640)
        self._on_model_changed(self.model_combo.currentText())

    # ------------------------------------------------------------------

    def _config_widgets(self):
        return (
            self.name_edit,
            self.target_list,
            self.tree,
            self.window_spin,
            self.tolerance_spin,
            self.max_iter_spin,
            self.lr_spin,
            self.crf_check,
        )

    def _on_model_changed(self, text: str):
        """New model: everything editable. Existing model: config is frozen —
        the stored feature layout defines the classifier's input columns, so
        editing it would invalidate every stored training trial."""
        is_new = self.model_combo.currentIndex() == 0
        for widget in self._config_widgets():
            widget.setEnabled(is_new)
        self._config = None
        self.session_list.clear()
        if is_new:
            self.tree.populate_from_loader(self.app_state)
            self.status_label.setText("")
            return
        self._config = om.load_config(text)
        self.name_edit.setText(self._config.name)
        self._show_config_targets(self._config)
        self.tree.populate_from_config(self._config.features)
        self.window_spin.setValue(self._config.window_s)
        self.tolerance_spin.setValue(self._config.tolerance_s)
        self.max_iter_spin.setValue(self._config.max_iter)
        self.lr_spin.setValue(self._config.learning_rate)
        self.crf_check.setChecked(self._config.use_crf)
        self._refresh_sessions()
        trained = "trained" if om.is_trained(text) else "not trained yet"
        self.status_label.setText(f"Config is frozen for an existing model ({trained}).")

    def _show_config_targets(self, config: om.OnsetModelConfig):
        """Read-only view of a frozen model's targets."""
        self.target_list.clear()
        for label_id, name in config.targets.items():
            item = QListWidgetItem(f"{name} ({label_id})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.target_list.addItem(item)

    def _selected_targets(self) -> dict[int, str]:
        """The ticked point-event classes as ``{label_id: name}``."""
        targets: dict[int, str] = {}
        for i in range(self.target_list.count()):
            item = self.target_list.item(i)
            if item.checkState() == Qt.Checked:
                targets[int(item.data(Qt.UserRole))] = item.text().rsplit(" (", 1)[0]
        return targets

    def _refresh_sessions(self):
        self.session_list.clear()
        if self._config is None:
            return
        for session, meta_dict in om.list_sessions(self._config.name).items():
            n = meta_dict.get("n_trials", "?")
            self.session_list.addItem(QListWidgetItem(f"{session} — {n} trials"))

    def _ensure_config(self) -> om.OnsetModelConfig | None:
        """The active config, creating + saving a new model's on first use."""
        if self._config is not None:
            return self._config
        name = self.name_edit.text().strip()
        if not name:
            notify("Give the model a name first.", severity="warning")
            return None
        if name in om.list_models():
            notify(f"A model named {name!r} already exists — pick it from the combo to extend it.",
                   severity="warning")
            return None
        targets = self._selected_targets()
        if not targets:
            notify(
                "Tick at least one point-event class — mark a label as a point event first if none are listed.",
                severity="warning",
            )
            return None
        try:
            features = self.tree.selected_features()
        except ValueError as e:
            notify(str(e), severity="warning")
            return None
        self._config = om.OnsetModelConfig(
            name=name,
            targets=targets,
            features=features,
            window_s=float(self.window_spin.value()),
            tolerance_s=float(self.tolerance_spin.value()),
            max_iter=int(self.max_iter_spin.value()),
            learning_rate=float(self.lr_spin.value()),
            use_crf=self.crf_check.isChecked(),
        )
        om.save_config(self._config)
        self.model_combo.addItem(name)
        self.model_combo.setCurrentText(name)  # re-enters _on_model_changed → frozen view
        return self._config

    # ------------------------------------------------------------------

    def _add_session(self):
        config = self._ensure_config()
        if config is None:
            return
        source_path = getattr(self.app_state, "nc_file_path", None)
        if not source_path:
            notify("No dataset loaded.", severity="warning")
            return
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            notify("This session has no labels.", severity="warning")
            return

        session = om.session_id(source_path)
        n_written = 0
        n_multi = 0
        per_target: dict[int, int] = {label: 0 for label in config.targets}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tid, loader, t0, t1, shift in _iter_trial_windows(self.app_state):
                trial_rows = df[df["trial"] == tid]
                y_times: dict[int, float] = {}
                for label in config.targets:
                    rows = _point_rows(trial_rows, label)
                    if rows.empty:
                        continue
                    if len(rows) > 1:
                        n_multi += 1
                    y_times[label] = float(rows.iloc[0]["onset_s"])
                    per_target[label] += 1
                if not y_times:
                    continue
                time, data = om.extract_features(loader, config.features, t0, t1)
                om.write_trial_training_data(config.name, session, tid, time - shift, data, y_times)
                n_written += 1
        except ValueError as e:
            notify(f"Extraction failed: {e}", severity="error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not n_written:
            notify(f"No trials carry any of: {config.describe_targets()}.", severity="warning")
            return
        om.write_session_meta(
            config.name,
            session,
            {
                "source_path": str(source_path),
                "n_trials": n_written,
                "columns": [c.name for c in om.enumerate_columns(config.features)],
                "added": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self._refresh_sessions()
        per_target_text = ", ".join(f"{config.target_name(label)}: {n}" for label, n in per_target.items())
        msg = f"Stored {n_written} training trials from this session ({per_target_text})."
        if n_multi:
            msg += f" {n_multi} trials had multiple events of one class — only the first was used."
        self.status_label.setText(msg)
        notify(msg)

    @staticmethod
    def _named_order(config: om.OnsetModelConfig, key: str) -> str:
        """``"3-4"`` as the class names the user knows, ``"peck→land"``."""
        return "→".join(config.target_name(int(label)) for label in key.split("-"))

    def _train(self):
        config = self._ensure_config()
        if config is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            summary = om.train_model(config.name)
        except ValueError as e:
            notify(f"Training failed: {e}", severity="error")
            return
        finally:
            QApplication.restoreOverrideCursor()
        per_target = ", ".join(
            f"{config.target_name(label)}: {stats['n_trials']} trials / {stats['n_positive']} positive frames"
            for label, stats in summary["targets"].items()
        )
        msg = (
            f"Trained {config.name!r} on {summary['n_trials']} trials "
            f"from {summary['n_sessions']} session(s) — {per_target}."
        )
        if "crf" in summary:
            orders = summary["crf"]["sequences"]
            shown = ", ".join(f"{self._named_order(config, key)} ×{n}" for key, n in list(orders.items())[:3])
            msg += f" Sequence model trained on the orders it saw: {shown}."
        self.status_label.setText(msg)
        notify(msg)


# ---------------------------------------------------------------------------
# Predict dialog
# ---------------------------------------------------------------------------


class PredictOnsetDialog(QDialog):
    """Apply a trained onset model to the current session.

    Every class the model was trained on is predicted in one pass, over the
    trials the trials table shows. A trial that already carries a class is
    skipped for that class — the model only fills gaps, it never overrides.
    Each written event carries the model's confidence in the label's
    ``confidence`` column.
    """

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self._review_dialog: LabelGridViewDialog | None = None
        self.setWindowTitle("LightGBM: Predict onsets")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.model_combo = QComboBox()
        for name in om.list_models():
            self.model_combo.addItem(name)
        self.model_combo.currentTextChanged.connect(self._refresh_info)
        self.model_combo.setToolTip(
            "Trained models from ~/.ethograph/models. The line below says which\n"
            "classes this one predicts and what it was trained on."
        )
        form.addRow("Model:", self.model_combo)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: grey; font-size: 10px;")
        form.addRow("", self.info_label)

        self.individual_combo = QComboBox()
        for name in self.app_state.label_individuals():
            self.individual_combo.addItem(name)
        current = self.app_state.selected_individual()
        if current is not None:
            idx = self.individual_combo.findText(current)
            if idx >= 0:
                self.individual_combo.setCurrentIndex(idx)
        self.individual_combo.setToolTip("Predicted events are written for this individual")
        form.addRow("Individual:", self.individual_combo)

        self.crf_check = QCheckBox("Use the sequence model")
        self.crf_check.setChecked(True)
        self.crf_check.setToolTip(
            "Decode the whole trial at once so the events come out in an order the\n"
            "training trials showed, instead of picking each class's best frame\n"
            "independently. A class the decoded order leaves out gets no prediction."
        )
        form.addRow("", self.crf_check)

        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setSingleStep(0.05)
        self.min_conf_spin.setValue(0.0)
        self.min_conf_spin.setToolTip(
            "A prediction scoring below this is not written at all: the trial is\n"
            "left unlabelled for that class rather than given a doubtful label.\n"
            "\n"
            "Confidence combines how strongly the model believes (the peak of its\n"
            "probability curve) with how localised that belief is in time — a\n"
            "strong belief smeared over two seconds scores low, and so does a\n"
            "sharp spike the model barely believes in.\n"
            "\n"
            "Leave it at 0 to write every prediction and triage afterwards with\n"
            "Review predictions — nothing is lost that way, and the confidence is\n"
            "on each label either way."
        )
        form.addRow("Min confidence:", self.min_conf_spin)
        layout.addLayout(form)

        # Which trials: the trials table's filters, and nothing else — the one
        # place trials are included or excluded for every operation.
        self.trials_note = QLabel("")
        self.trials_note.setWordWrap(True)
        self.trials_note.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.trials_note)

        self.run_btn = QPushButton("Predict missing onsets")
        self.run_btn.setAutoDefault(False)
        self.run_btn.setToolTip(
            "Predict every class of this model, for every trial the trials table\n"
            "shows that does not already carry that class. Existing labels are never\n"
            "overwritten. Nothing is saved to disk until you press Ctrl+S."
        )
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        #: What the last run wrote — the label classes and the trials that got
        #: one — so the review grid opens on exactly those predictions.
        self._reviewable: tuple[list[int], set[str]] = ([], set())
        self.review_btn = QPushButton("Review predictions…")
        self.review_btn.setAutoDefault(False)
        self.review_btn.setEnabled(False)
        self.review_btn.setToolTip(
            "Open the label-frames grid on the events just predicted: the video\n"
            "frame at each one, with its confidence, and a click to jump there."
        )
        self.review_btn.clicked.connect(self._review)
        layout.addWidget(self.review_btn)

        self.resize(420, 480)
        self._refresh_info(self.model_combo.currentText())
        self._refresh_trials_note()
        self.app_state.trials_changed.connect(self._refresh_trials_note)

    # ------------------------------------------------------------------

    def _refresh_info(self, name: str):
        if not name:
            self.info_label.setText("No models found in ~/.ethograph/models.")
            self.run_btn.setEnabled(False)
            return
        config = om.load_config(name)
        trained = "trained" if om.is_trained(name) else "NOT trained yet"
        n_sessions = len(om.list_sessions(name))
        self.info_label.setText(
            f"Predicts {config.describe_targets()} · "
            f"{len(om.enumerate_columns(config.features))} feature columns · "
            f"{n_sessions} training session(s) · {trained}"
            + (" · sequence model" if config.use_crf else "")
            + "."
        )
        self.crf_check.setEnabled(config.use_crf)
        if not config.use_crf:
            self.crf_check.setChecked(False)
        self.run_btn.setEnabled(om.is_trained(name))

    def _refresh_trials_note(self, *_args):
        """Say which trials the run covers: the ones the trials table shows."""
        n = len(getattr(self.app_state, "trials", None) or [])
        self.trials_note.setText(
            f"Runs over the {n} trial(s) the trials table currently shows — filter there "
            "(Navigation section) to include or exclude trials."
        )

    # ------------------------------------------------------------------

    def _run(self):
        name = self.model_combo.currentText()
        if not name:
            return
        try:
            bundle = om.load_bundle(name)
        except ValueError as e:
            notify(str(e), severity="warning")
            return
        config = om.bundle_config(bundle)
        individual = self.individual_combo.currentText()
        min_conf = float(self.min_conf_spin.value())

        use_crf = self.crf_check.isChecked()
        df = getattr(self.app_state, "_all_labels_df", None)
        n_predicted = n_existing = n_low = n_absent = 0
        per_target: dict[int, int] = {label: 0 for label in config.targets}
        predicted_trials: set[str] = set()
        errors: list[str] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tid, loader, t0, t1, shift in _iter_trial_windows(self.app_state):
                trial_rows = df[df["trial"] == tid] if df is not None else None
                # Each class is filled independently: a trial already carrying
                # one of them can still receive the others.
                wanted = [
                    label
                    for label in config.targets
                    if trial_rows is None or _point_rows(trial_rows, label).empty
                ]
                n_existing += len(config.targets) - len(wanted)
                if not wanted:
                    continue
                try:
                    time, data = om.extract_features(loader, config.features, t0, t1)
                    predictions = om.predict_events(bundle, time - shift, data, use_crf=use_crf)
                except ValueError as e:
                    errors.append(f"trial {tid}: {e}")
                    continue
                written = False
                for label in wanted:
                    prediction = predictions.get(label)
                    if prediction is None:
                        # Only the sequence model can leave a class out: the
                        # decoded order says it did not happen in this trial.
                        n_absent += 1
                        continue
                    if prediction.confidence < min_conf:
                        n_low += 1
                        continue
                    trial_df = get_trial_from_tsv(self.app_state._all_labels_df, tid)
                    trial_df = add_point(
                        trial_df,
                        prediction.time,
                        label,
                        individual,
                        NO_RECIPIENT,
                        confidence=prediction.confidence,
                        labeling_method=LABELING_AUTOMATED,
                    )
                    self.app_state.set_trial_intervals(tid, trial_df)
                    per_target[label] += 1
                    n_predicted += 1
                    written = True
                    predicted_trials.add(str(tid))
                    df = self.app_state._all_labels_df
                if written:
                    self.app_state.set_trial_meta_attr(tid, "prediction_source", f"lightgbm:{name}")
        finally:
            QApplication.restoreOverrideCursor()

        self._reviewable = ([label for label, n in per_target.items() if n], predicted_trials)
        self.review_btn.setEnabled(bool(n_predicted))
        if n_predicted:
            self.app_state.changes_saved = False
            current = getattr(self.app_state, "trials_sel", None)
            if current is not None:
                self.app_state.label_intervals = get_trial_from_tsv(self.app_state._all_labels_df, current)
            data_widget = getattr(self.meta, "data_widget", None)
            if data_widget is not None:
                data_widget.update_main_plot(preserve_x_range=True)
            labels_widget = getattr(self.meta, "labels_widget", None)
            if labels_widget is not None:
                labels_widget.refresh_labels_shapes_layer()

        per_target_text = ", ".join(f"{config.target_name(label)}: {n}" for label, n in per_target.items())
        parts = [f"Predicted {n_predicted} onsets ({per_target_text})."]
        if n_existing:
            parts.append(f"{n_existing} trial/class pairs already labelled (untouched).")
        if n_low:
            parts.append(f"{n_low} below the confidence threshold.")
        if n_absent:
            parts.append(f"{n_absent} left out by the decoded event order.")
        if errors:
            parts.append(f"{len(errors)} trials failed — first: {errors[0]}")
            logger.warning("Onset prediction failures: %s", "; ".join(errors))
        msg = " ".join(parts)
        self.status_label.setText(msg)
        notify(msg, severity="warning" if errors else "info")
        if n_predicted:
            notify("Review the predictions and save with Ctrl+S.")

    def _review(self):
        """Open the label-frames grid on what the last run predicted."""
        label_ids, trials = self._reviewable
        if not label_ids:
            return
        self._review_dialog = LabelGridViewDialog(
            self.meta,
            parent=self.parent() or self,
            label_ids=label_ids,
            trials=trials,
        )
        self._review_dialog.show()
        self._review_dialog.raise_()
        self._review_dialog.activateWindow()
