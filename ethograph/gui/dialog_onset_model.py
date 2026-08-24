"""LightGBM onset-model dialogs (Model menu).

Two non-modal dialogs around :mod:`ethograph.labels.onset_model`:

* **Train** — the user names a model, ticks one or more point-event classes
  to predict (state events are out of scope; the model assumes at most one
  event per class per trial), ticks features and, per dim (keypoints,
  individuals, space, …), which values to include — plus, per feature,
  whether its rate of change comes along as an extra input. The current session's
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

A user who knows the classes run in a stereotypic order says so when
creating the model: tick them, drag them into that order, and Predict then
*flags* the trials that came out otherwise (or got only part of the set)
instead of correcting them. The flag lands in the trials table's
``prediction_check`` column, so filtering there scopes a review to exactly the
trials worth a second look.

Extraction runs per trial through the same ``DataLoader.select`` path the
plots use, but on throwaway loaders holding no display offset, so it never
disturbs the GUI's navigation state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
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
    QHeaderView,
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

from ethograph.gui.dialog_label_gridview import ConfidenceEdit, confidence_display
from ethograph.gui.notify import notify
from ethograph.io.catalog import PynappleLoader, XarrayLoader
from ethograph.labels import onset_curves
from ethograph.labels import onset_model as om
from ethograph.labels.intervals import EVENT_TYPE_POINT, LABELING_AUTOMATED, NO_RECIPIENT, add_point
from ethograph.labels.tsv_store import get_trial_from_tsv
from ethograph.labels.workflow import DEFAULT_CONFIDENCE

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


@dataclass
class PredictionOutcome:
    """What one prediction run wrote, and what it deliberately did not.

    Returned by :func:`predict_onsets` so the Predict dialog and a curation
    workflow report the same run the same way, and so a workflow's later
    steps can pick up exactly the classes and trials it produced.
    """

    model: str
    per_target: dict[int, int]
    target_names: dict[int, str]
    n_predicted: int = 0
    n_existing: int = 0
    n_low: int = 0
    n_absent: int = 0
    trials: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    #: trial -> the declared expectations it broke (see
    #: :func:`~ethograph.labels.onset_model.check_expectations`). Only trials
    #: that broke one are listed; the column records every trial's verdict.
    flagged: dict[str, list[str]] = field(default_factory=dict)

    def label_ids(self) -> list[int]:
        """The classes this run actually wrote at least one event for."""
        return [label for label, n in self.per_target.items() if n]

    def message(self) -> str:
        per_target_text = ", ".join(f"{self.target_names[label]}: {n}" for label, n in self.per_target.items())
        parts = [f"Predicted {self.n_predicted} onsets ({per_target_text})."]
        if self.n_existing:
            parts.append(f"{self.n_existing} trial/class pairs already labelled (untouched).")
        if self.n_low:
            parts.append(f"{self.n_low} below the confidence threshold.")
        if self.n_absent:
            parts.append(f"{self.n_absent} left out by the decoded event order.")
        if self.flagged:
            listed = ", ".join(sorted(self.flagged)[:5])
            parts.append(
                f"{len(self.flagged)} trial(s) did not match what the model expects "
                f"({listed}) — filter the trials table on '{om.EXPECTATION_COLUMN}' to review them."
            )
        if self.errors:
            parts.append(f"{len(self.errors)} trials failed — first: {self.errors[0]}")
        return " ".join(parts)


def _save_curves(app_state, curves_by_trial: dict) -> Path | None:
    """Write this run's probability curves to its own folder under ``labels/``.

    An aid to review, not part of the prediction: a session with no path on
    disk yet simply keeps none, and a failed write is logged rather than
    losing the predictions it belongs to.
    """
    if not curves_by_trial or not app_state.nc_file_path:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = onset_curves.run_dir(app_state.nc_file_path, timestamp)
    try:
        return onset_curves.write_curves(folder / onset_curves.CURVES_FILE, curves_by_trial)
    except OSError as exc:
        logger.warning("Could not write onset curves to %s: %s", folder, exc)
        return None


def _write_expectation_column(meta, verdicts: dict[str, str]) -> None:
    """Record each trial's expectation verdict in the metadata table.

    A string per trial ("ok", "order", "order+missing"), so the trials
    table's funnel filter reads it as a categorical checklist — the same
    reason ``curated`` is "yes"/"no" — and filtering to the flagged trials
    scopes every downstream operation to them, review included.
    """
    if not verdicts:
        return
    trials_widget = getattr(meta, "trials_widget", None)
    if trials_widget is None:
        return
    trials_widget.ensure_tabular_metadata_file()
    trials_widget.set_column_values(om.EXPECTATION_COLUMN, verdicts)


def predict_onsets(
    meta,
    name: str,
    *,
    individual: str,
    min_confidence: float,
) -> PredictionOutcome | None:
    """Apply the trained model *name* to every trial the trials table shows.

    Each class is filled independently and a trial already carrying one is
    never overridden for it. Returns ``None`` when the model cannot be
    loaded — the reason is notified.
    """
    app_state = meta.app_state
    try:
        bundle = om.load_bundle(name)
    except ValueError as e:
        notify(str(e), severity="warning")
        return None
    config = om.bundle_config(bundle)
    outcome = PredictionOutcome(
        model=name,
        per_target={label: 0 for label in config.targets},
        target_names={label: config.target_name(label) for label in config.targets},
    )

    df = getattr(app_state, "_all_labels_df", None)
    #: Every trial's probability curves, written beside the labels once the
    #: run is over — what frame-by-frame review draws under each label.
    curves_by_trial: dict[object, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
    #: Every trial's expectation verdict, written to the metadata table at the
    #: end so a filter can scope a review to the surprising ones.
    verdicts: dict[str, str] = {}
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        for tid, loader, t0, t1, shift in _iter_trial_windows(app_state):
            trial_rows = df[df["trial"] == tid] if df is not None else None
            # Each class is filled independently: a trial already carrying
            # one of them can still receive the others.
            wanted = [label for label in config.targets if trial_rows is None or _point_rows(trial_rows, label).empty]
            outcome.n_existing += len(config.targets) - len(wanted)
            if not wanted:
                continue
            try:
                time, data = om.extract_model_features(loader, config, t0, t1)
                trial_time = time - shift
                result = om.predict_trial(bundle, trial_time, data)
                predictions = result.events
            except ValueError as e:
                outcome.errors.append(f"trial {tid}: {e}")
                continue
            curves_by_trial[tid] = (trial_time, result.curves)
            written = False
            #: What this trial ended up with, which is what the declared
            #: expectations are checked against — a class dropped for low
            #: confidence is absent here, and that absence is the point.
            trial_times: dict[int, float] = {}
            for label in wanted:
                prediction = predictions.get(label)
                if prediction is None:
                    outcome.n_absent += 1
                    continue
                if prediction.confidence < min_confidence:
                    outcome.n_low += 1
                    continue
                trial_df = get_trial_from_tsv(app_state._all_labels_df, tid)
                trial_df = add_point(
                    trial_df,
                    prediction.time,
                    label,
                    individual,
                    NO_RECIPIENT,
                    confidence=prediction.confidence,
                    labeling_method=LABELING_AUTOMATED,
                )
                app_state.set_trial_intervals(tid, trial_df)
                trial_times[label] = prediction.time
                outcome.per_target[label] += 1
                outcome.n_predicted += 1
                written = True
                outcome.trials.add(str(tid))
                df = app_state._all_labels_df
            if written:
                app_state.set_trial_meta_attr(tid, "prediction_source", f"lightgbm:{name}")
            if config.expected_order:
                flags = om.check_expectations(trial_times, config)
                verdicts[str(tid)] = om.expectation_verdict(flags)
                if flags:
                    outcome.flagged[str(tid)] = flags
    finally:
        QApplication.restoreOverrideCursor()

    _save_curves(app_state, curves_by_trial)
    _write_expectation_column(meta, verdicts)
    if outcome.n_predicted:
        refresh_after_prediction(meta)
    if outcome.errors:
        logger.warning("Onset prediction failures: %s", "; ".join(outcome.errors))
    notify(outcome.message(), severity="warning" if outcome.errors else "info")
    if outcome.n_predicted:
        notify("Review the predictions and save with Ctrl+S.")
    return outcome


def refresh_after_prediction(meta) -> None:
    """Show freshly written predictions: current trial, plots, label shapes."""
    app_state = meta.app_state
    app_state.changes_saved = False
    current = getattr(app_state, "trials_sel", None)
    if current is not None:
        app_state.label_intervals = get_trial_from_tsv(app_state._all_labels_df, current)
    data_widget = getattr(meta, "data_widget", None)
    if data_widget is not None:
        data_widget.update_main_plot(preserve_x_range=True)
    labels_widget = getattr(meta, "labels_widget", None)
    if labels_widget is not None:
        labels_widget.refresh_labels_shapes_layer()


def _point_mappings(app_state) -> dict[int, str]:
    """Point-event label classes: ``{label_id: name}``."""
    mappings = getattr(app_state, "_label_mappings", None) or {}
    return {
        int(label_id): str(info.get("name", label_id))
        for label_id, info in sorted(mappings.items())
        if isinstance(label_id, int) and label_id != 0 and info.get("event_type", "state") == EVENT_TYPE_POINT
    }


# ---------------------------------------------------------------------------
# Feature tree — feature ▸ dim ▸ value, all checkable
# ---------------------------------------------------------------------------


#: Role holding a feature item's implicit dims: those with only one possible
#: value, so there is nothing to choose and no row is drawn for them at all.
_IMPLICIT_DIMS_ROLE = Qt.UserRole

#: Column carrying each feature's "include its derivative too" tick. It is a
#: checkbox *widget*, not the item's own check state: an auto-tristate item
#: with children reports its children's state in every column, which would
#: leave the box unrendered on any feature that has dim rows.
_DERIVATIVE_COLUMN = 1

_DERIVATIVE_TOOLTIP = (
    "Also feed the classifier this feature's rate of change (np.gradient —\n"
    "central differences, centred on the frame, so a turn in the signal shows\n"
    "up at the frame it happened). The window taps are seen in isolation, so a\n"
    "boosted tree cannot difference them itself."
)


class FeatureTree(QTreeWidget):
    """Checkable tree of features, their dims, and each dim's values.

    Auto-tristate makes the feature row an "All" toggle and dim rows show the
    partial state — the checked values per dim are exactly the subset the
    model config stores (see ``onset_model.enumerate_columns``). A dim with
    only one possible value (e.g. a single-individual dataset's "individual")
    is never drawn as a row — it is included automatically whenever its
    feature is checked, since there is nothing to choose between.

    Each feature row also carries a **d/dt** tick: that feature's time
    derivative then joins its value as an extra input column
    (``OnsetModelConfig.derivatives``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Feature", "d/dt"])
        self.headerItem().setToolTip(_DERIVATIVE_COLUMN, _DERIVATIVE_TOOLTIP)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(_DERIVATIVE_COLUMN, QHeaderView.ResizeToContents)

    def _add_derivative_check(self, item: QTreeWidgetItem, checked: bool = False) -> None:
        box = QCheckBox()
        box.setChecked(checked)
        box.setToolTip(_DERIVATIVE_TOOLTIP)
        self.setItemWidget(item, _DERIVATIVE_COLUMN, box)

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
            self._add_derivative_check(item)
            implicit: dict[str, list[str]] = {}
            for dim, values in (base.feature_dims(feature) or {}).items():
                if len(values) == 1:
                    implicit[dim] = [str(values[0])]
                    continue
                dim_item = QTreeWidgetItem(item, [dim])
                dim_item.setFlags(dim_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Unchecked)
            if implicit:
                item.setData(0, _IMPLICIT_DIMS_ROLE, implicit)

    def populate_from_config(self, config: om.OnsetModelConfig) -> None:
        """Read-only view of a frozen config (existing model)."""
        self.clear()
        for feature, dims in config.features.items():
            item = QTreeWidgetItem(self, [feature])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(0, Qt.Checked)
            self._add_derivative_check(item, feature in config.derivatives)
            implicit: dict[str, list[str]] = {}
            for dim, values in dims.items():
                if len(values) == 1:
                    implicit[dim] = list(values)
                    continue
                dim_item = QTreeWidgetItem(item, [dim])
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Checked)
            if implicit:
                item.setData(0, _IMPLICIT_DIMS_ROLE, implicit)
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
            dims: dict[str, list[str]] = dict(item.data(0, _IMPLICIT_DIMS_ROLE) or {})
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

    def selected_derivatives(self) -> list[str]:
        """Ticked features whose d/dt is ticked too, in tree order."""
        out: list[str] = []
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            box = self.itemWidget(item, _DERIVATIVE_COLUMN)
            if item.checkState(0) != Qt.Unchecked and box is not None and box.isChecked():
                out.append(item.text(0))
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
        # Dragging reorders: the ticked classes, top to bottom, ARE the order
        # the expectation below is declared in — one list, not two.
        self.target_list.setDragDropMode(QListWidget.InternalMove)
        self.target_list.setToolTip("Tick the classes to predict; drag them into the order you expect them in.")
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

        feature_group = QGroupBox("2 — Features (tick dims/values; d/dt adds the rate of change)")
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
        self.order_check = QCheckBox("Expect them in the order they are listed above")
        self.order_check.setToolTip(
            "Tick when the classes follow a stereotypic order in a trial (A then B\n"
            "then C), and drag the list above into that order.\n"
            "\n"
            "This never changes a prediction — every event still lands on its own\n"
            "curve's tallest peak. A trial whose events come out in another order is\n"
            "flagged in the trials table's 'prediction_check' column, so you can\n"
            "filter to those trials and look at them first."
        )
        params_lay.addRow("Expect:", self.order_check)
        self.together_check = QCheckBox("…and if a trial has one of them, expect the rest")
        self.together_check.setToolTip(
            "Tick when the classes come as a set: a trial with one has the others.\n"
            "\n"
            "A trial that ends up with only SOME of them — because the rest fell\n"
            "below the confidence floor — is flagged. A trial with NONE of them is\n"
            "not: the behaviour simply did not happen there, which is no surprise."
        )
        params_lay.addRow("", self.together_check)
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
            self.order_check,
            self.together_check,
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
        self.tree.populate_from_config(self._config)
        self.window_spin.setValue(self._config.window_s)
        self.tolerance_spin.setValue(self._config.tolerance_s)
        self.max_iter_spin.setValue(self._config.max_iter)
        self.lr_spin.setValue(self._config.learning_rate)
        self.order_check.setChecked(bool(self._config.expected_order))
        self.together_check.setChecked(self._config.expect_together)
        self._refresh_sessions()
        trained = "trained" if om.is_trained(text) else "not trained yet"
        self.status_label.setText(f"Config is frozen for an existing model ({trained}).")

    def _show_config_targets(self, config: om.OnsetModelConfig):
        """Read-only view of a frozen model's targets, in its declared order."""
        self.target_list.clear()
        ordered = [label for label in config.expected_order if label in config.targets]
        ordered += [label for label in config.targets if label not in ordered]
        for label_id, name in ((label, config.targets[label]) for label in ordered):
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
            notify(f"A model named {name!r} already exists — pick it from the combo to extend it.", severity="warning")
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
        derivatives = self.tree.selected_derivatives()
        self._config = om.OnsetModelConfig(
            name=name,
            targets=targets,
            features=features,
            derivatives=derivatives,
            window_s=float(self.window_spin.value()),
            tolerance_s=float(self.tolerance_spin.value()),
            max_iter=int(self.max_iter_spin.value()),
            learning_rate=float(self.lr_spin.value()),
            expected_order=list(targets) if self.order_check.isChecked() else [],
            expect_together=self.together_check.isChecked(),
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
                time, data = om.extract_model_features(loader, config, t0, t1)
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
                "columns": [c.name for c in config.columns()],
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
        # The held-out record is what every predicted confidence is scaled by,
        # so it belongs in the one message the user reads after training.
        def _held_out(label: int, cal: dict) -> str:
            ceiling = confidence_display((cal["n_hits"] + 1) / (cal["n_trials"] + 2))
            return (
                f"{config.target_name(label)}: {cal['n_hits']}/{cal['n_trials']} "
                f"within {config.tolerance_s:g} s (confidence ceiling {ceiling})"
            )

        held_out = ", ".join(_held_out(label, cal) for label, cal in summary.get("calibration", {}).items())
        if held_out:
            msg += f" On trials it did not see — {held_out}."
        orders = summary.get("sequences") or {}
        if orders:
            shown = ", ".join(f"{self._named_order(config, key)} ×{n}" for key, n in list(orders.items())[:3])
            msg += f" Orders seen in the training trials: {shown}."
        expected = config.describe_expectations()
        if expected:
            msg += f" Expecting {expected}."
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


        self.min_conf_edit = ConfidenceEdit(DEFAULT_CONFIDENCE)
        self.min_conf_edit.setToolTip(
            "A prediction scoring below this is not written at all: the trial is\n"
            "left unlabelled for that class rather than given a doubtful label.\n"
            "\n"
            "Confidence is the height of the tallest peak of the model's probability\n"
            "curve for that class — a point on the curve frame-by-frame review draws,\n"
            "so you can see what a good one and a bad one look like before choosing.\n"
            "\n"
            "Leave it at 0 to write every prediction and triage afterwards with\n"
            "Review predictions — nothing is lost that way, and the confidence is\n"
            "on each label either way."
        )
        form.addRow("Min confidence:", self.min_conf_edit)
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
        #: one — so review drops exactly those classes into curation scope.
        self._reviewable: tuple[list[int], set[str]] = ([], set())
        self.review_btn = QPushButton("Review predictions…")
        self.review_btn.setAutoDefault(False)
        self.review_btn.setEnabled(False)
        self.review_btn.setToolTip(
            "Open the Labels tab with the classes just predicted dropped into the\n"
            "Curation section's scope area, ready to review from there."
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
            f"{len(config.columns())} feature columns · "
            f"{n_sessions} training session(s) · {trained}"
            + (f" · expects {config.describe_expectations()}" if config.expected_order else "")
            + "."
        )
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
        outcome = predict_onsets(
            self.meta,
            name,
            individual=self.individual_combo.currentText(),
            min_confidence=float(self.min_conf_edit.value()),
        )
        if outcome is None:
            return
        self._reviewable = (outcome.label_ids(), outcome.trials)
        self.review_btn.setEnabled(bool(outcome.n_predicted))
        self.status_label.setText(outcome.message())

    def _review(self):
        """Drop what was just predicted into curation scope and open the Labels tab."""
        label_ids, _trials = self._reviewable
        if not label_ids:
            return
        labels_widget = getattr(self.meta, "labels_widget", None)
        curation_panel = getattr(labels_widget, "curation_panel", None)
        if curation_panel is not None:
            curation_panel.set_scope(label_ids, reason="labels predicted by lightgbm")
        collapsible_widgets = getattr(self.meta, "collapsible_widgets", None)
        if collapsible_widgets:
            collapsible_widgets[1].expand()  # "Labels" — see grid_section_container._SHORT_LABELS
