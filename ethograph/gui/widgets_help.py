"""Help widget with documentation links and debug tools."""

import warnings
import webbrowser

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.io import schema

TUTORIALS = [
    {
        "title": "Video playback",
        "url": "https://www.youtube.com/watch?v=hErA8c_BMUY&list=PLAI16F70Jqg0yE5LNO0lKouVIXkSwQkTN&index=3",
    },
    {
        "title": "Labeling guide",
        "url": "https://www.youtube.com/watch?v=s2UAfVRuJKY&list=PLAI16F70Jqg0yE5LNO0lKouVIXkSwQkTN&index=2",
    },
    {
        "title": "Adjusting axes limits",
        "url": "https://www.youtube.com/watch?v=oXc7bCoY6G0&list=PLAI16F70Jqg0yE5LNO0lKouVIXkSwQkTN",
    },
]


class HelpWidget(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state

        help_layout = QHBoxLayout()
        self.docs_button = QPushButton("📚 Documentation")
        self.docs_button.clicked.connect(lambda: webbrowser.open("https://Akseli-Ilmanen.github.io/ethograph"))
        help_layout.addWidget(self.docs_button)

        self.shortcuts_button = QPushButton("⌨ Shortcuts")
        self.shortcuts_button.clicked.connect(
            lambda: webbrowser.open("https://Akseli-Ilmanen.github.io/ethograph/advanced/shortcuts.html")
        )
        help_layout.addWidget(self.shortcuts_button)

        self.github_button = QPushButton("🔗 GitHub Issues")
        self.github_button.clicked.connect(
            lambda: webbrowser.open("https://github.com/akseli-ilmanen/ethograph/issues")
        )
        help_layout.addWidget(self.github_button)

        help_layout2 = QHBoxLayout()

        self.print_debug_button = QPushButton("🖨️Print for debugging")
        self.print_debug_button.setToolTip("Print app state, session, and trial alignment to console for debugging")
        self.print_debug_button.clicked.connect(self._on_print_debug)
        help_layout2.addWidget(self.print_debug_button)

        self.alignment_button = QPushButton("📊Visualize data alignment")
        self.alignment_button.setToolTip(
            "Show a timeline of how all loaded data sources (video, audio, features, pose)\n"
            "align across trials.  Useful for verifying session-level timing setup."
        )
        self.alignment_button.clicked.connect(self._on_show_alignment)
        help_layout2.addWidget(self.alignment_button)

        help_layout3 = QHBoxLayout()
        self.filter_warnings_checkbox = QCheckBox("Suppress library warnings")
        self.filter_warnings_checkbox.setObjectName("filter_warnings_checkbox")
        self.filter_warnings_checkbox.setChecked(app_state.get_with_default("filter_warnings"))
        self.filter_warnings_checkbox.setToolTip(
            "Suppress Python warnings from third-party libraries\n"
            "(e.g. NumPy deprecation notices, codec warnings).\n"
            "Does not affect ethograph's own notifications."
        )
        self.filter_warnings_checkbox.toggled.connect(self._on_filter_warnings_changed)
        self._apply_warning_filters(app_state.get_with_default("filter_warnings"))
        help_layout3.addWidget(self.filter_warnings_checkbox)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addLayout(help_layout)
        main_layout.addLayout(help_layout2)
        main_layout.addLayout(help_layout3)
        main_layout.addWidget(self._build_tutorials_group())
        main_layout.addStretch()
        self.setLayout(main_layout)

    def _build_tutorials_group(self) -> QGroupBox:
        group = QGroupBox("Video Tutorials")
        grid = QGridLayout()
        grid.setSpacing(8)
        cols = 3
        for i, entry in enumerate(TUTORIALS):
            btn = QPushButton(entry["title"])
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, url=entry["url"]: webbrowser.open(url))
            grid.addWidget(btn, i // cols, i % cols)
        group.setLayout(grid)
        return group

    def _on_filter_warnings_changed(self, checked: bool):
        self.app_state.filter_warnings = checked
        self._apply_warning_filters(checked)

    def _apply_warning_filters(self, enabled: bool):
        if enabled:
            warnings.filterwarnings("ignore")
        else:
            warnings.resetwarnings()

    def _on_show_alignment(self):
        from .widgets_navigation import _DataAlignmentDialog

        dlg = _DataAlignmentDialog(self.app_state, parent=self)
        dlg.exec()

    def _on_print_debug(self):
        SEP = "\n" * 4

        print(SEP)
        print("=" * 60)
        print("  APP STATE  (yaml-persisted)")
        print("=" * 60)
        self.app_state.print_state()

        print(SEP)
        print("=" * 60)
        print("  Labels (TSV store)")
        print("=" * 60)
        all_labels = getattr(self.app_state, "_all_labels_df", None)
        if all_labels is None or all_labels.empty:
            print("  No labels loaded.")
        else:
            print(f"  {len(all_labels)} label rows across trials")
            print(f"  Trials in labels: {sorted(all_labels['trial'].unique())}")

        if all_labels is not None and not all_labels.empty:
            print(all_labels.to_string(max_rows=20, index=False))

        print(SEP)
        print("=" * 60)
        print("  CURRENT TRIAL  labels + meta")
        print("=" * 60)
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None:
            print("  No trial selected.")
        else:
            trial_intervals = self.app_state.get_trial_intervals(trial)
            print(f"  Trial {trial!r}: {len(trial_intervals)} intervals")
            if not trial_intervals.empty:
                print(trial_intervals)
            trial_meta = self.app_state.get_trial_meta(trial)
            if trial_meta:
                print(f"\n  Trial {trial!r} metadata:")
                for k, v in trial_meta.items():
                    print(f"    {k}: {v!r}")

        print(SEP)
        print("=" * 60)
        print("  DATA LOADER / DATASET")
        print("=" * 60)
        store = getattr(self.app_state, "data_loader", None)
        ds = getattr(self.app_state, "ds", None)
        if store is None:
            print("  No data_loader.")
        else:
            print(f"  Store type: {type(store).__name__} (backend={store.backend})")
            print(f"  Features: {store.features}")
            print(f"  Dims: {list(store.dims.keys())}")
            print(f"  Changepoint names: {store.changepoint_names}")
            if hasattr(store, "_ds"):
                store_ds = store._ds
                if store_ds is not None:
                    cp_vars = schema.changepoint_vars(store_ds)
                    print(f"  Store._ds changepoint vars: {cp_vars}")
                    print(f"  Store._ds is app_state.ds: {store_ds is ds}")
                else:
                    print("  Store._ds is None")
        if ds is not None:
            cp_vars = schema.changepoint_vars(ds)
            for v in cp_vars:
                da = ds[v]
                import numpy as np

                n_pos = int(np.sum(da.values > 0))
                print(f"    {v}: shape={da.shape}, n_positive={n_pos}, target={da.attrs.get('target_feature')}")
        else:
            print("  app_state.ds is None")

        print(SEP)
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is None:
            dt = getattr(self.app_state, "dt", None)
            sio = getattr(dt, "nwb_alignment", None) if dt is not None else None
        if sio is None:
            print("  No nwb_alignment available.")
        else:
            sio.print_session()

        print(SEP)
        print("=" * 60)
        print("Trial Interval set")
        print("=" * 60)
        trials_ep = getattr(self.app_state.nwb_alignment, "trials_ep", None)
        if trials_ep is None:
            print("  No trials_ep available.")
        else:
            df = trials_ep.as_dataframe()
            print(df.to_string(max_rows=20))

        print(SEP)
        print("=" * 60)
        print("  SOURCE COLLECTION (time model)")
        print("=" * 60)
        sc = getattr(self.app_state, "source_collection", None)
        if sc is None:
            print("  No source_collection.")
        else:
            print(f"  Sources ({len(sc.sources)}):")
            for name, src in sc.sources.items():
                sr = src.sampling_rate
                sr_str = f"{sr:.1f} Hz" if sr else "irregular"
                print(f"    {name}: {src.time_range}  ({sr_str})")
            ur = sc.union_range
            ir = sc.intersection_range
            print(f"  Union range:        {ur}")
            print(f"  Intersection range: {ir}")
            print(f"  Trials: {sc.n_trials}")
            for i in range(min(sc.n_trials, 10)):
                tid = sc.trial_ids[i] if i < len(sc.trial_ids) else "?"
                print(f"    [{i}] trial={tid}  {sc.trial_range(i)}")
            if sc.n_trials > 10:
                print(f"    ... ({sc.n_trials - 10} more)")
        rw = getattr(self.app_state, "restrict_window", None)
        print(f"  Navigate mode:   {getattr(self.app_state, 'navigate_mode', '?')}")
        print(f"  Slider scope:    {getattr(self.app_state, 'slider_scope', '?')}")
        print(f"  Restrict window: {rw}")
        print(f"  Window bounds:   {self.app_state.window_bounds}")

        print(SEP)
        print("=" * 60)
        print("  TRIAL/ VIDEO BOUNDS")
        print("=" * 60)
        alignment = getattr(self.app_state, "trial_alignment", None)
        if alignment is None:
            print("  No trial alignment available.")
        else:
            print(alignment.summary())
