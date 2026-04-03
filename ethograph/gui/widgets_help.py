"""Help widget with documentation links and debug tools."""

import warnings
import webbrowser

from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HelpWidget(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state

        help_layout = QHBoxLayout()
        self.docs_button = QPushButton("📚 Documentation")
        self.docs_button.clicked.connect(lambda: webbrowser.open("https://ethograph.readthedocs.io/en/latest/"))
        help_layout.addWidget(self.docs_button)

        self.shortcuts_button = QPushButton("⌨ Shortcuts")
        self.shortcuts_button.clicked.connect(lambda: webbrowser.open("https://ethograph.readthedocs.io/en/latest/user_guide/shortcuts.html"))
        help_layout.addWidget(self.shortcuts_button)

        self.github_button = QPushButton("🔗 GitHub Issues")
        self.github_button.clicked.connect(lambda: webbrowser.open("https://github.com/akseli-ilmanen/ethograph/issues"))
        help_layout.addWidget(self.github_button)

        help_layout2 = QHBoxLayout()
        self.print_debug_button = QPushButton("🖨 Print for debugging")
        self.print_debug_button.setToolTip("Print app state, session, and trial alignment to console for debugging")
        self.print_debug_button.clicked.connect(self._on_print_debug)
        help_layout2.addWidget(self.print_debug_button)

        self.alignment_button = QPushButton("📊 Visualize data alignment")
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
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.addLayout(help_layout)
        main_layout.addLayout(help_layout2)
        main_layout.addLayout(help_layout3)
        self.setLayout(main_layout)

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
        all_labels = getattr(self.app_state, '_all_labels_df', None)
        if all_labels is None or all_labels.empty:
            print("  No labels loaded.")
        else:
            print(f"  {len(all_labels)} label rows across trials")
            print(f"  Trials in labels: {sorted(all_labels['trial'].unique())}")

        if all_labels is not None and not all_labels.empty:
            print("\n  Per-trial metadata columns:")
            for col in ["human_verified", "changepoint_corrected", "prediction_source"]:
                if col in all_labels.columns:
                    print(f"    {col}: {all_labels.groupby('trial')[col].first().to_dict()}")

        print(SEP)
        print("=" * 60)
        print("  CURRENT TRIAL  labels + meta")
        print("=" * 60)
        trial = getattr(self.app_state, 'trials_sel', None)
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
        print("  TRIAL ALIGNMENT")
        print("=" * 60)
        alignment = getattr(self.app_state, 'trial_alignment', None)
        if alignment is None:
            print("  No trial alignment available.")
        else:
            print(alignment.summary())

        print(SEP)
        print("=" * 60)
        print("  NWB ALIGNMENT (session_io)")
        print("=" * 60)
        dt = getattr(self.app_state, 'dt', None)
        sio = getattr(dt, 'session_io', None) if dt is not None else None
        if sio is None:
            print("  No session_io available.")
        else:
            sio.print_session()

        print(SEP)
        print("=" * 60)
        print(f"Trial Interval set")
        print("=" * 60)
        trials_ep = getattr(self.app_state.dt, 'trials_ep', None)
        if trials_ep is None:
            print("  No trials_ep available.")
        else:
            df = trials_ep.as_dataframe()
            print(df.to_string())

        print(SEP)
        print("=" * 60)
        print("  FEATURE STORE / DATASET")
        print("=" * 60)
        store = getattr(self.app_state, 'feature_store', None)
        ds = getattr(self.app_state, 'ds', None)
        if store is None:
            print("  No feature_store.")
        else:
            print(f"  Store type: {type(store).__name__} (backend={store.backend})")
            print(f"  Features: {store.features}")
            print(f"  Dims: {list(store.dims.keys())}")
            print(f"  Colors: {store.colors}")
            print(f"  Changepoint names: {store.changepoint_names}")
            if hasattr(store, '_ds'):
                store_ds = store._ds
                if store_ds is not None:
                    cp_vars = list(store_ds.filter_by_attrs(type="changepoints").data_vars)
                    print(f"  Store._ds changepoint vars: {cp_vars}")
                    print(f"  Store._ds is app_state.ds: {store_ds is ds}")
                else:
                    print("  Store._ds is None")
        if ds is not None:
            cp_vars = list(ds.filter_by_attrs(type="changepoints").data_vars)
            print(f"  app_state.ds changepoint vars: {cp_vars}")
            for v in cp_vars:
                da = ds[v]
                import numpy as np
                n_pos = int(np.sum(da.values > 0))
                print(f"    {v}: shape={da.shape}, n_positive={n_pos}, target={da.attrs.get('target_feature')}")
        else:
            print("  app_state.ds is None")
