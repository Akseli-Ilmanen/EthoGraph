"""Top menu bar for the ethograph main window.

Reorganises actions that previously lived on the right sidebar into a
conventional application menu bar:

    File | Changepoints | Tools | Docs | Help

**Docs** holds only external links (browser); **Help** holds only in-app
diagnostic/recovery actions — no nested submenus in either.

Menu items come in three flavours:

* **Executable** — run a command immediately (e.g. *Save labels*).
* **Boolean** — a checkable action kept in sync with a sidebar checkbox
  (e.g. *Show changepoints*).
* **Pop-up** — open a dialog that *hosts an existing sidebar section*.  The
  section widget is borrowed out of the sidebar's stacked widget while the
  dialog is open and returned to its original slot on close, so the sidebar
  is never left in a broken state (see :class:`SectionPopup`).

The whole thing is built from ``shell.meta_widget`` after it is attached, so
it degrades gracefully (guarded with ``getattr``) if a widget is missing.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QDialog,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify
from ethograph.utils.paths import ethograph_home

logger = logging.getLogger(__name__)

# Tools → the one screen-recording entry, relabelled per recorder state.
# Plain text, no glyphs: ⏺/⏹ render as tofu boxes in Windows menu fonts.
# Entries carry a "Category:" prefix (Demo/Labels/Neural) so the flat Tools
# menu reads as groups.
_RECORD_ACTION_LABELS = {
    "idle": "Demo: Screen-record the GUI…",
    "recording": "Demo: Stop screen recording  (Ctrl+Space)",
    "rendering": "Demo: Rendering recording…",
}

DOCS_URL = "https://Akseli-Ilmanen.github.io/ethograph"
SHORTCUTS_URL = "https://Akseli-Ilmanen.github.io/ethograph/advanced/shortcuts.html"
ISSUES_URL = "https://github.com/akseli-ilmanen/ethograph/issues"
TUTORIALS_URL = "https://www.youtube.com/playlist?list=PLAI16F70Jqg0yE5LNO0lKouVIXkSwQkTN"


class SectionPopup(QDialog):
    """Non-modal dialog that borrows a widget and returns it on close.

    The hosted widget is reparented into this dialog while it is visible and
    handed back to ``on_restore(widget)`` when the dialog closes, so its home
    (a sidebar stack slot, or the detached-widget holder) stays consistent.
    """

    def __init__(self, title: str, widget: QWidget, on_restore, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self._widget = widget
        self._on_restore = on_restore

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignTop)
        scroll.setFrameShape(QScrollArea.NoFrame)

        # Host the borrowed widget in a padded wrapper so its controls don't
        # sit flush against the dialog edges. The wrapper reparents the widget
        # (removes it from its home); closeEvent pulls it back out.
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(14, 14, 14, 14)
        host_layout.setSpacing(8)
        host_layout.addWidget(widget)
        host_layout.addStretch()
        scroll.setWidget(host)
        widget.setVisible(True)
        layout.addWidget(scroll)

        # Size to the widget's content (capped) rather than a fixed tall box, so
        # short panels (e.g. the I/O importer) don't float in empty space.
        hint = widget.sizeHint()
        width = min(max(hint.width() + 72, 460), 940)
        height = min(max(hint.height() + 72, 220), 780)
        self.resize(width, height)

    def closeEvent(self, event):
        if self._widget is not None and self._on_restore is not None:
            self._widget.setParent(None)
            self._on_restore(self._widget)
        super().closeEvent(event)


def _sidebar_stack(meta) -> QWidget | None:
    """The GridSectionContainer's internal QStackedWidget (holds sections)."""
    return getattr(meta, "_stack", None)


class TopBarBuilder:
    """Builds and owns the menu bar for :class:`EthographMainWindow`."""

    def __init__(self, shell):
        self.shell = shell
        self.meta = shell.meta_widget
        self.app_state = getattr(self.meta, "app_state", None)
        self._open_popups: dict[str, SectionPopup] = {}
        #: Refine-labels dialog, rebuilt when reopened so its label list and
        #: individuals always reflect the currently loaded labels.
        self._refine_dialog = None
        #: GradBoost onset-model dialogs, rebuilt when reopened for the same reason.
        self._onset_train_dialog = None
        self._onset_predict_dialog = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self):
        menu_bar = self.shell.menuBar()
        menu_bar.clear()
        self._build_file_menu(menu_bar)
        self._build_changepoints_menu(menu_bar)
        self._build_tools_menu(menu_bar)
        self._build_model_menu(menu_bar)
        self._build_docs_menu(menu_bar)
        self._build_help_menu(menu_bar)
        self._add_sidebar_toggle_button(menu_bar)

    def _build_tools_menu(self, menu_bar):
        """Tools menu — screen recorder + neural (PSTH / firing rates) actions."""
        from .dialog_screen_recorder import RecordController

        menu = menu_bar.addMenu("&Tools")
        # The menu entry IS the recorder: clicking it opens the settings dialog
        # and starts recording (and stops a running one) — no nested button.
        self._record_controller = RecordController(self.shell, parent=self.shell)
        self._record_action = menu.addAction(
            _RECORD_ACTION_LABELS["idle"],
            self._record_controller.toggle,
        )
        self._record_controller.state_changed.connect(self._on_record_state)

        menu.addSeparator()
        # Boundary refinement: existing labels become seeds the user nudges
        # frame-by-frame — the exception to ethograph's plot-first labelling.
        menu.addAction("Labels: Refine via frame-by-frame labelling…", self._open_refine_labels)

        menu.addSeparator()
        menu.addAction("Pose tracking (from scratch)…", self._open_keypoint_labelling)
        # Correcting an imported pose file (DLC/SLEAP/…) rather than labelling
        # from scratch — writes {stem}_refined copies beside the sources.
        menu.addAction("Pose correction (DLC, SLEAP, …)…", self._open_pose_refinement)

        menu.addSeparator()
        ephys = getattr(self.meta, "ephys_widget", None)
        # Phy TraceView controls live in the right sidebar's "Phy viewer"
        # context (shown when the Phy trace panel is clicked). Tools keeps
        # only the interactive PSTH launcher and the firing-rate popup.
        psth_open = self._first_method(ephys, "_open_psth")
        act = (
            menu.addAction("Neural: Interactive PSTH…", psth_open)
            if psth_open
            else menu.addAction("Neural: Interactive PSTH…")
        )
        if psth_open is None:
            act.setEnabled(False)
        menu.addAction("Neural: Compute firing rates…", lambda: self._popup_section("firing", "Firing rates", ephys))

    def _build_model_menu(self, menu_bar):
        """Model menu — supervised point-event onset detection (GradBoost).

        Train collects the session's existing point events as training data
        for a ``HistGradientBoostingClassifier``; Predict fills the target
        event into trials that don't carry it yet.
        """
        menu = menu_bar.addMenu("&Model")
        menu.addAction("Train onset detector (GradBoost)…", self._open_onset_train)
        menu.addAction("Predict onsets (GradBoost)…", self._open_onset_predict)

    def _open_onset_train(self):
        from .dialog_onset_model import TrainOnsetDialog

        # Rebuilt when reopened so the feature tree and point-event classes
        # always reflect the currently loaded session.
        if self._onset_train_dialog is None or not self._onset_train_dialog.isVisible():
            self._onset_train_dialog = TrainOnsetDialog(self.meta, parent=self.shell)
        self._onset_train_dialog.show()
        self._onset_train_dialog.raise_()
        self._onset_train_dialog.activateWindow()

    def _open_onset_predict(self):
        from .dialog_onset_model import PredictOnsetDialog

        if self._onset_predict_dialog is None or not self._onset_predict_dialog.isVisible():
            self._onset_predict_dialog = PredictOnsetDialog(self.meta, parent=self.shell)
        self._onset_predict_dialog.show()
        self._onset_predict_dialog.raise_()
        self._onset_predict_dialog.activateWindow()

    def _open_keypoint_labelling(self):
        """Open the keypoint labelling dialog (owned by the DataWidget, so the
        Tools entry and the Pose sidebar button raise the same instance)."""
        open_dialog = self._first_method(getattr(self.meta, "data_widget", None), "open_keypoint_labelling")
        if open_dialog is not None:
            open_dialog()

    def _open_pose_refinement(self):
        """Open the pose refinement dialog (owned by the DataWidget)."""
        open_dialog = self._first_method(getattr(self.meta, "data_widget", None), "open_pose_refinement")
        if open_dialog is not None:
            open_dialog()

    def _open_refine_labels(self):
        """Open the frame-by-frame label refinement dialog."""
        from .dialog_refine import RefineLabelsDialog

        if self._refine_dialog is None or not self._refine_dialog.isVisible():
            self._refine_dialog = RefineLabelsDialog(self.meta, parent=self.shell)
        self._refine_dialog.show()
        self._refine_dialog.raise_()
        self._refine_dialog.activateWindow()

    def _reset_video_view(self):
        """Rebuild the primary video panel — recovery for a frozen image."""
        vm = getattr(getattr(self.meta, "data_widget", None), "video_mgr", None)
        if vm is not None:
            vm.reset_primary_video()

    def _on_record_state(self, state: str):
        """Relabel the single Tools entry as the recorder changes state."""
        self._record_action.setText(_RECORD_ACTION_LABELS.get(state, _RECORD_ACTION_LABELS["idle"]))
        self._record_action.setEnabled(state != "rendering")

    def _add_sidebar_toggle_button(self, menu_bar):
        """Checkable button at the far right of the menu bar toggling the
        right control sidebar — a discoverable alternative to Shift+Z."""
        btn = QToolButton(menu_bar)
        btn.setText("◨ Sidebar")
        btn.setToolTip("Show/hide the right sidebar (Shift+Z)")
        btn.setCheckable(True)
        btn.setAutoRaise(True)
        sidebar_action = getattr(self.shell, "_sidebar_toggle", None)
        visible = (sidebar_action is None or sidebar_action.isChecked()) and not getattr(self.shell, "_zen_mode", False)
        btn.setChecked(visible)
        btn.toggled.connect(lambda vis: self.shell.set_zen_mode(not vis))
        menu_bar.setCornerWidget(btn, Qt.TopRightCorner)
        self.shell._sidebar_corner_btn = btn

    # ------------------------------------------------------------------
    # Pop-up helper
    # ------------------------------------------------------------------

    def _popup_section(self, key: str, title: str, widget: QWidget | None):
        """Open (or raise) a dialog hosting a borrowed section/detached widget."""
        if widget is None:
            return
        existing = self._open_popups.get(key)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        on_restore = self._restore_cb_for(widget)
        dlg = SectionPopup(title, widget, on_restore, parent=self.shell)
        dlg.finished.connect(lambda _=0, k=key: self._open_popups.pop(k, None))
        self._open_popups[key] = dlg
        dlg.show()
        dlg.raise_()

    def _restore_cb_for(self, widget: QWidget):
        """Build the callable that returns *widget* to its home on close.

        A widget currently living in the sidebar stack is re-inserted at its
        original index; an I/O sub-panel goes back to its slot inside the
        I/O widget; a detached widget is parked back in the hidden holder.
        """
        io = getattr(self.meta, "io_widget", None)
        if io is not None and widget in (
            getattr(io, "labels_group", None),
            getattr(io, "pred_group", None),
            getattr(io, "export_panel", None),
        ):
            return io.restore_subpanel
        stack = _sidebar_stack(self.meta)
        idx = stack.indexOf(widget) if stack is not None else -1
        if idx >= 0:
            return lambda w, s=stack, i=idx: s.insertWidget(i, w)
        holder = getattr(self.meta, "_detached_holder", None)
        if holder is not None and holder.layout() is not None:

            def _restore(w, h=holder):
                h.layout().addWidget(w)
                w.setVisible(False)

            return _restore
        return lambda w: None

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------

    def _build_file_menu(self, menu_bar):
        menu = menu_bar.addMenu("&File")
        io = getattr(self.meta, "io_widget", None)

        menu.addAction("Open settings folder (.ethograph)…", self._open_ethograph_home)
        menu.addSeparator()

        # Each I/O sub-panel pops up on its own (no unrelated sections):
        # labels import (mapping.txt + tsv/crowsetta) is separate from
        # predictions import and from label export. Data loading itself
        # happens only on the cover page.
        menu.addAction(
            "Import labels…",
            lambda: self._popup_section("import_labels", "Import labels", getattr(io, "labels_group", None)),
        )
        menu.addAction(
            "Import predictions…",
            lambda: self._popup_section("import_predictions", "Import predictions", getattr(io, "pred_group", None)),
        )
        menu.addAction(
            "Export labels…",
            lambda: self._popup_section("export_labels", "Export labels", getattr(io, "export_panel", None)),
        )
        menu.addSeparator()
        save_labels = self._first_method(io, "_save_labels")
        if save_labels is not None:
            # The key itself is bound once, in gui/shortcuts.py. A QAction
            # shortcut here would be a *second* Ctrl+S on the same window, and
            # Qt answers an ambiguous overload by firing neither -- the menu
            # entry kept working, the key stopped. So the key is named in the
            # entry's own text, exactly like the I/O panel's Save button.
            menu.addAction("Save labels (Ctrl+S)", save_labels)
        menu.addSeparator()
        menu.addAction("Exit", self.shell.close)

    def _open_ethograph_home(self):
        """Open the global ``.ethograph`` settings/cache folder in the OS file browser."""
        home = ethograph_home()
        home.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(home))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(home)])
            else:
                subprocess.Popen(["xdg-open", str(home)])
        except OSError:
            logger.warning("Could not open %s in the file browser", home)
            notify(f"Could not open {home} in the file browser.", severity="warning")

    # ------------------------------------------------------------------
    # Changepoints menu
    # ------------------------------------------------------------------

    def _build_changepoints_menu(self, menu_bar):
        menu = menu_bar.addMenu("&Changepoints")
        cp = getattr(self.meta, "changepoints_widget", None)

        self._add_checkbox_action(menu, "Show changepoints", getattr(cp, "show_cp_checkbox", None))
        self._add_checkbox_action(
            menu,
            "Changepoint correction",
            getattr(cp, "changepoint_correction_checkbox", None),
        )
        menu.addSeparator()
        menu.addAction(
            "Run changepoint correction…",
            lambda: self._popup_section("cp", "Changepoint correction", cp),
        )

    # ------------------------------------------------------------------
    # Docs + Help menus
    # ------------------------------------------------------------------

    def _build_docs_menu(self, menu_bar):
        """External links only — everything here opens the browser."""
        menu = menu_bar.addMenu("&Docs")
        menu.addAction("Documentation", lambda: webbrowser.open(DOCS_URL))
        menu.addAction("Shortcuts", lambda: webbrowser.open(SHORTCUTS_URL))
        menu.addAction("Git Issues", lambda: webbrowser.open(ISSUES_URL))
        menu.addAction("Tutorials", lambda: webbrowser.open(TUTORIALS_URL))

    def _build_help_menu(self, menu_bar):
        """In-app diagnostic and recovery actions — flat, no submenus."""
        menu = menu_bar.addMenu("&Help")
        help_w = getattr(self.meta, "help_widget", None)

        print_state = self._first_method(help_w, "_on_print_debug")
        if print_state is not None:
            menu.addAction("Print current state", print_state)
        show_align = self._first_method(help_w, "_on_show_alignment")
        if show_align is not None:
            menu.addAction("Visualize data alignment", show_align)

        menu.addSeparator()
        # Escape hatch for a frozen video image (dead pynaviz render chain):
        # rebuilds the primary PlotVideo without closing/re-adding the panel.
        menu.addAction("Reset video view", self._reset_video_view)
        reset_gui = self._first_method(getattr(self.meta, "io_widget", None), "_on_reset_gui_clicked")
        if reset_gui is not None:
            menu.addAction("Reset global settings (gui_settings.yaml)", reset_gui)
        menu.addAction("Reset local settings (this dataset)", self._reset_local_settings)

    def _reset_local_settings(self):
        """Clear the loaded dataset's local_settings.yaml + in-memory local vars."""
        if self.app_state is None:
            return
        if self.app_state._local_settings_path() is None:
            notify("No dataset loaded — there are no local settings to reset.", severity="warning")
            return
        self.app_state.reset_local_settings()
        notify("Local settings reset for this dataset — reload it for a clean layout.")

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _add_checkbox_action(self, menu, label: str, checkbox):
        """Create a checkable QAction bound bidirectionally to a QCheckBox."""
        act = QAction(label, self.shell, checkable=True)
        if checkbox is None:
            act.setEnabled(False)
            menu.addAction(act)
            return act
        act.setChecked(checkbox.isChecked())
        act.toggled.connect(checkbox.setChecked)
        checkbox.toggled.connect(act.setChecked)
        menu.addAction(act)
        return act

    @staticmethod
    def _first_method(obj, *names):
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
        return None


def build_menu_bar(shell):
    """Construct the top menu bar for *shell* (an EthographMainWindow)."""
    builder = TopBarBuilder(shell)
    builder.build()
    shell._top_bar = builder
    return builder
