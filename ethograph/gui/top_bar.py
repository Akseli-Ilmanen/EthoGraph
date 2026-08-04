"""Top menu bar for the ethograph main window.

Reorganises actions that previously lived on the right sidebar into a
conventional application menu bar:

    File | Changepoints | Tools | Help

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
import webbrowser

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import (
    QDialog,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Tools → the one screen-recording entry, relabelled per recorder state.
# Plain text, no glyphs: ⏺/⏹ render as tofu boxes in Windows menu fonts.
_RECORD_ACTION_LABELS = {
    "idle": "Screen-record the GUI…",
    "recording": "Stop screen recording  (Ctrl+Space)",
    "rendering": "Rendering recording…",
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
        #: The tag sheet, kept only while it is open — a closed one is rebuilt
        #: so it always reopens on the current video's resolution and the tag
        #: family the Detect tab is set to.
        self._tag_sheet = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self):
        menu_bar = self.shell.menuBar()
        menu_bar.clear()
        self._build_file_menu(menu_bar)
        self._build_changepoints_menu(menu_bar)
        self._build_tools_menu(menu_bar)
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
        menu.addAction("Keypoint labelling…", self._open_keypoint_labelling)
        # Printing tags is its own errand — it happens once, before any video
        # exists — so it is reachable without opening the labelling dialog.
        menu.addAction("Print tag sheet…", self._open_tag_sheet)

        menu.addSeparator()
        ephys = getattr(self.meta, "ephys_widget", None)
        # Phy TraceView controls live in the right sidebar's "Phy viewer"
        # context (shown when the Phy trace panel is clicked). Tools keeps
        # only the interactive PSTH launcher and the firing-rate popup.
        psth_open = self._first_method(ephys, "_open_psth")
        act = (
            menu.addAction("Open interactive PSTH…", psth_open)
            if psth_open
            else menu.addAction("Open interactive PSTH…")
        )
        if psth_open is None:
            act.setEnabled(False)
        menu.addAction("Firing rates…", lambda: self._popup_section("firing", "Firing rates", ephys))

    def _open_keypoint_labelling(self):
        """Open the keypoint labelling dialog (owned by the DataWidget, so the
        Tools entry and the Pose sidebar button raise the same instance)."""
        open_dialog = self._first_method(getattr(self.meta, "data_widget", None), "open_keypoint_labelling")
        if open_dialog is not None:
            open_dialog()

    def _open_tag_sheet(self):
        """Open the fiducial tag sheet, sized against the loaded video."""
        from .dialog_tag_sheet import TagSheetDialog

        if self.app_state is None:
            return
        if self._tag_sheet is None or not self._tag_sheet.isVisible():
            from .pose_fill import video_size

            # Seeded from the video FILE when one is loaded — never from what is
            # on screen, which may be a low-resolution proxy.
            path = getattr(self.app_state, "video_path", None)
            size = video_size(path) if path else None
            self._tag_sheet = TagSheetDialog(
                self.app_state, image_width_px=size[0] if size else None, parent=self.shell
            )
        self._tag_sheet.show()
        self._tag_sheet.raise_()

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
            act = menu.addAction("Save labels", save_labels)
            act.setShortcut(QKeySequence("Ctrl+S"))
        menu.addSeparator()
        menu.addAction("Exit", self.shell.close)

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
    # Help menu
    # ------------------------------------------------------------------

    def _build_help_menu(self, menu_bar):
        menu = menu_bar.addMenu("&Help")
        help_w = getattr(self.meta, "help_widget", None)

        links = menu.addMenu("Links")
        links.addAction("Documentation", lambda: webbrowser.open(DOCS_URL))
        links.addAction("Shortcuts", lambda: webbrowser.open(SHORTCUTS_URL))
        links.addAction("Git Issues", lambda: webbrowser.open(ISSUES_URL))
        links.addAction("Tutorials", lambda: webbrowser.open(TUTORIALS_URL))

        menu.addSeparator()
        debug = menu.addMenu("Debug")
        print_state = self._first_method(help_w, "_on_print_debug")
        if print_state is not None:
            debug.addAction("Print current state", print_state)
        show_align = self._first_method(help_w, "_on_show_alignment")
        if show_align is not None:
            debug.addAction("Visualize data alignment", show_align)
        reset_gui = self._first_method(getattr(self.meta, "io_widget", None), "_on_reset_gui_clicked")
        if reset_gui is not None:
            debug.addAction("Reset gui_settings.yaml", reset_gui)

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
