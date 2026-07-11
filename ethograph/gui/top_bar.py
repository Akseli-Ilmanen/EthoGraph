"""Top menu bar for the ethograph main window.

Reorganises actions that previously lived on the right sidebar into a
conventional application menu bar:

    File | Layout | Changepoints | Neural | Help

Menu items come in three flavours:

* **Executable** — run a command immediately (e.g. *Save layout*).
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
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

DOCS_URL = "https://Akseli-Ilmanen.github.io/ethograph"
SHORTCUTS_URL = "https://Akseli-Ilmanen.github.io/ethograph/user_guide/shortcuts.html"
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
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignTop)
        scroll.setWidget(widget)  # reparents widget -> removes it from its home
        widget.setVisible(True)
        layout.addWidget(scroll)

        # Size to the widget's content (capped) rather than a fixed tall box, so
        # short panels (e.g. the I/O importer) don't float in empty space.
        hint = widget.sizeHint()
        width = min(max(hint.width() + 36, 420), 900)
        height = min(max(hint.height() + 36, 200), 760)
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self):
        menu_bar = self.shell.menuBar()
        menu_bar.clear()
        self._build_file_menu(menu_bar)
        self._build_layout_menu(menu_bar)
        self._build_changepoints_menu(menu_bar)
        self._build_neural_menu(menu_bar)
        self._build_help_menu(menu_bar)

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
        original index; a detached widget is parked back in the hidden holder.
        """
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

        menu.addAction("Load data…", lambda: self._popup_section("io", "Load data", io))
        menu.addSeparator()
        menu.addAction("Import labels…", lambda: self._popup_section("io", "Import labels", io))
        menu.addAction("Export labels…", lambda: self._popup_section("io", "Export labels", io))
        menu.addAction(
            "Import predictions…",
            lambda: self._popup_section("io", "Import predictions", io),
        )
        menu.addSeparator()
        save_labels = self._first_method(io, "_save_labels")
        if save_labels is not None:
            act = menu.addAction("Save labels", save_labels)
            act.setShortcut(QKeySequence("Ctrl+S"))
        menu.addSeparator()
        menu.addAction("Exit", self.shell.close)

    # ------------------------------------------------------------------
    # Layout menu
    # ------------------------------------------------------------------

    def _build_layout_menu(self, menu_bar):
        # Layout persistence is automatic (local_settings.yaml / gui_settings.yaml)
        # — no save/load actions.
        menu = menu_bar.addMenu("&Layout")
        menu.addAction("Reset layout (last clicked plot)", self._on_reset_last_plot)
        menu.addAction("Reset layout (all plots)", self._on_reset_all_plots)
        menu.addSeparator()
        zen = QAction("Zen mode (hide sidebars)", self.shell, checkable=True)
        zen.setShortcut(QKeySequence("Ctrl+Z"))
        zen.toggled.connect(self._on_zen_toggled)
        self.shell._zen_action = zen
        menu.addAction(zen)

    def _on_reset_all_plots(self):
        reset = self._first_method(self.meta, "_on_reset_layout")
        if reset is not None:
            reset()

    def _on_reset_last_plot(self):
        container = getattr(self.meta, "plot_container", None)
        last = getattr(container, "last_clicked_plot", None) or getattr(
            self.app_state, "active_plot", None
        )
        reset = self._first_method(last, "reset_view", "autoRange")
        if reset is not None:
            reset()
        else:
            self._on_reset_all_plots()

    def _on_zen_toggled(self, on: bool):
        self.shell.set_zen_mode(on)

    # ------------------------------------------------------------------
    # Changepoints menu
    # ------------------------------------------------------------------

    def _build_changepoints_menu(self, menu_bar):
        menu = menu_bar.addMenu("&Changepoints")
        cp = getattr(self.meta, "changepoints_widget", None)

        self._add_checkbox_action(
            menu, "Show changepoints", getattr(cp, "show_cp_checkbox", None)
        )
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
    # Neural menu
    # ------------------------------------------------------------------

    def _build_neural_menu(self, menu_bar):
        menu = menu_bar.addMenu("&Neural")
        ephys = getattr(self.meta, "ephys_widget", None)
        menu.addAction(
            "Phy TraceView…", lambda: self._popup_section("ephys", "Phy TraceView", ephys)
        )
        psth = getattr(self.meta, "psth_widget", None) or ephys
        menu.addAction(
            "Firing rates…", lambda: self._popup_section("firing", "Firing rates", psth)
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
