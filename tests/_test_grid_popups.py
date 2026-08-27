"""Ad-hoc: log every top-level window shown while the label grid generates."""

import traceback

import pytest
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QApplication, QWidget

import ethograph.gui.dialog_label_gridview as _gridmod
from ethograph.gui.dialog_label_gridview import LabelGridViewDialog


class _Spy(QObject):
    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    def eventFilter(self, obj, event):
        if isinstance(obj, QWidget) and event.type() in (QEvent.Show, QEvent.Hide, QEvent.Close):
            if obj.isWindow() or obj.parent() is None:
                kind = {QEvent.Show: "SHOW", QEvent.Hide: "HIDE", QEvent.Close: "CLOSE"}[event.type()]
                self.events.append(
                    f"{kind:5} {type(obj).__name__:28} title={obj.windowTitle()!r} "
                    f"parent={type(obj.parent()).__name__ if obj.parent() else None}"
                )
                if kind == "SHOW" and obj.parent() is None and type(obj).__name__ == "QWidget":
                    self.events.append("".join(traceback.format_stack(limit=12)))
        return False


@pytest.mark.parametrize("skip_video", [True, False])
def test_grid_popups(moll2025_gui, qtbot, skip_video, monkeypatch):
    viewer, meta = moll2025_gui
    calls = []
    monkeypatch.setattr(_gridmod, "notify", lambda msg, severity="info": calls.append(f"{severity}: {msg}"))
    df = meta.app_state._all_labels_df
    print("labels:", None if df is None else len(df), "trials:", meta.app_state.trials)
    dialog = meta.labels_widget.curation_panel.open_grid_view()
    dialog.setup.method_combo.setCurrentIndex(dialog.setup.method_combo.findData("all"))
    print("scope:", dialog.setup.selected_label_ids(), "cams:", dialog.setup.selected_cameras(), "methods:", dialog.setup.selected_methods())
    dialog.show()
    qtbot.waitExposed(dialog)
    if dialog.panel_list is not None:
        for i in range(dialog.panel_list.count()):
            dialog.panel_list.item(i).setCheckState(Qt.Checked)
        print("panels:", [dialog.panel_list.item(i).text() for i in range(dialog.panel_list.count())])
    if dialog.skip_video_cb is not None:
        dialog.skip_video_cb.setChecked(skip_video)
    spy = _Spy()
    QApplication.instance().installEventFilter(spy)
    try:
        dialog._generate()
    finally:
        QApplication.instance().removeEventFilter(spy)
    entries = dialog.grid_view._entries if dialog.grid_view else []
    print(f"entries: {len(entries)} with_image={sum(e.image is not None for e in entries)} "
          f"with_panels={sum(bool(e.panels) for e in entries)} errors={[e.error for e in entries if e.error][:3]}")
    print(f"\n--- skip_video={skip_video}: {len(spy.events)} window events, {len(calls)} notifies ---")
    for line in spy.events:
        print(line)
    for c in calls:
        print("NOTIFY", c)
    dialog.close()
