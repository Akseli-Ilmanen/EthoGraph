"""Ad-hoc: the top-level-window test must go red when tiles are parentless again."""
import pytest

from ethograph.gui import dialog_label_gridview as gv
from tests.test_unit.test_label_gridview import TestConfigDialog  # noqa: F401


@pytest.fixture(autouse=True)
def _unfix(monkeypatch):
    original = gv.LabelGridView._make_cell

    def parentless(self, entry):
        cell = original(self, entry)
        cell.setParent(None)
        return cell

    monkeypatch.setattr(gv.LabelGridView, "_make_cell", parentless)
