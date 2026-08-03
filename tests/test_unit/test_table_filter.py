"""The shared table column filters (funnel headers + filtering proxy).

Driven over a plain ``QStandardItemModel``, the way the Kilosort cluster table
uses it; the keypoint labelling dialog drives the same proxy over a virtual
model, which is why the proxy reads through indices rather than items.
"""

from __future__ import annotations

import pytest
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication

from ethograph.gui.table_filter import (
    SORT_ROLE,
    CategoryFilterDialog,
    FilterHeaderView,
    MultiColumnFilterProxy,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def proxy(qapp) -> MultiColumnFilterProxy:
    """Three rows: a name, and a number whose text is not its value."""
    model = QStandardItemModel()
    for name, value in (("good", 1.0), ("mua", 12.5), ("good", 30.0)):
        number = QStandardItem(f"{value:.1f} ms")
        number.setData(value, SORT_ROLE)
        model.appendRow([QStandardItem(name), number])
    filtering = MultiColumnFilterProxy()
    filtering.setSourceModel(model)
    return filtering


def _column(proxy, col: int) -> list[str]:
    return [proxy.index(row, col).data() for row in range(proxy.rowCount())]


def test_a_categorical_filter_keeps_only_the_allowed_values(proxy):
    proxy.set_cat_filter(0, {"good"})
    assert _column(proxy, 0) == ["good", "good"]


def test_allowing_everything_is_the_same_as_no_filter(proxy):
    """Otherwise "all unchecked" and "all checked" both mean an empty table."""
    proxy.set_cat_filter(0, set())
    assert proxy.rowCount() == 3
    assert proxy.active_filters() == set()


def test_a_numeric_filter_compares_the_value_not_the_text(proxy):
    proxy.set_numeric_filter(1, ">=", 12.5)
    assert _column(proxy, 1) == ["12.5 ms", "30.0 ms"]


def test_filters_on_several_columns_are_combined(proxy):
    proxy.set_cat_filter(0, {"good"})
    proxy.set_numeric_filter(1, ">=", 12.5)

    assert _column(proxy, 0) == ["good"]
    assert proxy.active_filters() == {0, 1}


def test_filters_are_readable_back_and_clearable(proxy):
    proxy.set_cat_filter(0, {"mua"})
    proxy.set_numeric_filter(1, "<=", 20.0)

    assert proxy.cat_filter(0) == {"mua"}
    assert proxy.num_filter(1) == ("<=", 20.0)

    proxy.clear_filters()
    assert proxy.rowCount() == 3
    assert proxy.num_filter(1) is None


def test_a_row_missing_its_sort_value_is_filtered_out(qapp):
    model = QStandardItemModel()
    model.appendRow([QStandardItem("no value")])
    proxy = MultiColumnFilterProxy()
    proxy.setSourceModel(model)

    proxy.set_numeric_filter(0, ">=", 1.0)

    assert proxy.rowCount() == 0


def test_the_category_dialog_reports_a_partial_choice(qapp):
    dialog = CategoryFilterDialog(0, ["good", "mua", "noise"], {"good"})
    assert dialog.get_allowed() == {"good"}


def test_the_category_dialog_reports_no_filter_when_all_are_checked(qapp):
    dialog = CategoryFilterDialog(0, ["good", "mua"], set())
    assert dialog.get_allowed() == set()


def test_the_header_reserves_a_zone_only_for_filterable_columns(qapp):
    header = FilterHeaderView({0}, {1})
    assert header.filterable == {0, 1}
    assert header.is_categorical(0) and not header.is_numeric(0)
    assert header.is_numeric(1) and not header.is_categorical(1)

    header.set_filterable(set(), set())
    assert header.filterable == set()
