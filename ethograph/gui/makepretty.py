from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QGuiApplication, QFont
from qtpy.QtWidgets import QComboBox, QDockWidget, QStyledItemDelegate, QWidget

from .app_constants import (
    LAYER_DOCK_WIDTH_RATIO,
    LAYOUT_RELEASE_DELAY_MS,
    MAX_WIDGET_SIZE,
    NO_VIDEO_PANEL_WIDTH_RATIO,
    SIDEBAR_MIN_WIDTH_PX,
    VERTICAL_SPLIT_RATIO,
)

_LINK_COLOR = "#87CEEB"


def styled_link(url: str, text: str) -> str:
    return f'<a href="{url}" style="color: {_LINK_COLOR}; text-decoration: none;">{text}</a>'


REDUNDANT_PREFIXES = [
    "PoseEstimationSeries",
]


class ElidedDelegate(QStyledItemDelegate):
    def __init__(self, elide_mode=Qt.ElideMiddle, parent=None):
        super().__init__(parent)
        self._elide_mode = elide_mode

    def paint(self, painter, option, index):
        text = index.data(Qt.DisplayRole)
        if text:
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(text, self._elide_mode, option.rect.width())
            painter.drawText(option.rect, Qt.AlignVCenter | Qt.AlignLeft, elided)
        else:
            super().paint(painter, option, index)


def clean_display_labels(labels: list[str]) -> list[str]:
    """Strip a prefix from all labels when every label shares that prefix."""
    for prefix in REDUNDANT_PREFIXES:
        if labels and all(label.startswith(prefix) for label in labels):
            labels = [label[len(prefix):] for label in labels]
    return labels


def get_combo_value(combo: QComboBox) -> str:
    data = combo.currentData(Qt.ItemDataRole.UserRole)
    return data if data is not None else combo.currentText()


def find_combo_index(combo: QComboBox, value: str) -> int:
    idx = combo.findData(value, Qt.ItemDataRole.UserRole)
    if idx < 0:
        idx = combo.findText(str(value))
    return idx


def set_combo_to_value(combo: QComboBox, value: str) -> None:
    idx = find_combo_index(combo, str(value))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def apply_compact_widget_style(widget: QWidget, font_size: int = 8) -> None:
    """Apply compact font and stylesheet to a widget subtree."""
    font = QFont()
    font.setPointSize(font_size)
    widget.setFont(font)

    widget.setStyleSheet(f"""
        * {{
            font-size: {font_size}pt;
            padding: 2px;
            margin: 1px;
        }}
        QLabel {{
            font-size: {font_size}pt;
            padding: 2px;
        }}
        QPushButton {{
            font-size: {font_size}pt;
            padding: 4px 8px;
        }}
        QComboBox {{
            font-size: {font_size}pt;
            padding: 2px 4px;
        }}
        QSpinBox, QDoubleSpinBox {{
            font-size: {font_size}pt;
            padding: 2px;
        }}
        QLineEdit {{
            font-size: {font_size}pt;
            padding: 2px 4px;
        }}
        QGroupBox {{
            margin-top: 4px;
            margin-bottom: 2px;
            padding-top: 12px;
        }}
        QGroupBox::title {{
            padding: 2px 4px;
        }}
        QFrame {{
            margin: 1px;
            padding: 1px;
        }}
        QCollapsible {{
            margin: 1px;
            padding: 1px;
            border: none;
            spacing: 2px;
        }}
        QCollapsible > QToolButton {{
            padding: 2px 6px;
            margin: 1px;
            min-height: 18px;
            max-height: 20px;
            border: none;
            border-bottom: 1px solid palette(mid);
        }}
        QCollapsible > QFrame {{
            margin: 2px;
            padding: 2px;
            border: none;
        }}
    """)


def normalize_child_layouts(root: QWidget, spacing: int, margin: int) -> None:
    """Apply consistent spacing/margins to direct child widget layouts."""
    layout = root.layout()
    if layout is None:
        return
    for i in range(layout.count()):
        item = layout.itemAt(i)
        child = item.widget() if item else None
        child_layout = child.layout() if child is not None else None
        if child_layout is None:
            continue
        child_layout.setSpacing(spacing)
        child_layout.setContentsMargins(margin, margin, margin, margin)



class LayoutManager:

    def __init__(self, qt_window: QWidget, plot_container: QWidget):
        self._qt_window = qt_window
        self._plot_container = plot_container
        self._plot_dock: QDockWidget | None = None
        self._layer_docks: list[QDockWidget] = []
        self._sidebar_dock: QDockWidget | None = None

    def register_docks(self) -> None:
        self._plot_dock = None
        self._layer_docks = []
        for dock in self._qt_window.findChildren(QDockWidget):
            if dock.widget() is self._plot_container:
                self._plot_dock = dock
            else:
                title = (dock.windowTitle() or "").lower()
                if "layer" in title:
                    self._layer_docks.append(dock)

    @property
    def plot_dock(self) -> QDockWidget | None:
        return self._plot_dock

    @property
    def layer_docks(self) -> list[QDockWidget]:
        return self._layer_docks

    def set_vertical_ratio(self, ratio: float = VERTICAL_SPLIT_RATIO) -> None:
        if self._plot_dock is None:
            return

        def _resize():
            total_h = self._qt_window.height()
            if total_h <= 0:
                return
            ratio_clamped = max(0.15, min(0.85, float(ratio)))
            min_h = max(0, self._plot_dock.minimumSizeHint().height())
            plot_h = max(min_h, int(total_h * ratio_clamped))
            self._qt_window.resizeDocks([self._plot_dock], [plot_h], Qt.Vertical)

        QTimer.singleShot(100, _resize)

    def with_preserved_height(self, fn: callable) -> None:
        saved_height = self._plot_dock.height() if self._plot_dock else None
        fn()
        if self._plot_dock is not None and saved_height is not None:
            def _restore():
                self._qt_window.resizeDocks(
                    [self._plot_dock], [saved_height], Qt.Vertical,
                )
            QTimer.singleShot(0, _restore)

    def show_layer_docks(self) -> None:
        for dock in self._layer_docks:
            dock.setVisible(True)
        if len(self._layer_docks) >= 2:
            for i in range(1, len(self._layer_docks)):
                self._qt_window.tabifyDockWidget(
                    self._layer_docks[0], self._layer_docks[i],
                )
            self._layer_docks[0].raise_()

    def hide_layer_docks(self) -> None:
        for dock in self._layer_docks:
            dock.setVisible(False)

    def cap_layer_width(self, ratio: float = LAYER_DOCK_WIDTH_RATIO) -> None:
        if not self._layer_docks:
            return
        ratio_clamped = max(0.05, min(0.6, float(ratio)))
        max_w = int(self._qt_window.width() * ratio_clamped)
        for dock in self._layer_docks:
            dock.setMaximumWidth(max_w)

        def _release():
            for d in self._layer_docks:
                d.setMaximumWidth(MAX_WIDGET_SIZE)

        QTimer.singleShot(LAYOUT_RELEASE_DELAY_MS, _release)

    def freeze_layer_widths(self) -> None:
        if not self._layer_docks:
            return
        for dock in self._layer_docks:
            if dock.isVisible():
                dock.setMaximumWidth(dock.width())

        def _release():
            for d in self._layer_docks:
                d.setMaximumWidth(MAX_WIDGET_SIZE)

        QTimer.singleShot(LAYOUT_RELEASE_DELAY_MS, _release)

    def save_dock_widths(self) -> dict[QDockWidget, int]:
        saved: dict[QDockWidget, int] = {}
        for dock in self._layer_docks:
            if dock.isVisible():
                saved[dock] = dock.width()
        if self._sidebar_dock is not None and self._sidebar_dock.isVisible():
            saved[self._sidebar_dock] = self._sidebar_dock.width()
        return saved

    def restore_dock_widths(self, saved: dict[QDockWidget, int]) -> None:
        docks = [d for d in saved if d.isVisible()]
        sizes = [saved[dock] for dock in docks]
        if docks:
            self._qt_window.resizeDocks(docks, sizes, Qt.Horizontal)

    def configure_no_video(self, navigation_widget: QWidget) -> None:
        for dock in self._layer_docks:
            dock.hide()

        central = self._qt_window.centralWidget()
        if central:
            central.hide()

        if self._plot_dock is not None:
            self._qt_window.removeDockWidget(self._plot_dock)
            self._qt_window.addDockWidget(Qt.LeftDockWidgetArea, self._plot_dock)
            self._plot_dock.show()

            def _apply_dock_ratio():
                total_width = self._qt_window.width()
                if total_width <= 0:
                    return
                panel_width = int(total_width * NO_VIDEO_PANEL_WIDTH_RATIO)
                sidebar_width = total_width - panel_width
                for dock in self._qt_window.findChildren(QDockWidget):
                    if dock is self._plot_dock:
                        continue
                    if dock.isVisible() and dock.widget() is not None:
                        self._qt_window.resizeDocks(
                            [self._plot_dock, dock],
                            [panel_width, sidebar_width],
                            Qt.Horizontal,
                        )
                        break

            QTimer.singleShot(0, _apply_dock_ratio)



    def set_video_viewer_visible(self, visible: bool) -> None:
        central = self._qt_window.centralWidget()
        if central is None:
            return
        if visible:
            central.show()
            central.setMaximumHeight(MAX_WIDGET_SIZE)
        else:
            central.setMaximumHeight(0)
            central.hide()

    def set_sidebar_default_width(
        self, sidebar_widget: QWidget, ratio: float,
    ) -> None:
        sidebar_dock = next(
            (d for d in self._qt_window.findChildren(QDockWidget) if d.widget() is sidebar_widget),
            None,
        )
        if sidebar_dock is None:
            return
        self._sidebar_dock = sidebar_dock

        def _apply_sidebar_ratio() -> None:
            total_w = self._qt_window.width()
            if total_w <= 0:
                return

            ratio_clamped = max(0.15, min(0.6, float(ratio)))
            target_w = int(total_w * ratio_clamped)

            screen = self._qt_window.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry().width()
                target_w = min(target_w, int(avail * 0.7))

            target_w = max(SIDEBAR_MIN_WIDTH_PX, target_w)
            self._qt_window.resizeDocks([sidebar_dock], [target_w], Qt.Horizontal)

        # Apply once after the initial layout settles to avoid repeated dock churn.
        QTimer.singleShot(0, _apply_sidebar_ratio)
