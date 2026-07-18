"""Backwards-compat re-exports — the left sidebar became :mod:`source_popup`.

The permanent "Sources" dock was replaced by the transient add-panel popup
(``SourcePopup``), opened from the bottom bar's "➕ Add panel" button or Ctrl+N.
"""

from .source_popup import (  # noqa: F401
    SOURCE_MIME,
    PlotTypePicker,
    SourcePopup,
    allowed_plot_types,
    feature_ncols,
)
