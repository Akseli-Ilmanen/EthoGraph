"""Debug: trace space.yaml reference rendering."""
import numpy as np
import pyqtgraph.opengl as gl
from qtpy.QtWidgets import QApplication

from ethograph.gui.plots_space import (
    load_space_config, _parse_references, _render_reference_3d, _color_to_rgba,
)

app = QApplication.instance() or QApplication([])

# Load the actual space.yaml
from pathlib import Path
cfg = load_space_config(
    Path(r"C:\Users\aksel\.ethograph\example_data\Moll2025\.ethograph\space.yaml")
)
print(f"Config loaded: {cfg is not None}")

refs = _parse_references(cfg)
print(f"References parsed: {len(refs)}")

for ref in refs:
    print(f"  Name: {ref.name}")
    print(f"  Vertices shape: {ref.vertices.shape}")
    print(f"  Edges: {len(ref.edges)}")
    print(f"  Color: {ref.color}")
    print(f"  Color RGBA: {_color_to_rgba(ref.color)}")

# Try rendering on GL widget
w = gl.GLViewWidget()
print(f"\nBefore render: {len(w.items)} items")

try:
    for ref in refs:
        _render_reference_3d(w, ref)
    print(f"After render: {len(w.items)} items")
    for item in w.items:
        print(f"  {type(item).__name__}, mode={getattr(item, 'mode', 'N/A')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
