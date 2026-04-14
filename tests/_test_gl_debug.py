"""Debug script: check what's actually on the GL widget after 3D space plot render."""
import numpy as np
import pyqtgraph.opengl as gl
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

w = gl.GLViewWidget()
w.show()

# Test with NaN data
X = np.array([0, 1, 2, np.nan, 4, 5], dtype=np.float32)
Y = np.array([0, 1, 0, np.nan, 1, 0], dtype=np.float32)
Z = np.array([0, 0, 1, np.nan, 1, 0], dtype=np.float32)
xyz = np.column_stack([X, Y, Z])
line = gl.GLLinePlotItem(pos=xyz, color=(0, 0, 1, 1), width=3)
line._is_trajectory = True
w.addItem(line)

print(f"Items on widget: {len(w.items)}")
for item in w.items:
    print(f"  {type(item).__name__}, _is_trajectory={getattr(item, '_is_trajectory', False)}")

gl_items = [i for i in w.items if isinstance(i, gl.GLLinePlotItem)]
print(f"GLLinePlotItem count: {len(gl_items)}")

# Now test with clean data (no NaN)
from ethograph.features.preprocessing import interpolate_nans
X2 = interpolate_nans(X)
Y2 = interpolate_nans(Y)
Z2 = interpolate_nans(Z)
print(f"\nAfter interpolate_nans:")
print(f"  X has NaN: {np.any(np.isnan(X2))}")
print(f"  Y has NaN: {np.any(np.isnan(Y2))}")
print(f"  Z has NaN: {np.any(np.isnan(Z2))}")
