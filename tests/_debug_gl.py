import sys
from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

import pyqtgraph.opengl as gl
import numpy as np

w = gl.GLViewWidget()
w.setBackgroundColor('w')
print("bg:", w.opts.get('bgcolor'))

X = np.linspace(0, 10, 100).astype(np.float32)
Y = np.sin(X).astype(np.float32)
Z = np.cos(X).astype(np.float32)

xyz = np.column_stack([X, Y, Z])
line = gl.GLLinePlotItem(pos=xyz, color=(0, 0, 1, 1), width=3, antialias=True)
w.addItem(line)
print("items:", len(w.items))
print("xyz range:", xyz.min(axis=0), xyz.max(axis=0))

from pyqtgraph import Vector
cx, cy, cz = float(np.mean(X)), float(np.mean(Y)), float(np.mean(Z))
extent = float(max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min())) * 1.5
w.setCameraPosition(pos=Vector(cx, cy, cz), distance=max(extent, 1.0), elevation=30, azimuth=200)
print("camera distance:", w.cameraParams()['distance'])
print("OK - all good")
