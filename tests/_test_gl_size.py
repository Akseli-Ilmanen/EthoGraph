"""Debug: check GLViewWidget vs PlotWidget size policies."""
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy.QtWidgets import QApplication, QSizePolicy

app = QApplication.instance() or QApplication([])

pw = pg.PlotWidget()
gw = gl.GLViewWidget()

print(f"PlotWidget sizeHint: {pw.sizeHint()}")
print(f"PlotWidget sizePolicy: h={pw.sizePolicy().horizontalPolicy()}, v={pw.sizePolicy().verticalPolicy()}")
print(f"PlotWidget minimumSizeHint: {pw.minimumSizeHint()}")
print()
print(f"GLViewWidget sizeHint: {gw.sizeHint()}")
print(f"GLViewWidget sizePolicy: h={gw.sizePolicy().horizontalPolicy()}, v={gw.sizePolicy().verticalPolicy()}")
print(f"GLViewWidget minimumSizeHint: {gw.minimumSizeHint()}")
print()
print(f"QSizePolicy.Expanding = {QSizePolicy.Expanding}")
print(f"QSizePolicy.Preferred = {QSizePolicy.Preferred}")
