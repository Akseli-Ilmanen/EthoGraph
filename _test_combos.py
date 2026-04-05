"""Quick check: RGB excluded from combos, colors combo exists."""
import numpy as np
import pynapple as nap
from ethograph.io.catalog import catalog_from_pynapple

t = np.arange(100)
data = {
    "speed": nap.Tsd(t=t, d=np.random.randn(100)),
    "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
    "angle_rgb": nap.TsdFrame(t=t, d=np.random.rand(100, 3), columns=["R", "G", "B"]),
}

cat = catalog_from_pynapple(data)
print("combos:", list(cat.combos.keys()))
print("colors combo:", cat.combo_values("colors"))
print("RGB hidden:", "angle_rgb_columns" not in cat.combos)
assert "colors" in cat.combos, "colors combo missing"
assert "angle_rgb" in cat.combo_values("colors"), "angle_rgb not in colors combo"
assert "angle_rgb_columns" not in cat.combos, "RGB dim leaked into combos"
print("ALL OK")
