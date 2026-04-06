"""Quick smoke test for TrialTree.from_continuous()."""
import numpy as np
import pandas as pd
import xarray as xr
from ethograph.io.trialtree import TrialTree

# Create a 'long recording' - 300s at 10Hz
time = np.linspace(0, 300, 3000)
ds = xr.Dataset({
    "speed": xr.DataArray(np.sin(time), dims=["time"], coords={"time": time}),
    "position": xr.DataArray(np.cos(time), dims=["time"], coords={"time": time}),
})
ds.attrs["fps"] = 10

# Define 3 trial epochs
epochs = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 100.0, 200.0],
    "stop_time": [100.0, 200.0, 300.0],
})

dt = TrialTree.from_continuous(ds, epochs)

# Test trials list
assert dt.trials == [1, 2, 3], f"Expected [1,2,3], got {dt.trials}"
print("trials:", dt.trials)

# Test trial access — time should be shifted to 0
ds1 = dt.trial(1)
assert float(ds1.time[0]) < 0.1, f"Trial 1 time should start near 0, got {float(ds1.time[0])}"
assert ds1.attrs["trial"] == 1
print(f"trial(1) time: {float(ds1.time[0]):.2f} - {float(ds1.time[-1]):.2f}")

ds2 = dt.trial(2)
assert float(ds2.time[0]) < 0.1, f"Trial 2 time should start near 0, got {float(ds2.time[0])}"
print(f"trial(2) time: {float(ds2.time[0]):.2f} - {float(ds2.time[-1]):.2f}")

# Test itrial
ds0 = dt.itrial(0)
assert ds0.attrs["trial"] == 1
print(f"itrial(0) trial attr: {ds0.attrs['trial']}")

# Test trial_items
items = list(dt.trial_items())
assert len(items) == 3
print(f"trial_items count: {len(items)}")

# Test _is_continuous
assert dt._is_continuous
print(f"is_continuous: {dt._is_continuous}")

# Test materialise
mat = dt.materialise()
assert mat.trials == [1, 2, 3]
assert not mat._is_continuous
print(f"materialised: trials={mat.trials}, is_continuous={mat._is_continuous}")

# Test get_all_trials
all_t = dt.get_all_trials()
assert sorted(all_t.keys()) == [1, 2, 3]

# Test pynapple IntervalSet input
import pynapple as nap
ep = nap.IntervalSet(start=[0, 100, 200], end=[99, 199, 299])
dt2 = TrialTree.from_continuous(ds, ep)
assert dt2.trials == [1, 2, 3]
ds_nap = dt2.trial(2)
assert float(ds_nap.time[0]) < 0.1
print(f"pynapple epochs: trials={dt2.trials}")

# Test public API
import ethograph as eto
dt3 = eto.from_continuous(ds, epochs)
assert dt3.trials == [1, 2, 3]
print("eto.from_continuous: OK")

print("\nAll tests passed!")
