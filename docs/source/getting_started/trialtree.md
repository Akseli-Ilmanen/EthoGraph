(target-trialtree-getting-started)=
# TrialTree

{class}`~ethograph.io.trialtree.TrialTree` is the core data structure in ethograph — a wrapper around {class}`xarray.DataTree` that stores one {class}`xarray.Dataset` per trial.

```python
import numpy as np
import xarray as xr
import ethograph as eto

# Build: one xr.Dataset per trial
datasets = []
for i in range(1, 4):
    ds = xr.Dataset({"speed": xr.DataArray(np.random.rand(300), dims=["time"],
                                            coords={"time": np.arange(300) / 30.0})})
    ds.attrs["trial"] = i
    ds.attrs["fps"] = 30.0
    datasets.append(ds)

dt = eto.from_datasets(datasets)

dt.trial(2)          # xr.Dataset for trial 2  (label-based)
dt.itrial(0)         # xr.Dataset for trial 1  (0-based index)
dt.trials            # [1, 2, 3]
```

```{seealso}
For the full API with all methods, parameters, and code examples, see {doc}`../api/trialtree`.
```

## Accessing trials

```python
# By trial ID (like xr.Dataset.sel)
ds = dt.trial(2)
ds.attrs["trial"]   # 2
ds["speed"]          # the speed DataArray for trial 2

# By integer index (like xr.Dataset.isel)
ds = dt.itrial(0)
ds.attrs["trial"]   # 1
```

## Iterating

```python
for trial_id, ds in dt.trial_items():
    print(f"Trial {trial_id}: {len(ds.time)} timepoints")

# Apply a function to every trial, returning a new TrialTree
dt_smooth = dt.map_trials(lambda ds: ds.rolling(time=5).mean())
```

## Modifying trials

In-place mutations (changing attribute values, modifying existing arrays) work directly through {meth}`~ethograph.io.trialtree.TrialTree.trial`. Structural changes (adding/removing variables) require {meth}`~ethograph.io.trialtree.TrialTree.update_trial`:

```python
# In-place: modifying existing data
dt.trial(1).attrs["stimulus"] = "tone_A"
dt.trial(1)["speed"].values[:10] = 0.0

# Structural: adding a new variable
dt.update_trial(1, lambda ds: ds.assign(
    smoothed_speed=ds["speed"].rolling(time=5).mean()
))
```

## Saving and loading

```python
dt.save("session.nc")
dt = eto.open("session.nc")
```

## Continuous mode

For a single long recording with trial epochs, use {meth}`~ethograph.io.trialtree.TrialTree.from_continuous` — it slices on demand instead of copying data:

```python
import pandas as pd

epochs = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 60.0, 120.0],
    "stop_time": [60.0, 120.0, 180.0],
})
dt = eto.from_continuous(ds, epochs)
dt.trial(2)  # returns 60–120 s slice, time shifted to 0
```
