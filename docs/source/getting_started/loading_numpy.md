(target-loading-numpy)=
# From a numpy file

Use this path for pre-computed feature arrays stored as `.npy`.

Expected shape: `(n_samples, n_variables)` or `(n_variables, n_samples)`. The longer dimension is assumed to be `n_samples`.

---

## Load it — drag & drop

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`
```

1. On the start page, drag your **`.npy` file** (and optionally a **video**) onto the **Drag & drop** zone.
2. Click **Load**.

One follow-up popup asks for the **data sampling rate** (Hz) — a numpy array has no time axis, so this cannot be inferred. A `session.nc` is written next to your `.npy` and loaded. Frame rate, if a video is dropped, is read automatically.

---

## Adding named variables

Drag & drop creates generic variable names (`var_0`, `var_1`, ...). To give columns meaningful names, create the dataset via a short script instead:

```python
import numpy as np
import xarray as xr

data = np.load("features.npy")   # shape: (n_samples, n_vars)
sr = 1000.0

ds = xr.Dataset({
    "emg": xr.DataArray(
        data,
        dims=["time", "channels"],
        coords={
            "time": np.arange(data.shape[0]) / sr,
            "channels": ["biceps", "triceps"],
        },
    )
})

ds.to_netcdf("session.nc")
```

Then drop the resulting `session.nc` onto the start page.

---

## Data requirements

| Attribute | Value |
|-----------|-------|
| `attrs["fps"]` | Not required unless video is also loaded |

For the full {class}`xarray.Dataset` structure see {doc}`data_requirements`.
