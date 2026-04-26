(target-loading-numpy)=
# From a numpy file

Use this path for pre-computed feature arrays stored as `.npy`.

Expected shape: `(n_samples, n_variables)` or `(n_variables, n_samples)`. The longer dimension is assumed to be `n_samples`.

---

## Steps

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`
In the **I/O widget**, click **Create with own data** — the wizard opens.
```

1. Under **Single trial**, select: **4) Generate from npy file**
2. Click **Next** — the dialog opens
3. Set **Npy file** (`.npy`)
4. Set **Data sampling rate** (Hz)
5. Optionally set **Video file** — frame rate is auto-detected
6. Set **Output path** for the generated `session.nc`
7. Click **Generate .nc file**
8. The I/O widget auto-populates -> click **Load**

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Npy file** | 2D array — shape `(n_samples, n_vars)` or `(n_vars, n_samples)` |
| **Data sampling rate** | Hz |
| **Video file** | Optional |
| **Output path** | |

---

## Adding named variables

The dialog creates generic variable names (`var_0`, `var_1`, ...). To give columns meaningful names, create the dataset via a short script instead:

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

---

## Data requirements

| Attribute | Value |
|-----------|-------|
| `attrs["fps"]` | Not required unless video is also loaded |

For the full {class}`xarray.Dataset` structure see {doc}`data_requirements`.
