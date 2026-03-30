# From a numpy file

Use this path for pre-computed feature arrays stored as `.npy`.

Expected shape: `(n_samples, n_variables)` or `(n_variables, n_samples)`. The longer dimension is assumed to be `n_samples`.

---

## Steps

--8<-- "snippets/launch-single.md"

1. Under **Single trial**, select: **4) Generate from npy file**
2. Click **Next →** — the dialog opens
3. Set **Npy file** (`.npy`)
4. Set **Data sampling rate** (Hz)
5. Optionally set **Video file** — frame rate is auto-detected
6. Set **Output path** for the generated `trials.nc`
7. Click **Generate .nc file**
8. The I/O widget auto-populates → click **Load**

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Video file** | Optional |
| **Npy file** | 2D array — shape `(n_samples, n_vars)` or `(n_vars, n_samples)` |
| **Video frame rate** | Auto-detected from video |
| **Video onset in npy** | Time offset (s) where video starts relative to data stream |
| **Data sampling rate** | Hz — required |
| **Individuals** | Optional, comma-separated |
| **Load video motion features** | Extracts frame-to-frame motion signal from video |
| **Output path** | Where to save the generated `trials.nc` |

---

## Adding named variables

The dialog creates generic variable names (`var_0`, `var_1`, …). To give columns meaningful names, create the dataset via a short script instead:

```python
import numpy as np
import xarray as xr
import ethograph as eto

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
        attrs={"type": "features"},
    )
})
ds.attrs["trial"] = 1

dt = eto.dataset_to_basic_trialtree(ds)
dt.save("trials.nc")
```

---

## Data requirements

| Attribute | Value |
|-----------|-------|
| `attrs["fps"]` | Not required unless video is also loaded |
| `attrs["type"]` on the feature variable | `"features"` |

For the full xarray Dataset structure see [Data Format Requirements](data-requirements.md).
