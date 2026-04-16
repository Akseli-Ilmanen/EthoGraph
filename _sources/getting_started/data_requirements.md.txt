(target-data-requirements)=
# Data Format Requirements

EthoGraph supports three data backends. Pick the one that matches your workflow:

| Backend | Best for | Core object |
|---------|----------|-------------|
| **xarray** | Custom datasets, pose estimation, multi-dim arrays | {class}`xarray.Dataset` / {class}`~ethograph.io.trialtree.TrialTree` |
| **Pynapple** | Neuroscience time-series, NWB interop | {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsGroup` |
| **NWB** | Standardised neurodata, DANDI archives | `.nwb` file (loaded via pynapple) |

This page covers the **core** format: a minimal working example, required
attributes, how features are discovered, multi-subject datasets, and where
media/alignment metadata lives. Optional features (custom dimensions, color
variables) are at the bottom.

For multi-trial setups (multiple videos per trial, multi-camera, session-wide
audio, ephys alignment) see {doc}`multi_trial`.

---

## Minimal working example

::::{tab-set}

:::{tab-item} Xarray
```python
import numpy as np
import xarray as xr
import ethograph as eto

ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            np.random.randn(9000),
            dims=["time"],
            coords={"time": np.arange(9000) / 30.0},
        ),
    },
    coords={"individuals": ["mouse1"]},
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0

dt = eto.from_datasets([ds])
dt.save("session.nc")
```
:::

:::{tab-item} Pynapple
```python
import numpy as np
import pynapple as nap

speed = nap.Tsd(
    t=np.arange(9000) / 30.0,
    d=np.random.randn(9000),
)
nap.save_file({"speed": speed}, "session")
# Load in GUI: select the session.npz file
```
:::

:::{tab-item} NWB
```python
# NWB files from DANDI or NeuroConv are loaded directly — no conversion needed.
# In the GUI: select the .nwb file in the I/O widget and click Load.
#
# To create an NWB file programmatically, see the pynwb documentation:
# https://pynwb.readthedocs.io/en/stable/tutorials/general/plot_file.html
```
:::

::::

---

## Required attributes

::::{tab-set}

:::{tab-item} Xarray

Every trial's {class}`xarray.Dataset` **must** have:

| Attribute | Type | Description |
|-----------|------|-------------|
| `attrs["trial"]` | `int`, `str` | Trial identifier (1, 2, 3, ...). Must be unique across trials. |
| `attrs["fps"]` | `float` | Frame rate of the primary video. Not required for audio-only datasets. |

```python
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0
```
:::

:::{tab-item} Pynapple

Pynapple objects carry timestamps natively — no `fps` or `trial` attribute is needed.

- **Timestamps**: every {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsdTensor` stores its own time axis.
- **Trials**: defined by an {class}`~pynapple.IntervalSet` (either from NWB trials or created manually).

```python
import pynapple as nap
import numpy as np

trials = nap.IntervalSet(
    start=[0.0, 300.0, 600.0],
    end=[299.5, 599.5, 899.5],
)
speed = nap.Tsd(t=np.arange(27000) / 30.0, d=np.random.randn(27000))
```
:::

:::{tab-item} NWB

NWB files follow the [NWB standard](https://www.nwb.org/). EthoGraph reads:

- **Trials**: `nwb.trials` table (`start_time`, `stop_time`, plus custom columns).
- **Behavioural data**: {class}`~pynwb.TimeSeries` in `nwb.processing` modules.
- **Electrophysiology**: {class}`~pynwb.ecephys.ElectricalSeries` in `nwb.acquisition`.
- **Pose estimation**: `PoseEstimation` containers (ndx-pose extension).

If `nwb.trials` is absent, the entire recording is treated as a single trial.
:::

::::

---

## Features (plottable variables)

::::{tab-set}

:::{tab-item} Xarray

Any `data_var` with at least one dimension whose name contains `"time"` appears
in the GUI's **Feature** dropdown:

```python
ds["speed"] = xr.DataArray(
    speed_values,
    dims=["time", "keypoints", "individuals"],
)
```

Different features can use different time coordinates with different sampling
rates (e.g. `time`, `time_accelerometer`, `time_video`).
:::

:::{tab-item} Pynapple

All {class}`~pynapple.Tsd`, {class}`~pynapple.TsdFrame`, and
{class}`~pynapple.TsdTensor` objects in the loaded data dict are detected
automatically. Column names become selectable dimensions.

```python
speed = nap.Tsd(t=time_s, d=speed_values)

position = nap.TsdFrame(
    t=time_s,
    d=pos_array,                        # shape: (n_time, 3)
    columns=["x", "y", "z"],
)
```
:::

:::{tab-item} NWB

EthoGraph discovers features automatically via
{func}`~ethograph.io.nwb_import.probe_behavioral_series`, which lists every
{class}`~pynwb.TimeSeries` in processing modules (excluding `ecephys`, `ophys`,
`ogen`). NWB data is loaded via pynapple, so all {class}`~pynwb.TimeSeries`
become pynapple objects internally.
:::

::::

---

## Specifying individuals

::::{tab-set}

:::{tab-item} Xarray

Individuals are stored as a **coordinate**, not an attribute. With multi-animal
data, this allows the GUI to store separate labels and feature data for
different individuals.

```python
ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            speed_array,                # shape: (time, individuals)
            dims=["time", "individuals"],
        ),
    },
    coords={
        "time": time_values,
        "individuals": ["mouse1", "mouse2", "mouse3"],
    },
)
```

When labelling, the selected individual filters which labels are shown and
created.
:::

:::{tab-item} Pynapple

Pynapple has no built-in concept of "individuals". Multi-subject data is
typically stored as separate objects:

```python
data = {
    "speed_mouse1": nap.Tsd(t=time_s, d=speed_mouse1),
    "speed_mouse2": nap.Tsd(t=time_s, d=speed_mouse2),
}
```

Each object appears as a separate feature in the GUI. Individual selection is
not available for pynapple backends.
:::

:::{tab-item} NWB

NWB files represent a single subject per file —
{attr}`~pynwb.file.NWBFile.subject` is a singular
{class}`~pynwb.file.Subject` object
([PyNWB docs](https://pynwb.readthedocs.io/en/stable/pynwb.file.html)).
Multi-subject experiments use separate `.nwb` files per subject. Individual
selection is not available when loading a single NWB file.
:::

::::

---

## Media files and alignment

Media filenames (video, audio, pose), trial timing, and stream offsets are
stored in an **NWB alignment file** (`.ethograph/alignment.nwb`) — not inside
individual trial datasets. This keeps data and metadata separate, and filenames
are stored as **filename-only strings** so datasets remain portable.

For single-trial recordings, the {doc}`wizard` creates the alignment file for
you. For multi-trial, multi-camera, or session-wide media, see
{doc}`multi_trial`.

{class}`~ethograph.io.nwb_alignment.NWBAlignment` provides programmatic access
to session metadata:

```python
from ethograph.io.nwb_alignment import NWBAlignment

alignment = NWBAlignment("my_project/.ethograph/alignment.nwb")
print(alignment.trials_df)
print(alignment.cameras)
print(alignment.start_time(trial=1))
alignment.close()
```

For **NWB sources**, session metadata lives in the source NWB file directly —
no sidecar alignment is needed for standalone `.nwb` loading. For project
directories (e.g. DANDI), the GUI creates `.ethograph/alignment.nwb`
automatically.

---

## Optional: custom dimensions

Any dimension that co-occurs with a time dimension in at least one feature
variable is automatically discovered and gets a selection
[combo box](https://www.pythonguis.com/docs/qcombobox/) in the GUI.

::::{tab-set}

:::{tab-item} Xarray

```python
ds["emg"] = xr.DataArray(
    emg_data,                            # shape: (time, channels)
    dims=["time", "channels"],
    coords={"channels": ["biceps", "triceps"]},
)
```

Dimensions **do not need to match across features**. For example, `position`
may have `(time, keypoints, space, individuals)` while `speed` only has
`(time, keypoints, individuals)`. The GUI creates combo boxes for the union of
all discovered dimensions. When a feature doesn't have a selected dimension,
that selection is silently ignored via
{func}`~ethograph.utils.xr_utils.sel_valid`:

```python
import ethograph as eto

# "keypoints" and "individuals" are applied; "space" is silently dropped
data, used_kwargs = eto.sel_valid(
    ds["speed"],
    {"keypoints": "nose", "space": "x", "individuals": "mouse1"},
)
```
:::

:::{tab-item} Pynapple

Column names in a {class}`~pynapple.TsdFrame` become a selectable dimension.
Objects with identical column names share a single combo in the GUI.

```python
position = nap.TsdFrame(t=time_s, d=pos, columns=["x", "y", "z"])
velocity = nap.TsdFrame(t=time_s, d=vel, columns=["x", "y", "z"])
```
:::

::::

---

## Optional: color variables

Color variables are identified by **name**: any feature with `"rgb"` in its
name (case-insensitive) is automatically offered in the GUI's **Colors** combo.
Values should lie in `[0, 1]` (float) or `[0, 255]` (int).

::::{tab-set}

:::{tab-item} Xarray

The variable should have an `RGB` dimension of size 3:

```python
ds["angle_rgb"] = xr.DataArray(
    rgb_values,                          # shape: (time, keypoints, individuals, 3)
    dims=["time", "keypoints", "individuals", "RGB"],
)
```

To compute angle-based RGB automatically from pose data, use
{func}`~ethograph.io.dataset.add_angle_rgb_to_ds`:

```python
import ethograph as eto

ds = eto.add_angle_rgb_to_ds(ds, smoothing_params={"sigma": 3})
```
:::

:::{tab-item} Pynapple

Store RGB as a {class}`~pynapple.TsdFrame` with columns `["R", "G", "B"]` (or
any 3-column frame whose name contains `"rgb"`):

```python
import pynapple as nap

angle_rgb = nap.TsdFrame(
    t=time_s,
    d=rgb_values,                        # shape: (n_time, 3)
    columns=["R", "G", "B"],
)
data = {"angle_rgb": angle_rgb}
```

To compute angle-based RGB automatically from a position
{class}`~pynapple.TsdFrame`, use
{func}`~ethograph.io.pynapple.add_angle_rgb_to_nap`:

```python
from ethograph.io.pynapple import add_angle_rgb_to_nap

angle_rgb = add_angle_rgb_to_nap(position, smoothing_params={"sigma": 3})
data["angle_rgb"] = angle_rgb
```
:::

::::
