# Multi-trial setup (Python script)

You need a short script when:

- You have **multiple trials** — separate video/audio/pose files per trial
- You recorded from **multiple cameras**
- You have **multiple separate microphone files** (one `.wav` per mic)

The Create dialog handles only single files. Everything else uses `eto.from_datasets()`.

---

## Minimal example

```python
import numpy as np
import xarray as xr
import ethograph as eto

datasets = []
for trial_id in range(1, 6):
    n_time = 9000
    ds = xr.Dataset(
        {"speed": xr.DataArray(
            np.random.randn(n_time),
            dims=["time"],
            coords={"time": np.arange(n_time) / 30.0},
            attrs={"type": "features"},
        )},
    )
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = 30.0
    datasets.append(ds)

dt = eto.from_datasets(datasets)
dt.set_media("video", [[f"trial{i:03d}.mp4"] for i in range(1, 6)])
dt.save("trials.nc")
```

---

## Multiple cameras

```python
dt.set_media("video",
    [["cam1_trial001.mp4", "cam2_trial001.mp4"],
     ["cam1_trial002.mp4", "cam2_trial002.mp4"]],
    device_labels=["left", "right"],
)
dt.set_media("pose",
    [["dlc_cam1_trial001.h5", "dlc_cam2_trial001.h5"],
     ["dlc_cam1_trial002.h5", "dlc_cam2_trial002.h5"]],
    device_labels=["left", "right"],
)
```

Camera index determines which pose file is shown: `dt.cameras[i]` maps to `dt.get_media(trial, "pose", cameras[i])`.

---

## Session-wide audio

One continuous audio file covering all trials (with optional per-mic split):

```python
dt.set_media("audio",
    ["session_ch1.wav", "session_ch2.wav"],
    device_labels=["mic-1", "mic-2"],
    per_trial=False,
)
dt.set_stream_offset("audio", 0.23)   # if audio starts 230 ms after reference
```

For per-trial audio files:

```python
dt.set_media("audio",
    [["mic1_trial001.wav", "mic2_trial001.wav"],
     ["mic1_trial002.wav", "mic2_trial002.wav"]],
    device_labels=["mic-1", "mic-2"],
)
```

---

## Ephys with multiple trials

Ephys is session-wide — select the file in the GUI rather than embedding it in `trials.nc`. If clocks differ, record the offset:

```python
dt.set_stream_offset("ephys", 0.0)   # seconds; adjust to match your setup
```

See [Ephys data](loading-ephys.md) and [TrialTree — Stream offsets](trialtree.md#stream-offsets).

---

## Session table and trial timing

```python
import pandas as pd

session_table = pd.DataFrame({
    "trial": list(range(1, 6)),
    "start_time": [i * 300.0 for i in range(5)],
    "stop_time":  [(i + 1) * 300.0 - 0.5 for i in range(5)],
})
dt = eto.from_datasets(datasets, session_table=session_table)
```

Timing enables `dt.start_time(trial)`, `dt.trial_epoch(trial)`, and restricting neural data to trial windows.

---

## Trial conditions

Add metadata to each trial for filtering in the Navigation widget and for export:

```python
ds.attrs["stimulus"] = "tone_A"
```

See [Data requirements — Trial conditions](data-requirements.md#trial-condition-attributes-optional).

---

## Full worked example

For a complete multi-trial example including all streams, see [Data requirements — Full example](data-requirements.md#full-example-multi-trial-dataset).

---

## Other / unsupported formats

If your data format is not covered by the options above:

--8<-- "snippets/other-formats.md"

---

## References

- [TrialTree API](trialtree.md) — `from_datasets()`, `set_media()`, offsets, timing, iteration
- [Data Format Requirements](data-requirements.md) — xarray Dataset structure and `attrs["type"]` conventions
- [Dataset tutorials](tutorials.md) — notebooks with real data
