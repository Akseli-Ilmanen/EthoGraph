(target-loading-script)=
# Multi-trial setup (Python script)

You need a short script when:

- You have **multiple trials** — separate video/audio/pose files per trial
- You recorded from **multiple cameras**
- You have **multiple separate microphone files** (one `.wav` per mic)

The Create dialog handles only single files. Everything else uses {func}`eto.from_datasets() <ethograph.from_datasets>`.

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

See {doc}`loading_ephys` and {ref}`TrialTree API — Stream offsets <target-trialtree-offsets-api>`.

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

See {ref}`Data requirements — Trial conditions <target-data-requirements>`.

---

## Full worked example

For a complete multi-trial example including all streams, see {ref}`Data requirements — Full example <target-data-requirements>`.

---

## Other / unsupported formats

If your data format is not covered by the options above:

### Option A — Convert to `.npy`

Save your data as a `.npy` file (shape `(n_samples,)` or `(n_samples, n_variables)`) and use the **4) Generate from npy file** dialog.

- Column names are assigned in the dialog.
- **High sampling rate?** Enable the **Downsample** checkbox in the I/O widget (e.g. factor 100 keeps 1 in 100 samples). In a script: {func}`eto.downsample_trialtree(dt, factor) <ethograph.downsample_trialtree>`.

See {doc}`loading_numpy` for full steps.

### Option B — High sampling-rate periodic data -> `.wav`

For signals you want to visualise quickly (e.g. 1 kHz pressure sensor, EMG), convert to `.wav` with [audioio](https://github.com/bendalab/audioio):

```python
import audioio
audioio.write_audio("signal.wav", data, sample_rate)
```

Load via **3) Generate from audio file**. Audio is displayed with min/max downsampling — the waveform and spectrogram render fast at any zoom level, no manual downsample step needed.

See {doc}`loading_audio` for full steps.

### Option C — Multi-dimensional data -> xarray script

For arrays with 3 or more dimensions (e.g. `time x individuals x keypoints x space`), create an {class}`xarray.Dataset` and wrap it in a {class}`~ethograph.io.trialtree.TrialTree`:

```python
import xarray as xr
import ethograph as eto

da = xr.DataArray(
    data,  # e.g. shape: (n_time, n_individuals, n_keypoints, 3)
    dims=["time", "individuals", "keypoints", "space"],
    coords={"time": time_vec},
)
ds = xr.Dataset({"position": da})
ds.attrs["fps"] = sample_rate

dt = eto.dataset_to_basic_trialtree(ds)
dt.save("data.nc")
```

**High sampling rate?** Enable the **Downsample** checkbox in the I/O widget, or call `eto.downsample_trialtree(dt, factor)` in your script before `dt.save()`.

See the {doc}`../api/trialtree` and {doc}`data_requirements` for the full xarray format.

---

## References

- {doc}`../api/trialtree` — `from_datasets()`, `set_media()`, offsets, timing, iteration
- {doc}`data_requirements` — {class}`xarray.Dataset` structure and `attrs["type"]` conventions
