## Option A — Convert to `.npy`

Save your data as a `.npy` file (shape `(n_samples,)` or `(n_samples, n_variables)`) and use the **4) Generate from npy file** dialog.

- Column names are assigned in the dialog.
- **High sampling rate?** Enable the **Downsample** checkbox in the I/O widget (e.g. factor 100 keeps 1 in 100 samples). In a script: `eto.downsample_trialtree(dt, factor)`.

See [From a numpy file](loading-numpy.md) for full steps.

---

## Option B — High sampling-rate periodic data → `.wav`

For signals you want to visualise quickly (e.g. 1 kHz pressure sensor, LFP, EMG), convert to `.wav` with [`audioio`](https://github.com/bendalab/audioio):

```python
import audioio
audioio.write_audio("signal.wav", data, sample_rate)
```

Load via **3) Generate from audio file**. Audio is displayed with min/max downsampling — the waveform and spectrogram render fast at any zoom level, no manual downsample step needed.

See [From an audio file](loading-audio.md) for full steps.

---

## Option C — Multi-dimensional data → xarray script

For arrays with 3 or more dimensions (e.g. `time × individuals × keypoints × space`), create an `xr.Dataset` and wrap it in a `TrialTree`:

```python
import xarray as xr
import ethograph as eto

da = xr.DataArray(
    data,  # e.g. shape: (n_time, n_individuals, n_keypoints, 3)
    dims=["time", "individuals", "keypoints", "space"],
    coords={"time": time_vec},
    attrs={"type": "features"},
)
ds = xr.Dataset({"position": da})
ds.attrs["fps"] = sample_rate

dt = eto.dataset_to_basic_trialtree(ds)
dt.save("data.nc")
```

**High sampling rate?** Enable the **Downsample** checkbox in the I/O widget, or call `eto.downsample_trialtree(dt, factor)` in your script before `dt.save()`.

See [TrialTree](trialtree.md) and [Data requirements](data-requirements.md) for the full xarray format.
