# Ephys Data

EthoGraph loads extracellular electrophysiology data as a **single session-wide file** — there is no per-trial splitting of the raw recording. The ephys file is selected directly in the GUI (the "Ephys" file browser in the I/O tab) rather than embedded in the `TrialTree` via `dt.set_media()`. If the recording started at a different time than the behavioural reference clock, use `dt.set_stream_offset("ephys", offset_s)` in your dataset creation script; see [TrialTree — Stream offsets](trialtree.md#stream-offsets).

---

## Raw trace: two loading paths

The raw waveform viewer uses one of two loaders depending on the file format.

### Neo loader — known formats

For files with a recognised extension, [Neo](https://neo.readthedocs.io) extracts dtype, gain, sample rate, and channel count directly from the file header. No extra parameters are required. The following formats are supported:

| Extension(s) | System |
|---|---|
| `.rhd`, `.rhs` | Intan |
| `.oebin` | Open Ephys Binary |
| `.ns1`–`.ns6`, `.nev`, `.nsx` | Blackrock |
| `.abf` | Axon (pCLAMP) |
| `.edf`, `.bdf` | EDF/BDF |
| `.vhdr` | BrainVision |
| `.smr`, `.smrx` | Spike2 (CED) |
| `.ncs`, `.nse`, `.ntt` | Neuralynx |
| `.plx`, `.pl2` | Plexon |
| `.rec` | SpikeGadgets |
| `.meta` | SpikeGLX |
| `.xdat` | NeuroNexus |
| `.tbk` / `.tev` / `.tsq` / … | TDT |
| `.trc` | Micromed |
| `.edr`, `.wcp` | WinEDR / WinWCP |
| `.nwb` | NWB (local or remote URL) |

When a format supports multiple signal streams (e.g. amplifier vs. auxiliary channels in Intan), the GUI lets you select the desired stream from a combo box.

### Phylib loader — raw binary (`.dat` / `.bin` / `.raw`)

Raw binary formats produced by Kilosort cannot be auto-detected: the file header carries no metadata. These files are **loaded internally** when you provide a Kilosort output folder (see below) — you do not browse to the `.dat` file directly. The loader uses [phylib](https://github.com/cortex-lab/phylib) and requires `n_channels` and `sample_rate`, both read from `params.py`.

**Do not** select a `.dat`/`.bin`/`.raw` file via the normal ephys file browser; use the Kilosort folder picker instead.

---

## Kilosort spike sorting output

If you have run Kilosort (any version that produces the standard output folder layout), point the GUI at the output folder via the "Kilosort folder" picker in the Ephys tab.

### Expected files

| File | Required | Description |
|---|---|---|
| `spike_times.npy` | Yes | Sample indices of each spike |
| `spike_clusters.npy` | Yes | Cluster ID per spike |
| `cluster_info.tsv` | Yes | Per-cluster metadata (group, ch, depth, …) |
| `params.py` | Auto-created if missing | Sample rate, channel count, raw data path |
| `channel_positions.npy` | Recommended | Probe site coordinates (x, y) in µm |
| `channel_map.npy` | Recommended | Site index → hardware channel mapping |

### `params.py` — what it provides and what happens when it is missing

`params.py` is a plain Python file (written by Kilosort) containing:

```python
dat_path = r'C:\data\recording.dat'
n_channels_dat = 385
dtype = 'int16'
sample_rate = 30000.0
hp_filtered = False
```

EthoGraph reads `dat_path`, `n_channels_dat`, and `sample_rate` from this file. When `params.py` is missing or `dat_path` no longer points to a valid file, a dialog prompts you to enter the values manually. On confirmation, a new `params.py` is written to the folder so the step is not repeated.

**Sample-rate validation:** If an ephys file is already open in the trace panel, EthoGraph checks that the sample rate in `params.py` matches the rate reported by the loader. A warning is shown if they differ by more than 1 Hz.

### What is loaded from the Kilosort folder

Once the folder is accepted:

- `cluster_info.tsv` is parsed for cluster groups, best hardware channel (`ch`), depth, and firing-rate statistics. Two label columns are read: `KSLabel` contains the automatic Kilosort classification (`good`, `mua`, `noise`), while `group` contains the human-curated labels from [phy](https://github.com/cortex-lab/phy) (if the sort has been manually reviewed). Both columns are imported; phy curation does not overwrite `KSLabel`. 
- `channel_positions.npy` and `channel_map.npy` are used to map spike clusters to probe geometry for the raster and probe-channel dialog.
- The raw `.dat` file (from `dat_path`) is opened via the **phylib loader** and displayed in the Phy-Viewer trace panel.


---

## Stream alignment

When the ephys amplifier runs on a different clock from the behavioural reference, record the offset in your dataset creation script:

```python
# Audio started 230 ms after the reference clock
dt.set_stream_offset("ephys", 0.0)   # ephys is the reference
dt.set_stream_offset("audio", 0.23)  # audio lags by 230 ms
```

`source_start_time(trial, "ephys")` then returns the trial-relative time of ephys sample 0, so the trace is correctly aligned to the behaviour. See [TrialTree — Stream offsets](trialtree.md#stream-offsets) for the full API.
