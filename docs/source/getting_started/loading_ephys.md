(target-loading-ephys)=
# From an ephys recording

Use this path for extracellular electrophysiology data, with optional Kilosort spike-sorting output.

At least one of: an ephys file **or** a Kilosort folder is required.

Ephys is a **session-wide stream** — the raw recording file is selected in the GUI rather than embedded in the `session.nc`. For datasets with multiple behavioural trials alongside ephys, see {ref}`Ephys with multiple trials <target-ephys-multi-trial>`.

---

## Steps

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`

In the **I/O widget**, click **Create with own data** — the wizard opens.
```

1. Under **Single trial**, select: **5) Generate from ephys file and/or kilosort folder**
2. Click **Next** — the dialog opens
3. Set **Ephys file** and/or **Kilosort folder**
4. Optionally set **Video file** or **Audio file**
5. Set **Output path** for the generated `session.nc`
6. Click **Generate .nc file**
7. The I/O widget auto-populates -> click **Load**

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Ephys file** | See {ref}`supported formats <target-ephys-formats>` below |
| **Kilosort folder** | Auto-detected if `kilosort4/` or `kilosort/` exists next to ephys file |
| **Ephys sampling rate** | Auto-detected from file or `params.py`; set manually for raw binary |
| **N channels** | Auto-detected for known formats; set manually for raw binary |
| **Video file** | Optional |
| **Video onset in ephys** | Time offset (s) where video starts relative to ephys recording |
| **Audio file** | Optional |
| **Audio onset in ephys** | Time offset (s) where audio starts relative to ephys recording |
| **Output path** | Where to save the generated `session.nc` |

---

(target-ephys-formats)=
## Supported formats

EthoGraph uses [Neo](https://neo.readthedocs.io) to read files with recognised headers — sample rate, channel count, and dtype are extracted automatically. Raw binary files are handled via phylib and require a Kilosort folder.

### Known formats (headers auto-detected)

| Extension(s) | System |
|---|---|
| `.rhd`, `.rhs` | Intan |
| `.oebin` | Open Ephys Binary |
| `.ns1`-`.ns6`, `.nev`, `.nsx` | Blackrock |
| `.abf` | Axon (pCLAMP) |
| `.edf`, `.bdf` | EDF/BDF |
| `.vhdr` | BrainVision |
| `.smr`, `.smrx` | Spike2 (CED) |
| `.ncs`, `.nse`, `.ntt` | Neuralynx |
| `.plx`, `.pl2` | Plexon |
| `.rec` | SpikeGadgets |
| `.meta` | SpikeGLX |
| `.xdat` | NeuroNexus |
| `.tbk` / `.tev` / `.tsq` / ... | TDT |
| `.trc` | Micromed |
| `.edr`, `.wcp` | WinEDR / WinWCP |
| `.nwb` | NWB (local or remote URL) |

When a format supports multiple signal streams (e.g. amplifier vs auxiliary channels in Intan), the GUI lets you select the desired stream from a combo box.

### Raw binary (`.dat` / `.bin` / `.raw`)

Raw binary files produced by Kilosort carry no metadata. They are loaded via [phylib](https://github.com/cortex-lab/phylib) using `n_channels` and `sample_rate` read from `params.py`. Use the **Kilosort folder** picker rather than the ephys file browser — EthoGraph resolves the `.dat` path from `params.py` internally.

---

## Kilosort spike sorting output

Point the GUI at a Kilosort output folder via the **Kilosort folder** picker in the Ephys tab.

**Auto-detection:** If a `kilosort4/` or `kilosort/` directory exists next to your ephys file, EthoGraph fills the field automatically on selection.

### Expected files

| File | Required | Description |
|---|---|---|
| `spike_times.npy` | Yes | Sample indices of each spike |
| `spike_clusters.npy` | Yes | Cluster ID per spike |
| `cluster_info.tsv` | Yes | Per-cluster metadata (group, ch, depth, ...) |
| `params.py` | Auto-created if missing | Sample rate, channel count, raw data path |
| `channel_positions.npy` | Yes | Probe site coordinates (x, y) in um |
| `channel_map.npy` | Yes | Site index -> hardware channel mapping |

### `params.py`

`params.py` is a plain Python file written by Kilosort:

```python
dat_path = r'C:\data\recording.dat'
n_channels_dat = 385
dtype = 'int16'
sample_rate = 30000.0
hp_filtered = False
```

EthoGraph reads `dat_path`, `n_channels_dat`, and `sample_rate` from it. If the file is missing or `dat_path` no longer points to a valid file, a dialog prompts for the values and writes a new `params.py` so the step is not repeated.


### What gets loaded

- `cluster_info.tsv` — cluster groups, best hardware channel (`ch`), depth, firing-rate statistics. Both `KSLabel` (automatic Kilosort classification) and `group` (phy manual curation) are imported.
- `channel_positions.npy` + `channel_map.npy` — probe geometry for the raster and probe-channel dialog.
- The raw `.dat` file (from `dat_path`) — displayed in the trace panel via the phylib loader.

---

(target-ephys-multi-trial)=
## Ephys with multiple trials

Ephys is session-wide. If you have separate video/audio files per trial, build a `session.nc` first (see {doc}`multi_trial`), then select the ephys file separately in the I/O widget.
