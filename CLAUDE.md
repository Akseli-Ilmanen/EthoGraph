## Frame Rate Guidelines

Never hardcode frame rates (e.g., 30 fps) anywhere in the codebase. Always use actual source metadata (e.g., video.fps, audio sample rate) or user settings.

# CLAUDE.md

## Continue

You always keep proposing things, and not implementing. Stop waiting for me to say 'yes go'.

## System prompt

---
name: python-pro
description: Write idiomatic Python code with advanced features like decorators, generators, and async/await. Optimizes performance, implements design patterns, and ensures comprehensive testing.
---

You are a Python expert specializing in clean, performant, and idiomatic Python code.

## Focus Areas
- Advanced Python features (decorators, metaclasses, descriptors)
- Performance optimization and profiling
- Design patterns and SOLID principles
- Type hints and static analysis (mypy, ruff)

## Import statements

- Remove unused imports, add missing imports, sort: stdlib → third-party → local
- Use explicit imports, never wildcard (`from x import *`)
- Never place imports inside functions/methods (only exception: avoiding circular imports)

## Philosophy for adding comments
Self-documenting code. Comments only when logic is not obvious, which should be very rare.

Never remove human-authored comments (TODO, FIXME, NOTE, explanatory comments). These represent decisions or reminders from the developer. Only remove comments that you yourself added.

## Error Handling: Fail Fast

- BUG (wrong type, missing key, None where value expected) → Let it crash
- RUNTIME CONDITION (file not found, invalid user input) → Handle gracefully
- Never wrap code in try/except that silently returns None
- Catch broad exceptions ONLY at the outermost GUI boundary

## Managing Claude.md

After making major design changes, update this file to match the current state.

## Development Notes

Claude Code has permission to make any necessary changes to files in this repository.

## Project Overview

ethograph-GUI is a napari plugin for labeling start/stop times of animal movements. It integrates with ethograph, a workflow using action segmentation transformers to predict movement segments. The GUI loads NetCDF datasets, displays synchronized video/audio/ephys, and allows interactive labeling.

## Import Convention

```python
import ethograph as eto

dt = eto.open("data.nc")
dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da)
data, filt = eto.sel_valid(da, kwargs)
```

## File Structure

```
ethograph/
    __init__.py               # Public API

ethograph/gui/
    modality.py               # Unified data source + buffering (ModalitySource, FileSource, XarraySource, WindowedBuffer)
    app_state.py              # Central state management (AppStateSpec + ObservableAppState)
    data_sources.py           # build_audio_source() -> FileSource
    data_loader.py            # Dataset loading utilities
    plots_container.py        # UnifiedPanelContainer — multi-panel layout
    plots_base.py             # Abstract base class for all plots (BasePlot)
    plots_audiotrace.py       # Audio waveform (WindowedBuffer + FileSource)
    plots_spectrogram.py      # Spectrogram (SpectrogramBuffer + ModalitySource)
    plots_ephystrace.py       # Ephys multichannel trace (custom pyramid buffer + FileSource)
    plots_lineplot.py         # Time-series line plot (WindowedBuffer + XarraySource)
    plots_heatmap.py          # N-dim heatmap (WindowedBuffer + XarraySource for features)
    plots_raster.py           # Spike raster plot
    plots_space.py            # 2D/3D position visualization
    plots_timeseriessource.py # TimeRange, TrialAlignment, compute_trial_alignment()
    label_drawing_mixin.py    # Shared label/changepoint drawing
    video_sync.py             # Napari video/audio synchronization (NapariVideoSync)
    video_manager.py          # Multi-camera video loading
    widgets_meta.py           # Main orchestrator (MetaWidget)
    widgets_data.py           # Dataset controls (DataWidget — central orchestrator)
    widgets_io.py             # File loading, I/O controls
    widgets_labels.py         # Label labeling interface
    widgets_navigation.py     # Trial navigation
    widgets_changepoints.py   # Changepoint detection + correction
    widgets_ephys.py          # Ephys controls, Kilosort, firing rates
    widgets_plot_settings.py  # Plot settings controls
    widgets_transform.py      # Energy envelope + noise reduction

ethograph/labels/
    intervals.py              # Interval operations, mapping loaders, find_blocks (merged from core.py)
    ml.py                     # Dense↔interval conversion, ML post-processing (stitch_gaps, purge_small_blocks, fix_endings)
    tsv_store.py              # TSV file I/O, per-trial access, validation (n_samples per-trial metadata)
    predictions.py            # Load model predictions (.npy/.pickle), confidence via 1-entropy
    crowsetta_format.py       # EthographSeq Crowsetta format (export adapter, int→string labels)
    converters.py             # Crowsetta/NWB import converters
    export.py                 # enrich_labels_df(), correct_offsets_trial()

ethograph/utils/
    trialtree.py              # TrialTree (xr.DataTree subclass)
    io.py                     # Standalone I/O functions
    xr_utils.py               # sel_valid(), get_time_coord()
```

## Architecture

### Unified Data Source + Buffering: `modality.py`

All modalities (audio, ephys, features) use a shared abstraction for data access and viewport caching:

**`ModalitySource`** (Protocol) — uniform interface: `name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`

**Concrete sources:**
- `FileSource` — wraps any loader with `rate`/`__len__`/`__getitem__` (audioio, ephys, memmap)
- `XarraySource` — wraps `xr.Dataset`, returns time-sliced datasets from `get_data()`

**`WindowedBuffer`** — generic viewport-aware cache. Loads data wider than the viewport (configurable `buffer_multiplier`), reloads only when panning past the buffered region. Identity-based invalidation on source change.

**Which plots use what:**
| Plot | Source | Buffer |
|------|--------|--------|
| AudioTracePlot | `FileSource` | `WindowedBuffer` |
| SpectrogramPlot | `ModalitySource` (FileSource) | `SpectrogramBuffer` (caches FFT output) |
| EphysTracePlot | `FileSource` (via buffer) | `EphysTraceBuffer` (custom: multi-resolution pyramid) |
| LinePlot | `XarraySource` | `WindowedBuffer` |
| HeatmapPlot (features) | `XarraySource` | `WindowedBuffer` |
| HeatmapPlot (envelope) | Direct loader access | Inline cache |

### TrialTree: `trialtree.py`

`TrialTree` inherits from `xr.DataTree`. Each trial is a child node with `attrs["trial"]`.

Key: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`

Media: `dt.set_media(video=, audio=, pose=)`, `dt.get_video(trial, camera)`, `dt.get_audio(trial, mic)`. 2-D arrays indexed by `(trial, cameras/mics)`.

### State Management: `app_state.py`

**AppStateSpec** — type-checked spec with ~40 variables.
**ObservableAppState** — Qt signals auto-generated per variable (e.g., `current_frame_changed`). Dynamic `*_sel` attributes for xarray selections. Auto-saves to YAML.

Key signals: `trial_changed`, `labels_modified`, `verification_changed`

### Plot System

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`): time marker, x-axis range management, click handling, axes locking.

`UnifiedPanelContainer` holds all panels in a `QSplitter`, links x-axes, manages panel visibility.

### Video Sync: `video_sync.py`

`NapariVideoSync`: `frame_to_time(frame)` / `time_to_frame(t)` with `time_offset`. All widgets use these — never raw `frame / fps`.

### Labels

**Storage:** TSV file (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, trial, human_verified, changepoint_corrected, prediction_source, n_samples`. The `n_samples` column stores per-trial sample count for dense conversion. Label names managed centrally in `mapping.txt`.

**Module structure:** `intervals.py` has interval operations + mapping loaders + `find_blocks`. `ml.py` has dense↔interval conversion + ML post-processing (`stitch_gaps`, `purge_small_blocks`, `fix_endings`). Old `core.py` merged into `intervals.py`; old `dense.py` renamed to `ml.py`.

**In-memory:** `app_state._all_labels_df` (all trials), `app_state.label_intervals` (current trial view). Per-trial metadata stored as columns, not a separate dict.

**Predictions:** Per-trial `.npy`/`.pickle` files in prediction folders. Shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`. Confidence stays in memory for GUI overlay, never stored in our format.

**Crowsetta interop:** `EthographSeq` format registered at import time. Export adapter converts int→string labels via mapping for sharing. Internal storage stays integer-based.

**Migration:** Old `.nc` files with embedded labels auto-migrate to TSV on first load.

### Widget Orchestration

`MetaWidget` creates all widgets and wires signals. `DataWidget` is the central orchestrator — handles trial changes, plot updates, video/audio loading.

**Signal flow:** `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → updates everything.

### Kilosort Channel Mapping

Two index spaces: **site index** (0..n_sites-1, indexes `channel_positions.npy`) vs **hardware channel** (from `channel_map.npy`, can exceed n_sites). `cluster_info.tsv` `ch` column = hardware channel. Always index `channel_positions` by site index.

### Changepoint Correction

Bridge pattern: intervals→dense→correct→intervals. Kinematic CPs stored as dense `int8` arrays. Audio CPs stored as onset/offset float pairs (compact at 44kHz).

---

## Dataset Structure

- NetCDF with trials. Time coords: `time`, `time_aux`, etc. (any containing 'time')
- Variables with `type='features'` for feature selection
- Media at session level via `dt.set_media()`
- Labels: stored in `_labels.tsv` (not inside `.nc`). Legacy `.nc` labels auto-migrate on first load.

## Roadmap

### Testing
Add tests for changepoints, plot content verification, model predictions loaded.

### Integration with models
Audio: vocalpy/vak. Video: DLC2Action.

### Labels I/O
TODO: Crowsetta export (CSV, Audacity, BORIS, Raven). Interval-native changepoint correction (eliminate dense bridge).
