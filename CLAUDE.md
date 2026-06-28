## Frame Rate Guidelines

Never hardcode frame rates (e.g., 30 fps) or sample rates (e.g., 44100 Hz) anywhere in the codebase — not even as fallbacks. Always use actual source metadata (e.g., video.fps, audio sample rate, ImageSeries.rate) or user-specified settings. If a rate is unknown, raise an error or return None — never silently default to a hardcoded value.

Never hardcode a 1-second fallback for time windows, trial durations, or trial start/stop timing. If timing metadata is missing, propagate unknown timing (or raise) instead of using 1.0s placeholders.

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

## Test Files

All test and debug scripts go in `tests/`. Never leave `test_*.py` or `_test_*.py` files in the project root. Prefix with `_test_` for ad-hoc debug scripts (pytest won't discover them).

## Development Notes

Claude Code has permission to make any necessary changes to files in this repository.

## Writing docs

Don't mention indivdiuals Poppy, Freddy, Ivy anywhere in docstrings, docs.

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
    plot_sources               # Plot-facing data sources + buffering (PlotSource, FileSource, XarraySource, WindowedBuffer)
    app_state.py              # Central state management (AppStateSpec + ObservableAppState)
    data_sources.py           # build_audio_source() -> FileSource
    data_loader.py            # Dataset loading: .nc, .nwb, pynapple (DANDI files loaded after local download)
    pose_render.py            # Pose loading (direct NWB + movement), filtering, PoseDisplayManager
    plots_container.py        # UnifiedPanelContainer — multi-panel layout
    plots_base.py             # Abstract base class for all plots (BasePlot)
    plots_audiotrace.py       # Audio waveform (WindowedBuffer + FileSource)
    plots_spectrogram.py      # Spectrogram (SpectrogramBuffer + PlotSource)
    plots_ephystrace.py       # Ephys multichannel trace (custom pyramid buffer + FileSource)
    plots_lineplot.py         # Time-series line plot (WindowedBuffer + XarraySource)
    plots_heatmap.py          # N-dim heatmap (WindowedBuffer + XarraySource for features)
    plots_raster.py           # Spike raster plot
    plots_space.py            # 2D/3D position visualization
    plots_timeseriessource.py # Re-exports from io/time_model.py (backwards compat)
    label_drawing_mixin.py    # Shared label/changepoint drawing
    video_sync.py             # Napari video/audio synchronization (NapariVideoSync)
    video_manager.py          # Multi-camera video loading
    widget_intervals.py       # Interval navigation: browse by trial/label/sequence (IntervalNavigationWidget)
    widgets_meta.py           # Main orchestrator (MetaWidget)
    widgets_data.py           # Dataset controls (DataWidget — central orchestrator)
    widgets_io.py             # File loading, I/O controls (.nc, .nwb, .npz, pynapple folders)
    widgets_labels.py         # Label labeling interface
    widgets_navigation.py     # Navigation: Session/Trial/Label/Sequence mode ("Time slider:" combo)
    widgets_changepoints.py   # Changepoint detection + correction
    widgets_ephys.py          # Ephys controls, neurons (Kilosort/Pynapple), firing rates
    widgets_plot_settings.py  # Plot settings controls
    widgets_transform.py      # Energy envelope + noise reduction

ethograph/labels/
    intervals.py              # Interval operations, mapping loaders, find_blocks (merged from core.py)
    ml.py                     # Dense↔interval conversion, ML post-processing (stitch_gaps, purge_small_blocks, fix_endings)
    tsv_store.py              # TSV file I/O, per-trial access, validation (n_samples per-trial metadata)
    predictions.py            # Load model predictions (.npy/.pickle), confidence via 1-entropy
    crowsetta_format.py       # EthographSeq Crowsetta format (export adapter, int→string labels)
    converters.py             # Crowsetta/pynapple import converters
    export.py                 # enrich_labels_df(), correct_offsets_trial()

ethograph/io/
    catalog.py                # Unified DataCatalog + DataLoader (XarrayLoader, PynappleLoader), pose series discovery
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TrialAlignment, TimeSource, SourceCollection, restriction builders
    time_sources.py           # Concrete adapters: XarrayTrialSource, PynappleSource
    dataset.py                # downsample_trialtree
    validation.py             # validate_datatree, extract_type_vars (delegates to catalog)
    pynapple.py               # Pynapple/NWB loading: load_nap_data, detect_trials, changepoint helpers


ethograph/utils/
    io.py                     # Standalone I/O functions
    xr_utils.py               # sel_valid(), get_time_coord()
    sequences.py              # match_sequences, get_label_instances, get_unique_sequences
```

## Architecture

### Two Data-Source Layers

The codebase has two distinct source protocols:

**Rendering layer** (`io/plot_sources`) — what plots use to load and cache viewport data:
- **`PlotSource`** (Protocol) — `name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`
- `FileSource` — wraps any loader with `rate`/`__len__`/`__getitem__` (audioio, ephys, memmap)
- `XarraySource` — wraps `xr.Dataset`, returns time-sliced datasets from `get_data()`
- `PynappleSource` — lazy access to pynapple Tsd/TsdFrame objects via `restrict()`
- `WindowedBuffer` — viewport-aware cache. Loads wider than viewport, reloads on pan past buffer. Works with all PlotSource implementations.

**Navigation layer** (`io/time_model.py` + `io/time_sources.py`) — session-level time metadata:
- **`TimeSource`** (Protocol) — `name`, `time_range`, `sampling_rate`, `get_data(t0, t1)`
- `SourceCollection` — registry that computes `union_range`, `intersection_range`, `find_trial(t)`
- Concrete adapters: `XarrayTrialSource`, `PynappleSource`

`SourceCollection` only uses `time_range` metadata — it never calls `get_data()`.

**Which plots use what:**
| Plot | Source | Buffer |
|------|--------|--------|
| AudioTracePlot | `FileSource` | `WindowedBuffer` |
| SpectrogramPlot | `PlotSource` (FileSource) | `SpectrogramBuffer` (caches FFT output) |
| EphysTracePlot | `FileSource` (via buffer) | `EphysTraceBuffer` (custom: multi-resolution pyramid) |
| LinePlot | `XarraySource` | `WindowedBuffer` |
| HeatmapPlot (features) | `XarraySource` | `WindowedBuffer` |
| HeatmapPlot (envelope) | Direct loader access | Inline cache |

### TrialTree: `trialtree.py`

`TrialTree` inherits from `xr.DataTree`. Each trial is a child node with `attrs["trial"]`.

Key: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`

Media & Session: Session metadata (trial timing, media file paths, FPS, stream offsets) accessed via `app_state.nwb_alignment` (NWBAlignment). For NWB sources, the source NWB file is used directly — no separate alignment.nwb is needed. For non-NWB sources (.nc, .npz), a sidecar `.ethograph/alignment.nwb` provides this metadata.

### State Management: `app_state.py`

**AppStateSpec** — type-checked spec with ~40 variables.
**ObservableAppState** — Qt signals auto-generated per variable (e.g., `current_frame_changed`). Dynamic `*_sel` attributes for xarray selections. Auto-saves to YAML.

Key signals: `trial_changed`, `restrict_window_changed`, `labels_modified`, `verification_changed`

### Time Model + Navigation: `time_model.py`

Core types in `ethograph/io/time_model.py` (canonical home, re-exported from `gui/plots_timeseriessource.py` for backwards compat):

**`TimeRange`** — immutable time interval with `union()`, `intersect()`, `contains()`, `overlaps()`.

**`TimeSource`** (Protocol) — one time-aligned data source: `name`, `time_range`, `sampling_rate`, `get_data(t0, t1) → (timestamps, values)`. Concrete adapters in `time_sources.py`: `XarrayTrialSource`, `PynappleSource`.

**`SourceCollection`** — Neurosift-inspired registry of `TimeSource` objects. Provides `union_range` (full navigable extent), `intersection_range` (overlap of all sources), `session_range` (min trial start to max trial end), `sources_at(t)`, trial bookmarks (`trial_range`, `find_trial`, `trial_offset`). Built during dataset loading in `data_loader.py`, stored as `app_state.source_collection`.

**`RestrictionWindow`** — display window with mode: `"session"`, `"trial"`, `"label"`, or `"sequence"`.

`app_state.window_bounds` returns the current `RestrictionWindow.time_range` (or falls back to `trial_bounds`). `app_state.session_time_range` returns the full session extent from `SourceCollection`. All plots use `window_bounds` for x-axis limits.

**Navigation modes** (UI label: "Time slider:"):
- Session mode: slider covers entire session (inter-trial gaps navigable)
- Trial mode: standard trial navigation
- Label mode: navigate between instances of a specific label class across trials
- Sequence mode: navigate between trials matching a label sequence pattern (e.g. "1-2-3-5")

**Restriction builders** (in `time_model.py`): `build_trial_window()`, `build_label_window()`, `build_sequence_window()`, `find_closest_trial()`.

**Sequence matching** (`ethograph/utils/sequences.py`): `match_sequences()`, `get_label_instances()`, `get_unique_sequences()`.

### Unified Data Catalog + Loader: `catalog.py`

Replaces the old `type_vars_dict` pattern with two explicit abstractions:

**`DataCatalog`** — declares what's available: features, dimensions (combos), streams. Built by `catalog_from_xarray()` or `catalog_from_pynapple()`. Features are auto-detected: all `data_vars` with a time dimension (xarray) or all `Tsd`/`TsdFrame`/`TsdTensor` objects (pynapple) — no `attrs["type"]` annotation needed. The GUI creates combo boxes from `catalog.combos`. Colors are handled separately: the GUI always creates a "Colors" combo populated with all features, with an "rgb suffix" checkbox (default on) that filters to features containing "rgb" in their name.

**`DataLoader`** (Protocol) — backend-agnostic data access. `select(feature, selections, t0, t1) → PlotData`. Callers always pass `t0, t1` (absolute session times for pynapple, trial-relative for xarray). Follows the `sel_valid` principle: combo selections can be overspecified, loaders ignore dimensions that don't exist on the target feature.

**`PlotData`** — source-agnostic dataclass: `time`, `data` (numpy `(T,)` or `(T,D)`), `dim_labels`, `title`, `ylabel`, `color_data`, `changepoints`. Consumed by `render_plot_data()` in `plots_lineplot.py`.

**Concrete loaders:**
- `XarrayLoader` — wraps `xr.Dataset`, delegates to `sel_valid()`. Updated on trial change via `update_ds()`. t0/t1 are optional (viewport slicing within the already-per-trial dataset).
- `PynappleLoader` — stateless: no trial state. Callers pass absolute session times as `t0, t1` to `select()`. The loader `restrict()`-s to that range, subtracts `t0` so returned time starts near 0. Trial management lives in `SourceCollection` / the GUI, not the loader.

NWB files are loaded via pynapple (which handles NWB → in-memory conversion). No dedicated NWB loader exists — pynapple's native NWB support is sufficient. The only direct NWB access is `_find_pose_series_names()` / `_discover_pose_keypoints()` in `catalog.py`, which scan NWB files with h5py to identify `PoseEstimationSeries` leaf names for the keypoints combo.

**Shared column dimensions:** `_compute_shared_column_dims()` (in `catalog.py`) groups TsdFrame objects by their column values. Objects with identical columns (e.g. position & velocity both with x/y/z) share one dimension name → one combo in the GUI.

### Alignment System: `nwb_alignment.py`

**Rule: `.nwb` sources are read directly; `.ethograph/alignment.nwb` sidecars only exist for non-NWB sources.** Since NWB files are local and writable, edits go into the source NWB (via `edit_nwb` in `nwb_alignment.py`). No bootstrap copies, no two-file divergence. Wizards still produce `.ethograph/alignment.nwb` for `.nc` / `.npz` / pynapple-folder projects that have no NWB to write into.

**`NWBAlignment`** reads any NWB file for session metadata. Constructed from a file path (`NWBAlignment(path)`). Key methods:
- `get_stream_rate(stream, device)` — read `.rate` from any ImageSeries
- `resolve_media_path(trial, stream, device, fallback_folder)` — try ImageSeries path → NWB-relative → fallback folder + filename.
- `stream_offset_for_trial(trial, stream, device)` — trial-relative offset derived from ImageSeries timing

**Priority order** (`_resolve_alignment` in `data_loader.py`): For `.nwb` sources: source NWB trials → sidecar metadata TSV (timing only). For non-NWB sources: sidecar `.ethograph/alignment.nwb` → sidecar metadata TSV.

**Path fallback**: ImageSeries stores original paths. If files move, `resolve_media_path` extracts the filename and joins with a user-specified fallback folder (`video_folder`, `audio_folder`, etc.).

**`align_media_per_trial`** (`utils/nwb.py`) creates ImageSeries for ALL streams via `sync_acquisition_for_streams(nwbfile, stream_rates)`. Takes `stream_rates: dict[str, float]` — no hardcoded FPS values.

**`align_media_from_streams`** (`io/nwb_alignment.py`) — flexible alignment creation accepting per-trial or session-wide files. Used by the NWB wizard to create alignment.nwb.

### Data Loading: `data_loader.py`

The GUI supports loading `.nc` (NetCDF), `.npz`, and pynapple folders. NWB files are loaded through pynapple. Dispatch in `data_loader.py`:
- `.npz`/folder (including NWB via pynapple) → `load_nap_data()` + `catalog_from_pynapple()` + `PynappleLoader`.
- `.nc` → `eto.open()` + `catalog_from_xarray()` + `XarrayLoader`.

`load_dataset()` returns a `LoadResult` dataclass with `dt`, `all_labels_df`, `catalog`, `data_loader`, `source_collection`, and metadata.

### Pose Rendering: `pose_render.py`

Two loading paths for pose data:
- **File-based** (DLC, SLEAP, etc.): `load_pose_from_file()` via `movement.io.load_dataset` + `ds_to_napari_layers`
- **NWB-based**: `load_pose_from_nwb_direct()` reads `PoseEstimationSeries.data` and `.confidence` directly via lazy HDF5 slicing — no xarray/movement intermediate

`PoseRenderData` is the unified result type. `apply_confidence_filter()` and `apply_keypoint_filter()` work on the `data_not_nan` mask. `PoseDisplayManager` orchestrates display via `shown` mask — filtering never recreates layers.

The NWB wizard stores `nwb_pose_keys` (e.g. `["LeftCamera", "RightCamera"]`) in the project config. At render time, `PoseDisplayManager._load_pose_for_camera()` maps camera index → pose key → direct NWB loading.

### Skeleton Visualization: `ethograph/skeleton/` + pose_render

Skeleton rendering reuses the `ethograph/skeleton/` module (ported from movement PR #763): `PrecomputedRenderer` consumes a movement poses `xr.Dataset` and emits a napari Vectors layer; `SkeletonState`/`config.py` manage connections, colors, widths. Only the **config layer** is ethograph-specific:
- `nwb_skeleton_to_config(nodes, edges)` (in `skeleton/config.py`) converts an ndx-pose `Skeleton` (nodes + edge index-pairs) into the standard config dict — so the renderer/state/validation are reused unchanged. This is the default source: `_read_skeleton_config()` reads `container.skeleton` during NWB pose loading and stores it on `PoseRenderData.skeleton_config`.
- `pose_render_to_movement_ds()` un-flattens napari points back into the `(time, space, keypoints, individuals)` poses dataset the renderer needs.
- `PoseDisplayManager._display_skeleton_direct()` builds that dataset and calls `add_skeleton_layer()`. Gated by the "Show skeleton" checkbox. Colour precedence: an active `app_state.skeleton_config_override` (user-drawn, per-segment colours) wins; otherwise the NWB-derived config is recoloured uniformly with `app_state.skeleton_base_color`.
- **Confidence filtering of edges is automatic**: the skeleton is built from the same confidence/keypoint-filtered `PoseRenderData`, so a low-confidence (or hidden) endpoint becomes NaN and the renderer drops any edge touching it on that frame — no skeleton-specific filter code needed.

**Skeleton editor** (`dialog_skeleton_editor.py`): `SkeletonEditorDialog` lets the user draw a skeleton on real pose data — a frame slider scrubs frames, the pyqtgraph canvas shows keypoint XY, drag between keypoints creates an edge, and color categories are assigned to selected edges (click or rubber-band). `get_config()` returns a config dict stored in `skeleton_config_override`. Launched from the Pose tab's "Create / edit skeleton…" button; data via `PoseDisplayManager.primary_pose_for_editor()`.

**Anchored shapes** (`ethograph/skeleton/shapes.py`): shapes (square/triangle/circle templates with named control points) that deform to follow the pose. The user binds ≥2 control points to keypoints (`ShapeAnchorDialog`, a visual template picker); each frame a transform is fit from the template's anchor points to the live keypoint positions — **2 anchors → similarity** (angle-preserving, so e.g. a triangle's base stays perpendicular to its median), **3+ → affine** (deformable). `fit_transform()` + `shape_outline_for_frame()` precompute per-frame outlines; `PoseDisplayManager._display_shapes_direct()` renders them as a napari Shapes layer (frame index as first vertex coord → per-frame visibility, same pattern as bboxes). Shapes live under a `"shapes"` key in the skeleton config and render alongside edges when "Show skeleton" is on.

### Plot System

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`): time marker, x-axis range management, click handling, axes locking.

`UnifiedPanelContainer` holds all panels in a `QSplitter`, links x-axes, manages panel visibility.

### Rendering Sources & Streaming (`io/plot_sources`)

The rendering layer provides **lazy, viewport-aware data streaming** without converting to xarray. All sources implement the `PlotSource` protocol and work with `WindowedBuffer` for efficient caching.

**Core Pattern:**
```python
# Create a source (lazy, no data loaded yet)
source = NWBSource("data.nwb", "acquisition/video_cam-1")

# Wrap in buffer for viewport caching
buffer = WindowedBuffer(buffer_multiplier=5.0)
buffer.set_source(source)

# Viewport pan/zoom triggers lazy load
data = buffer.get(t0=10.0, t1=20.0)  # SampleSlice(timestamps, values)
```

**Source Types:**
- `FileSource` — audioio, ephys, memmap loaders (existing)
- `XarraySource` — xr.Dataset slicing (existing)
- `PynappleSource` — Direct pynapple Tsd/TsdFrame access
  - Wraps `restrict()` for time slicing
  - No xarray intermediate

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

### Neuron Loading (Kilosort / Pynapple)

Two loading paths, both producing a `nap.TsGroup` + cluster table:
- **Kilosort folder**: loads `.npy` files, `cluster_info.tsv`, raw `.dat` trace. Full features (probe map, spike waveforms, raster).
- **Pynapple file** (`.npz`/`.nwb`): loads via `nap.load_file()`, extracts `data["units"]` TsGroup. Raster-only (no raw traces, no probe map). TsGroup metadata columns populate the cluster table.

State: `app_state.neurons_path` (was `kilosort_folder`), `app_state.has_neurons` (was `has_kilosort`). `EphysWidget._neurons_source` is `"kilosort"` or `"pynapple"`.

### Kilosort Channel Mapping

Two index spaces: **site index** (0..n_sites-1, indexes `channel_positions.npy`) vs **hardware channel** (from `channel_map.npy`, can exceed n_sites). `cluster_info.tsv` `ch` column = hardware channel. Always index `channel_positions` by site index.

### Changepoint Correction

Bridge pattern: intervals→dense→correct→intervals. Kinematic CPs stored as dense `int8` arrays. Audio CPs stored as onset/offset float pairs (compact at 44kHz).

---

## Dataset Structure

- NetCDF with trials. Time coords: `time`, `time_aux`, etc. (any containing 'time')
- All `data_vars` with a time dimension are features (no `attrs["type"]` required). Changepoints still use `attrs["type"] = "changepoints"`.
- Color variables are identified by name: any feature with "rgb" in the name (case-insensitive) is offered in the Colors combo. No `attrs["type"] = "colors"` needed.
- Media/session metadata: for `.nwb` sources, read from the source NWB directly (no sidecar created); for non-NWB sources, from `.ethograph/alignment.nwb`
- Labels: stored in `_labels.tsv` (not inside `.nc`). Legacy `.nc` labels auto-migrate on first load.
