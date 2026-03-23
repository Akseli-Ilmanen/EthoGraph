## Frame Rate Guidelines

Never hardcode frame rates (e.g., 30 fps) anywhere in the codebase. Always use actual source metadata (e.g., video.fps, audio sample rate) or user settings. If a default is needed, make it configurable and document the rationale.

# CLAUDE.md

## Continue

You always keep proposing things, and not implementing. Stop waiting for me to say 'yes go'.


## System prompt

---
name: python-pro
description: Write idiomatic Python code with advanced features like decorators, generators, and async/await. Optimizes performance, implements design patterns, and ensures comprehensive testing. Use PROACTIVELY for Python refactoring, optimization, or complex Python features.
---

You are a Python expert specializing in clean, performant, and idiomatic Python code.

## Focus Areas
- Advanced Python features (decorators, metaclasses, descriptors)
- Async/await and concurrent programming
- Performance optimization and profiling
- Design patterns and SOLID principles in Python
- SOLID stands for:
    Single-responsibility principle (SRP)
    Open-closed principle (OCP)
    Liskov substitution principle (LSP)
    Interface segregation principle (ISP)
    Dependency inversion principle (DIP)
- Comprehensive testing (pytest, mocking, fixtures)
- Type hints and static analysis (mypy, ruff)

## Approach
1. Pythonic code - follow PEP 8 and Python idioms
2. Prefer composition over inheritance
3. Use generators for memory efficiency

## Import statements

When modifying a Python file, always clean up the import statements at the top:
- Remove unused imports
- Add any missing imports needed by new code
- Sort imports: stdlib → third-party → local (following isort conventions)
- Use explicit imports rather than wildcard (`from x import *`)
- Never place import statements inside functions, methods, or any local scope. All imports belong at the top of the file, regardless of how rarely the code path is executed.
- E.g. never do the following. Only if this would avoid a circular import, that's the only exception.
def load_pose_from_file(...):
    from movement.io import load_poses


## Philosophy for adding comments
"Write code with the philosophy of self-documenting code, where the names of functions, variables, and the overall structure should make the purpose clear without the need for excessive comments. This follows the principle outlined by Robert C. Martin in 'Clean Code,' where the code itself expresses its intent. Therefore, comments should be used very sparingly and only when the code is not obvious, which should occur very, very rarely, as stated in 'The Pragmatic Programmer': 'Good code is its own best documentation. Comments are a failure to express yourself in code.'"

## Error Handling: Fail Fast

Distinguish BUGS from RUNTIME CONDITIONS:
- BUG (wrong type, missing key, None where value expected) → Let it crash. 
  The developer needs the traceback.
- RUNTIME CONDITION (file not found, invalid user input, device disconnected) 
  → Handle gracefully.

Rules:
- Never wrap code in try/except that returns None or defaults when the 
  operation MUST succeed for correctness
- Never add `if x is not None` guards against your own code's output
- Catch broad exceptions ONLY at the outermost GUI boundary (to show 
  error dialogs, not to silently degrade)
- This codebase has defined data flow contracts — trust them, don't 
  defensively re-check upstream outputs

## Output
- Clean Python code with type hints
- Unit tests with pytest and fixtures
- Performance benchmarks for critical paths
- Documentation with docstrings and examples
- Refactoring suggestions for existing code
- Memory and CPU profiling results when relevant

Leverage Python's standard library first. Use third-party packages judiciously.

## Managing Claude.md

After making major design changes, change this claude.md file to match the current state of the repo.

## Development Notes

Claude Code has permission to read make any necessary changes to files in this repository during development tasks.
It has also permissions to read (but not edit!) the folders:
C:\Users\Admin\Documents\Akseli\Code\ethograph
C:\Users\Admin\anaconda3\envs\ethograph-gui



## Project Overview

ethograph-GUI is a napari plugin for labeling start/stop times of animal movements. It integrates with ethograph, a workflow using action segmentation transformers to predict movement segments. The GUI loads NetCDF datasets containing behavioral features, displays synchronized video/audio, and allows interactive labeling.


## Import Convention

The codebase uses `import ethograph as eto` (like `import numpy as np`). Common functions are accessible directly:

```python
import ethograph as eto

dt = eto.open("data.nc")                    # TrialTree.open()
dt = eto.from_datasets([ds1, ds2])          # TrialTree.from_datasets()
time = eto.get_time_coord(da)               # xr_utils.get_time_coord()
data, filt = eto.sel_valid(da, kwargs)       # xr_utils.sel_valid()
align = build_trial_alignment(dt, "1", ds)   # plots_timeseriessource.build_trial_alignment()
dt_new = eto.dataset_to_basic_trialtree(ds) # io.dataset_to_basic_trialtree()
dt.set_media_files(video=..., audio=...)     # trialtree.set_media_files()
eto.add_changepoints_to_ds(ds, ...)         # io.add_changepoints_to_ds()
```

Less common imports use full paths: `from ethograph.features.neural import firing_rate_by_cluster`

## File Structure

```
ethograph/
    __init__.py               # Public API: TrialTree, open(), from_datasets(), get_time_coord(), sel_valid(), etc.

ethograph/gui/
    app_state.py              # Central state management (AppStateSpec + ObservableAppState)
    nwb_loader.py             # NWB file loader — direct PoseEstimation extraction, DANDI video matching
    data_loader.py            # Dataset loading utilities
    label_drawing_mixin.py    # Mixin for label/changepoint drawing (shared by containers)
    napari.yaml               # Napari plugin manifest
    plots_container.py      # Flexible panel container (replaces PlotContainer + MultiPanelContainer)
    plots_audiotrace.py       # Audio waveform with min/max downsampling
    plots_base.py             # Abstract base class for plots
    plots_ephystrace.py       # Ephys multichannel trace
    plots_heatmap.py          # N-dim heatmap (e.g., firing rates per cluster)
    plots_lineplot.py         # Time-series line plot
    plots_raster.py           # Spike raster plot (dots per spike/channel)
    plots_space.py            # 2D/3D position visualization
    plots_spectrogram.py      # Audio spectrogram + caching (SpectrogramBuffer, SharedAudioCache)
    shortcuts_dialog.py       # Keyboard shortcuts help dialog
    video_sync.py             # Napari video/audio synchronization
    widget_ephys.py           # Ephys controls: trace, neuron jumping, preprocessing, firing rates
    widgets_data.py           # Dataset controls and combo boxes (DataWidget)
    widgets_documentation.py  # Help/tutorial interface
    widgets_io.py             # File/folder selection, 2-tab layout: Load data + I/O controls
    widgets_labels.py         # Label labeling interface (LabelsWidget)
    widgets_meta.py           # Main orchestrator widget (MetaWidget)
    widgets_navigation.py     # Trial navigation (NavigationWidget)
    widgets_plot_settings.py  # Plot settings controls (PlotSettingsWidget)
    widgets_transform.py      # Energy envelope + noise reduction controls (TransformWidget)

ethograph/features/
    neural.py             # Firing rate computation from Kilosort spike data (pynapple)

ethograph/utils/
    trialtree.py          # TrialTree class (xr.DataTree subclass with trial convenience methods)
    io.py                 # Standalone I/O functions: dataset_to_basic_trialtree, downsample, changepoints
    xr_utils.py           # sel_valid(), get_time_coord(), trees_to_df()
    timeseries_source.py  # Neurosift-inspired data source abstractions (TimeRange, TimeseriesSource, TrialAlignment)
    label_intervals.py    # Interval-based label representation + crowsetta I/O
    labels.py             # Label utilities (mapping, purge, stitch - legacy dense ops)
    validation.py         # Dataset validation utilities
```

## Architecture

### TrialTree: `trialtree.py`

`TrialTree` inherits from `xr.DataTree`. Each trial is a child node containing an `xr.Dataset` with `attrs["trial"]` as the trial identifier.

**Key methods:**
- `TrialTree.open(path)` / `eto.open(path)`: Load from NetCDF
- `TrialTree.from_datasets(datasets, session_table=)` / `eto.from_datasets(datasets, session_table=)`: Create from list of Datasets
- `dt.trial(trial_id) -> xr.Dataset`: Access a trial's dataset by ID
- `dt.itrial(index) -> xr.Dataset`: Access by integer index
- `dt.trials -> list`: List all trial IDs
- `dt.trial_items() -> Iterator[(trial_id, dataset)]`: Iterate over all trials (replaces manual children filtering)
- `dt.map_trials(func) -> TrialTree`: Apply function to each trial dataset, return new TrialTree
- `dt.update_trial(trial, func)`: Read-modify-write a single trial (for structural changes like adding variables)
- `dt.get_label_dt()`: Extract label TrialTree (auto-converts legacy dense labels to interval format)
- `dt[int_key]` / `dt[int_key] = ds`: Integer indexing supported, auto-wraps Dataset in DataTree


**When to use `dt.trial()` vs `dt.update_trial()`:**
- `dt.trial(id)` returns a mutable Dataset reference — in-place mutations (e.g., `ds["var"].values[:] = ...`) work directly
- `dt.update_trial(id, func)` is needed for structural changes (adding/removing variables) since those require replacing the entire node

---

### Core State Management: `app_state.py`

**Two-class system:**

1. **AppStateSpec** - Type-checked specification with ~40 variables
   - Each variable: `(type_hint, default_value, save_to_yaml)` tuple
   - Categories: Video, Data, Paths, Plotting
   - `saveable_attributes()` returns set of keys to persist

2. **ObservableAppState** - Qt-based reactive state container
   - Inherits from QObject for signal support
   - Stores values in `self._values` dict
   - Auto-generates Signal for each variable (e.g., `current_frame_changed`)
   - **Dynamic `*_sel` attributes**: Created on-the-fly for xarray selections (e.g., `features_sel`, `individuals_sel`)
   - Auto-saves to `gui_settings.yaml` every 30 seconds via QTimer
   - Type validation on `__setattr__` via `check_type()`

**Event signals for decoupled widget communication:**
- `trial_changed`: Emitted by NavigationWidget when trial changes
- `labels_modified`: Emitted by LabelsWidget when labels are created/deleted/modified
- `verification_changed`: Emitted when human verification status changes

**Key methods:**
- `get_ds_kwargs()`: Builds selection dict from all `*_sel` attributes
- `set_key_sel(type_key, value)`: Sets selection for a given dimension key
- `cycle_key_sel(type_key, data_widget)`: Cycles to the next combo box item
- `save_to_yaml()` / `load_from_yaml()`: YAML persistence
- `video_fps` (property): Returns `video.fps` (read from video file via PyAV) or 30.0 if no video loaded


**Notable state variables (added recently):**
- `audio_playback_speed` / `av_speed_coupled`: Audio playback rate control
- `kilosort_folder` (str): Path to Kilosort output for firing rate computation
- `fr_bin_size` / `fr_sigma`: Firing rate bin width and smoothing parameters

---

### Time Coordinate System

Different DataArrays can have different time coordinates (e.g., `time`, `time_aux`, `time_labels`) with different sampling rates. The system handles this transparently:

**Core utility** (`xr_utils.py`, also available as `eto.get_time_coord()`):
```python
def get_time_coord(da: xr.DataArray) -> xr.DataArray | None:
    """Select whichever time coord is available for a given DataArray."""
    coords = da.coords
    time_coord = next((c for c in coords if 'time' in c), None)
    return coords[time_coord].values
```

**AppState time variables:**
- `app_state.label_intervals` (pd.DataFrame | None): Working DataFrame for the current trial's interval-based labels.

**Usage pattern:**
```
Feature DataArray    ->  time coord: "time" or "time_aux"  ->  app_state.time
Labels (intervals)   ->  onset_s/offset_s in seconds       ->  app_state.label_intervals
```

**Where used:**
- `LinePlot._get_buffered_ds()`: Uses `app_state.time` for buffer range calculations
- `BasePlot.set_x_range()`: Uses `app_state.time` for x-axis limits
- `DataWidget.update_label_plot()`: Passes `app_state.label_intervals` DataFrame to plot
- `LabelsWidget`: All operations work directly in seconds (no index conversion needed)

Labels are decoupled from any specific sampling rate since they store onset/offset in seconds.



**X-axis limit enforcement:**
- `_apply_zoom_constraints(x_bounds_override=)` and `toggle_axes_lock(x_bounds_override=)` accept an optional bounds tuple to override each plot's own `_get_time_bounds()`.
- Container computes **tight bounds** via `_compute_tight_x_bounds()` — the intersection of all visible panels' time ranges — so no panel scrolls past another's data.
- Called from `_apply_all_zoom_constraints()`, `toggle_axes_lock()`, `_rebuild_splitter()`, and `_swap_feature_panel()`.
- Individual plots still use their own bounds when `update_plot()` is called outside the container context.

---

### Widget Orchestration: `widgets_meta.py` (MetaWidget)

Central coordinator that creates and wires all widgets together.

**Responsibilities:**
- Creates shared `ObservableAppState` and passes to all widgets
- Sets up signal connections for decoupled communication
- Binds all global keyboard shortcuts via `@viewer.bind_key()`
- Manages unsaved changes dialog on close
- Collapsible layout refresh via `eventFilter` + debounced `QTimer` (50ms)

**Widget creation order (sidebar):**
1. IOWidget (file loading — 2-tab: "Load data" / "I/O controls")
2. DataWidget (dataset controls)
3. TransformWidget (energy envelope + noise reduction)
4. EphysWidget (ephys trace, neuron jumping, preprocessing, firing rates)
5. LabelsWidget (labeling)
6. ChangepointsWidget (CP detection + correction)
7. PlotSettingsWidget (plot settings)
8. NavigationWidget (trial navigation / help)

**Bottom dock:** `UnifiedPanelContainer` (flexible multi-panel layout)

**Signal connections (decoupled communication):**
- `app_state.trial_changed` -> `data_widget.on_trial_changed()`
- `app_state.trial_changed` -> `changepoints_widget._update_cp_status()`
- `app_state.labels_modified` -> `MetaWidget._on_labels_modified()` -> updates plots
- `app_state.verification_changed` -> `MetaWidget._on_verification_changed()` -> updates UI indicators
- `app_state.verification_changed` -> `labels_widget._update_human_verified_status()`
- `plot_container.labels_redraw_needed` -> `MetaWidget._on_labels_redraw_needed()`

**Direct references (DataWidget as central orchestrator):**
- `data_widget.set_references(plot_container, labels_widget, plot_settings_widget, navigation_widget, transform_widget, changepoints_widget, ephys_widget)`
- `labels_widget.set_plot_container(plot_container)` - for drawing labels
- `plot_settings_widget.set_plot_container(plot_container)` - for applying settings
- `transform_widget.set_plot_container(plot_container)` - for envelope overlays
- `ephys_widget.set_plot_container(plot_container)` - for ephys trace
- `io_widget.wire_label_signals()` - connects prediction checkbox to labels_widget
- `io_widget.wire_ephys_signals(ephys_widget)` - connects kilosort/stream combos

---

### Data Loading: `data_loader.py` -> `widgets_io.py` -> `widgets_data.py`

**load_dataset() workflow:**
1. Validate .nc file extension
2. Load via `eto.open(file_path)` -> returns TrialTree
3. Extract label_dt via `dt.get_label_dt()`
4. Get first trial: `ds = dt.itrial(0)`
5. Categorize variables by `type` attribute (features, colors, changepoints)
6. Extract device info (cameras, mics) from session table via `dt.cameras`, `dt.mics`
7. Return: `(dt, label_dt, type_vars_dict)`


**DataWidget** - The central orchestrator widget:
- `on_load_clicked()`: Triggers loading, creates dynamic UI controls
- `on_trial_changed()`: Handles all consequences of trial change (called via signal). Loads interval-based labels via `app_state.get_trial_intervals()` into `app_state.label_intervals`.
- `_create_trial_controls()`: Creates combos for all dimensions (including dynamic ones)
- `_on_combo_changed()`: Central handler for all selection changes
- `update_main_plot()`: Updates active plot with current selections
- `update_label_plot()`: Passes `app_state.label_intervals` DataFrame to `labels_widget.plot_all_labels()`
- `update_video_audio()`: Loads/switches video/audio files
- Stores video sync object on `app_state.video` for access by other widgets

--

---

### Plot System

**Hierarchy:**
```
UnifiedPanelContainer (plots_container.py)
    |  inherits LabelDrawingMixin (label_drawing_mixin.py)
    |
    +-- AudioTracePlot (plots_audiotrace.py) — min/max downsampled waveform
    +-- SpectrogramPlot (plots_spectrogram.py) — audio spectrogram
    +-- EphysTracePlot (plots_ephystrace.py) — multichannel ephys
    +-- RasterPlot (plots_raster.py) — spike raster (dots per spike/channel)
    +-- LinePlot (plots_lineplot.py) — time-series features
    +-- HeatmapPlot (plots_heatmap.py) — N-dim heatmap (e.g., firing rates)
    |
    All inherit BasePlot (plots_base.py)
```

**Panel visibility:** Configured dynamically via `configure_panels(has_audio, has_ephys, has_video)` after data load. Panels show/hide based on available data. Feature panel always present, toggles between LinePlot and HeatmapPlot. Raster panel auto-shows when Kilosort spike data is loaded, positioned between EphysTrace and Feature.

**RasterPlot** (`plots_raster.py`):
- Displays one dot per spike at `(spike_time, channel_y_position)` using `ScatterPlotItem`
- Y-axis synced bidirectionally with EphysTracePlot (guard flag prevents infinite loop)
- Single-cluster mode: gray dots; multi-cluster mode: colored per `_CLUSTER_COLORS`
- Viewport culling via `np.searchsorted` on time + y-range filtering, capped at 5000 dots/cluster
- `sync_y_axis()`: Mirrors `_hw_to_global_y` mapping from ephys trace
- `set_spike_data(times, best_channels)`: Single-cluster display
- `set_multi_cluster_spike_data(entries)`: Multi-cluster with per-cluster colors
- Auto-populated with all spikes on Kilosort load via `_populate_raster_all_spikes()`

**LabelDrawingMixin** (`label_drawing_mixin.py`):
- Extracted from old PlotContainer — provides `draw_all_labels()`, `_draw_single_label()`, `draw_audio_changepoints()`, `draw_dataset_changepoints()`
- Smart styling: uses bottom-strip for spectrogram/heatmap/ephys, vertical bars for line plots
- Shared by UnifiedPanelContainer

**BasePlot** (`plots_base.py`) - Abstract base:
- Time marker (red vertical line for video sync)
- X-axis range modes: `'default'`, `'preserve'`, `'center'`
- Click handling: Emits `plot_clicked` signal with `{x: time, button}`
- Axes locking: Prevents zoom, allows pan

Subclasses implement:
- `update_plot_content(t0, t1)` - Draw actual content
- `apply_y_range(ymin, ymax)` - Set y-axis limits
- `_apply_y_constraints()` - Optional y zoom constraints

**LinePlot** (`plots_lineplot.py`):
- Calls `plot_ds_variable()` to render xarray data
- Stores items in `plot_items` (lines) and `label_items` (labels)
- Y-constraints based on data percentile (default 99.5%)

**Single vs Multi-line Plotting** (`plot_qtgraph.py`):
- Controlled by dimension selection combos with "All" checkbox option
- When a dimension has a specific value selected: `data.ndim == 1` -> single line via `plot_singledim()`
- When "All" is checked for a dimension (e.g., space): `data.ndim == 2` -> multiple lines via `plot_multidim()`
- `plot_multidim()` adds a legend showing coordinate labels (e.g., 'x', 'y', 'z' for space)
- `eto.sel_valid()` handles dimension selection:
  - Dimensions with coordinates use `.sel()` (label-based)
  - Dimensions without coordinates use `.isel()` (integer-based)
  - Returns only `.sel()`-compatible kwargs for title display

**SpectrogramPlot** (`plots_spectrogram.py`):
- Renders audio spectrogram as 2D image
- **SharedAudioCache**: Thread-safe singleton preventing repeated file opens
- **SpectrogramBuffer**: Smart time-based caching with buffer multiplier (default 5x)
- Updates on view range changes via `sigRangeChanged`

**HeatmapPlot** (`plots_heatmap.py`):
- Renders N-dim data (e.g., firing rates per cluster) as 2D color-mapped image
- Configurable colormap and normalization (per_channel, global)

**UnifiedPanelContainer** (`plots_container.py`):
- Replaces old PlotContainer + MultiPanelContainer
- Dynamic panel layout: audio, spectrogram, ephys, feature panels
- X-axis linking across all visible panels
- Time slider shown in no-video mode (manual time browsing)
- Audio playback via `sounddevice` when no video
- Envelope overlay support (on audio or feature panel)
- Emits `plot_changed` and `labels_redraw_needed` signals

---

### Video Synchronization: `video_sync.py`

**NapariVideoSync class:**
- Connects to napari `dims.events.current_step` for frame tracking
- `frame_to_time(frame) -> float`: Converts frame to time accounting for video offset (`frame / fps + time_offset`)
- `time_to_frame(time_s) -> int`: Converts time to frame accounting for video offset
- `seek_to_frame()`: Updates napari dims
- `start()/stop()`: Controls napari's dims.play() with fps_playback
- `play_segment(start, end)`: Synchronized audio/video playback
- `_on_napari_step_change()`: Updates `app_state.current_frame`, emits `frame_changed`
- `time_offset`: Set from `TrialAlignment.video_offset` on construction

**Frame↔time conversion:** All widgets use `video.frame_to_time()` / `video.time_to_frame()` instead of raw `frame / fps`. This ensures video offset is accounted for everywhere.

**Playback rate coupling:** Audio rate = `(fps_playback / fps) * sample_rate`

---


**LabelsWidget** - Label labeling interface:

**State:**
- `_mappings`: Dict[int, {color, name}] from mapping.txt
- `ready_for_label_click`: Activated by label key press
- `first_click` / `second_click`: Float times in seconds from two clicks
- `current_labels_pos`: int | None — DataFrame index of selected interval

**Label creation workflow:**
1. `activate_label(labels)` -> sets `ready_for_label_click = True`
2. User clicks plot twice -> `_on_plot_clicked()` captures time in seconds
3. Optional snap to changepoint via `_snap_to_changepoint_time()` (works in time domain)
4. `_apply_label()` calls `add_interval()` which handles overlap resolution
5. Stores result in `app_state.label_intervals` and writes to `label_dt` via `set_trial_intervals()`
6. `plot_all_labels(intervals_df)` redraws all labels on all plots

**Label selection**: `_check_labels_click()` uses `find_interval_at(df, time_s, individual)` to find the clicked interval. Returns onset/offset/labels directly — no dense array scanning.

**plot_all_labels(intervals_df, predictions_df=None):**
- Delegates to `PlotContainer.draw_all_labels(intervals_df)`
- `_draw_intervals_on_plot()` iterates DataFrame rows directly
- `_draw_single_label(plot, start_time, end_time, labels)` unchanged — already works in time domain

**Note:** Labels redraw on plot switch via `labels_redraw_needed` signal connected in MetaWidget.

---

### Navigation: `widgets_navigation.py`


**Trial Conditions:**
- `setup_trial_conditions(type_vars_dict)`: Called by DataWidget after loading
- Creates combo boxes for each trial condition attribute (e.g., poscat, num_pellets)
- Condition values extracted from dataset attributes (not coordinates)
- Filtering: When a condition is selected, Previous/Next buttons skip non-matching trials

**On trial change:**
1. NavigationWidget sets `app_state.trials_sel` and emits `trial_changed`
2. DataWidget.on_trial_changed() handles all consequences (via signal connection):
   - Loads new trial from datatree
   - Emits `verification_changed` for UI updates
   - Updates video/audio, tracking, plots

---

### Changepoint Correction System: `widgets_changepoints.py` + `changepoints.py`

The correction system refines raw label boundaries by snapping them to detected changepoints in the kinematic/audio data. Uses a **bridge pattern**: intervals are converted to dense arrays for correction, then back to intervals.

**UI Location:** Correction tab in `ChangepointsWidget` (4th toggle alongside Kinematic, Ruptures, Audio).

**Parameters** (persisted in `configs/changepoint_settings.yaml`):
- `min_label_length`: Global minimum label length in samples (labels shorter are removed)
- `label_thresholds`: Per-label overrides for min length (`{labels: min_length}`)
- `stitch_gap_len`: Max gap between same-label segments to merge
- `changepoint_params.max_expansion`: Max samples a boundary can expand toward a changepoint
- `changepoint_params.max_shrink`: Max samples a boundary can shrink toward a changepoint

**Correction pipeline** (`correct_changepoints_dense` in `changepoints.py` — unchanged, operates on dense arrays):
1. Merge all dataset changepoints into a single binary array via `merge_changepoints()`
2. `purge_small_blocks()` — remove labels shorter than their threshold
3. `stitch_gaps()` — merge adjacent same-label segments separated by small gaps
4. For each label block, snap start/end to nearest changepoint index, constrained by `max_expansion`/`max_shrink`
5. Final `purge_small_blocks()` + `fix_endings()` cleanup

**Undo/snapshot**: Stores DataFrame copies (not dense arrays) for revert.

**Modes:**
- *Single Trial*: Corrects current trial's labels only
- *All Trials*: Corrects every trial; sets `label_dt.attrs["changepoint_corrected"] = 1` to prevent double-application

**Signal flow:**
```
User clicks "All Trials" -> ChangepointsWidget._cp_correction("all_trials")
    |
    _correct_trial_intervals() for each trial (bridge: intervals->dense->correct->intervals)
    |
    app_state.set_trial_intervals(trial, corrected_df)
    |
    label_dt.attrs["changepoint_corrected"] = 1
    |
    _update_cp_status() -> green status
    |
    app_state.labels_modified.emit() -> plots refresh
```

---

### Changepoint Storage Architecture

Two distinct storage formats for changepoints, reflecting different data characteristics:

**Normal (kinematic) changepoints** — dense binary arrays in the trial dataset:
- Stored as `int8` DataArrays sharing the feature's time dimension (e.g., `speed_troughs`)
- `attrs: type="changepoints", target_feature=..., method=...`
- Per-feature, multi-dimensional (can have space, keypoints dims)
- Created via `add_changepoints_to_ds()` or ruptures detection
- Typical size: ~100KB/trial at 30Hz

**Audio changepoints** — onset/offset time pairs in the trial dataset:
- Stored as two `float64` DataArrays: `audio_cp_onsets`, `audio_cp_offsets` sharing an `audio_cp` dimension
- `attrs: type="audio_changepoints", target_feature=..., method=...`
- Times in seconds (not indices), decoupled from any sampling rate
- Created via VocalPy/VocalSeg detection in `ChangepointsWidget._compute_audio_changepoints()`
- Typical size: a few KB (hundreds of onset/offset pairs vs ~25MB for dense at 44kHz)

**Why two formats:** Audio rates (44kHz) would make dense binary storage prohibitively large. Onset/offset pairs are compact and naturally map to the seconds-based label system.

**Storage/retrieval in ChangepointsWidget:**
- `_store_audio_cps_to_ds(onsets, offsets, target_feature, method)` — writes to ds via `dt.update_trial()`
- `_get_audio_cps_from_ds()` — reads `audio_cp_onsets`/`audio_cp_offsets` from ds
- Audio CPs persist across trial switches (stored in DataTree) and survive save/reload


**What if you would like to apply audio changepoint detection on non sound files?**
- Audio changepoints are in many ways optimized for audio data. However, if you have high SR periodic data, these methods may be helpful. In this case, we would recommend saving this data as `.wav` files using `audioio`. The `plots_audiotrace.py` also uses smart min/max downsampling, hence the GUI will act much faster than if you load in as conventional `pyqtgraph` lineplot.



---

### Plot Controls: `widgets_plot_settings.py`

**PlotSettingsWidget** - Real-time parameter adjustment:
- Y-axis limits (separate for lineplot/spectrogram)
- Window size (visible time range)
- Buffer settings for audio/spectrogram
- Autoscale and Lock axes checkboxes

---

### Energy & Noise: `widgets_transform.py`

**TransformWidget** - Energy envelope + noise reduction:
- Toggle buttons for Energy/Noise panels
- Energy metric combo: lowpass, highpass, bandpass, meansquared (vocalpy), AVA (vocalpy)
- "Configure..." button opens `dialog_function_params.py` for per-metric parameters
- Envelope target combo (no-video mode only): Audio (top panel) or Feature (bottom panel)
- Noise reduction (noisereduce library) — audio only

---

### Ephys: `widget_ephys.py`

**EphysWidget** - Four toggle tabs:
1. **Ephys trace**: Channel selection, multichannel toggle
2. **Neuron jumping**: Navigate to spike-sorted neuron clusters
3. **Preprocessing**: Z-score, Gaussian smoothing, bandpass filter
4. **Firing rates**: Load Kilosort data, compute firing rates via pynapple, display as heatmap

**Kilosort channel mapping — IMPORTANT:**

Kilosort outputs use two distinct index spaces that must not be confused:

- **Site index** (0..n_sites-1): Position in the `channel_positions.npy` array. Row `i` of `channel_positions` is the (x, y) coordinate of site `i`. This is also the row index into `templates.npy`.
- **Hardware channel** (arbitrary integers): The actual channel ID on the recording device. `channel_map.npy[i]` gives the hardware channel for site `i`. Hardware channel values can exceed `n_sites` (e.g., site index 62 might map to hardware channel 127) because Kilosort may drop channels during sorting.

```
channel_map[site_index] -> hardware_channel    (site 0 -> hw ch 47, site 1 -> hw ch 46, ...)
channel_positions[site_index] -> (x, y)        (NOT indexed by hardware channel!)
```

**Rules:**
- Index `channel_positions` by **site index**, never by hardware channel value
- `cluster_info.tsv` column `ch` contains **hardware channel** IDs (Kilosort's "best channel")
- `spike_clusters.npy` contains **cluster IDs** (not channels) — cluster IDs can have gaps (Kilosort removes clusters during curation), so `max(cluster_id)` != `n_clusters`
- When filtering clusters (e.g., "good" only), always use `xr.DataArray.sel(cluster_id=ids)` rather than passing filtered IDs through the pynapple pipeline, since the cached `_tsgroup` contains all clusters

**Firing rate workflow:**
- Load cluster table from Kilosort (spike_times.npy, spike_clusters.npy, cluster_info.tsv)
- On Kilosort load: `_register_kilosort_features()` adds "Firing rate" and "PCA" to features combo as **disabled** (greyed-out) items, plus "PCA 2D"/"PCA 3D" to slot1 combo
- User clicks "Compute" button → `_compute_firing_rates(force=True)`
- Bin spikes via `firing_rate_by_cluster()` using pynapple
- Store as `ds["firing_rate"]` xr.DataArray with `(cluster_id, time_fr)` dims
- Filter to selected groups (good/mua) via `.sel(cluster_id=...)` after building the full DataArray
- `_enable_feature_item("Firing rate")` makes it selectable; auto-switches combo to "Firing rate"
- Display via HeatmapPlot
- On trial change: `on_trial_changed()` disables "Firing rate"/"PCA" items (user must re-compute)

---

---

## Data Flow Diagrams

**On data load:**
```
User clicks Load -> DataWidget.on_load_clicked()
    |
load_dataset(nc_path) -> TrialTree.open()
    |
dt.get_label_dt() -> auto-converts dense to interval format if needed
    |
app_state.dt, label_dt, ds set
    |
_create_trial_controls() -> combos created
    |
app_state.ready = True
    |
DataWidget.on_trial_changed() -> loads intervals, video/audio/plots
```

**On trial change (signal-based):**
```
User changes trial combo -> NavigationWidget._on_trial_changed()
    |
app_state.trials_sel = new_trial
    |
app_state.trial_changed.emit()  <- Signal emitted
    |
DataWidget.on_trial_changed()   <- Connected listener
    |
    +-- Update datasets (ds, label_ds, pred_ds)
    +-- app_state.label_intervals = app_state.get_trial_intervals(trial)
    +-- _build_trial_alignment() -> app_state.trial_alignment (discovers all sources)
    +-- app_state.verification_changed.emit()
    +-- update_video_audio() -> update_audio_panels() passes alignment audio source
    +-- update_tracking()
    +-- update_main_plot() -> update_label_plot(intervals_df)
    +-- update_space_plot()
```

**On label creation (signal-based):**
```
User presses '1' -> labels_widget.activate_label(1)
    |
labels_widget.ready_for_label_click = True
    |
User clicks plot twice -> _on_plot_clicked() (captures time in seconds)
    |
_apply_label() -> add_interval(df, onset_s, offset_s, labels, individual)
    |
app_state.label_intervals = df  (working DataFrame)
app_state.set_trial_intervals(trial, df)  (persists to label_dt)
    |
app_state.labels_modified.emit()  <- Signal emitted
    |
MetaWidget._on_labels_modified()  <- Connected listener
    |
DataWidget.update_main_plot() -> plot_all_labels(intervals_df)
```


---

## Keyboard Shortcuts

See `docs/shortcuts.md`

---

## Key Design Patterns

1. **Observer Pattern**: AppState emits signals, widgets react
2. **Centralized State**: All data flows through ObservableAppState
3. **Dynamic Attributes**: `*_sel` attributes created as needed for xarray selections
4. **Signal-based Decoupling**: Widgets emit event signals (`trial_changed`, `labels_modified`, `verification_changed`) instead of calling each other directly
5. **Central Orchestrator**: DataWidget handles complex multi-step operations, other widgets are decoupled
6. **Resource Sharing**: SharedAudioCache singleton prevents file handle leaks; video sync stored on `app_state.video`
7. **Smart Caching**: SpectrogramBuffer with buffer multiplier for efficiency
8. **State Persistence**: Auto-saving to YAML every 30 seconds

**Widget Coupling Summary:**
| Widget | Dependencies | Communication |
|--------|--------------|---------------|
| NavigationWidget | `app_state`, `viewer` only | Emits `trial_changed` signal |
| LabelsWidget | `app_state`, `plot_container`, `changepoints_widget`, `io_widget` | Emits `labels_modified`, `verification_changed` |
| ChangepointsWidget | `app_state`, `plot_container` | CP detection, correction, emits `labels_modified` |
| DataWidget | All widgets (orchestrator) | Listens to signals, updates UI |
| PlotSettingsWidget | `app_state`, `plot_container` | Direct plot manipulation |
| TransformWidget | `app_state`, `plot_container` | Energy envelope + noise reduction |
| EphysWidget | `app_state`, `plot_container`, `io_widget`, `data_widget` | Ephys trace, firing rates |
| IOWidget | `app_state`, `data_widget`, `labels_widget`, `changepoints_widget` | Load, predictions, crowsetta I/O |

---

## Dataset Structure Requirements

- NetCDF format with `trials` dimension
- Time coordinates: Can be `time`, `time_aux`, `time_labels`, etc. (any coord containing 'time')
  - Different variables can use different time coordinates with different sampling rates
- Expected coordinates: `keypoints`, `individuals`, `features`
- **Media files at session level:** Video, audio, and pose filenames stored via `dt.set_media_files()` in the session node as 2-D arrays indexed by `(trial, cameras)` or `(trial, mics)`. Access via `dt.get_video(trial, camera)`, `dt.get_audio(trial, mic)`, `dt.get_pose(trial, camera)`. Camera/mic labels via `dt.cameras`, `dt.mics`.
- Variables with `type='features'` attribute for feature selection (features are optional — audio-only datasets supported)
- Video files matched by filename in session table to video folder (video is optional — no-video mode supported)
- **Camera–pose ordering:** `dt.cameras[i]` corresponds to `dt.get_pose(trial, cameras[i])`. Camera index determines pose index — no separate pose combo. Empty string at a pose index means no pose for that camera. For NWB-embedded pose, camera index selects `view=i` or `position_{nwb_pose_keys[i]}`.
- "Audio Waveform" auto-added to features when audio is loaded
- "Firing rate" and "PCA" added (greyed out) when Kilosort folder is loaded; enabled after user clicks "Compute"

**Label format (interval-based):**
- Labels stored as xarray Dataset with `segment` dimension containing:
  - `onset_s` (float64) — start time in seconds
  - `offset_s` (float64) — end time in seconds
  - `labels` (int32) — label class ID (nonzero)
  - `individual` (str) — individual identifier
- **Backward compat**: Old files with dense `labels` DataArray (time x individuals) are auto-converted on load
- Working representation: `pd.DataFrame` with columns `["onset_s", "offset_s", "labels", "individual"]`
- Dense arrays generated on demand via `intervals_to_dense(df, sample_rate, duration, individuals)` for ML pipelines and changepoint correction

---




## RoadMap: Future work



### Testing

Work through claude test functions, and only keep important ones. Add some for changepoints and checking if plot content is there, e.g. spectrogram, etc. Add for model predictions loaded. 


### Integration with models

For audio models, use https://github.com/vocalpy/vak

For vidoe models, DLC2Action, ...


### Labels I/O

**Implemented:**
- **Crowsetta import**: Via IOWidget "I/O controls" tab — format combo (aud-seq, simple-seq, generic-seq, notmat, textgrid, timit, yarden) + file browse → auto-creates/updates mapping → imports to label_dt
- Key functions in `label_intervals.py`: `crowsetta_to_intervals()`, `build_mapping_from_labels()`, `extract_crowsetta_labels()`, `write_mapping_file()`, `resolve_crowsetta_mapping()`

**TODO:**
- For model, check that intervals to dense conversion works correctly, before giving dense to ML
- **Crowsetta export**: `intervals_df_to_crowsetta(df) → crowsetta.Annotation` for:
  - **CSV export**: Via `crowsetta.formats.seq.GenericSeq` transcriber
  - **Audacity label track export**: Via `crowsetta.formats.seq.AudSeq` transcriber
  - **BORIS export**: Via crowsetta or direct DataFrame `to_csv()` with BORIS column mapping
  - **Raven selection table**: Via `crowsetta.formats.bbox.Raven` format
- **Interval-native changepoint correction**: Rewrite `purge_small_blocks()`, `stitch_gaps()`, and boundary snapping to operate directly on interval boundaries in seconds, eliminating the dense bridge entirely
- **Per-label `purge_short_intervals`** with seconds-based thresholds (already implemented in `label_intervals.py`, not yet wired to UI)


