# CLAUDE.md


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
4. Comprehensive error handling with custom exceptions
5. Test coverage above 90% with edge cases

## Import statements

When modifying a Python file, always clean up the import statements at the top:
- Remove unused imports
- Add any missing imports needed by new code
- Sort imports: stdlib → third-party → local (following isort conventions)
- Use explicit imports rather than wildcard (`from x import *`)

## Philosophy for adding comments
"Write code with the philosophy of self-documenting code, where the names of functions, variables, and the overall structure should make the purpose clear without the need for excessive comments. This follows the principle outlined by Robert C. Martin in 'Clean Code,' where the code itself expresses its intent. Therefore, comments should be used very sparingly and only when the code is not obvious, which should occur very, very rarely, as stated in 'The Pragmatic Programmer': 'Good code is its own best documentation. Comments are a failure to express yourself in code.'"

## Output
- Clean Python code with type hints
- Unit tests with pytest and fixtures
- Performance benchmarks for critical paths
- Documentation with docstrings and examples
- Refactoring suggestions for existing code
- Memory and CPU profiling results when relevant

Leverage Python's standard library first. Use third-party packages judiciously.


## Development Notes

Claude Code has permission to read make any necessary changes to files in this repository during development tasks.
It has also permissions to read (but not edit!) the folders:
C:\Users\Admin\Documents\Akseli\Code\ethograph
C:\Users\Admin\anaconda3\envs\ethograph-gui


## Project Overview

ethograph-GUI is a napari plugin for labeling start/stop times of animal movements. It integrates with ethograph, a workflow using action segmentation transformers to predict movement segments. The GUI loads NetCDF datasets containing behavioral features, displays synchronized video/audio, and allows interactive labeling.

## Development Commands
! for bash mode
### Testing
```bash
# Run tests with tox
tox

# Run tests with pytest directly
pytest
```

### Code Quality
```bash
# Format code with black
black src/

# Lint with ruff (auto-fix enabled in config)
ruff check src/

# Line length: 120 characters (configured in pyproject.toml)
```

### Installation
```bash
# Development install
pip install -e .

# With napari and Qt dependencies
pip install -e ".[all]"

# Testing dependencies
pip install -e ".[testing]"
```

## File Structure

```
ethograph/gui/
    app_state.py          # Central state management (AppStateSpec + ObservableAppState)
    data_loader.py        # Dataset loading utilities
    napari.yaml           # Napari plugin manifest
    plot_container.py     # Manages LinePlot/SpectrogramPlot switching
    plots_base.py         # Abstract base class for plots
    plots_lineplot.py     # Time-series line plot
    plots_space.py        # 2D/3D position visualization
    plots_spectrogram.py  # Audio spectrogram + caching (SpectrogramBuffer, SharedAudioCache)
    shortcuts_dialog.py   # Keyboard shortcuts help dialog
    video_sync.py         # Napari video/audio synchronization
    widgets_data.py       # Dataset controls and combo boxes (DataWidget)
    widgets_documentation.py  # Help/tutorial interface
    widgets_io.py         # File/folder selection (IOWidget)
    widgets_labels.py     # Label labeling interface (LabelsWidget)
    widgets_meta.py       # Main orchestrator widget (MetaWidget)
    widgets_navigation.py # Trial navigation (NavigationWidget)
    widgets_plot.py       # Plot settings controls (PlotsWidget)

ethograph/utils/
    label_intervals.py    # Interval-based label representation (core module)
    labels.py             # Label utilities (mapping, purge, stitch - legacy dense ops)
    io.py                 # TrialTree I/O with auto-conversion of dense → interval format
    data_utils.py         # sel_valid(), get_time_coord()
```

## Architecture

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
- `set_key_sel(type_key, value)`: Sets selection with previous value tracking
- `toggle_key_sel(type_key)`: Swaps current/previous selection
- `save_to_yaml()` / `load_from_yaml()`: YAML persistence
- `get_trial_intervals(trial) -> pd.DataFrame`: Extracts interval labels from label_dt for a trial
- `set_trial_intervals(trial, df)`: Writes interval DataFrame back to label_dt (preserves attrs)

---

### Time Coordinate System

Different DataArrays can have different time coordinates (e.g., `time`, `time_aux`, `time_labels`) with different sampling rates. The system handles this transparently:

**Core utility** (`data_utils.py`):
```python
def get_time_coord(da: xr.DataArray) -> np.ndarray:
    """Select whichever time coord is available for a given DataArray."""
    coords = da.coords
    time_coord = next((c for c in coords if 'time' in c), None)
    return coords[time_coord].values
```

**AppState time variables:**
- `app_state.time` (np.ndarray): Time array for the currently selected feature. Updated when feature selection changes via `set_key_sel("features", ...)`.
- `app_state.label_sr` (float): Derived from the feature time coordinate. Used for audio changepoint quantization in plot_container.
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

---

### Widget Orchestration: `widgets_meta.py` (MetaWidget)

Central coordinator that creates and wires all widgets together.

**Responsibilities:**
- Creates shared `ObservableAppState` and passes to all widgets
- Sets up signal connections for decoupled communication
- Binds all global keyboard shortcuts via `@viewer.bind_key()`
- Manages unsaved changes dialog on close

**Widget creation order:**
1. DocumentationWidget (help/shortcuts)
2. IOWidget (file loading)
3. DataWidget (dataset controls)
4. LabelsWidget (labeling)
5. PlotsWidget (plotting settings)
6. NavigationWidget (trial navigation)
7. PlotContainer (bottom dock - hidden until data loads)

**Signal connections (decoupled communication):**
- `app_state.trial_changed` -> `data_widget.on_trial_changed()`
- `app_state.trial_changed` -> `changepoints_widget._update_cp_status()`
- `app_state.labels_modified` -> `MetaWidget._on_labels_modified()` -> updates plots
- `app_state.verification_changed` -> `MetaWidget._on_verification_changed()` -> updates UI indicators
- `app_state.verification_changed` -> `labels_widget._update_human_verified_status()`
- `plot_container.labels_redraw_needed` -> `MetaWidget._on_labels_redraw_needed()`

**Direct references (DataWidget as central orchestrator):**
- `data_widget.set_references(plot_container, labels_widget, plots_widget, navigation_widget)`
- `labels_widget.set_plot_container(plot_container)` - for drawing labels
- `plots_widget.set_plot_container(plot_container)` - for applying settings

---

### Data Loading: `data_loader.py` -> `widgets_io.py` -> `widgets_data.py`

**load_dataset() workflow:**
1. Validate .nc file extension
2. Load via `TrialTree.open(file_path)` -> returns DataTree
3. Extract label_dt via `dt.get_label_dt()`
4. Get first trial: `ds = dt.itrial(0)`
5. Categorize variables by `type` attribute (features, colors, changepoints)
6. Extract device info (cameras, mics, tracking) from dataset attrs
7. Return: `(dt, label_dt, type_vars_dict)`

**IOWidget** - File/folder selection:
- Manages paths via QLineEdit + Browse + Clear buttons
- Stores device combos in `self.combos` dict: `{cameras, mics, tracking}`

**DataWidget** - The central orchestrator widget:
- `on_load_clicked()`: Triggers loading, creates dynamic UI controls
- `on_trial_changed()`: Handles all consequences of trial change (called via signal). Loads interval-based labels via `app_state.get_trial_intervals()` into `app_state.label_intervals`.
- `_create_trial_controls()`: Creates combos for all dimensions (including dynamic ones)
- `_on_combo_changed()`: Central handler for all selection changes
- `update_main_plot()`: Updates active plot with current selections
- `update_label_plot()`: Passes `app_state.label_intervals` DataFrame to `labels_widget.plot_all_labels()`
- `update_video_audio()`: Loads/switches video/audio files
- Stores video sync object on `app_state.video` for access by other widgets

**Dynamic Dimension Handling:**
- `find_temporal_dims()` in `validation.py`: Identifies all dimensions that co-occur with time
- Creates combo boxes for any dimension found in feature variables (e.g., `channels`, `space`)
- Dimensions with coordinates: Display coordinate labels in combo
- Dimensions without coordinates: Display integer indices (0, 1, 2, ...)
- `get_ds_kwargs()` filters out invalid selections (None, "", "None") before building selection dict

---

### Plot System

**Hierarchy:**
```
PlotContainer (plot_container.py)
    |
    +-- LinePlot (plots_lineplot.py)
    |       +-- inherits BasePlot (plots_base.py)
    |
    +-- SpectrogramPlot (plots_spectrogram.py)
            +-- inherits BasePlot (plots_base.py)
            +-- uses SpectrogramBuffer (caching)
            +-- uses SharedAudioCache (singleton)
```

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
- `sel_valid()` handles dimension selection:
  - Dimensions with coordinates use `.sel()` (label-based)
  - Dimensions without coordinates use `.isel()` (integer-based)
  - Returns only `.sel()`-compatible kwargs for title display

**SpectrogramPlot** (`plots_spectrogram.py`):
- Renders audio spectrogram as 2D image
- **SharedAudioCache**: Thread-safe singleton preventing repeated file opens
- **SpectrogramBuffer**: Smart time-based caching with buffer multiplier (default 5x)
- Updates on view range changes via `sigRangeChanged`

**PlotContainer** (`plot_container.py`):
- Manages LinePlot/SpectrogramPlot visibility switching
- Preserves x-range and time marker on switch
- Emits `plot_changed` and `labels_redraw_needed` signals

---

### Video Synchronization: `video_sync.py`

**NapariVideoSync class:**
- Connects to napari `dims.events.current_step` for frame tracking
- `seek_to_frame()`: Updates napari dims
- `start()/stop()`: Controls napari's dims.play() with fps_playback
- `play_segment(start, end)`: Synchronized audio/video playback
- `_on_napari_step_change()`: Updates `app_state.current_frame`, emits `frame_changed`

**Playback rate coupling:** Audio rate = `(fps_playback / fps) * sample_rate`

---

### Label System: Interval-Based (`label_intervals.py` + `widgets_labels.py`)

Labels are stored as a pandas DataFrame with columns `(onset_s, offset_s, labels, individual)` — times in seconds, decoupled from any sampling rate.

**Core module** (`ethograph/utils/label_intervals.py`):
```python
INTERVAL_COLUMNS = ["onset_s", "offset_s", "labels", "individual"]

empty_intervals() -> pd.DataFrame              # Empty DataFrame with correct dtypes
dense_to_intervals(arr, time, individuals)     # Legacy dense array -> intervals
intervals_to_dense(df, sr, duration, individuals)  # Intervals -> dense (for CP correction)
intervals_to_xr(df) -> xr.Dataset             # DataFrame -> xarray (segment dim) for persistence
xr_to_intervals(ds) -> pd.DataFrame           # xarray -> DataFrame for working
add_interval(df, onset, offset, labels, ind) # Add with overlap resolution (split/trim/delete)
delete_interval(df, idx) -> pd.DataFrame       # Drop by DataFrame index
find_interval_at(df, time_s, individual) -> int | None  # Find interval containing time
get_interval_bounds(df, idx) -> (onset, offset, labels)
purge_short_intervals(df, min_s, per_label=None)  # Drop short intervals
stitch_intervals(df, max_gap_s, individual)    # Merge adjacent same-label
```

**Persistence**: xarray Dataset with `segment` dimension inside the DataTree (same `.nc` file).
**Backward compat**: Old dense `.nc` files auto-convert on load in `get_label_dt()`.

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

**NavigationWidget** - Fully decoupled, only knows `app_state` and `viewer`:
- Trial selection combo (color-coded: green=verified, red=unverified)
- Previous/Next buttons with confidence filtering
- Playback FPS control
- Trial condition combos (managed here, not in DataWidget)
- Emits `app_state.trial_changed` signal when trial changes

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

**Per-label thresholds** are managed via a popup `LabelThresholdsDialog` (button shows count of custom overrides). Values matching the global min are automatically excluded.

**Bridge pattern** (`_correct_trial_intervals` in `widgets_changepoints.py`):
```python
# For each individual:
dense_1d = intervals_to_dense(ind_df, sr, duration, [ind])  # intervals -> dense
corrected_1d = correct_changepoints_dense(dense_1d, ds, all_params)  # correct
corrected_df = dense_to_intervals(corrected_1d, time_coord, [ind])  # dense -> intervals
```

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

### Plot Controls: `widgets_plot.py`

**PlotsWidget** - Real-time parameter adjustment:
- Y-axis limits (separate for lineplot/spectrogram)
- Window size (visible time range)
- Buffer settings for audio/spectrogram
- Autoscale and Lock axes checkboxes

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
    +-- app_state.verification_changed.emit()
    +-- update_video_audio()
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

**On verification change (signal-based):**
```
User marks trial verified -> labels_widget._human_verification_true()
    |
app_state.verification_changed.emit()  <- Signal emitted
    |
    +-- labels_widget._update_human_verified_status()  <- Updates status button
    +-- MetaWidget._on_verification_changed()
        +-- update_labels_widget_title()  <- Updates emoji in title
        +-- data_widget.update_trials_combo()  <- Updates color coding
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
| LabelsWidget | `app_state`, `plot_container`, `changepoints_widget` | Emits `labels_modified`, `verification_changed` |
| ChangepointsWidget | `app_state`, `plot_container` | CP detection, correction, emits `labels_modified` |
| DataWidget | All widgets (orchestrator) | Listens to signals, updates UI |
| PlotsWidget | `app_state`, `plot_container` | Direct plot manipulation |
| IOWidget | `app_state`, `data_widget`, `labels_widget` | Load operations |

---

## Dataset Structure Requirements

- NetCDF format with `trials` dimension
- Time coordinates: Can be `time`, `time_aux`, `time_labels`, etc. (any coord containing 'time')
  - Different variables can use different time coordinates with different sampling rates
- Expected coordinates: `cameras`, `mics`, `keypoints`, `individuals`, `features`
- Variables with `type='features'` attribute for feature selection
- Video files matched by filename in dataset to video folder

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

## Interval Labels Migration — Context & Roadmap

### Why interval-based labels?

EthoGraph labels were originally a dense `np.int8` array of shape `(n_time_samples, n_individuals)` locked to a single sampling rate via `label_sr`. This had three limitations:
1. **Rate-locked**: Cannot label across data types with different rates (video 30Hz vs audio 44kHz)
2. **Export friction**: Converting to standard formats (BORIS, crowsetta, Audacity) required index→time conversion
3. **Memory**: Dense arrays waste space for sparse annotations

The migration to interval-based labels `(onset_s, offset_s, labels, individual)` decouples labels from any sampling rate while preserving the "one file to rule them all" NetCDF philosophy.

### Architecture (implemented)

- **Core module**: `ethograph/utils/label_intervals.py` — 11 pure functions, no GUI dependencies
- **Persistence**: xarray Dataset with `segment` dimension inside `label_dt` → same `.nc` file
- **Working representation**: `pd.DataFrame` in `app_state.label_intervals` (fast filtering, easy crowsetta mapping)
- **Bridge pattern**: Changepoint correction converts intervals→dense→correct→intervals via `_correct_trial_intervals()`
- **Auto-conversion**: Old dense `.nc` files transparently convert on load in `get_label_dt()`
- **Individual filtering**: `update_label_plot()` filters intervals by selected individual from `ds_kwargs`

### Files modified in migration

| File | Changes |
|------|---------|
| `ethograph/utils/label_intervals.py` | **NEW** — all interval logic (empty_intervals, dense_to_intervals, intervals_to_dense, add_interval, delete_interval, find_interval_at, etc.) |
| `tests/test_label_intervals.py` | **NEW** — 32 unit tests |
| `ethograph/utils/io.py` | `get_label_dt()` detects format, auto-converts legacy; `overwrite_with_labels()` writes interval format |
| `ethograph/gui/app_state.py` | Added `label_intervals` state, `get_trial_intervals()`, `set_trial_intervals()` helpers |
| `ethograph/gui/widgets_labels.py` | All label CRUD operations → interval-based (clicks store seconds, not indices) |
| `ethograph/gui/widgets_data.py` | `update_label_plot()` passes filtered intervals DataFrame |
| `ethograph/gui/plot_container.py` | `draw_all_labels()` / `_draw_intervals_on_plot()` iterates DataFrame rows |
| `ethograph/gui/widgets_changepoints.py` | Bridge: intervals↔dense for correction, undo snapshots store DataFrames |
| `ethograph/gui/data_loader.py` | `minimal_basics()` creates empty interval structure |



## RoadMap: Future work

### Audio changepoints

Fix envelope plotting, see also https://github.com/issues/created?issue=vocalpy%7Cvocalpy%7C229, test with cricket audio


### Testing

Work through claude test functions, and only keep important ones. Add some for changepoints and checking if plot content is there, e.g. spectrogram, etc. Add for model predictions loaded. 


### Integration with models

For audio models, use https://github.com/vocalpy/vak

For vidoe models, DLC2Action, ...


### Labels I/0

Thoughts:
- I/O -> Get ./labels should also try to be able to load crowsetta, audacity, boris, raven, all by attempting to convert them into crowsetta and from there into label_dt.nc format. 

- For model, check that intervals to dense conversion works correctly, before giving dense to ML
- 


- **Crowsetta integration**: `intervals_df_to_crowsetta(df) → crowsetta.Annotation` and vice versa. Trivial since DataFrame columns map 1:1 to crowsetta's `Segment(onset_s, offset_s, label)`. This enables:
  - **CSV export/import**: Via `crowsetta.formats.seq.GenericSeq` transcriber
  - **Audacity label track export**: Via `crowsetta.formats.seq.AudSeq` transcriber
  - **BORIS export**: Via crowsetta or direct DataFrame `to_csv()` with BORIS column mapping
  - **Raven selection table**: Via `crowsetta.formats.bbox.Raven` format
- **Interval-native changepoint correction**: Rewrite `purge_small_blocks()`, `stitch_gaps()`, and boundary snapping to operate directly on interval boundaries in seconds, eliminating the dense bridge entirely
- **Per-label `purge_short_intervals`** with seconds-based thresholds (already implemented in `label_intervals.py`, not yet wired to UI)
