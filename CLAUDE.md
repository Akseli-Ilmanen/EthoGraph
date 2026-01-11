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

ethograph-GUI is a napari plugin for labeling start/stop times of animal movements. It integrates with ethograph, a workflow using action segmentation transformers to predict movement segments. The GUI loads NetCDF datasets containing behavioral features, displays synchronized video/audio, and allows interactive motif labeling.

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
    parameters.py         # Parameter dialogs (magicgui-based)
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
    widgets_labels.py     # Motif labeling interface (LabelsWidget)
    widgets_meta.py       # Main orchestrator widget (MetaWidget)
    widgets_navigation.py # Trial navigation (NavigationWidget)
    widgets_plot.py       # Plot settings controls (PlotsWidget)
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

**Key methods:**
- `get_ds_kwargs()`: Builds selection dict from all `*_sel` attributes
- `set_key_sel(type_key, value)`: Sets selection with previous value tracking
- `toggle_key_sel(type_key)`: Swaps current/previous selection
- `save_to_yaml()` / `load_from_yaml()`: YAML persistence

---

### Widget Orchestration: `widgets_meta.py` (MetaWidget)

Central coordinator that creates and wires all widgets together.

**Responsibilities:**
- Creates shared `ObservableAppState` and passes to all widgets
- Sets up cross-references between widgets
- Binds all global keyboard shortcuts via `@viewer.bind_key()`
- Manages unsaved changes dialog on close

**Widget creation order:**
1. DocumentationWidget (help/shortcuts)
2. IOWidget (file loading)
3. DataWidget (dataset controls)
4. LabelsWidget (motif labeling)
5. PlotsWidget (plotting settings)
6. NavigationWidget (trial navigation)
7. PlotContainer (bottom dock - hidden until data loads)

**Cross-references set:**
- `data_widget.set_references(plot_container, labels_widget, plots_widget, navigation_widget)`
- `labels_widget.set_plot_container(plot_container)`
- `navigation_widget.set_data_widget(data_widget)`
- `plot_container.labels_redraw_needed` signal -> `update_motif_plot()`

---

### Data Loading: `data_loader.py` -> `widgets_io.py` -> `widgets_data.py`

**load_dataset() workflow:**
1. Validate .nc file extension
2. Load via `TrialTree.load(file_path)` -> returns DataTree
3. Extract label_dt via `dt.get_label_dt()`
4. Get first trial: `ds = dt.isel(trials=0)`
5. Categorize variables by `type` attribute (features, colors, changepoints)
6. Extract device info (cameras, mics, tracking) from dataset attrs
7. Return: `(dt, label_dt, type_vars_dict)`

**IOWidget** - File/folder selection:
- Manages paths via QLineEdit + Browse + Clear buttons
- Stores device combos in `self.combos` dict: `{cameras, mics, tracking}`

**DataWidget** - The orchestrator widget:
- `on_load_clicked()`: Triggers loading, creates dynamic UI controls
- `_create_trial_controls()`: Creates combos for all dimensions
- `_on_combo_changed()`: Central handler for all selection changes
- `update_main_plot()`: Updates active plot with current selections
- `update_motif_plot()`: Draws motif rectangles on plot
- `update_video_audio()`: Loads/switches video/audio files

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
- Stores items in `plot_items` (lines) and `label_items` (motifs)
- Y-constraints based on data percentile (default 99.5%)

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

### Label Management: `widgets_labels.py`

**LabelsWidget** - Motif labeling interface:

**State:**
- `motif_mappings`: Dict[int, {color, name}] from mapping.txt
- `ready_for_label_click`: Activated by motif key press
- `first_click` / `second_click`: Time positions from two clicks

**Motif creation workflow:**
1. `activate_motif(motif_id)` -> sets `ready_for_label_click = True`
2. User clicks plot twice -> `_on_plot_clicked()` fires
3. Creates rectangle in `app_state.label_ds['labels']` array
4. `plot_all_motifs()` redraws all labels

**plot_all_motifs():**
1. Gets current plot from plot_container
2. Clears existing `label_items`
3. Draws motif rectangles via `_draw_motif_rectangle()`
4. Different styling: LinePlot uses `LinearRegionItem`, Spectrogram uses edge lines

**Note:** Labels redraw on plot switch via `labels_redraw_needed` signal connected in MetaWidget.

---

### Navigation: `widgets_navigation.py`

**NavigationWidget:**
- Trial selection combo (color-coded: green=verified, red=unverified)
- Previous/Next buttons with confidence filtering
- Playback FPS control

**_trial_change_consequences()** - On trial change:
1. Loads new trial from datatree
2. Updates video/audio files
3. Updates plots and motif labels
4. Resets current_frame to 0

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
load_dataset(nc_path) -> TrialTree.load()
    |
app_state.dt, label_dt, ds set
    |
_create_trial_controls() -> combos created
    |
app_state.ready = True
    |
navigation_widget._trial_change_consequences() -> video/audio/plots
```

**On feature selection change:**
```
User changes feature combo -> _on_combo_changed()
    |
app_state.features_sel = new_value -> Signal emitted
    |
If "Spectrogram" -> plot_container.switch_to_spectrogram()
Else -> plot_container.switch_to_lineplot()
    |
update_main_plot() -> current_plot.update_plot()
    |
update_motif_plot() -> plot_all_motifs()
```

**On motif creation:**
```
User presses '1' -> widgets_meta.activate_motif(1)
    |
labels_widget.ready_for_label_click = True
    |
User clicks plot twice -> _on_plot_clicked()
    |
Create label in app_state.label_ds['labels']
    |
plot_all_motifs() -> redraws labels
    |
app_state.changes_saved = False
```

---

## Keyboard Shortcuts

Bound in `MetaWidget._bind_global_shortcuts()`:

| Key | Action |
|-----|--------|
| 1-0, Q-P, A-L, Z-M | Activate motif 1-30 |
| Ctrl+S | Save labels |
| Space | Play/pause video |
| Up/Down Arrow | Navigate trials |
| Ctrl+F | Toggle feature selection |
| Ctrl+I | Toggle individual selection |
| Ctrl+K | Toggle keypoint selection |
| Ctrl+C | Toggle camera selection |
| Ctrl+M | Toggle mic selection |
| Ctrl+T | Toggle tracking selection |
| Ctrl+E | Edit selected motif |
| Ctrl+D | Delete selected motif |
| Ctrl+V | Mark motif as verified |

---

## Key Design Patterns

1. **Observer Pattern**: AppState emits signals, widgets react
2. **Centralized State**: All data flows through ObservableAppState
3. **Dynamic Attributes**: `*_sel` attributes created as needed for xarray selections
4. **Signal-based Communication**: Widgets communicate via Qt signals
5. **Resource Sharing**: SharedAudioCache singleton prevents file handle leaks
6. **Smart Caching**: SpectrogramBuffer with buffer multiplier for efficiency
7. **State Persistence**: Auto-saving to YAML every 30 seconds

---

## Dataset Structure Requirements

- NetCDF format with `time`, `trials` dimensions
- Expected coordinates: `cameras`, `mics`, `keypoints`, `individuals`, `features`
- Variables with `type='features'` attribute for feature selection
- Video files matched by filename in dataset to video folder
