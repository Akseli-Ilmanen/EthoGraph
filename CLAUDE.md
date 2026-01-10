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
    Open–closed principle (OCP)
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

## Philosophy for adding commetns
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
C:\Users\Admin\Documents\Akseli\Code\moveseg
C:\Users\Admin\anaconda3\envs\moveseg-gui


## Project Overview

moveseg-GUI is a napari plugin for labeling start/stop times of animal movements. It integrates with moveseg, a workflow using action segmentation transformers to predict movement segments. The GUI loads NetCDF datasets containing behavioral features, displays synchronized video/audio, and allows interactive motif labeling.

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

## Architecture

### Core Components

**MetaWidget** (`meta_widget.py`) - Main container widget that orchestrates all other widgets:
- Creates and manages `ObservableAppState` with YAML persistence
- Sets up cross-references between widgets
- Binds global keyboard shortcuts for napari
- Uses CollapsibleWidgetContainer for UI organization

**ObservableAppState** (`app_state.py`) - Central state management:
- Qt signal-based reactive state container
- Automatic YAML persistence (every 10 seconds)
- Manages dataset, file paths, plot settings, current selections
- Dynamic `_sel` attributes for user selections (trials_sel, keypoints_sel, etc.)

**IOWidget** (`io_widget.py`) - Data and media file management:
- File/folder selection dialogs for NetCDF datasets, video, and audio files
- Handles loading and saving of datasets and user sessions
- Validates file formats and paths before loading
- Provides feedback on IO operations and errors

**DataWidget** (`data_widget.py`) - Dataset exploration and trial navigation:
- Displays dataset structure and metadata
- Dynamic combo boxes for selecting trials, individuals, and features based on loaded dataset
- Trial filtering by user-defined conditions
- Coordinates selection changes with other widgets
- Synchronizes trial and selection state with video/audio playback

## Plot System Architecture (LinePlot & Spectrogram)

### Component Hierarchy

```
PlotContainer (plot_container.py)
    │
    ├── LinePlot (plots_lineplot.py)
    │       └── inherits → BasePlot (plots_base.py)
    │
    └── SpectrogramPlot (plots_spectrogram.py)
            ├── inherits → BasePlot (plots_base.py)
            └── uses → SpectrogramBuffer (same file)
```

### BasePlot (`plots_base.py`)
Abstract base class providing shared functionality:
- Time marker (red vertical line for video sync)
- X-axis range management (default, preserve, center modes)
- Click handling via `plot_clicked` signal
- Axes locking (prevent zoom, allow pan)
- Zoom constraints based on data bounds

Subclasses must implement:
- `update_plot_content(t0, t1)` - Draw actual content
- `apply_y_range(ymin, ymax)` - Set y-axis limits
- `_apply_y_constraints()` - Optional y-axis zoom limits

### LinePlot (`plots_lineplot.py`)
- Calls `plot_ds_variable()` to draw xarray data as lines
- Stores line items in `self.plot_items`
- Stores motif rectangles in `self.label_items`
- Y-constraints based on data percentiles

### SpectrogramPlot (`plots_spectrogram.py`)
- Uses `pg.ImageItem` to display spectrogram as image
- Connects `sigRangeChanged` to recompute on pan/zoom
- Has `SpectrogramBuffer` for caching computed spectrograms
- Frequency axis (y) based on audio sample rate

### SpectrogramBuffer (in `plots_spectrogram.py`)
Smart caching system for spectrogram computation:
- Dictionary cache with `(buffer_t0, buffer_t1)` keys
- Buffer multiplier expands requested range (default 5x)
- Uses `SharedAudioCache` (audio_cache.py) for audio data
- Cache limit: 10 entries with simple LRU eviction

### PlotContainer (`plot_container.py`)
Simple container for switching between plots:
- Holds both LinePlot and SpectrogramPlot instances
- Only one visible at a time
- Pass-through methods: `get_current_xlim()`, `set_x_range()`, etc.
- Emits `plot_changed` signal on switch

### Data Flow for Plot Updates

```
DataWidget.update_main_plot(**kwargs)
    │
    ├─→ current_plot.update_plot(t0, t1)
    │       │
    │       └─→ BasePlot.update_plot()
    │               │
    │               ├─→ update_plot_content()  [subclass specific]
    │               │       ├─→ LinePlot: plot_ds_variable()
    │               │       └─→ Spectrogram: buffer.compute()
    │               │
    │               ├─→ set_x_range()
    │               └─→ toggle_axes_lock()
    │
    └─→ update_motif_plot(ds_kwargs)
            │
            └─→ labels_widget.plot_all_motifs(time, labels, predictions)
                    │
                    └─→ _draw_motif_rectangle() → adds to current_plot.label_items
```

### Label Drawing (widgets_labels.py)

`plot_all_motifs()` workflow:
1. Get `current_plot` from plot_container
2. Clear existing `label_items` from current_plot
3. Iterate through label data finding motif segments
4. Call `_draw_motif_rectangle()` for each segment
5. Rectangles added to `current_plot.plot_item` and `current_plot.label_items`

**CRITICAL ISSUE**: Labels are only drawn on whichever plot is currently active.
When switching from LinePlot to SpectrogramPlot, labels don't transfer.

---

## Known Issues & Diagnosis (December 2025)

### Issue 1: Labels Only Show on LinePlot, Not Spectrogram

**Root cause**: `_draw_motif_rectangle()` in `widgets_labels.py:215-262` only draws on `self.plot_container.get_current_plot()`.

**Why it happens**:
- Each plot has its own `label_items` list
- `plot_all_motifs()` clears and redraws on current plot only
- When you switch plots, the other plot has empty `label_items`

**Location**: `widgets_labels.py` lines 145-152, 215-262

**Potential fixes**:
1. Store label data centrally (not as plot items), redraw on both plots
2. Call `plot_all_motifs()` again when switching plots
3. Have PlotContainer manage labels and draw on both plots simultaneously

### Issue 2: Spectrogram Buffering Inefficient for Zoom

**Root cause**: Cache keyed by exact time ranges, no multi-resolution support.

**Why it happens**:
- `SpectrogramBuffer._get_cache_key()` creates keys from buffered time range
- Zooming out requires new computation even if zoomed-in data exists
- No mechanism to combine multiple cached ranges
- Computing full resolution spectrogram for zoomed-out views is wasteful

**Location**: `plots_spectrogram.py` lines 191-366 (SpectrogramBuffer class)

**Potential fixes**:
1. Pre-compute spectrogram for entire trial at low resolution
2. Implement tile-based caching (like map software)
3. Use downsampled spectrogram for zoomed-out views
4. Implement smarter cache that combines overlapping ranges

### Issue 3: LinePlot/Spectrogram Sync Drift

**Root cause**: Independent ViewBoxes, no sync on switch.

**Why it happens**:
- Each plot has its own `vb` (ViewBox)
- When switching, the hidden plot doesn't receive updates
- X-range can drift between the two plots

**Location**: `plot_container.py` switch methods (lines 41-73)

**Potential fix**: Copy x-range from previous plot when switching.

---

## Widget Component Summary

**LabelsWidget** (`widgets_labels.py`) - Motif labeling interface:
- Interactive motif creation (click-based boundary definition)
- Motif editing and deletion
- Color-coded motif visualization (LinearRegionItem for main, PlotDataItem for predictions)
- Export functionality for labeled data
- Connects to `plot_clicked` signal from both LinePlot and SpectrogramPlot

**PlotsWidget** (`widgets_plot.py`) - Plot configuration and control:
- Y-axis limits and window size settings for line plots and spectrograms
- Buffer settings for audio and spectrogram caching
- Autoscale and lock axes checkboxes
- Real-time parameter updates via `_on_edited()`

**NavigationWidget** (`widgets_navigation.py`) - Trial navigation:
- Trial selection combo box with prev/next buttons
- Playback FPS control
- Confidence-based trial filtering

**NapariVideoSync** (`video_sync.py`) - Napari-integrated video player:
- Uses napari-pyav for video loading (FastVideoReader)
- Segment playback with synchronized audio using audioio
- Frame-based seeking through napari's built-in dims controls
- Emits `frame_changed` signal connected to plot updates

### Data Flow

1. **Loading**: DataWidget loads NetCDF dataset → populates AppState → creates dynamic UI controls
2. **Selection**: User changes selections → AppState signals → widgets update plots/video
3. **Labeling**: User presses motif key → LabelsWidget activates → click twice on LinePlot → motif created
4. **Playback**: Right-click on motif → triggers video/audio playback at that time segment

### Widget Communication

Widgets communicate through:
- **AppState signals** for data changes
- **Direct references** set in MetaWidget (e.g., lineplot.set_plots_widget())
- **Cross-widget method calls** for complex interactions

### Key Design Patterns

- **Observer Pattern**: AppState emits signals, widgets react to changes
- **Centralized State**: All application state flows through ObservableAppState
- **Widget Composition**: MetaWidget composes and coordinates specialized widgets
- **Dynamic UI**: UI controls generated based on dataset structure

## Important Implementation Details

### AppState Persistence
- Only attributes in `_saveable_attributes` are persisted to YAML
- All `*_sel` attributes (user selections) are automatically saved
- Numeric values converted to Python types (not numpy) for YAML compatibility

### Audio/Video Synchronization
- Audio playback controlled by napari animation thread events
- Spectrogram data cached in buffers for performance
- Buffer cleared when switching trials or changing audio settings

### Keyboard Shortcuts
- Bound at napari viewer level in MetaWidget._bind_global_shortcuts()
- Motif keys (1-9, 0, Q, W, R, T) mapped to motif numbers 1-14
- Navigation keys (M/N for trials, arrows for plot navigation)

### Dataset Structure Requirements
- NetCDF format with time, trials dimensions
- Expected coordinates: cameras, mics, keypoints, individuals, features
- Video files matched by filename in dataset to video folder
