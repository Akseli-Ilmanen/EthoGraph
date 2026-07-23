

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

## Frame Rate Guidelines

Never hardcode frame rates (e.g., 30 fps) or sample rates (e.g., 44100 Hz) anywhere in the codebase — not even as fallbacks. Always use actual source metadata (e.g., video.fps, audio sample rate, ImageSeries.rate) or user-specified settings. If a rate is unknown, raise an error or return None — never silently default to a hardcoded value.

Never hardcode a 1-second fallback for time windows, trial durations, or trial start/stop timing. If timing metadata is missing, propagate unknown timing (or raise) instead of using 1.0s placeholders.

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

Two distinct source protocols:

**Rendering layer** (`io/plot_sources`) — plots load/cache viewport data through the `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`):
- `FileSource` — wraps a loader with `rate`/`__len__`/`__getitem__` (audioio, ephys, memmap)
- `XarraySource` — wraps `xr.Dataset`, time-sliced datasets from `get_data()`
- `PynappleSource` — lazy pynapple Tsd/TsdFrame access via `restrict()` (no xarray intermediate)
- `WindowedBuffer` — viewport-aware cache; loads wider than viewport, reloads on pan past buffer. Works with all sources.

**Navigation layer** (`io/time_model.py` + `io/time_sources.py`) — session-level time metadata via the `TimeSource` protocol; `SourceCollection` is the registry. It uses only `time_range` metadata — it never calls `get_data()`.

**Which buffer each plot uses:** AudioTrace/LinePlot/HeatmapPlot(features) → `WindowedBuffer`; Spectrogram → `SpectrogramBuffer` (caches FFT); EphysTrace → `EphysTraceBuffer` (multi-resolution pyramid); Heatmap(envelope) → inline cache.

### TrialTree: `trialtree.py`

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. Key API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`.

Session metadata (trial timing, media paths, FPS, offsets) is accessed via `app_state.nwb_alignment` (NWBAlignment): NWB sources read directly; non-NWB (.nc/.npz) use a `.ethograph/alignment.nwb` sidecar.

### State Management: `app_state.py`

**AppStateSpec** — type-checked spec (~40 variables). **ObservableAppState** — Qt signals auto-generated per variable (e.g. `current_frame_changed`), dynamic `*_sel` attributes for xarray selections, auto-saves to YAML. Key signals: `trial_changed`, `restrict_window_changed`, `labels_modified`, `verification_changed`.

Most things that determine the plot x-extent — `fixed_window_s`, `navigate_mode`, `slider_scope` — are `SCOPE_LOCAL` (per-dataset `local_settings.yaml`). A view mode picked for one dataset must never follow the user into the next: a "fixed 10 s window size" chosen for one dataset would otherwise clamp an unrelated dataset's window size too. `load_from_yaml` strips local-scope keys from the global file so stale entries written by older versions can't act as sticky defaults.

`xlim_mode` ("interval" i.e. follows slider scope, or "fixed" window) is the one exception: it's `SCOPE_GLOBAL`, a plain user preference (edited via the "X-limits:" combo in the Navigate section) that defaults to "interval" and is never inferred or overridden by the load path (drag & drop, template, wizard). Earlier versions guessed "fixed" for drag-dropped media and "interval" for multi-trial setups in `cover_page.py` — that per-load-path guessing was removed because it was a source of surprising behavior; the setting now simply persists as the user last left it.

### Time Model + Navigation: `time_model.py`

Canonical home for the core types (re-exported from `gui/plots_timeseriessource.py` for compat):
- **`TimeRange`** — immutable interval (`union`, `intersect`, `contains`, `overlaps`).
- **`TimeSource`** (Protocol) + adapters `XarrayTrialSource`, `PynappleSource` (in `time_sources.py`).
- **`SourceCollection`** — registry of TimeSources; provides `union_range`, `intersection_range`, `session_range`, `sources_at(t)`, trial bookmarks (`trial_range`, `find_trial`, `trial_offset`). Built in `data_loader.py`, stored as `app_state.source_collection`.
- **`RestrictionWindow`** — display window with mode `"session"|"trial"|"label"|"sequence"`.
- Restriction builders: `build_trial_window()`, `build_label_window()`, `build_sequence_window()`, `find_closest_trial()`.

`app_state.window_bounds` = current window's time_range (falls back to `trial_bounds`); all plots use it for x-limits. `app_state.session_time_range` = full session extent.

**Navigation modes** ("Time slider:"): Session (whole session, gaps navigable), Trial, Label (instances of one class across trials), Sequence (trials matching a label pattern, e.g. "1-2-3-5"). Sequence matching in `utils/sequences.py`.

### Unified Data Catalog + Loader: `catalog.py`

**`DataCatalog`** — declares features, dimensions (combos), streams. Built by `catalog_from_xarray()` / `catalog_from_pynapple()`. Features are auto-detected (all `data_vars` with a time dim, or all pynapple `Tsd`/`TsdFrame`/`TsdTensor`) — no `attrs["type"]` needed. The GUI builds combos from `catalog.combos`; the "Colors" combo lists all features, filtered to names containing "rgb" via a default-on checkbox.

**`DataLoader`** (Protocol) — backend-agnostic: `select(feature, selections, t0, t1) → PlotData`. Callers always pass `t0, t1` (absolute for pynapple, trial-relative for xarray). Follows `sel_valid`: selections may be overspecified; loaders ignore dims the feature lacks.

**`PlotData`** — source-agnostic dataclass: `time`, `data` `(T,)`/`(T,D)`, `dim_labels`, `title`, `ylabel`, `color_data`, `changepoints`. Consumed by `render_plot_data()` in `plots_lineplot.py`.

**Loaders:** `XarrayLoader` wraps `xr.Dataset`, delegates to `sel_valid()`, updated per trial via `update_ds()`. `PynappleLoader` is stateless — `restrict()`s to `t0,t1` and returns time in absolute session coordinates (same as the plot x-axis; never re-based to the requested window, so any viewport position renders correctly). NWB is loaded via pynapple; no dedicated NWB loader. The only direct NWB access is `_find_pose_series_names()`/`_discover_pose_keypoints()` (h5py scan for `PoseEstimationSeries` leaf names). `_compute_shared_column_dims()` groups TsdFrames with identical columns into one shared dim/combo.

### Alignment System: `nwb_alignment.py`

**Rule: `.nwb` sources are read/edited directly (`edit_nwb`); `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources.** Wizards produce sidecars for `.nc`/`.npz`/pynapple-folder projects.

**`NWBAlignment(path)`** reads any NWB for session metadata: `get_stream_rate(stream, device)`, `resolve_media_path(...)` (ImageSeries path → NWB-relative → fallback folder + filename), `stream_offset_for_trial(...)`.

**Resolution priority** (`_resolve_alignment` in `data_loader.py`): explicit `alignment_path` (from `app_state.nwb_file_path` or a drag-drop tmp alignment) wins; then NWB sources use source NWB trials → sidecar TSV; non-NWB use sidecar `.ethograph/alignment.nwb` → sidecar TSV.

**Single-trial loading = drag & drop (`cover_page.py`).** `classify_files()` buckets dropped paths by extension. `_collect_drop_details()` shows ONE popup surfacing only unresolvable values (npy sample rate; pose `source_software` for ambiguous `.h5`/`.csv`) plus, when a dropped video has an embedded audio track, an "extract audio" checkbox (default on unless separate audio was dropped; extraction writes a throwaway `.wav` into the drop dir that joins the normal `audio_mic-N` pipeline); everything else is auto-detected. Each drop gets a FRESH temp subdir via `_prepare_drop_dir()` (`%TEMP%/ethograph_tmp_alignment/{uuid}/`) holding both the tmp alignment and any video-motion `.nc` — mandatory, because throwaway `.nc` files share `local_settings` by parent dir, so a shared dir would leak a prior drop's stale layout. A drop resets `video_folder`/`audio_folder`/`pose_folder` to `None` before repopulating only the dropped modalities.

`align_media_per_trial` (`utils/nwb.py`) and `align_media_from_streams` (`nwb_alignment.py`) create ImageSeries for all streams from `stream_rates: dict[str, float]` — no hardcoded FPS.

### Data Loading: `data_loader.py`

Dispatch: `.npz`/folder (incl. NWB via pynapple) → `load_nap_data()` + `catalog_from_pynapple()` + `PynappleLoader`; `.nc` → `eto.open()` + `catalog_from_xarray()` + `XarrayLoader`. `load_dataset()` returns a `LoadResult` (`dt`, `all_labels_df`, `catalog`, `data_loader`, `source_collection`, metadata).

### Pose Rendering: `pose_render.py`

Two loading paths, unified into `PoseRenderData`: file-based (`load_pose_from_file()` via movement) and NWB-based (`load_pose_from_nwb_direct()`, lazy HDF5 slicing of `PoseEstimationSeries.data`/`.confidence`). `apply_confidence_filter()`/`apply_keypoint_filter()` act on the `data_not_nan` mask; `PoseDisplayManager` displays via a `shown` mask — filtering never recreates layers. NWB pose keys (`nwb_pose_keys`) map camera index → pose key in `_load_pose_for_camera()`.

### Skeleton Visualization: `ethograph/skeleton/` + pose_render

Reuses `ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses `xr.Dataset` into a napari Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. Only the config layer is ethograph-specific: `nwb_skeleton_to_config(nodes, edges)` converts an ndx-pose `Skeleton` to the standard config dict (default source, read by `_read_skeleton_config()` → `PoseRenderData.skeleton_config`). `PoseDisplayManager._display_skeleton_direct()` renders it, gated by "Show skeleton". Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. Edge confidence filtering is automatic — a NaN endpoint drops any edge touching it.

**Skeleton editor** (`dialog_skeleton_editor.py`): draw a skeleton on real pose data (frame slider + pyqtgraph canvas, drag between keypoints = edge, color categories per edge); `get_config()` → `skeleton_config_override`.

**Anchored shapes** (`skeleton/shapes.py`): square/triangle/circle templates that deform to follow the pose. Bind ≥2 control points to keypoints (`ShapeAnchorDialog`); per frame a transform is fit — 2 anchors → similarity (angle-preserving), 3+ → affine. Rendered as a napari Shapes layer via `_display_shapes_direct()` (frame index as first vertex coord = per-frame visibility). Shapes live under a `"shapes"` key in the skeleton config.

### Plot System

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`): time marker, x-range management, click handling, axes locking. `UnifiedPanelContainer` holds all panels, links x-axes, manages visibility.

### Panels Are Layout Instances — No Per-Plot-Type Toggles

Panels are instances created via the layout, never on/off toggles:
- The add-panel popup (`SourcePopup`, bottom bar "➕ Add panel" / Ctrl+N) drags a Media/Feature source onto the plot area (or Enter for the default spot); every panel has a ✕. Templates define layouts in the same instance terms.
- **Video motion (pixel-change) is a drop-time `(time, camera)` feature.** The cover page's "Compute video motion" checkbox runs `extract_video_motion()` per camera and writes a throwaway `.nc` stacked on a `camera` dim (`cam-1`, `cam-2`, …; NaN-padded; needs an `individuals` coord). That `.nc` becomes `nc_file_path`; catalog auto-detects `video_motion` with a `camera` combo (lineplot-per-camera + heatmap-all-cameras for free); media/pose still come from the tmp alignment via `nwb_file_path`.
- Initial visibility after load derives purely from data availability (`DataWidget._setup_panel_controls`): audio → trace + spectrogram, features → feature plot, neo/neurons → neural panels. `_create_default_audio_panels()` makes one trace+spectrogram pair per mic (pinned to its first channel) when several exist, else one global-following pair.
- There is NO saved per-panel yes/no state, no panel checkboxes, no Shift+A/S/E/F/C toggles. Never assume a boolean visibility toggle per plot type.
- **Duplicates are never prevented.** Dropping an already-shown source always creates another instance; the user removes extras via ✕. Never add "already shown → just reveal it" dedup logic.
- **Audio panels are instances.** Collections `plot_container.audio_trace_plots`/`spectrogram_plots`; created via `add_audio_panel("audiotrace"|"spectrogram", mic_name=None)`, removed via `remove_audio_panel(plot)`. `audio_trace_plot`/`spectrogram_plot` are compat properties (first or None). Each instance may pin a mic/channel: `plot.mic_name` is an `audio_source_map` key (`None` follows the global Mic combo). Popup lists ONE entry per mic (`Audio (Mic1)`); on drop a channel picker pins file+channel. Sidebar "Channel:" combo re-pins the active panel via `set_audio_panel_mic(plot, key)`. `audio_mic_channels` maps mic → ordered keys. Spectrogram settings apply to all instances. When removing an instance, stop its `ThrottleDebounce` (`plot._td.stop()`).
- **Extra camera views are instances, each in its OWN closable shell dock.** `VideoArea._extras` keyed by unique instance key (`"cam"`, `"cam (2)"`, …) with the real name on `view.camera_name` (always read via `getattr(view, "camera_name", key)`). Each extra gets its own `QDockWidget` (`CameraViewDock {key}`); only the primary lives in the "Video" dock. Closing defers `remove_extra(key)` → `camera_view_removed(view)`; `_on_camera_view_removed` (only when the LAST view of a camera closes) drops its pose layers and resets any combo naming it. `add_camera(..., duplicate=True)` always creates a new view; without it, existing views reload. `remove_camera(name)` removes all views of that camera.
- **Static images are camera-like media (`IMAGE_EXTENSIONS` in `io/validation.py`).** A dropped/browsed image is stored in `app_state.image_paths` (SCOPE_LOCAL) and listed as `Image (name.png)` + a permanent "Image — browse…" entry (`IMAGE_BROWSE` sentinel). Each drop creates a static view via `add_image_view()`. Static views are timeless (no playback), but the PRIMARY camera's pose/skeleton overlays and animates: `_display_pose_on_image()` + `CameraView.set_overlay_time(t)`, driven by `_on_time_marker_updated`. An image + pose drop with NO video works — the details dialog asks pose fps, the image is written as a static `video_cam-N` stream at that fps, session duration from the pose file; image-only drops are rejected ("no time axis").
- **All line plots are equal instances.** `plot_container.line_plots` is the only collection; `add_lineplot(feature=None)` / `remove_lineplot(plot)`. No `line_plot` attribute, no built-in/extra distinction. The heatmap is a fixed singleton; `set_feature_view("heatmap"|"lineplot")` shows/hides it. `get_current_plot()` = active feature plot, else first line plot, else heatmap.
- **One canonical feature list.** `catalog.feature_choices()` is the single source for the features combo, popup, and panel creation — never list features from raw `ds.data_vars` (it contains bookkeeping vars like `onset_s`). Any feature offered anywhere must be displayable everywhere.
- **Feature plots render ONLY from their own `panel_state`** (`PanelStateMixin` in `plots_base.py`, used by LinePlot AND HeatmapPlot; forked from globals on first render via `_ensure_panel_state()`). Never make a feature plot read `app_state.features_sel`/`get_selections()` for rendering — that recreates cross-panel coupling. The sidebar edits the active plot via `set_panel_control()`; the global `*_sel` mirror is only for shared consumers (labels, changepoints). Changing a feature must never auto-switch lineplot⇄heatmap; only the explicit "Feature plot type:" combo converts the active panel (carrying `panel_settings()` over, removing a converted line-plot instance). An "All" checkbox = absence of that dim from `panel_state["selections"]`.
- **Space plots are instances like line plots.** `DataWidget.space_plots` collection; `add_space_plot(feature=None, view_3d=None)` (each drop = a NEW dock, never reconfigures), removed by closing its shell dock (`SpacePlot.closed` → `remove_space_plot`). `DataWidget.space_plot` = active instance. Only the active instance's X/Y/Z + dim controls show in the sidebar's Space context. New space docks go in the shell's **top** area (never "left" — its title bar collided with the top bar). Persist per dataset in `panel_layout["space_plots"]` (each instance's `space_settings()`), recreated by `apply_space_layout_state()`, dock positions via objectNames `SpacePlotDock_{i}`. Shell dock arrangement lives ONLY per dataset (`panel_layout["shell_dock_state_b64"]` → local_settings.yaml, travels with templates); `window_state` → gui_settings.yaml holds only machine-local prefs (window geometry; right sidebar always starts visible), never layout. Applied via `shell.apply_dock_state_b64()` — **deferred to show when hidden**, since restoreState must run on a VISIBLE window or Qt evicts docks created in between; later docks are placed via `shell.restoreDockWidget()`.
- **Space plot controls are catalog-driven** (`plots_space.py`): Feature + "Space dim:" combo (default: a dim named `space`, else the first) + X/Y/Z combos of that dim's values (default x/y/z, else first three) + one combo per remaining dim + Color combo ("Labels" = label-highlight; any feature = per-point coloring, disables highlight). Renders purely from its own combos — never reads global `*_sel`.
- **Space reference geometry** comes solely from the **geometry library** at `~/.ethograph/geometries/*.yaml` (each file has a `references:` list — name/vertices/edges/color). **One file = one selectable geometry, keyed by filename stem**; all of a file's references draw together. `app_state.space_library_geometry` (SCOPE_LOCAL) holds that stem, set via the Space "Library geometry:" combo or a default in gui/local settings. `load_library_geometries()`/`_load_references()` in `plots_space.py`. The library is seeded on first GUI launch (`ensure_geometry_library()`) by copying `ethograph/geometries/*.yaml` (e.g. `moll2025_geometry.yaml` → `space_library_geometry: moll2025_geometry`); it never re-seeds an existing dir, so user deletions stick. Templates set their default via `DATASETS[...]["library_geometry"]`.
- **Panels are dock widgets (pynaviz-style).** `UnifiedPanelContainer` hosts a nested `QMainWindow` (`_dock_host`, nesting enabled); every panel is a `QDockWidget` with a slim title bar (name + ⠿ move menu + ✕). Arrange freely (side by side, stacked, tabbed, floated). Default: vertical stack in `_PANEL_ORDER`, line plots at the bottom.
- **Layout persistence is automatic; NO JSON layout files exist.** `app_state.panel_layout` (open panels + each `panel_settings()` + a `dock_state_b64` blob) is SCOPE_LOCAL → dataset's `local_settings.yaml`; `app_state.window_state` (outer geometry) is global → `gui_settings.yaml`. Both refreshed by `MetaWidget._snapshot_layouts` (on 10s auto-save + close) and re-applied automatically. No Save/Load layout actions.
- **Templates ship layouts via `local_settings.yaml`.** Selecting a template downloads data + optionally the release asset `local_settings.yaml` into `.ethograph/` (`download_template_local_settings()`, never overwrites a local file). Never special-case a `dataset_key` in GUI code — put per-dataset settings in `DATASETS` metadata.

### Video Sync: `video_sync.py`

`NapariVideoSync`: `frame_to_time(frame)` / `time_to_frame(t)` with `time_offset`. All widgets use these — never raw `frame / fps`.

### Labels

**Storage:** TSV (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, trial, human_verified, changepoint_corrected, prediction_source, n_samples` (`n_samples` = per-trial count for dense conversion). Label names in `mapping.txt`. Legacy `.nc` embedded labels auto-migrate to TSV on first load.

**Modules:** `intervals.py` (interval ops + mapping loaders + `find_blocks`), `ml.py` (dense↔interval + `stitch_gaps`/`purge_small_blocks`/`fix_endings`).

**In-memory:** `app_state._all_labels_df` (all trials), `app_state.label_intervals` (current trial). Per-trial metadata as columns, not a dict.

**Per-plot-type rendering:** `app_state.label_overlay_modes` maps plot type key (`lineplot`, `audio`, `spectrogram`, `heatmap`, `ephys`, `neo`) → `"full"|"bottom"|"none"`, applied to every instance of that type — no visibility hierarchy. Defaults in `DEFAULT_LABEL_OVERLAY_MODES` (app_constants.py); edited via "Show labels per plot type" (`LabelsPerPlotDialog`).

**Labels on new panels:** Any path that creates/shows a panel must end with `plot_container.schedule_labels_redraw()` — deferred (coalesced, next tick) because it must run AFTER content render, or "bottom"-strip rectangles (positioned from the y viewRange) are invisible. Never emit `labels_redraw_needed` synchronously from a panel-creation path.

**Predictions:** Per-trial `.npy`/`.pickle`. Shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`. Confidence stays in memory, never stored. Dotted confidence curve on by default, gated per feature plot by "Predictions: Show predictions" (`show_predictions` in `panel_state`).

**Top-bar File menu:** "Import labels…" / "Import predictions…" / "Export labels…" each borrow their own I/O sub-panel (`IOWidget.restore_subpanel()` returns it on close). No "Load data…" entry — data loading happens only on the cover page.

**Crowsetta interop:** `EthographSeq` registered at import; export adapter converts int→string labels via mapping. Internal storage stays integer-based.

### Widget Orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator (trial changes, plot updates, video/audio loading). Signal flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → updates everything.

**Context-sensitive right sidebar:** which setting sections the "Data" section shows per plot type is defined solely by `_CONTEXT_MAP` (+ `_CONTEXT_TITLE`) in `gui/right_context.py`; `RightContextPanel.set_context()` shows/hides the sections. `MetaWidget._build_context_panel()` borrows the actual group widgets from `DataWidget`/`PlotSettingsWidget`/`NavigationWidget` into section keys, and `MetaWidget._on_active_panel()` swaps the context on panel click (only for kinds in `_CONTEXT_KINDS`; raster keeps the current sidebar). Contexts: `ephys` (Phy viewer) → `phy` section = `EphysWidget.traceview_panel` (channel/gain/pyramid/probe/cluster table); `neo` (Neo viewer) → `neocontrols` (`PlotSettingsWidget.neo_controls_group`: per-panel gain + auto-gain + channel spacing, editing the active Neo panel via `set_active_neo_plot`) — channels are chosen at drop time. Neither ephys nor neo includes the `shared` (autoscale / lock-axes) group — those don't apply to the trace views. The top-bar **Neural** menu keeps only "Open interactive PSTH…" (`EphysWidget._open_psth`) and "Firing rates…" (pops `EphysWidget.firing_rate_panel`) — the Phy TraceView popup is gone.

**Neo trace panels are dynamic instances (one per stream/modality).** A Neo file exposes multiple `signal_streams` (EMG, accelerometer, amplifier…) via `EphysData.stream_info`; each is a modality with its own channel set. `plots_container._DYNAMIC_PANEL_SPECS["neo"]` (`cls=EphysTracePlot`, `group="neo"`) makes `add_panel("neo", stream_name=…, channels=…)` create a new instance per modality (collection `plot_container.neo_trace_plots`; there is NO `neo_trace_plot` singleton). `add_panel` calls back into `DataWidget.configure_neo_plot(plot)` (needs `ephys_source_map` + `load_ephys`) to load the stream, set the `FileSource`, and restrict to `plot.neo_channels` via `set_custom_channel_set` (None = all). The add-panel popup lists one "Neo (Ephys)" source per stream (`DataWidget.neo_stream_names()`, excluding the Phy/kilosort stream); dropping one opens `ChannelSelectDialog` (multi-select, default all) → `DataWidget.add_neo_panel`. `on_kilosort_loaded()` drops any Neo panel that duplicates the Phy stream; `refresh_neo_panels()` re-renders on trial change. Layout persistence stores `{type: "neo", stream_name, channels}` per instance.

**Neo AND Phy trace panels are NOT auto-loaded (they are heavy).** On dataset/Kilosort load the code only resolves the Phy loader stream (`_ensure_default_ephys_stream`) and pre-wires the Phy source (`configure_ephys_trace_plot`, hidden). Both panels are added on demand from the "➕ Add panel" popup: the **Ephys** header offers "Ephys (Phy-like viewer)" (kind `"phy"`, shown when `MetaWidget._phy_available()` / `EphysWidget.has_phy_trace()` — a raw `.bin`/`.dat` Kilosort loader resolves) and one "Neo (…)" per stream. `MetaWidget._add_phy_panel()` re-shows the Phy singleton (`set_neural_panel_mode("trace")` + `configure_ephys_trace_plot`), so closing it via ✕ and re-adding works. `refresh_source_popup()` repopulates the popup (called on each open and after Kilosort load). Saved layouts still restore whatever panels were open.

### Neuron Loading (Kilosort / Pynapple)

Two paths, both → `nap.TsGroup` + cluster table: **Kilosort folder** (`.npy` + `cluster_info.tsv` + raw `.dat`; full features — probe map, waveforms, raster) and **Pynapple file** (`.npz`/`.nwb` via `nap.load_file()`, `data["units"]`; raster-only). State: `app_state.neurons_path`, `app_state.has_neurons`; `EphysWidget._neurons_source` is `"kilosort"`/`"pynapple"`.

### Kilosort Channel Mapping

Two index spaces: **site index** (0..n_sites-1, indexes `channel_positions.npy`) vs **hardware channel** (`channel_map.npy`, can exceed n_sites; `cluster_info.tsv` `ch` column). Always index `channel_positions` by site index.

### Changepoint Correction

Bridge pattern: intervals→dense→correct→intervals. Kinematic CPs stored as dense `int8`; audio CPs as onset/offset float pairs.

---

## Dataset Structure

- NetCDF with trials. Time coords: `time`, `time_aux`, etc. (any containing 'time')
- All `data_vars` with a time dimension are features (no `attrs["type"]` required). Changepoints still use `attrs["type"] = "changepoints"`.
- Color variables are identified by name: any feature with "rgb" in the name (case-insensitive) is offered in the Colors combo. No `attrs["type"] = "colors"` needed.
- Media/session metadata: for `.nwb` sources, read from the source NWB directly (no sidecar created); for non-NWB sources, from `.ethograph/alignment.nwb`
- Labels: stored in `_labels.tsv` (not inside `.nc`). Legacy `.nc` labels auto-migrate on first load.
