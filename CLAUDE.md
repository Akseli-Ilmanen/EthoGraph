# CLAUDE.md

## Working style

- Implement, don't propose. Don't wait for "yes go".
- Write idiomatic, typed Python (mypy/ruff clean). Prefer clean design patterns over cleverness.
- Imports: sorted stdlib → third-party → local, explicit (never `from x import *`), never inside a function (only exception: circular imports).
- Comments only where the logic isn't obvious — which should be rare. **Never remove human-authored comments** (TODO/FIXME/NOTE/explanatory); only remove comments you added yourself.
- **Fail fast**: bugs (wrong type, missing key, unexpected `None`) crash; runtime conditions (missing file, bad user input) are handled. Never `try/except` into a silent `None`. Catch broad exceptions only at the outermost GUI boundary.
- Test/debug scripts live in `tests/`, never the project root. Prefix ad-hoc debug scripts `_test_` so pytest skips them.
- Docs/docstrings: don't name individuals (Poppy, Freddy, Ivy).
- After major design changes, update this file.
- Claude Code may change any file in this repo.

## Project Overview

EthoGraph is a GUI for labelling start/stop times of animal movements, paired with a workflow using action-segmentation transformers to predict segments. It loads NetCDF/NWB/pynapple datasets and displays synchronized video/audio/ephys.

```python
import ethograph as eto
dt = eto.open("data.nc"); dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da); data, filt = eto.sel_valid(da, kwargs)
```

## Hard Rules

**Never hardcode rates.** No frame rates (30 fps), sample rates (44100 Hz), or 1-second fallbacks for windows/trial durations — not even as fallbacks. Use source metadata (`video.fps`, `ImageSeries.rate`, audio rate) or user settings; if unknown, raise or return `None`.

**Never hardcode a device.** `resolve_device()` picks CUDA → MPS → CPU.

**Never special-case a `dataset_key` in GUI code** — put per-dataset settings in `DATASETS` metadata.

## File Structure

```
ethograph/__init__.py         # Public API

ethograph/gui/
    app_state.py              # AppStateSpec + ObservableAppState
    data_loader.py            # Dataset dispatch (.nc / .nwb / pynapple / DANDI)
    data_sources.py, plot_sources
    pose_render.py            # Pose loading (NWB + movement), PoseDisplayManager
    pose_annotate.py          # KeypointStore + movement export
    pose_fill.py              # Fill backends: Spline, OpticalFlow (+ private CoTracker3)
    pose_refine.py            # PosePAL: CoTracker3 + test-time refinement, GPU only
    pose_detect.py            # Detect stage: AprilTag tag36h11 (pupil-apriltags), assignment learning
    pose_detect_preview.py    # PreviewPanel — what the detector sees on this frame
    pose_tagsheet.py          # Tag sheet: layout maths + vector PDF/SVG/printer output
    dialog_tag_sheet.py       # Print tag sheet… (cover page pre-recording tools + Tools menu)
    pose_edit_mixin.py        # KeypointLabelMode (canvas anchor editing)
    dialog_pose_labelling.py, dialog_skeleton_editor.py
    plots_base.py             # BasePlot, PanelStateMixin
    plots_container.py        # UnifiedPanelContainer
    plots_{audiotrace,spectrogram,ephystrace,lineplot,heatmap,raster,space}.py
    plots_timeseriessource.py # Re-exports io/time_model.py (compat)
    label_drawing_mixin.py, video_sync.py, video_manager.py
    widgets_meta.py           # MetaWidget — creates + wires everything
    widgets_data.py           # DataWidget — central orchestrator
    widgets_{io,labels,navigation,changepoints,ephys,plot_settings,transform}.py
    widget_intervals.py, right_context.py, main_window.py, top_bar.py, cover_page.py
    table_filter.py           # Funnel-header column filters (ephys + keypoint tables)
    nwb_alignment.py, shortcuts.py

ethograph/labels/
    intervals.py              # Interval ops, mapping loaders, find_blocks
    ml.py                     # Dense↔interval, stitch_gaps/purge_small_blocks/fix_endings
    tsv_store.py, predictions.py, crowsetta_format.py, converters.py, export.py

ethograph/io/
    catalog.py                # DataCatalog + XarrayLoader/PynappleLoader, pose discovery
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TimeSource, SourceCollection
    time_sources.py           # XarrayTrialSource, PynappleSource
    dataset.py, validation.py, pynapple.py, metadata_table.py, ephys_loader.py

ethograph/utils/              # io.py, xr_utils.py (sel_valid, get_time_coord), sequences.py
ethograph/skeleton/           # PrecomputedRenderer, SkeletonState, config.py, shapes.py
```

## Architecture

### Two data-source layers

**Rendering** (`io/plot_sources`) — `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`): `FileSource` (loader with `rate`/`__len__`/`__getitem__`), `XarraySource`, `PynappleSource` (lazy `restrict()`, no xarray intermediate). `WindowedBuffer` caches wider than the viewport. Per-plot buffers: AudioTrace/LinePlot/Heatmap(features) → `WindowedBuffer`; Spectrogram → `SpectrogramBuffer` (caches FFT); EphysTrace → `EphysTraceBuffer` (pyramid); Heatmap(envelope) → inline.

**Navigation** (`io/time_model.py` + `time_sources.py`) — session-level time metadata via `TimeSource`; `SourceCollection` is the registry. Uses only `time_range`, **never** calls `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`. Session metadata (trial timing, media paths, FPS, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec (~40 vars); `ObservableAppState` auto-generates a Qt signal per variable (`current_frame_changed`, `trial_changed`, `restrict_window_changed`, `labels_modified`, `verification_changed`), exposes dynamic `*_sel` attributes, and auto-saves to YAML.

Anything defining the plot x-extent (`fixed_window_s`, `navigate_mode`, `slider_scope`) is `SCOPE_LOCAL` (per-dataset `local_settings.yaml`) — a view mode picked for one dataset must never follow the user to the next. `load_from_yaml` strips local-scope keys from the global file.

`xlim_mode` ("interval" | "fixed") is the exception: `SCOPE_GLOBAL`, a plain preference set only via the "X-limits:" combo. **Never infer it from the load path** (drag & drop, template, wizard).

### Time model + navigation

`TimeRange` (immutable: `union`/`intersect`/`contains`/`overlaps`), `TimeSource` (Protocol), `SourceCollection` (`union_range`, `intersection_range`, `session_range`, `sources_at(t)`, `trial_range`, `find_trial`, `trial_offset`; built in `data_loader.py` → `app_state.source_collection`), `RestrictionWindow` (mode `"session"|"trial"|"label"|"sequence"`), builders `build_trial_window()` / `build_label_window()` / `build_sequence_window()` / `find_closest_trial()`.

`app_state.window_bounds` (falls back to `trial_bounds`) drives every plot's x-limits; `app_state.session_time_range` is the full extent. Navigation modes ("Time slider:"): Session / Trial / Label (one class across trials) / Sequence (label pattern, e.g. `1-2-3-5`; matching in `utils/sequences.py`).

### Catalog + loader: `catalog.py`

`DataCatalog` declares features, dimensions (combos) and streams; built by `catalog_from_xarray()` / `catalog_from_pynapple()`. Features are **auto-detected** (any `data_var` with a time dim; any pynapple `Tsd`/`TsdFrame`/`TsdTensor`) — no `attrs["type"]`. The "Colors" combo lists all features, filtered to names containing "rgb" via a default-on checkbox.

`DataLoader` (Protocol): `select(feature, selections, t0, t1) → PlotData`. Callers always pass `t0, t1` (absolute for pynapple, trial-relative for xarray). Selections may be overspecified — loaders ignore dims a feature lacks, like `sel_valid`.

`PlotData`: `time`, `data` `(T,)`/`(T,D)`, `dim_labels`, `title`, `ylabel`, `color_data`, `changepoints` → `render_plot_data()` in `plots_lineplot.py`.

`XarrayLoader` wraps `xr.Dataset` (`update_ds()` per trial). `PynappleLoader` is stateless and returns **absolute session time**, never re-based to the requested window. NWB loads via pynapple — the only direct NWB access is `_find_pose_series_names()` / `_discover_pose_keypoints()` (h5py scan). `_compute_shared_column_dims()` groups identical-column TsdFrames into one combo.

### Alignment: `nwb_alignment.py`

**`.nwb` sources are read/edited directly (`edit_nwb`); `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources** (`.nc`/`.npz`/pynapple folders).

`NWBAlignment(path)`: `get_stream_rate(stream, device)`, `resolve_media_path(...)` (ImageSeries path → NWB-relative → fallback folder + filename), `stream_offset_for_trial(...)`. Resolution priority (`_resolve_alignment`): explicit `alignment_path` wins → NWB sources use source NWB trials → sidecar TSV; non-NWB use sidecar `alignment.nwb` → sidecar TSV. `align_media_per_trial` / `align_media_from_streams` build ImageSeries from `stream_rates: dict[str, float]`.

**Drag & drop = single-trial loading** (`cover_page.py`). `classify_files()` buckets by extension; `_collect_drop_details()` shows ONE popup for unresolvable values only (npy sample rate; `source_software` for ambiguous `.h5`/`.csv`) plus an "extract audio" checkbox when the video has an embedded track (throwaway `.wav` joining the normal `audio_mic-N` pipeline). Each drop gets a **fresh** temp subdir (`_prepare_drop_dir()`, `%TEMP%/ethograph_tmp_alignment/{uuid}/`) — mandatory: throwaway `.nc` files share `local_settings` by parent dir, so a shared dir leaks the previous drop's layout. A drop resets `video_folder`/`audio_folder`/`pose_folder` to `None` first.

### Pose rendering

Two paths unified into `PoseRenderData`: `load_pose_from_file()` (movement) and `load_pose_from_nwb_direct()` (lazy HDF5 slicing). `apply_confidence_filter()` / `apply_keypoint_filter()` act on the `data_not_nan` mask; `PoseDisplayManager` displays via a `shown` mask — **filtering never recreates layers**. `nwb_pose_keys` maps camera index → pose key.

**Colour encodes ONE axis, chosen by the user** (SLEAP's model): `app_state.pose_color_by` ∈ `{"keypoint", "individual"}` (constants in `pose_convert.py`, SCOPE_GLOBAL, "Colour by" in the pose sidebar + the labelling dialog). Keypoint mode = one hue per body part shared across individuals; individual mode = one hue per animal shared across its keypoints. **Never re-derive it from the number of individuals** — that was the old auto-rule, and it silently dropped keypoint identity as soon as a second animal appeared. Text labels carry the *other* axis (`text_prop`), falling back only when that axis has one value. The same setting drives the labelling canvas, so both surfaces always agree; **there is no per-individual marker shape** (the shape alphabet was removed — colour is the only identity channel).

### Keypoint labelling + fill

Label a few frames by clicking the video, let a point tracker fill the rest — single video, 2D, one or more individuals; no training, no GPU for the default backend. **Tools ▸ Keypoint labelling…** or the Pose sidebar → `DataWidget.open_keypoint_labelling()`, one non-modal dialog. With markers, an optional **Detect** stage (`pose_detect.py`) supplies the same kind of evidence a click does, ahead of the same fill backends.

Scope: **one camera, one trial** — the dialog follows `app_state.video_path` and is keyed by frame index on that video's grid; there is no trial/camera axis, so `TrialTree` datasets are not supported here.

**The design rules for this whole area live in `docs/source/advanced/keypoint_labelling/`** — store/provenance model, fill backends, PosePAL, detection, tag printing, canvas editing, keys, dialog, points table, frame suggestion, export. **Read those pages before editing any of `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`**; the rules there are binding exactly like the ones here.

### Skeleton visualization

`ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses Dataset into a Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. Only the config layer is ethograph-specific: `nwb_skeleton_to_config(nodes, edges)` converts an ndx-pose `Skeleton` (read by `_read_skeleton_config()` → `PoseRenderData.skeleton_config`), rendered by `_display_skeleton_direct()` behind "Show skeleton". Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. A NaN endpoint automatically drops any edge touching it.

`dialog_skeleton_editor.py` draws a skeleton on real pose data → `skeleton_config_override`. **Anchored shapes** (`skeleton/shapes.py`): square/triangle/circle templates deforming to follow the pose; bind ≥2 control points (`ShapeAnchorDialog`) → per-frame transform (2 anchors = similarity, 3+ = affine), rendered as a Shapes layer with the frame index as first vertex coord. Shapes live under `"shapes"` in the skeleton config.

### Panels are layout instances — no per-plot-type toggles

Panels are instances created via the layout, never on/off toggles. **There is NO saved per-panel yes/no state, no panel checkboxes, no Shift+A/S/E/F/C toggles.** **Duplicates are never prevented**: dropping an already-shown source always creates another instance (removed via ✕) — never add "already shown → just reveal it" dedup.

- Created via the add-panel popup (`SourcePopup`, bottom bar ➕ / Ctrl+N): drag a source onto the plot area, or Enter for the default spot. Every panel has a ✕. Templates define layouts in the same terms.
- Initial visibility derives purely from data availability (`DataWidget._setup_panel_controls`): audio → trace + spectrogram, features → feature plot, neo/neurons → neural panels.
- **Panels are dock widgets** (pynaviz-style): `UnifiedPanelContainer` hosts a nested `QMainWindow` with nesting enabled; each panel is a `QDockWidget` with a slim title bar. Default is a vertical stack in `_PANEL_ORDER`, line plots at the bottom.
- **The media/plots separator drags across the whole window (~10/90 either way).** Qt clamps a separator drag at the minimum size of the widgets on each side, so every minimum in the split is deliberately a sliver: `PLOT_CONTAINER_MIN_HEIGHT`, `PANEL_MIN_HEIGHT` (every panel widget in `_make_dock`, plus the nested dock host) and `MEDIA_VIEW_MIN_WIDTH`/`_HEIGHT`. **Never raise a minimum to get a default proportion** — defaults come from sizeHints and `resizeDocks`. Covered by `tests/test_integration/test_split_ratio.py`.
- **Layout persistence is automatic; NO JSON layout files exist.** `app_state.panel_layout` (open panels + `panel_settings()` + `dock_state_b64`) is SCOPE_LOCAL → dataset `local_settings.yaml`; `app_state.window_state` (outer geometry only) → `gui_settings.yaml`. Refreshed by `MetaWidget._snapshot_layouts` (10s auto-save + close). No Save/Load layout actions. Applied via `shell.apply_dock_state_b64()`, **deferred to show when hidden** — `restoreState` must run on a visible window or Qt evicts docks created in between; later docks go through `shell.restoreDockWidget()`.
- **Audio panels are instances**: `audio_trace_plots`/`spectrogram_plots`, `add_audio_panel("audiotrace"|"spectrogram", mic_name=None)` / `remove_audio_panel(plot)`; `audio_trace_plot`/`spectrogram_plot` are compat properties (first or None). `plot.mic_name` pins an `audio_source_map` key (`None` follows the global Mic combo); a channel picker pins file+channel on drop. `_create_default_audio_panels()` makes one pair per mic. Spectrogram settings apply to all instances. When removing, stop `plot._td`.
- **Extra camera views are instances, each in its OWN closable dock** (`CameraViewDock {key}`); only the primary lives in the `VideoDock`. `VideoArea._extras` is keyed by instance key (`"cam"`, `"cam (2)"`…) with the real name on `view.camera_name` (always read via `getattr(view, "camera_name", key)`). Closing defers `remove_extra(key)` → `camera_view_removed`; pose layers/combos reset only when the LAST view of a camera closes. `add_camera(..., duplicate=True)` always creates a new view. **Anything that must follow the trial iterates the live views, never `_extra_camera_combos`** — the combos are hidden vestigial state that a popup-dropped view never enters, so combo-driven refresh froze extras on the trial that was open when they were created (right pose, stale video, because `PoseDisplayManager.update_pose()` already iterated views). `VideoManager.refresh_extra_videos()` reloads every extra from `extra_widgets` on trial change; the combos only still *create* views, for the saved `extra_cameras` restore path. Covered by `tests/test_integration/test_camera_trial_follow.py`.
- **The primary is a camera view like any other — never a generic "Video" panel.** `update_video()` stamps `primary_view.camera_name`, and every panel (primary + extras) is titled `camera_dock_title()` → `cam-1 (front.mp4)` via `VideoManager.refresh_view_title()`; only the objectNames (`VideoDock`, `CameraViewDock {key}`) stay fixed, because they key layout persistence. The primary also takes its **fps from the probe** and its **offset from its own `stream_offset_for_trial`**, exactly like an extra — reading either from `trial_alignment` desynced it whenever the primary camera changed. `app_state.video_fps` prefers the loaded view's fps over the stored stream rate for the same reason (every time↔frame conversion goes through it). Switching the primary camera **must** rebuild `trial_alignment` first (`DataWidget._on_primary_camera_changed`). Covered by `tests/test_integration/test_camera_panel_identity.py`.
- **Static images are camera-like media** (`IMAGE_EXTENSIONS` in `io/validation.py`): `app_state.image_paths` (SCOPE_LOCAL), listed as `Image (name.png)` plus a permanent `IMAGE_BROWSE` entry; each drop creates a view via `add_image_view()`. Static views are timeless, but the primary camera's pose/skeleton overlay animates via `_display_pose_on_image()` + `CameraView.set_overlay_time(t)`. Image + pose with no video works (dialog asks pose fps, image written as a static `video_cam-N` stream); image-only drops are rejected ("no time axis").
- **Video motion is a drop-time `(time, camera)` feature**: the cover page's checkbox runs `extract_video_motion()` per camera into a throwaway `.nc` stacked on a `camera` dim (`cam-1`…, NaN-padded, needs an `individuals` coord) which becomes `nc_file_path`; the catalog then gives lineplot-per-camera and heatmap-all-cameras for free. Media/pose still come from the tmp alignment via `nwb_file_path`.
- **All line plots are equal instances**: `plot_container.line_plots`, `add_lineplot(feature=None)` / `remove_lineplot(plot)`. No `line_plot` attribute, no built-in/extra distinction. The heatmap is a fixed singleton toggled by `set_feature_view("heatmap"|"lineplot")`.
- **One canonical feature list**: `catalog.feature_choices()` feeds the combo, popup and panel creation — never list features from raw `ds.data_vars` (it holds bookkeeping vars like `onset_s`). Anything offered anywhere must be displayable everywhere.
- **Feature plots render ONLY from their own `panel_state`** (`PanelStateMixin`, forked from globals via `_ensure_panel_state()`). **Never make a feature plot read `app_state.features_sel`/`get_selections()` for rendering** — that recreates cross-panel coupling. The sidebar edits the active plot via `set_panel_control()`; the global `*_sel` mirror serves only shared consumers (labels, changepoints). Changing a feature must never auto-switch lineplot⇄heatmap — only the "Feature plot type:" combo converts a panel. An "All" checkbox = absence of that dim from `panel_state["selections"]`.
- **Space plots are instances**: `DataWidget.space_plots`, `add_space_plot(feature=None, view_3d=None)` (each drop = a NEW dock, never reconfigures), removed by closing its dock. New space docks go in the shell's **top** area (never "left" — its title bar collides with the top bar). Persisted per dataset in `panel_layout["space_plots"]`, dock positions via objectNames `SpacePlotDock_{i}`. Controls are catalog-driven (Feature + "Space dim:" + X/Y/Z + one combo per remaining dim + Color) and render purely from their own combos — never global `*_sel`.
- **Space reference geometry** comes solely from `~/.ethograph/geometries/*.yaml` (each file has a `references:` list; **one file = one selectable geometry keyed by filename stem**, all its references drawn together). `app_state.space_library_geometry` (SCOPE_LOCAL) holds the stem. Seeded on first launch by `ensure_geometry_library()` copying `ethograph/geometries/*.yaml`; it never re-seeds an existing dir, so user deletions stick. Templates set a default via `DATASETS[...]["library_geometry"]`.
- **Templates ship layouts via `local_settings.yaml`**: selecting one downloads data plus the optional release asset (`download_template_local_settings()`, never overwrites a local file).

### Plot system

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`): time marker, x-range management, click handling, axes locking. `UnifiedPanelContainer` holds all panels and links x-axes.

### Video sync

`NapariVideoSync.frame_to_time(frame)` / `time_to_frame(t)` apply `time_offset`. All widgets use these — **never raw `frame / fps`**.

**Reloading the same file must never rebuild the `PlotVideo`.** `CameraView.set_video()` reuses the loaded plot when `_video_path` is unchanged (trial change, camera re-apply) and only re-clips `start_frame`/`end_frame`/`time_offset`; only a genuinely different decode path (proxy swap, other camera) goes through `clear()`. Each `PlotVideo` owns a **spawned** pynaviz worker that must re-import `av`/`pygfx`/`pynapple` (~1.5–2 s) before attaching to the parent's shared memory, while `PlotVideo.close()` waits only `join(timeout=2)` before dropping the parent's handle — which destroys the mapping on Windows — so a close-then-create cycle kills the new worker with `FileNotFoundError: [WinError 2] … 'wnsm_…'`. `update_video()` therefore drops only the `VideoSync` (`_teardown_primary_sync`); `_cleanup_primary_video` (= teardown + `view.clear()`) is for genuinely unloading the video, and any aborted setup must clear the view itself. On the reuse path the renderer handlers and the `_update_extra_objects` overlay hook are **not** re-registered (they survive with the plot); `_detach_load_state()` drops the per-load state — labelling mode and pose overlay — that `clear()` would have. Covered by `tests/test_integration/test_video_reload.py`.

### Labels

**Storage:** TSV (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, trial, human_verified, changepoint_corrected, prediction_source, n_samples`. Label names in `mapping.txt`. Legacy `.nc` embedded labels auto-migrate on first load. In memory: `app_state._all_labels_df` (all trials) and `app_state.label_intervals` (current trial), per-trial metadata as columns rather than a dict.

**Per-plot-type rendering:** `app_state.label_overlay_modes` maps plot type key (`lineplot`, `audio`, `spectrogram`, `heatmap`, `ephys`, `neo`) → `"full"|"bottom"|"none"`, applied to every instance of that type — no visibility hierarchy. Defaults in `DEFAULT_LABEL_OVERLAY_MODES`; edited via `LabelsPerPlotDialog`.

**Labels on new panels:** any path creating/showing a panel must end with `plot_container.schedule_labels_redraw()` — deferred because it must run AFTER content render, or "bottom"-strip rectangles (positioned from the y viewRange) are invisible. **Never emit `labels_redraw_needed` synchronously from a panel-creation path.**

**Predictions:** per-trial `.npy`/`.pickle`, shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`. Confidence stays in memory, never stored. Dotted confidence curve on by default, gated per feature plot by `show_predictions` in `panel_state`.

**Crowsetta:** `EthographSeq` registered at import; the export adapter converts int→string via the mapping. Internal storage stays integer-based.

**Top-bar File menu:** "Import labels…" / "Import predictions…" / "Export labels…" each borrow their own I/O sub-panel (`IOWidget.restore_subpanel()`). No "Load data…" — data loading happens only on the cover page.

### Widget orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator. Flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → everything else.

**Context-sensitive right sidebar:** the sections shown per plot type are defined solely by `_CONTEXT_MAP` (+ `_CONTEXT_TITLE`) in `gui/right_context.py`. `MetaWidget._build_context_panel()` borrows the real group widgets from `DataWidget`/`PlotSettingsWidget`/`NavigationWidget`; `_on_active_panel()` swaps context on panel click (only for `_CONTEXT_KINDS`; raster keeps the current sidebar). `ephys` → `phy` section (`EphysWidget.traceview_panel`); `neo` → `neocontrols` (`PlotSettingsWidget.neo_controls_group`, editing the active panel via `set_active_neo_plot`). Neither includes the `shared` autoscale/lock-axes group. The **Neural** menu keeps only "Open interactive PSTH…" and "Firing rates…".

**Neo trace panels are dynamic instances, one per stream/modality.** `_DYNAMIC_PANEL_SPECS["neo"]` makes `add_panel("neo", stream_name=…, channels=…)` create an instance per modality (collection `plot_container.neo_trace_plots` — there is NO `neo_trace_plot` singleton). `add_panel` calls back into `DataWidget.configure_neo_plot(plot)` to load the stream and restrict to `plot.neo_channels` via `set_custom_channel_set` (`None` = all). The popup lists one "Neo (…)" per stream (excluding the Phy/kilosort stream); dropping opens `ChannelSelectDialog`. `on_kilosort_loaded()` drops Neo panels duplicating the Phy stream; `refresh_neo_panels()` re-renders on trial change.

**Neo and Phy trace panels are NOT auto-loaded (heavy).** Load only resolves the Phy stream (`_ensure_default_ephys_stream`) and pre-wires the source (`configure_ephys_trace_plot`, hidden). Both are added on demand from the popup — "Ephys (Phy-like viewer)" (kind `"phy"`, shown when `MetaWidget._phy_available()`) and one "Neo (…)" per stream. `_add_phy_panel()` re-shows the Phy singleton so ✕-then-re-add works. `refresh_source_popup()` repopulates on each open and after Kilosort load.

### Neurons + ephys

Two paths → `nap.TsGroup` + cluster table: **Kilosort folder** (`.npy` + `cluster_info.tsv` + raw `.dat`; full features — probe map, waveforms, raster) and **Pynapple file** (`.npz`/`.nwb`, `data["units"]`; raster only). State: `app_state.neurons_path`, `app_state.has_neurons`, `EphysWidget._neurons_source`.

**Kilosort has two index spaces**: site index (0..n_sites-1, indexes `channel_positions.npy`) vs hardware channel (`channel_map.npy`, can exceed n_sites; the `ch` column of `cluster_info.tsv`). **Always index `channel_positions` by site index.**

### Changepoint correction

Bridge pattern: intervals → dense → correct → intervals. Kinematic CPs stored dense `int8`; audio CPs as onset/offset float pairs.

## Dataset Structure

- NetCDF with trials. Time coords: anything containing `time` (`time`, `time_aux`, …).
- Every `data_var` with a time dim is a feature — no `attrs["type"]` needed. Changepoints still use `attrs["type"] = "changepoints"`; colour vars are identified by "rgb" in the name.
- Media/session metadata: `.nwb` sources read directly; non-NWB read `.ethograph/alignment.nwb`.
- Labels live in `_labels.tsv`, not the `.nc`.
