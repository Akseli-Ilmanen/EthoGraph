# CLAUDE.md

## Working style

- Implement, don't propose. Don't wait for "yes go".
- Write idiomatic, typed Python (mypy/ruff clean). Prefer clean design patterns over cleverness.
- Imports: sorted stdlib → third-party → local, explicit (never `from x import *`), never inside a function (only exception: circular imports).
- Comments only where the logic isn't obvious — which should be rare. **Never remove human-authored comments** (TODO/FIXME/NOTE/explanatory); only remove comments you added yourself.
- **Fail fast**: bugs (wrong type, missing key, unexpected `None`) crash; runtime conditions (missing file, bad user input) are handled. Never `try/except` into a silent `None`. Catch broad exceptions only at the outermost GUI boundary.
- Test/debug scripts live in `tests/`, never the project root. Prefix ad-hoc debug scripts `_test_` so pytest skips them.
- Docs/docstrings: don't name individuals (Poppy, Freddy, Ivy).
- Claude Code may change any file in this repo.

### How to maintain this file

This file describes **how the system is**, not the history of how it got here. State each rule positively — the invariant that must hold — and keep it short.

- **Do not add war-stories.** No "the old code did X, which caused bug Y, so never do X" narratives. That knowledge belongs in a **test** that fails when the invariant breaks, not in prose here (the test is enforced; prose is not, and a model has to re-read and remember it). If a rule is worth writing down, it is worth a test — add the test and reference it in one clause (`Covered by tests/...`).
- **Prefer deletion over accumulation.** When you fix a class of bug, encode the fix as a test and, if needed, a one-line positive rule here — do not append another exception to a growing list.
- Update this file after a genuine architectural change, not after every bug fix.

## Project Overview

EthoGraph is a GUI for labelling start/stop times of animal movements, paired with a workflow using action-segmentation transformers to predict segments. It loads NetCDF/NWB/pynapple datasets and displays synchronized video/audio/ephys.

```python
import ethograph as eto
dt = eto.open("data.nc"); dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da); data, filt = eto.sel_valid(da, kwargs)
```

## Hard Rules

**Never hardcode rates.** No frame rates (30 fps), sample rates (44100 Hz), or 1-second fallbacks for windows/trial durations. Use source metadata (`video.fps`, `ImageSeries.rate`, audio rate) or user settings; if unknown, raise or return `None`.

**Never hardcode a device.** `resolve_device()` picks CUDA → MPS → CPU.

**Never special-case a `dataset_key` in GUI code** — put per-dataset settings in `DATASETS` metadata.

**Never call `QFileDialog` directly** — go through `gui/file_dialogs.py` (`browse_open_file` / `browse_open_dir` / `browse_save_file`), which start at the caller's `preferred_dir` (else `app_state.last_browse_dir`, SCOPE_GLOBAL) and record what was picked. (Wizard tabs use raw dialogs — they hold no `app_state`.)

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
    dialog_tag_sheet.py       # Print tag sheet… (cover page pre-recording tools only)
    pose_edit_mixin.py        # KeypointLabelMode (canvas anchor editing)
    dialog_pose_labelling.py, dialog_skeleton_editor.py
    widgets_curation.py       # Curation section of the Labels tab: scope, modes, frame-by-frame review, Ctrl+C
    dialog_label_gridview.py  # Label grid view: frames at label times; click = navigate / curated / uncurated
    dialog_video_grid.py      # Video grid: clips of one label class played side by side, sorted by duration
    dialog_pose_refinement.py # Refine imported poses: correct DLC/SLEAP files, _refined copies
    dialog_onset_model.py     # Model menu: GradBoost onset-detector train/predict dialogs
    plots_base.py             # BasePlot, PanelStateMixin
    plots_container.py        # UnifiedPanelContainer
    plots_{audiotrace,spectrogram,ephystrace,lineplot,heatmap,raster,space}.py
    plots_timeseriessource.py # Re-exports io/time_model.py (compat)
    plots_console.py          # ConsolePanel — REPL over what the clicked panel plots
    plots_radial.py           # RadialPlot — compass arrow for a heading at the current time
    label_drawing_mixin.py, video_sync.py, video_manager.py
    widgets_meta.py           # MetaWidget — creates + wires everything
    widgets_data.py           # DataWidget — central orchestrator
    widgets_{io,labels,navigation,changepoints,ephys,plot_settings,transform}.py
    widget_intervals.py, right_context.py, main_window.py, top_bar.py, cover_page.py
    table_filter.py           # Funnel-header column filters (ephys + keypoint tables)
    file_dialogs.py           # Browse dialogs that remember the last folder
    nwb_alignment.py, shortcuts.py

ethograph/labels/
    intervals.py              # Interval ops, mapping loaders, find_blocks
    ml.py                     # Dense↔interval, stitch_gaps/purge_small_blocks/fix_endings
    onset_model.py            # GradBoost point-event onset detection (train/predict core)
    curation.py               # labeling_method transitions, per-trial verdicts, review queues (Qt-free)
    tsv_store.py, predictions.py, crowsetta_format.py, converters.py, export.py

ethograph/io/
    catalog.py                # DataCatalog + XarrayLoader/PynappleLoader, pose discovery
    derived.py                # TracedArray + DerivedFeature + DerivedLoader (console features)
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TimeSource, SourceCollection
    time_sources.py           # XarrayTrialSource, PynappleSource
    audio_extract.py          # Container audio (AAC/MP4) → cached WAV, resolve_audio_path
    dataset.py, validation.py, pynapple.py, metadata_table.py, ephys_loader.py

ethograph/utils/              # io.py, xr_utils.py (sel_valid, get_time_coord), sequences.py
ethograph/skeleton/           # PrecomputedRenderer, SkeletonState, config.py, shapes.py
```

## Architecture

### Two data-source layers

**Rendering** (`io/plot_sources`) — `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`): `FileSource`, `XarraySource`, `PynappleSource`. `WindowedBuffer` caches wider than the viewport; per-plot buffers specialise (Spectrogram → `SpectrogramBuffer`, EphysTrace → `EphysTraceBuffer` pyramid).

**Navigation** (`io/time_model.py` + `time_sources.py`) — session-level time metadata via `TimeSource`; `SourceCollection` is the registry. Uses only `time_range`, never calls `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`. Session metadata (trial timing, media paths, FPS, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec; `ObservableAppState` auto-generates a Qt signal per variable, exposes dynamic `*_sel` attributes, and auto-saves to YAML.

- **`SCOPE_LOCAL`** (per-dataset `local_settings.yaml`): anything defining the plot x-extent (`fixed_window_s`, `navigate_mode`, `slider_scope`) — a view mode picked for one dataset must not follow the user to the next.
- **`SCOPE_GLOBAL`** (`gui_settings.yaml`): plain preferences. `xlim_mode` is global and set only via the "X-limits:" combo — never inferred from the load path.
- **Path settings are validated on load.** `AppStateSpec.PATH_VARS` maps every saveable path to what must exist (`"file"`/`"dir"`/`"any"`); `load_from_dict` filters through `sanitize_path_state()` so a path from another machine is dropped, not restored. Dropped values are remembered in `_unavailable_paths` and written back, so an unplugged drive does not erase the setting. Covered by `tests/test_unit/test_path_sanitize.py`.

### Time model + navigation

`TimeRange` (immutable: `union`/`intersect`/`contains`/`overlaps`), `TimeSource` (Protocol), `SourceCollection` (`union_range`, `intersection_range`, `session_range`, `sources_at(t)`, `trial_range`, `find_trial`, `trial_offset`; built in `data_loader.py`), `RestrictionWindow` (mode `"session"|"trial"|"label"|"sequence"`), builders `build_trial_window()` / `build_label_window()` / `build_sequence_window()`.

`app_state.window_bounds` (falls back to `trial_bounds`) drives every plot's x-limits; `app_state.session_time_range` is the full extent. Navigation modes ("Time slider:"): Session / Trial / Label / Sequence (pattern e.g. `1-2-3-5`, matched in `utils/sequences.py`).

**The one clock rule** (docs: `advanced/time_slider_trial_session.md`): `app_state.display_basis` (derived) is `"session"` iff `slider_scope == "session"` and `navigate_mode` is not label/sequence; else `"trial"`. It is the authority on which clock the plot axis speaks. Conversions go through `app_state.to_display(trial_id, t_rel)` / `from_display(t_display, strict=)` — **never hand-roll `trial_offset + t`**. Session scope is disabled for multi-trial xarray (`NavigationWidget.update_scope_availability`). `VideoSync.frame_to_time`/`time_to_frame` speak the display clock; consumers needing the trial clock convert back via `from_display`. Audio indexes through `audio_display_offset()`, ephys through `ephys_display_offset()`. Labels draw from `app_state.get_display_intervals()`. Covered by `tests/test_unit/test_time_basis.py` + `tests/test_integration/test_session_scope.py`.

### Catalog + loader: `catalog.py`

`DataCatalog` declares features, dimensions and streams; built by `catalog_from_xarray()` / `catalog_from_pynapple()`. Features are auto-detected (any `data_var` with a time dim; any pynapple `Tsd`/`TsdFrame`/`TsdTensor`) — no `attrs["type"]`.

- **A combo is named after the dim it selects from.** `INDIVIDUAL_DIMS` lists the spellings most-preferred first; read whichever a dataset uses via `catalog.individual_combo` / `app_state.selected_individual()` — never hardcode one. A selection key either IS a dim or is inert, on both sides (`_selections_for_var`, `_sanitize_selections`).
- `DataLoader` Protocol: `select(feature, selections, t0, t1) → PlotData`, always in **display (plot-axis) coordinates**. Loaders ignore dims a feature lacks.
- `PanelStateMixin._sanitize_selections` guarantees **at most one free multi-value dim**, re-applied by `set_panel_control` on every change. A saved `panel_layout` is untrusted: `MetaWidget.apply_saved_panel_layout` catches failures and rebuilds defaults (`_rebuild_default_panels`).
- `PlotData`: `time`, `data` `(T,)`/`(T,D)`, `dim_labels`, `title`, `ylabel`, `color_data`, `changepoints` → `render_plot_data()` in `plots_lineplot.py`.
- `XarrayLoader` wraps `xr.Dataset` (`update_ds()` per trial). `PynappleLoader` holds no trial state; pynapple objects live in absolute session time and the loader bridges to display time via a **display-offset provider** (`set_display_offset_provider`, installed by `DataWidget`). NWB loads via pynapple.
- **A pynapple feature's columns are a dim named in exactly one place**: `_column_axes()` returns a `ColumnAxis(dim, labels)` per feature, read by `catalog_from_pynapple`, `feature_dims()` and `select()`. `x`/`y`/`z` → `SPACE_DIM`; objects sharing a column tuple share one dim; a lone object gets `{name}_columns`. `select` resolves the column strictly under `axis.dim`.

### Alignment: `nwb_alignment.py`

`.nwb` sources are read/edited directly (`edit_nwb`); `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources.

`NWBAlignment(path)`: `get_stream_rate(stream, device)`, `resolve_media_path(...)`, `stream_offset_for_trial(...)`.

- **Trial timing has one source: the alignment NWB trials table.** For pynapple data, `_load_pynapple_dataset` reads `trials_ep` from `nwb_alignment.trials_ep` only. The one sanctioned bridge: the cover page offers, with explicit consent, to convert a trials IntervalSet in a pynapple folder into `.ethograph/alignment.nwb` (`alignment_from_trials_ep`).
- **Metadata is purely additive** — a tabular file joined on its `trial` column; no merge/conflict machinery. Resolution priority (`_resolve_alignment`): explicit `alignment_path` → NWB source trials → sidecar TSV.
- **Metadata is edited in the trials table** (`widget_trials.py`): a per-dataset "Edit values on double-click" toggle, only the current trial's row editable and tinted. Edits are written back to the source they were read from (`resolve_metadata_target` in `io/metadata_edit.py`); NWB trials tables are written in append mode. Debounced, flushed on `trial_changed` + close. Covered by `tests/test_unit/test_metadata_edit.py` + `tests/test_integration/test_metadata_edit_gui.py`.
- **The trials table's filters are the one trial filter, and every operation runs over `app_state.trials`.** `TrialsWidget._apply_filters` writes the visible subset to `app_state.trials`; navigation (`NavigationWidget._only_visible_trials`), curation queues/grids (`LabelSetupPage.allowed_trials`, `CurationPanel.build_queue`), onset-model training and inference (`_iter_trial_windows`) and the cross-trial label operations all read that list and nothing else. **No dialog gets a metadata filter of its own** — it shows how many trials it will run over (`trials_note`, refreshed on `trials_changed`). A filter change clamps the index without navigating. Documented as the `{important}` box at the top of `docs/source/advanced/metadata.md` and in the user manual's *Trial metadata* section. Covered by `tests/test_unit/test_onset_dialog.py` (`TestTrialsScope`) + `tests/test_unit/test_label_gridview.py` (`TestConfigDialog`).
- **Drag & drop = single-trial loading** (`cover_page.py`): `classify_files()` buckets by extension; each drop gets a fresh temp subdir and resets `video_folder`/`audio_folder`/`pose_folder` to `None`.
- **Video-container audio is decoded once to a cached WAV** (`io/audio_extract.py`, `resolve_audio_path()`, keyed by `media_cache_key`). Every audio reader goes through it. Never add a video extension to `AUDIO_EXTENSIONS`. Covered by `tests/test_unit/test_audio_extract.py`.

### Pose rendering

Two paths unified into `PoseRenderData`: `load_pose_from_file()` (movement) and `load_pose_from_nwb_direct()` (lazy HDF5). `apply_confidence_filter()` / `apply_keypoint_filter()` act on the `data_not_nan` mask; `PoseDisplayManager` displays via a `shown` mask — filtering never recreates layers.

**Colour encodes one axis, chosen by the user**: `app_state.pose_color_by` ∈ `{"keypoint", "individual"}` (SCOPE_GLOBAL, "Colour by"). Text labels carry the other axis (`text_prop`). The same setting drives the labelling canvas. There is no per-individual marker shape — colour is the only identity channel.

### Keypoint labelling + fill

Label a few frames by clicking the video, let a point tracker fill the rest — single video, 2D, one or more individuals. **Tools ▸ Keypoint labelling…** (`DataWidget.open_keypoint_labelling()`), one non-modal dialog. Scope: one camera, one trial (keyed by frame index on `app_state.video_path`); `TrialTree` datasets are not supported here.

**The binding design rules live in `docs/source/advanced/keypoint_labelling/`** — store/provenance, fill backends, PosePAL, detection, calibration, tag printing, canvas editing, keys, points table, export. **Read those pages before editing `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`.**

### Skeleton visualization

`ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses Dataset into a Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. `nwb_skeleton_to_config(nodes, edges)` converts an ndx-pose `Skeleton`, rendered by `_display_skeleton_direct()` behind "Show skeleton". Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. `dialog_skeleton_editor.py` draws a skeleton → `skeleton_config_override`. **Anchored shapes** (`skeleton/shapes.py`): templates bound to ≥2 control points → per-frame transform, stored under `"shapes"` in the skeleton config.

### Panels are layout instances — no per-plot-type toggles

Panels are instances created via the layout, never on/off toggles. There is no saved per-panel yes/no state and no panel checkboxes. Dropping an already-shown source creates another instance (removed via ✕).

- Created via the add-panel popup (`SourcePopup`, ➕ / Shift+N) or dropped onto the plot area. Plot-type gating (`allowed_plot_types`/`feature_ncols`) asks the loader's `feature_dims()` first (`app_state.ds` is `None` for pynapple).
- Initial visibility derives from data availability (`DataWidget._setup_panel_controls`).
- **Panels are dock widgets** (`UnifiedPanelContainer` hosts a nested `QMainWindow`); each is a `QDockWidget`. The media/plots separator drags across the whole window — minimums in the split are deliberately slivers (`PLOT_CONTAINER_MIN_HEIGHT`, `PANEL_MIN_HEIGHT`, `MEDIA_VIEW_MIN_*`); never raise a minimum to get a default proportion (defaults come from sizeHints and `resizeDocks`). Covered by `tests/test_integration/test_split_ratio.py`.
- **Layout persistence is automatic** (no JSON layout files): `app_state.panel_layout` (open panels + `panel_settings()` + `dock_state_b64`) is SCOPE_LOCAL; `app_state.window_state` is SCOPE_GLOBAL. Refreshed by `MetaWidget._snapshot_layouts`; applied via `shell.apply_dock_state_b64()`.
- **Audio panels are instances**: `audio_trace_plots`/`spectrogram_plots`, `add_audio_panel(...)` / `remove_audio_panel(...)`; `plot.mic_name` pins an `audio_source_map` key. Spectrogram settings apply to all instances.
- **Extra camera views are instances, each in its own closable dock** (`CameraViewDock {key}`); only the primary lives in the `VideoDock`. Anything that must follow the trial iterates the live views (`VideoManager.refresh_extra_videos()`), never `_extra_camera_combos`. Covered by `tests/test_integration/test_camera_trial_follow.py`.
- **The primary is a camera view like any other** — `update_video()` stamps `primary_view.camera_name`; every panel is titled `camera_dock_title()`. Primary fps comes from the probe, offset from its own `stream_offset_for_trial`. Switching the primary camera rebuilds `trial_alignment` first (`DataWidget._on_primary_camera_changed`). Covered by `tests/test_integration/test_camera_panel_identity.py`.
- **Static images are camera-like media** (`IMAGE_EXTENSIONS`): `app_state.image_paths` (SCOPE_LOCAL); each drop creates a view via `add_image_view()`, animated via `_display_pose_on_image()` + `CameraView.set_overlay_time(t)`. Pose with no video/image also works (a `pose_cam-N` stream alone; movement written to a throwaway `.nc` → `nc_file_path`).
- **Video motion is a drop-time `(time, camera)` feature** (`extract_video_motion()` → throwaway `.nc` → `nc_file_path`).
- **All line plots are equal instances**: `plot_container.line_plots`, `add_lineplot(...)` / `remove_lineplot(...)`. The heatmap is a fixed singleton toggled by `set_feature_view(...)`.
- **One canonical feature list**: `catalog.feature_choices()` feeds combo, popup and panel creation — never `ds.data_vars`.
- **Feature plots render only from their own `panel_state`** (`PanelStateMixin`, forked via `_ensure_panel_state()`). Never make a feature plot read `app_state.features_sel` for rendering. The sidebar edits the active plot via `set_panel_control()`; the global `*_sel` mirror serves shared consumers (labels, changepoints). An "All" checkbox = absence of that dim from `panel_state["selections"]`.
- **Space plots are instances**: `DataWidget.space_plots`, `add_space_plot(...)`, removed by closing its dock. Controls are catalog-driven and render from their own combos. **View sync** (`space_sync_views`) mirrors the coordinate frame across open space plots (2D never syncs with 3D). **Time window** (`space_window_s`, 0 = All) fetches ±N s around the marker.
- **Radial (compass) plots are instances**, modelled on space plots: `DataWidget.radial_plots`, `add_radial_plot(...)` / `remove_radial_plot(...)`. Shows one instant (`time_marker_updated` → `set_time`). Offered for any feature that pins down to one column whose values span a full turn (`plots_radial.angular_unit`); the unit is read off the span, never assumed. Pins all dims first, then judges the column (`default_selections` → `probe_angular_unit`); one combo per dim, at most one free. Arrow colours come from `app_constants.MULTIDIM_COLORS`.
- **Space reference geometry** comes from `~/.ethograph/geometries/*.yaml` (one file = one selectable geometry keyed by filename stem). `app_state.space_library_geometry` (SCOPE_LOCAL) holds the stem; seeded by `ensure_geometry_library()`.
- **Templates ship layouts via `local_settings.yaml`** (`download_template_local_settings()`, never overwrites a local file).

### Console panel + derived features

Added from the popup ("Tools ▸ Python console"), the **one singleton panel** (`plot_container.console_panel`).

- **The console binds what a panel renders**, not what backs it: `bind_panel(plot)` → `loader.select(...)` binds the resulting array to the feature's name, plus `t` (its time vector).
- Numpy, not xarray: `TracedArray` (`io/derived.py`) is an `np.ndarray` recording the expression graph. Each assigned name becomes a feature via `DerivedLoader.register()`, as a **recipe** (all-ufunc, re-evaluated per window) or a **snapshot** (frozen).
- `stack(sin, cos)` makes a `(T, D)` feature whose columns are exposed as `DERIVED_COLUMN_DIM` (`"Dimension"`) — this is what makes a derived feature usable as a space plot. Plain multi-assignment makes separate 1-D features.
- Commands: `features()`, `forget('name')`, `clear` / Ctrl+L, `clear(all=True)`. Help behind the `?` toggle, never printed.
- **Derived features live for one trial**: `trial_changed` → `ConsolePanel.reset_for_trial()`.
- **The console rebinds on `plot_container.panel_content_changed`** — that signal is its only rebind channel. `DerivedLoader` wraps the real loader at every `app_state.data_loader` assignment.

### Plot system

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`): time marker, x-range management, click handling, axes locking. `UnifiedPanelContainer` holds all panels and links x-axes.

### Video sync

`NapariVideoSync.frame_to_time(frame)` / `time_to_frame(t)` apply `time_offset` and the display offset — they speak the **display clock**. All widgets use these — never raw `frame / fps`. Consumers needing the trial/video clock convert with `from_display`. Video decode stays clipped to the current trial (`_trial_clip`; frame math is `trial_frame_window()` in `io/time_model.py`).

- **Playback seeks via `VideoSync._seek_playback_frame`.** While a freshly spawned decode worker is cold (`CameraView.decoder_ready()` False) each tick decodes synchronously in-process plus fires one async request; once warm it hands over to pure async. `play_segment`'s initial seek is async like regular Play. "Auto-play on navigate" → `autoplay_on_navigate` (SCOPE_GLOBAL). Covered by `tests/test_unit/test_video_sync_ready.py` + `test_autoplay_state.py`.
- **The playhead is always exact; only the video quantizes.** Audio is sliced at true sub-frame bounds and the marker sits on the true time; the video shows the nearest frame, and ←/→ (`step_frame_*` → `seek_to_frame`) reads off where frames fall. `VideoSync.play_segment(start_frame, end_frame, exact_t0=, exact_t1=)` takes bounds on the display clock and converts internally.
- **The audio-master clock (`gui/audio_clock.py`) is DAC-anchored, chunked and rate-bounded.** The playhead is what the ear hears, from the device's own timestamps (design note: `scripts/feedback/audio_clock_dac_timestamps.md`). It must never lead the sound. Resampling uses a bounded ratio (`limit_denominator(MAX_RESAMPLE_DENOM)`) and is chunked. A `playback_mic_key` change rebuilds the clock in place. Covered by `tests/test_unit/test_audio_clock.py` + `test_audio_playback_sync.py`.
- **The video render chain is guarded and watched.** `install_animate_guard` (pygfx_video.py) wraps `animate` on each fresh plot, re-arming on a raise and stamping a heartbeat; `VideoSync`'s 1 s watchdog re-arms a stale chain via `nudge_render_if_stalled`. Tools ▸ Reset video view (`VideoManager.reset_primary_video`) rebuilds the primary plot in place. Covered by `tests/test_unit/test_render_watchdog.py`.
- **Reloading the same file reuses the `PlotVideo`.** `CameraView.set_video()` reuses the loaded plot when `_video_path` is unchanged and only re-clips bounds; only a different decode path goes through `clear()`. `update_video()` drops only the `VideoSync` (`_teardown_primary_sync`); `_detach_load_state()` drops per-load state. Covered by `tests/test_integration/test_video_reload.py`.
- **Closing the primary video dock unloads the video** (`VideoArea.eventFilter` → `primary_close_requested` → `VideoManager.close_primary_video()`). Covered by `tests/test_unit/test_primary_dock_close.py`.
- **Cropping a camera is display-only, keyed by camera name** (`CameraView.set_crop` / `crop_clip_planes`): a rectangle becomes world-space clipping planes plus a camera reframe; the decoder/frame math/pose overlay are untouched. `VideoManager._camera_crops` re-applies per trial; session state, never saved. Covered by `tests/test_unit/test_video_crop.py`.

### Labels

**Storage:** TSV (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, individual_rec, event_type, confidence, labeling_method, trial, changepoint_corrected, prediction_source, n_samples`. A legacy `human_verified` column is carried along per trial and never read. **`onset_s`/`offset_s` are canonically trial-relative**, written and read as-is with no basis detection. A saved TSV carries no comment header (`load_labels_tsv` still passes `comment="#"` for legacy files). `enrich_labels_df` writes `trial_onset`/`onset_global`/`offset_global` alongside when the alignment knows the trial start. Label names in `mapping.txt`. In memory: `app_state._all_labels_df` (all trials, trial-relative) and `app_state.label_intervals` (current trial); `get_display_intervals()` is the read-only axis-clock view.

- **A label's subject is a pair: `individual` (actor) + `individual_rec` (recipient).** `NO_RECIPIENT` (`""`) means a solo behaviour. Each (actor, recipient) is an independent track: `add_interval`/`add_point`, overlap resolution and stitching group on `SUBJECT_COLUMNS`, and `select_subject`/`subject_mask` (`None` = "any") is the one place the filter is expressed. The GUI reads the pair from `app_state.selected_individual()` + `selected_recipient()`, never from `ds_kwargs`. A naming disjoint from the dataset's own skips the actor filter (`app_state.labels_name_our_individuals`). Covered by `tests/test_unit/test_label_intervals.py` (`TestRecipient`) + `tests/test_integration/test_individual_recipient.py`.
- **A visible label is a selectable label.** Click-selection (`_check_labels_click`) gates on `app_state.active_label_ids` (shown branches). Branch scope lives on mutation: `_delete_label`/`_edit_label` refuse a selection outside the active branch (`_refuse_foreign_branch`). Covered by `tests/test_unit/test_label_click_branches.py`.
- **Every label carries a `labeling_method`** ∈ {`manual`, `automated`, `curated`} (ndx-ethogram vocabulary; `labels/curation.py`, docs `advanced/labels/curation.md`). `add_point`/`add_interval` default to `manual`; a model passes `automated` (`predictions.py`, `dialog_onset_model.py`); a trimmed remnant keeps its row's method. The only transition the GUI performs is automated → curated (`curate_trial`/`curate_label`) — manual is never rewritten, and nothing runs backwards except an edit (→ manual) or a new prediction (→ automated). `ensure_labeling_method` reads a file without the column off its `confidence` (`< 1.0` → automated). A trial is **curated** iff none of its labels is automated (`trial_curation_status`); that verdict colours the trial combo + bottom bar (green/red, refreshed on `app_state.curation_changed`) and is written to the metadata table's `curated` column on a timer (`CurationPanel.sync_metadata`, `METADATA_SYNC_MS`) — never on the labelling path. Covered by `tests/test_unit/test_curation.py`.
- **Curation has one home: the Curation section of the Labels tab** (`widgets_curation.py`). *Scope* = label classes dragged out of the branch tables (`app_state.curation_label_ids`, SCOPE_LOCAL; `None` = all). *Mode* (`app_state.curation_mode`, SCOPE_LOCAL): `manual` (edits → manual, **Ctrl+C** → `curate_current_trial`), `inspect` (`trial_changed` → curate the trial, deferred a tick), `frame` (inline frame-by-frame review: queue from `build_review_queue` sorted (trial, onset), each reached via `NavigationWidget.jump_to_label_instance`; Enter commits `video.frame_to_time(current_frame)` → `from_display` — moved → manual + `HUMAN_CONFIDENCE`, unmoved → curated; Backspace deletes the whole row; B/N back/next, N curating the boundary it leaves when `curation_next_curates`). Enter/Backspace/Delete/B/N are ApplicationShortcuts installed on `_begin`, removed on `_stop`, disabled while typing; no button is default/autoDefault. The review view is `refine_window_s` (SCOPE_GLOBAL). Every label mutation in `LabelsWidget` ends with `curation_panel.note_labels_edited()`. Covered by `tests/test_unit/test_curation_review.py`.
- **Automated labels draw dotted, manual/curated solid** (`label_drawing_mixin.py`, `_method_style`). Drawn items are indexed by `draw_key` (class, display onset, subject); a one-label transition goes through `plot_container.restyle_label(key, automated)` (pens swapped in place, 0 → full redraw), a whole-trial transition through `schedule_labels_redraw()`.
- **Every label carries a `confidence`**: `HUMAN_CONFIDENCE` (1.0) for a hand-placed one, the producing model's own score otherwise. It is part of `INTERVAL_COLUMNS`, so it survives every per-trial round trip; `ensure_confidence` fills it in for files written without it. Covered by `tests/test_unit/test_label_intervals.py` (`TestConfidence`).
- **The grids are review surfaces over the scope, never separate curation systems.** `dialog_label_gridview.py` (Label grid view…) prints confidence + method per tile, outlines tiles below the threshold in red, plots `ConfidenceHistogramsDialog` sharing that threshold, and exports a PDF; `dialog_video_grid.py` (Video grid…) plays clips of **one label class at a time, sorted by duration, a non-scrolling screenful at a time** (Previous/Next label, Previous/Next clips — greyed out at the ends) driven by one Play/slider spanning the longest clip, played once then stopped at a speed that opens at the GUI's `playback_speed_pct` and is otherwise the grid's own (the tick advances wall-clock × speed, `_apply_timer_interval`) (shorter clips hold their last frame; point events get a red corner marker on their frame, `marker_visible`), decoded one screenful at a time at `CLIP_MAX_SIDE`; a window cut at the video's start/end is said on the tile (`clip_note`), as is where in the trial the label sits. Both share `LabelSetupPage` + `GridModeBar`/`TileVerdicts` (keyed by `entry_key`, so all tiles of a label share a verdict): *navigate* (jump, or `CurationPanel.start_review_at` in frame mode), *curate* (Done curates the clicked), *uncurate* (Done curates everything else; **Mark low-confidence as uncurated** pre-clicks the threshold's tiles and is enabled only in this mode — a low score argues for doubt, never approval); Done applies through `CurationPanel.curate_labels`. **Grid layout is a viewing habit, SCOPE_GLOBAL**: `video_grid_point_window_s` (0.5 s), `video_grid_per_page`, `video_grid_columns`, `label_grid_columns`, `label_grid_window_s` — each spin opens at the saved value and writes back on change. Covered by `tests/test_unit/test_label_gridview.py` + `tests/test_unit/test_video_grid.py`.
- **LightGBM onset models predict point events only, at most one per target class per trial** (`labels/onset_model.py` + `dialog_onset_model.py`, Model menu, docs `advanced/labels/onset_model.md`). One model holds `config.targets` (`{label_id: name}`) and fits one binary classifier per target over shared features. A model lives in `~/.ethograph/models/{name}`: `config.yaml` (frozen at creation — it defines the classifier's input columns), `train_data/{session_id}/trial_*.npz`, `model.joblib`. Feature columns come from the catalog's `select()` path with every dim pinned (explicit values, never "all"); all columns must share one sampling rate. A trial that does not carry a target contributes nothing to it. `predict_events` returns one `OnsetPrediction` per target, whose `confidence` is the geometric mean of the smoothed curve's peak and its `curve_sharpness`. **`use_crf` adds a linear-chain CRF (`sklearn-crfsuite`) over the classes' order**: every frame is tagged with the class of the most recent event (`phase_tags`), so the CRF's transitions are the sequence dependencies and `all_possible_transitions=False` makes an unseen order undecodable; its emissions are the per-class probabilities, cross-fitted by trial so the CRF never sees a classifier scoring its own training data. Decoding can legitimately leave a class out. Inference never overrides an existing target event and respects the trials-table filter + a metadata-column filter. Covered by `tests/test_unit/test_onset_model.py`.
- **The close prompt asks the file, not the flag.** `MetaWidget._check_unsaved_changes` calls `app_state.labels_dirty()`, comparing `_all_labels_df` against `labels_file_path()` via `labels_equal` (canonical `TSV_COLUMNS`, order-insensitive). Covered by `tests/test_unit/test_labels_dirty.py`.
- **A label edit is one undo step, snapshotted before it runs.** `Ctrl+Z` (`LabelsWidget.undo_last_label_edit`) walks `LabelHistory` in `labels/tsv_store.py`: a bounded stack of **per-trial** snapshots (rows + trial flags), never the whole table. Every handler that mutates labels calls `app_state.record_label_edit(...)` once at the top, before anything changes — so the trimming, sliver purge and changepoint correction that follow are inside the same step. `clear_label_history()` runs wherever `_all_labels_df` is replaced wholesale (dataset load, label/prediction import). Covered by `tests/test_unit/test_label_undo.py`.
- **Per-plot-type rendering:** `app_state.label_overlay_modes` maps plot type key → `"full"|"bottom"|"none"`, applied to every instance of that type. Defaults in `DEFAULT_LABEL_OVERLAY_MODES`; edited via `LabelsPerPlotDialog`.
- **Labels on new panels:** any path creating/showing a panel ends with `plot_container.schedule_labels_redraw()` (deferred — it must run after content render). Never emit `labels_redraw_needed` synchronously from a panel-creation path.
- **A state label being drawn is visible from the first click**: `LabelDrawingMixin.show_pending_label()` / `clear_pending_label()`; every path abandoning or committing a half-placed label goes through `LabelsWidget._reset_label_clicks()`. Covered by `tests/test_unit/test_pending_label_preview.py`.
- **Predictions:** per-trial `.npy`/`.pickle`, shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`. Confidence stays in memory. Dotted confidence curve gated per feature plot by `show_predictions` in `panel_state`.
- **Crowsetta:** `EthographSeq` registered at import; export converts int→string via the mapping. Internal storage stays integer-based.
- **Top-bar File menu:** "Import labels…" / "Import predictions…" / "Export labels…" borrow their own I/O sub-panel (`IOWidget.restore_subpanel()`). Data loading happens only on the cover page.

### Widget orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator. Flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → everything else.

- **Context-sensitive right sidebar:** the sections shown per plot type are defined by `_CONTEXT_MAP` (+ `_CONTEXT_TITLE`) in `gui/right_context.py`. The **Individual group sits above the context title and outside that map** (`RightContextPanel._individual`, shown for every context but `video`). Its actor combo IS the dataset's individual-dim combo (created into `DataPanel.individual_layout`); with no individual dim it is created under the singular spelling from the names the labels use (`DataWidget.refresh_individual_choices`). There is exactly one individual combo in the sidebar. `_on_active_panel()` swaps context on panel click (only for `_CONTEXT_KINDS`). `ActivePanelManager.set_active` re-emits `active_changed` even for the already-active panel (context swap is suppressed under zen mode / open Labels-Navigation section, re-synced on lift via `_sync_context_to_active`).
- **Guarded shortcuts are disabled while typing, never a no-op.** Every global shortcut (`shell.bind_shortcut`) has `Qt.ApplicationShortcut`; `EthographMainWindow._sync_guarded_shortcuts` (on `focusChanged`) toggles `setEnabled` for shortcuts bound with `guarded=True`; `typing_in_text_field()` (`gui/shortcuts.py`) decides. Keys a text editor owns are guarded automatically (`_TEXT_EDITING_KEYS`). Covered by `tests/test_unit/test_source_popup_nav.py`.
- **Neo trace panels are dynamic instances, one per stream/modality** (`_DYNAMIC_PANEL_SPECS["neo"]`, collection `plot_container.neo_trace_plots`). `add_panel` calls `DataWidget.configure_neo_plot(plot)`; the popup lists one "Neo (…)" per stream.
- **Neo and Phy trace panels are not auto-loaded** (heavy). Load resolves the Phy stream (`_ensure_default_ephys_stream`) and pre-wires the source; both are added on demand from the popup.

### Neurons + ephys

Two paths → `nap.TsGroup` + cluster table: **Kilosort folder** (`.npy` + `cluster_info.tsv` + raw `.dat`; full features) and **Pynapple file** (`.npz`/`.nwb`, `data["units"]`; raster only). State: `app_state.neurons_path`, `app_state.has_neurons`, `EphysWidget._neurons_source`.

**Kilosort has two index spaces**: site index (0..n_sites-1, indexes `channel_positions.npy`) vs hardware channel (`channel_map.npy`, the `ch` column of `cluster_info.tsv`). **Always index `channel_positions` by site index.**

### Changepoint correction

Bridge pattern: intervals → dense → correct → intervals. Kinematic CPs stored dense `int8`; audio CPs as onset/offset float pairs.

## Dataset Structure

- NetCDF with trials. Time coords: anything containing `time` (`time`, `time_aux`, …).
- Every `data_var` with a time dim is a feature — no `attrs["type"]`. Changepoints use `attrs["type"] = "changepoints"`; colour vars are identified by "rgb" in the name.
- Media/session metadata: `.nwb` sources read directly; non-NWB read `.ethograph/alignment.nwb`.
- Labels live in `_labels.tsv`, not the `.nc`.
