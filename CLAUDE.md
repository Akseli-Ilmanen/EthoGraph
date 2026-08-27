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

This file describes **the architecture**, not the history of how it got here, and not the feature you just built. Every line here is loaded into every session, so a line has to pay for itself.

- **Finishing a task never adds to this file.** A new setting, a new dialog, a fixed bug, a layout rule, a scope decision — none of these belong here. Encode the invariant in a test; the test is enforced, prose is not. If you feel the urge to write "X is now Y", write `tests/test_unit/test_x.py` instead and stop.
- **Only a genuine architectural change touches this file**: a new subsystem, a new layer between existing ones, or a rule that changes how every module must be written. And when it does, the edit is preferably a deletion.
- **No war-stories.** No "the old code did X, which caused Y, so never do X". No "was removed because". No lists of exceptions. State the invariant positively in one line, or not at all.
- **Prefer deletion over accumulation.** A rule the code already makes obvious (by name, type, or signature) is a restatement — remove it.

### What earns a test

A test earns its place if **its failure would surprise you**. If the only way it can go red is "someone deliberately moved this", it is a change-detector, not a test.

**Keep**: invariant guards (a rule stated here), contract guards (two things must agree and nothing forces them to), branching logic (clock conversions, filters, resolution priority), regression guards (it broke in a way nobody saw coming).

**Cut**: tombstones (`..._relocated`, `..._has_no_y`), structural pins (menu-title lists, widget parentage, `hasattr`), unasserted actions, restatements (the expected value is recomputed the way the code computes it).

**A test never pays for data it does not read.** Take the cheapest fixture that can still fail: `gui` for widget wiring, `qapp` for one widget, none for Qt-free logic. Behaviour that holds for every dataset is tested once, not once per dataset.

## Project Overview

EthoGraph is a GUI for labelling start/stop times of animal movements, paired with a workflow using action-segmentation transformers to predict segments. It loads NetCDF/NWB/pynapple datasets and displays synchronized video/audio/ephys.

```python
import ethograph as eto
dt = eto.open("data.nc"); dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da); data, filt = eto.sel_valid(da, kwargs)
```

## Hard Rules

**Never hardcode rates.** No frame rates, sample rates, or 1-second fallbacks. Use source metadata (`video.fps`, `ImageSeries.rate`, audio rate) or user settings; if unknown, raise or return `None`.

**Never hardcode a device.** `resolve_device()` picks CUDA → MPS → CPU.

**Never special-case a `dataset_key` in GUI code** — put per-dataset settings in `DATASETS` metadata.

**Never call `QFileDialog` directly** — go through `gui/file_dialogs.py` (wizard tabs excepted: they hold no `app_state`).

**Never hand-roll `trial_offset + t`.** Conversions go through `app_state.to_display` / `from_display`.

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
    pose_detect.py            # Detect stage: AprilTag tag36h11, assignment learning
    pose_detect_preview.py    # PreviewPanel — what the detector sees on this frame
    pose_tagsheet.py          # Tag sheet: layout maths + vector PDF/SVG/printer output
    dialog_tag_sheet.py       # Print tag sheet… (cover page pre-recording tools only)
    pose_edit_mixin.py        # KeypointLabelMode (canvas anchor editing)
    dialog_pose_labelling.py, dialog_skeleton_editor.py
    widgets_curation.py       # Curation section of the Labels tab
    dialog_label_gridview.py  # Label grid view: frames at label times
    dialog_video_grid.py      # Video grid: clips of one label class side by side
    dialog_label_table.py     # Label table: every trial's rows as a spreadsheet
    dialog_pose_refinement.py # Refine imported poses: correct DLC/SLEAP files, _refined copies
    dialog_onset_model.py     # Model menu: GradBoost onset-detector train/predict dialogs
    dialog_curation_workflow.py # Saved curation routines: step editor + WorkflowRunner
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
    intervals.py              # Interval ops, mapping loaders
    ml.py                     # Dense↔interval conversion only
    onset_model.py            # GradBoost point-event onset detection (train/predict core)
    curation.py               # labeling_method transitions, per-trial verdicts, review queues (Qt-free)
    workflow.py               # Curation workflows: the STEP_KINDS contract + the YAML store (Qt-free)
    tsv_store.py, predictions.py, crowsetta_format.py, converters.py, export.py

ethograph/io/
    catalog.py                # DataCatalog + XarrayLoader/PynappleLoader, pose discovery
    derived.py                # TracedArray + DerivedFeature + DerivedLoader (console features)
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TimeSource, SourceCollection
    time_sources.py           # XarrayTrialSource, PynappleSource
    audio_extract.py          # Container audio (AAC/MP4) → cached WAV, resolve_audio_path
    schema.py                 # Variable schema: kind / is_egocentric / normalise attrs (movement#978)
    dataset.py, validation.py, pynapple.py, metadata_table.py, ephys_loader.py

ethograph/video_features/     # S3D video features: plan.py (seconds → frames per video rate), frames.py (streaming PyAV), extract.py

ethograph/features/
    columns.py                # FeatureColumn + enumerate_columns/extract_features — the one input-layout definition
    geometry.py               # Egocentric/pairwise/heading/angle/area features (xarray in → xarray out)
    changepoints.py           # Changepoint detection + add_changepoint_features
    movement.py, preprocessing.py, energy.py, oscillatory.py, neural.py, audio_changepoints.py

ethograph/segment/            # Segmentation pipeline (docs: docs/add_to_docs_later/segment/, ADRs: docs/adr/)
    config.py                 # SegmentConfig dataclasses: YAML + base: merge + dotlist overrides
    sessions.py               # Headless session opening (GUI loaders, no Qt), trial filter, media paths
    samples.py                # ClassTable, ColumnLayout, one (trial, individual) sample
    materialise.py            # Stage 1: the literature layout + index.tsv/columns.yaml/classes.yaml
    preprocess.py, augment.py, dataset.py, losses.py, metrics.py, postprocess.py, plotting.py
    train.py                  # Stage 2: assign_roles (the split), stats, run dir, compare_runs
    infer.py                  # Stage 3: prediction sets ({stem}_labels.tsv + _probs.npz)
    search.py                 # Workflow stage 1: Optuna on the val split → searches/{name}/best.yaml
    crossval.py               # Workflow stage 2: leave-one-session-out folds
    project.py                # Project — the one entry point: a config + a method per stage
    video_features.py         # S3D: a folder of videos or a config's sessions, + merge into a session
    models/__init__.py        # Architecture registry + contract
    models/vendored.py        # DLC2Action adapters: the registry contract + upstream's YAML defaults
    dlc2action/               # Vendored AGPL: model/, loss/, config/ — see its NOTICE.md; excluded from ruff/mypy

ethograph/utils/              # io.py, xr_utils.py (sel_valid, get_time_coord), sequences.py, device.py (resolve_device)
ethograph/utils/system_check.py # Linux preflight for the GUI wheels' system libs (`ethograph check`)
ethograph/skeleton/           # PrecomputedRenderer, SkeletonState, config.py, shapes.py
```

## Architecture

### Two data-source layers

**Rendering** (`io/plot_sources`) — `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`): `FileSource`, `XarraySource`, `PynappleSource`. `WindowedBuffer` caches wider than the viewport; per-plot buffers specialise.

**Navigation** (`io/time_model.py` + `time_sources.py`) — session-level time metadata via `TimeSource`; `SourceCollection` is the registry. Uses only `time_range`, never `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`. Session metadata (trial timing, media paths, FPS, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec; `ObservableAppState` auto-generates a Qt signal per variable, exposes dynamic `*_sel` attributes, and auto-saves to YAML.

- **`SCOPE_LOCAL`** (per-dataset `local_settings.yaml`): anything that only means something for one dataset — the plot x-extent (`fixed_window_s`, `navigate_mode`, `slider_scope`), names of its individuals/keypoints, its paths.
- **`SCOPE_GLOBAL`** (`gui_settings.yaml`): viewing habits and plain preferences that should follow the user to the next dataset.
- **Path settings are validated on load** (`AppStateSpec.PATH_VARS`, `sanitize_path_state()`); a path from another machine is dropped, not restored, and remembered in `_unavailable_paths`. Covered by `tests/test_unit/test_path_sanitize.py`.

### Time model + navigation

`TimeRange` (immutable), `TimeSource` (Protocol), `SourceCollection` (`union_range`, `session_range`, `sources_at(t)`, `trial_range`, `find_trial`, `trial_offset`; built in `data_loader.py`), `RestrictionWindow` (mode `"session"|"trial"|"label"|"sequence"`).

`app_state.window_bounds` (falls back to `trial_bounds`) drives every plot's x-limits; `app_state.session_time_range` is the full extent. Navigation modes: Session / Trial / Label / Sequence (pattern e.g. `1-2-3-5`, `utils/sequences.py`).

**The one clock rule** (docs: `advanced/time_slider_trial_session.md`): `app_state.display_basis` is `"session"` iff `slider_scope == "session"` and `navigate_mode` is not label/sequence; else `"trial"`. It is the authority on which clock the plot axis speaks. `VideoSync.frame_to_time`/`time_to_frame` speak the display clock; audio indexes through `audio_display_offset()`, ephys through `ephys_display_offset()`; labels draw from `app_state.get_display_intervals()`. Session scope is disabled for multi-trial xarray. Covered by `tests/test_unit/test_time_basis.py` + `tests/test_integration/test_session_scope.py`.

### Catalog + loader: `catalog.py`

`DataCatalog` declares features, dimensions and streams; built by `catalog_from_xarray()` / `catalog_from_pynapple()`. Features are auto-detected (any `data_var` with a time dim; any pynapple `Tsd`/`TsdFrame`/`TsdTensor`).

- **A combo is named after the dim it selects from.** `INDIVIDUAL_DIMS` lists the spellings most-preferred first; read whichever a dataset uses via `catalog.individual_combo` / `app_state.selected_individual()` — never hardcode one.
- `DataLoader` Protocol: `select(feature, selections, t0, t1) → PlotData`, always in **display coordinates**. Loaders ignore dims a feature lacks.
- `PanelStateMixin._sanitize_selections` guarantees **at most one free multi-value dim**. A saved `panel_layout` is untrusted: `MetaWidget.apply_saved_panel_layout` catches failures and rebuilds defaults.
- `XarrayLoader` wraps `xr.Dataset` (`update_ds()` per trial). `PynappleLoader` holds no trial state; pynapple objects live in absolute session time and the loader bridges to display time via a display-offset provider. NWB loads via pynapple.
- **A pynapple feature's columns are a dim named in exactly one place**: `_column_axes()` returns a `ColumnAxis(dim, labels)` per feature, read by `catalog_from_pynapple`, `feature_dims()` and `select()`.

### Alignment: `nwb_alignment.py`

`.nwb` sources are read/edited directly (`edit_nwb`); `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources. `NWBAlignment(path)`: `get_stream_rate`, `resolve_media_path`, `stream_offset_for_trial`.

- **Trial timing has one source: the alignment NWB trials table.** The one sanctioned bridge: the cover page offers, with consent, to convert a pynapple trials IntervalSet into `.ethograph/alignment.nwb`.
- **Metadata is purely additive** — a tabular file joined on `trial`. Resolution priority (`_resolve_alignment`): explicit `alignment_path` → NWB source trials → sidecar TSV. Edited in the trials table (`widget_trials.py`), written back to the source it was read from (`io/metadata_edit.py`). Covered by `tests/test_unit/test_metadata_edit.py`.
- **The trials table's filters are the one trial filter, and every operation runs over `app_state.trials`.** Navigation, curation queues/grids, onset-model training/inference and cross-trial label operations read that list and nothing else; no dialog gets a metadata filter of its own. The label filter (Tools ▸ Find label inconsistencies…, `utils/sequences.trials_matching_labels`) is a second, session-only slot intersected after the column filters. Covered by `tests/test_unit/test_label_matching.py` + `test_label_inconsistencies.py`.
- **Drag & drop = single-trial loading** (`cover_page.py`, `classify_files()`).
- **Video-container audio is decoded once to a cached WAV** (`io/audio_extract.py`, `resolve_audio_path()`). Never add a video extension to `AUDIO_EXTENSIONS`. Covered by `tests/test_unit/test_audio_extract.py`.

### Pose rendering

Two paths unified into `PoseRenderData`: `load_pose_from_file()` (movement) and `load_pose_from_nwb_direct()` (lazy HDF5). Filtering acts on masks (`data_not_nan`, `shown`) — it never recreates layers.

**Colour encodes one axis, chosen by the user**: `app_state.pose_color_by` ∈ `{"keypoint", "individual"}`; text labels carry the other axis. The same setting drives the labelling canvas. Colour is the only identity channel.

### Keypoint labelling + fill

Label a few frames by clicking the video, let a point tracker fill the rest. **Tools ▸ Keypoint labelling…** (`DataWidget.open_keypoint_labelling()`). Scope: one camera, one trial; `TrialTree` datasets are not supported.

**The binding design rules live in `docs/source/advanced/keypoint_labelling/`. Read those pages before editing `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`.**

### Skeleton visualization

`ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses Dataset into a Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. **Anchored shapes** (`skeleton/shapes.py`): templates bound to ≥2 control points, stored under `"shapes"` in the skeleton config.

### Panels are layout instances — no per-plot-type toggles

Panels are instances created via the layout (`SourcePopup`, ➕ / Shift+N, or drop), never on/off toggles; dropping an already-shown source creates another instance.

- **Panels are dock widgets** (`UnifiedPanelContainer` hosts a nested `QMainWindow`). Minimums in the media/plots split are deliberately slivers; never raise a minimum to get a default proportion. Covered by `tests/test_integration/test_split_ratio.py`.
- **Layout persistence is automatic**: `app_state.panel_layout` is SCOPE_LOCAL, `app_state.window_state` SCOPE_GLOBAL.
- **Audio panels, extra camera views, line plots, space plots, radial plots are all instances** with `add_*`/`remove_*` on their owner; anything that must follow the trial iterates the live views. The heatmap is a fixed singleton toggled by `set_feature_view(...)`. Covered by `tests/test_integration/test_camera_trial_follow.py` + `test_camera_panel_identity.py`.
- **One canonical feature list**: `catalog.feature_choices()` feeds combo, popup and panel creation — never `ds.data_vars`.
- **Feature plots render only from their own `panel_state`** (`PanelStateMixin`). Never make a feature plot read `app_state.features_sel` for rendering.
- **Space reference geometry** comes from `~/.ethograph/geometries/*.yaml`; **templates ship layouts via `local_settings.yaml`** (never overwriting a local file).

### Console panel + derived features

The one singleton panel (`plot_container.console_panel`). **The console binds what a panel renders**, not what backs it. Numpy, not xarray: `TracedArray` (`io/derived.py`) records the expression graph; each assigned name becomes a feature via `DerivedLoader.register()`. `stack(a, b)` makes a `(T, D)` feature usable as a space plot. Derived features live for one trial. The console rebinds only on `plot_container.panel_content_changed`.

### Plot system

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`). `UnifiedPanelContainer` holds all panels and links x-axes.

### Video sync

`NapariVideoSync.frame_to_time(frame)` / `time_to_frame(t)` speak the **display clock**; all widgets use these — never raw `frame / fps`. Video decode stays clipped to the current trial (`trial_frame_window()` in `io/time_model.py`).

- **The playhead is always exact; only the video quantizes.** Audio is sliced at true sub-frame bounds; the video shows the nearest frame.
- **The audio-master clock (`gui/audio_clock.py`) is DAC-anchored, chunked and rate-bounded.** It must never lead the sound. Covered by `tests/test_unit/test_audio_clock.py` + `test_audio_playback_sync.py`.
- **The video render chain is guarded and watched** (`install_animate_guard`, `VideoSync`'s watchdog). Covered by `tests/test_unit/test_render_watchdog.py`.
- **Reloading the same file reuses the `PlotVideo`.** Covered by `tests/test_integration/test_video_reload.py`.
- **Cropping a camera is display-only**, keyed by camera name, session state. Covered by `tests/test_unit/test_video_crop.py`.

### Labels

**Storage:** TSV (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, individual_rec, event_type, confidence, labeling_method, trial, changepoint_corrected, prediction_source, n_samples`. **`onset_s`/`offset_s` are canonically trial-relative.** Label names in `mapping.txt`. In memory: `app_state._all_labels_df` (all trials) and `app_state.label_intervals` (current trial).

- **A label's subject is a pair: `individual` + `individual_rec`** (`NO_RECIPIENT` = solo). Each pair is an independent track; `select_subject`/`subject_mask` is the one place the filter is expressed. Covered by `tests/test_unit/test_label_intervals.py`.
- **A visible label is a selectable label**; mutation refuses a selection outside the active branch. Covered by `tests/test_unit/test_label_click_branches.py`.
- **Every label carries a `labeling_method`** ∈ {`manual`, `automated`, `curated`} and a `confidence` (`HUMAN_CONFIDENCE` for a hand-placed one). The only transition the GUI performs is automated → curated; an edit → manual, a new prediction → automated. A trial is curated iff none of its labels is automated; the verdict is written to the metadata table's `curated` column as `"yes"`/`"no"`, on a timer, only while `app_state.curation_active`. Covered by `tests/test_unit/test_curation.py` + `test_curation_review.py`.
- **Curation has one home: the Curation section of the Labels tab** (`widgets_curation.py`, docs `advanced/labels/curation.md`): scope (label classes), mode (`manual` / `inspect` / `frame`), frame-by-frame review. **The grids are review surfaces over the scope, never separate curation systems** (`dialog_label_gridview.py`, `dialog_video_grid.py`; `GridModeBar.entries_fn` returns what is on screen, and that is what every operation runs over). Covered by `tests/test_unit/test_label_gridview.py` + `test_video_grid.py`.
- **A curation workflow is a recording of the GUI, never a second way of doing anything** (`labels/workflow.py` + `dialog_curation_workflow.py`, docs `advanced/labels/workflows.md`). `STEP_KINDS` is the one contract; every handler drives the widget a user would. Covered by `tests/test_unit/test_curation_workflow.py`.
- **Onset models predict point events only, at most one per target class per trial** (`labels/onset_model.py`, docs `advanced/labels/onset_model.md`). Feature columns come from `features/columns.py` — the one input-layout definition shared by training and inference. A label sits on its curve's tallest peak and its `confidence` is the height there; every target is read independently off its own curve. Runs keep their curves (`labels/onset_curves.py`). Covered by `tests/test_unit/test_onset_model.py` + `test_onset_curves.py`.
- **The label table is one spreadsheet over the whole TSV** (`dialog_label_table.py`); only `INTERVAL_COLUMNS` are editable. Covered by `tests/test_unit/test_label_table.py`.
- **A label edit is one undo step, snapshotted before it runs** (`app_state.record_label_edit`, `LabelHistory` per trial). Curating is not an undoable edit. Covered by `tests/test_unit/test_label_undo.py`.
- **The close prompt asks the file, not the flag** (`app_state.labels_dirty()`). Covered by `tests/test_unit/test_labels_dirty.py`.
- **Automated labels draw dotted, manual/curated solid.** Covered by `tests/test_unit/test_label_method_style.py`.
- **Labels on new panels:** any path creating/showing a panel ends with `plot_container.schedule_labels_redraw()` (deferred). Never emit `labels_redraw_needed` synchronously from a panel-creation path.
- **Predictions:** per-trial `.npy`/`.pickle`, shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`.
- **Crowsetta:** `EthographSeq` registered at import; internal storage stays integer-based.

### Variable schema (`ethograph/io/schema.py`)

What a data variable *is*, following movement's proposal (issue #978). Docs: `docs/add_to_docs_later/variable_schema.md`.

- **`kind` is advisory and nothing may require it; it is a label, never a switch.** Anything that changes maths reads a behavioural attr (`normalise`). Covered by `tests/test_unit/test_schema.py`.
- **Flags are written `0`/`1`, never `True`/`False`** (NetCDF has no boolean attr).
- **Both backends use one vocabulary**; a pynapple `Tsd` declares its schema in `{session}/.ethograph/schema.yaml`. Every reader goes through `schema.attrs_of`.
- **For changepoints, the label and the mask marker are different attrs**: `kind="changepoint_feature"` labels the family, `changepoint_mask` marks a raw mask (`schema.is_changepoint`).
- **`train.drop_kinds` is the ablation axis, `train.subsample` the rate axis, `train.loss.tau` the smoothing truncation** — all run-level, so one materialised dataset serves every run. Covered by `tests/test_unit/test_segment_pipeline.py` + `test_segment_losses.py`.

### Segmentation pipeline (`ethograph/segment/`)

Code-first, never in the GUI, and **scripted — there is no CLI**. One YAML config becomes a `Project`; every stage is a method on it; overrides are dotted `key=value` strings. Vocabulary in `CONTEXT.md`; design in `docs/add_to_docs_later/segment/`; decisions in `docs/adr/`.

- **Two workflow stages**: `search()` (trials pooled and cut by `train.split`, Optuna on val, winner → `searches/{name}/best.yaml`) and `cross_validate()` (leave-one-session-out, each fold predicting its held-out session). A session has no role; `train.split` is three ratios summing to 1, drawn by whole trial. Covered by `tests/test_unit/test_segment_pipeline.py`.
- **A setting has exactly one spelling**: the dotted override path, in the file, in an override and in `search.params`. `model.params` is per architecture, validated against upstream's own YAML. Covered by `tests/test_unit/test_segment_architectures.py`.
- **Features are built with the session, never by the pipeline.** Stage 1 only selects existing variables through `features/columns.py`. The one exception is changepoint mask expansion (`features.changepoint_features`). Covered by `tests/test_unit/test_segment_changepoint_features.py`.
- **A sample is one (trial, individual)**; the individual dim is pinned per sample. Only `manual`/`curated` labels are training targets. Background is class 0.
- **The materialised dataset is role-agnostic and in the literature layout**; roles and normalisation statistics belong to the run.
- **The models and the loss are DLC2Action's, vendored in upstream's layout**, never edited beyond `NOTICE.md`. **Every architecture speaks one contract**: `model(x (B,F,T), mask (B,1,T)) → logits (S,B,C,T)`, `logits[-1]` is the prediction, read through `as_output(...)`. **A default is never written in our code** — read from `dlc2action/config/` via `DLC2ACTION_CONFIG`. Covered by `tests/test_unit/test_segment_architectures.py` + `test_segment_losses.py`.
- **Prediction sets are the GUI's labels TSV** (`labeling_method=automated`, `prediction_source=run name`) under `predictions/{run}/`. Post-processing goes through the GUI's own `correct_changepoints`; the GUI's `gui_settings.yaml` is the default source of its numbers, and the GUI never reads a project config (ADR 0006). Covered by `tests/test_unit/test_segment_gui_postprocess.py`.
- **S3D sidecars are on the video's clock; merging converts.** Covered by `tests/test_unit/test_segment_video_features.py`.

### Widget orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator. Flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → everything else.

- **Context-sensitive right sidebar:** `_CONTEXT_MAP` in `gui/right_context.py`. There is exactly one individual combo in the sidebar (`DataWidget.refresh_individual_choices`).
- **Guarded shortcuts are disabled while typing, never a no-op** (`shell.bind_shortcut(guarded=True)`, `typing_in_text_field()`). Covered by `tests/test_unit/test_source_popup_nav.py`.
- **Neo and Phy trace panels are dynamic instances, added on demand from the popup** (heavy, never auto-loaded).

### Neurons + ephys

Two paths → `nap.TsGroup` + cluster table: Kilosort folder (full features) and pynapple file (raster only). **Kilosort has two index spaces**: site index (indexes `channel_positions.npy`) vs hardware channel (`channel_map.npy`). **Always index `channel_positions` by site index.**

### Video features (S3D)

`ethograph/video_features/`: **configured in seconds, resolved per video** — `S3DConfig` → `plan_s3d(video_fps, cfg)` → `S3DPlan`, refused loudly when the rate cannot carry the window. The rate comes from `io/video_probe.py`, never a setting. Covered by `tests/test_unit/test_s3d_plan.py` + `test_s3d_extract.py`.

### Changepoint correction

Bridge pattern: intervals → dense → correct → intervals. **A click snaps to what the clicked panel draws** (`features/changepoints.changepoint_fired` is the one reading of a mask). Covered by `tests/test_unit/test_changepoint_snap.py`.

## Dataset Structure

- NetCDF with trials. Time coords: anything containing `time`.
- Every `data_var` with a time dim is a feature. Changepoints are found via `schema.is_changepoint`; colour vars by "rgb" in the name.
- Media/session metadata: `.nwb` sources read directly; non-NWB read `.ethograph/alignment.nwb`.
- Labels live in `_labels.tsv`, not the `.nc`.
