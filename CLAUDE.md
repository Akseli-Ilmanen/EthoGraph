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
    pose_annotate.py          # KeypointStore + movement/DLC export
    pose_fill.py              # Fill backends: Spline, OpticalFlow, CoTracker3
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

**Rendering** (`io/plot_sources`) — `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`): `FileSource` (loader with `rate`/`__len__`/`__getitem__`), `XarraySource`, `PynappleSource` (lazy `restrict()`, no xarray intermediate). `WindowedBuffer` is a viewport-aware cache that loads wider than the viewport.

Per-plot buffers: AudioTrace/LinePlot/Heatmap(features) → `WindowedBuffer`; Spectrogram → `SpectrogramBuffer` (caches FFT); EphysTrace → `EphysTraceBuffer` (pyramid); Heatmap(envelope) → inline.

**Navigation** (`io/time_model.py` + `time_sources.py`) — session-level time metadata via `TimeSource`; `SourceCollection` is the registry. Uses only `time_range`, **never** calls `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`. Session metadata (trial timing, media paths, FPS, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec (~40 vars); `ObservableAppState` auto-generates a Qt signal per variable (`current_frame_changed`, `trial_changed`, `restrict_window_changed`, `labels_modified`, `verification_changed`), exposes dynamic `*_sel` attributes, and auto-saves to YAML.

Anything defining the plot x-extent (`fixed_window_s`, `navigate_mode`, `slider_scope`) is `SCOPE_LOCAL` (per-dataset `local_settings.yaml`) — a view mode picked for one dataset must never follow the user to the next. `load_from_yaml` strips local-scope keys from the global file.

`xlim_mode` ("interval" | "fixed") is the exception: `SCOPE_GLOBAL`, a plain preference set only via the "X-limits:" combo. **Never infer it from the load path** (drag & drop, template, wizard) — that guessing was removed for being surprising.

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

**Drag & drop = single-trial loading** (`cover_page.py`). `classify_files()` buckets by extension; `_collect_drop_details()` shows ONE popup for unresolvable values only (npy sample rate; `source_software` for ambiguous `.h5`/`.csv`) plus an "extract audio" checkbox when the video has an embedded track (writes a throwaway `.wav` joining the normal `audio_mic-N` pipeline). Each drop gets a **fresh** temp subdir (`_prepare_drop_dir()`, `%TEMP%/ethograph_tmp_alignment/{uuid}/`) — mandatory, since throwaway `.nc` files share `local_settings` by parent dir and a shared dir leaks the previous drop's layout. A drop resets `video_folder`/`audio_folder`/`pose_folder` to `None` first.

### Pose rendering

Two paths unified into `PoseRenderData`: `load_pose_from_file()` (movement) and `load_pose_from_nwb_direct()` (lazy HDF5 slicing). `apply_confidence_filter()` / `apply_keypoint_filter()` act on the `data_not_nan` mask; `PoseDisplayManager` displays via a `shown` mask — **filtering never recreates layers**. `nwb_pose_keys` maps camera index → pose key.

### Keypoint labelling + fill

Label a few frames by clicking the video, let a point tracker fill the rest — no training, no GPU. Single video, 2D, one or more individuals. Opened via **Tools ▸ Keypoint labelling…** or the Pose sidebar (both → `DataWidget.open_keypoint_labelling()`, one non-modal dialog).

**Hierarchy is SLEAP-style**: one shared keypoint schema; each individual is an instance of it. The active target is an `(individual, keypoint)` pair; single-individual is just `n_individuals == 1` (`individual=None` means "the first one"). **`n_individuals == 0` is a legal state** — the user may delete the last individual (the store never resurrects one); nothing can then be labelled and the canvas mode's `active_individual` is `None`. **Naming is singular** for dims/columns/keys (`keypoint`, `individual`); plural only for Python containers (`keypoint_names`) and counts (`n_keypoints`).

**Asymmetric schemas**: `shared_keypoints=False` (the dialog's "Individuals share the same keypoints" checkbox) gives each individual its own subset in `keypoint_sets`; `keypoint_names` stays the **union**, so arrays stay rectangular and backends/exports/overlay are untouched. `keypoints_for(individual)` / `has_keypoint()` / `keypoint_mask()` are the only readers of the split; `set_keypoints_for(individual, names)` edits one set (appending new names to the union), `set_shared_keypoints()` toggles (never destructive — off gives everyone the union, on re-admits it). Pairs outside a set are permanently `NaN`: `set_point` refuses them and `set_fill` blanks whatever the backend tracked there. `n_points` is the flat grid (`n_individuals * n_keypoints`, what backends see); `n_schema_points` is what the user can actually label.

`KeypointStore` owns all state — the GUI never mutates the arrays. Anchors: `frame → (n_individuals, n_keypoints, 2)` in **`(x, y)` source-video pixels**, `NaN` where unlabelled. `pose_convert.poses_ds_to_points` emits `(track_id, frame, y, x)`; **the axis swap lives in `store_to_movement_ds()` and nowhere else**. A frame is an anchor if *any* point is labelled, so backends use per-point anchor sets (`anchor_frames_for(...)`), never one shared frame list. `filled` is `(n_frames, n_individuals, n_keypoints, 2)`; `confidence` is `(n_frames, n_individuals, n_keypoints)`, `1.0` on anchors. Schema edits carry points over by name and invalidate the fill.

**Provenance is the anchors/filled split — there is no third state.** A fill **never** feeds the next fill: `flat_anchors()` reads anchors only, so re-filling is a pure function of the labels (test: `test_refilling_never_feeds_on_the_previous_fill`). `human_mask(frame)` gives per-point provenance, `is_anchor(frame, kp, ind)` one point, `is_human(frame, ind)` the **row rule** — *one* hand-placed point anywhere in a `(frame, individual)` makes the whole row the user's, which is what the dialog's `Source` column shows. A prediction is "accepted" only by becoming a label: `promote_fill(frame, individual=None)` copies filled points into anchors (never overwriting one, one undo step each), and `nearest(..., include_fill=True)` makes them grabbable so a click does the same for one point. Deleting stays anchors-only — there is nothing to remove from a prediction.

**Anchors are project data, not settings** — persisted to `<video>.keypoints.json` next to the video (keys `keypoint`, `individual`, `shared_keypoints`, `keypoint_set`, `n_frames`, `anchors` — unknown keys are ignored, so sidecars carrying the removed `last_labelled_frame` still load; a missing `individual` key means a legacy one-individual sidecar, an empty list means the user deleted them all), never app_state. Only `labelling_keypoints` / `labelling_individuals` (SCOPE_LOCAL) and `labelling_backend` live in app_state.

**Fill backends** (`pose_fill.py`) share one protocol (`name`, `requires_video`, `fill(anchors, n_frames, frames, progress)`; `progress` returns `False` on cancel) and **know nothing about the hierarchy** — they track flat `(n_points, 2)` rows via `store.flat_anchors()` / `set_fill_from_flat()`. `SplineBackend` (PCHIP, no new deps, holds endpoints rather than extrapolating) is the default and the yardstick; `OpticalFlowBackend` (cv2) and `CoTrackerBackend` subclass `_GapBackend` (track each gap forward and backward, crossfade; spline pre-pass seeds keypoints missing on an endpoint). **Invariant asserted in tests: anchor frames come back exactly as labelled.** Gap-backend confidence is `min(visibility_fwd, visibility_bwd) × exp(−‖p_fwd − p_bwd‖ / disagreement_px)`, the disagreement measured in **source pixels** (positions are rescaled before the subtraction). `disagreement_px` is a constructor argument threaded through `build_backend`, exposed as `app_state.labelling_disagreement_px` and the dialog's "Disagreement tolerance" spin (shown only for `_TRACKING_BACKENDS`, since the spline scores by distance from the nearest anchor instead) — the right value depends on the footage, so it is not a constant. CoTracker's own visibility is **boolean**: `CoTrackerPredictor` thresholds it at `0.9` internally and that is not configurable, so the graded part of the score is entirely the disagreement term. `build_backend(key)` imports lazily; `available_backends()` reports install hints. `VideoFrameSource` decodes lazily via PyAV with `max_side` downscaling (`scale` back to source pixels) and `start_frame`.

**Canvas editing** (`pose_edit_mixin.py`): `KeypointLabelMode` attaches to a `CameraView`, which gained `scene()`, `screen_to_image()` (pygfx camera unprojection, correct under pan/zoom), `image_units_per_pixel()` and `set_label_mode()`. While attached, left-drag is labelling and panning moves to Shift+left-drag. Anchors draw in their own overlay — deliberately distinct from the pose overlay showing filled predictions — with shape = individual and colour = keypoint, inactive individuals dimmed, and the active point outlined in white — a **transparent-fill `circle` with a hairline edge**, never the `ring` marker (its donut is fixed-thickness and swamps the keypoint under it). **There is NO separate edit mode**: clicking an existing point always selects and drags it, and `Ctrl+Z` undoes — correcting must never require switching mode first. **Filled points are grabbable too** (`nearest(..., include_fill=True)`): grabbing one pins it as a label where the backend put it (`_drag_recorded = True`, so a drag collapses into that one undo step), which is how a prediction gets accepted or corrected. `on_released` fires once per pointer release so the dialog can rebuild the pose override then — rebuilding the whole poses dataset per mouse move is not viable, and `mode.dragging` gates the mid-drag path. `Backspace`/`Delete` → `_delete_selected_point()`: with a mode armed, `delete_selected()` = **the active point** (what the outline is drawn around), falling back to `delete_under_cursor()` when the active pair is unlabelled on this frame (deleting only what the cursor happens to hover was the surprising half); **with no mode armed, the Keypoints tree's selected pair on the current frame** — deleting must never require arming labelling first. The dialog's event filter is therefore installed for its **whole lifetime** (in `__init__`, removed in `closeEvent`), not only while a mode runs, and `_owned_key()` claims Backspace/Ctrl+Z always but Tab/arrows/`1`-`9` only while a mode is armed, so the main window keeps its own bindings. `_typing()` gives Backspace back to a focused spin box, which would otherwise delete a keypoint per keystroke. The two modes (after napari-deeplabcut) differ only in what happens *after* a placement: `SEQUENTIAL_MODE` advances to the **first** keypoint this individual lacks on this frame — schema order, which is the points table's left-to-right column order, so the table fills from the left even after labelling out of order; when the frame is complete the active keypoint stays put. It **never navigates**; `LOOP_MODE` keeps the keypoint and calls `on_advance_frame` — the dialog owns navigation and steps the suggestion list when one is active, else the next frame. `Tab` cycles keypoints, `1`–`9` select the individual; the modes are armed by their buttons only (no shortcuts — two were removed as unnecessary). The dialog's event filter accepts `ShortcutOverride` for those keys, otherwise the main window's global `1`–`9` (behaviour labels) would swallow them while labelling.

**Two visual channels: shape = individual, colour = keypoint.** `MARKER_SHAPES` / `marker_for_individual()` / `glyph_for_individual()` / `keypoint_colors()` in `pose_edit_mixin.py` are the single source for both the canvas and the dialog tree (branch text carries the glyph, each leaf's `●`/`·` mark is drawn in its keypoint colour). Colouring both axes the same way — as the pose display does — cannot show which beak belongs to which animal. A marker shape is a **material** property, not per-vertex, so the overlay holds **one `gfx.Points` layer per individual** (`AnchorOverlay._layers`); never collapse them back into one object.

**Two interaction modes** (`LABEL_MODE` / `EDIT_MODE`, `Shift+L` / `Shift+E`, pressing the running one again disarms): *label* is sequential — every click places the active keypoint and advances, never grabbing what is under the cursor; *edit* never creates — a click selects and drags the nearest existing anchor and an empty click does nothing. The dialog owns them via `set_interaction_mode(mode | None)` / `interaction_mode`; `KeypointLabelMode.set_mode()` switches without rebuilding the overlay. Shortcuts are bound **twice on purpose**: `QShortcut`s on the dialog (window context, so the tree/table type-search can't eat them) plus the canvas event filter (the canvas belongs to the main window). `keyPressEvent` covers events sent straight to the dialog.

**The dialog is a `QTabWidget`, one tab per stage**: "Keypoints" (schema), "Label" (mode buttons + points table + frame suggestions), "Fill and export". A single column of all groups was taller than a screen. Arming a mode jumps to the Label tab (its status line says what the next click does); disarming leaves the tab alone.

**Label tab header is one compact row** — `[Sequential] [Loop] [individual ▼] [keypoint ▼]` — over a prominent status chip showing the marker the next click drops (individual's glyph + keypoint name, both in the keypoint's canvas colour; individuals are shape-coded, not colour-coded, so inventing an individual colour would teach a mapping the canvas lacks). The keypoint picker and the **"Between clicks:" row** appear **only in Loop mode** — Sequential advances keypoints itself. The combos are inputs *and* readouts, so `_refresh_target_combos()` syncs them under `_blocked()` whenever a number key, Tab or a click on an existing point moves the target.

**"Between clicks" is explicit, not inferred** (`_AFTER_CLICK_CHOICES` → `AFTER_CLICK_FRAME` / `AFTER_CLICK_SUGGESTION` / `AFTER_CLICK_STAY`, read by `_advance_frame`): Loop used to follow the suggestion list whenever one existed and step a frame otherwise, so the same click did different things depending on invisible state. The old `+step` spin box (jump N frames) is **gone** — a numeric stride was a worse answer to "skip near-identical neighbours" than the suggestion list itself.

**Arrow keys split by modifier**: plain `←`/`→` are left entirely to the main window (single-frame stepping); `Shift+←`/`Shift+→` step the **suggested** frames, claimed from the main window's window-stepping via `ShortcutOverride` **only while `_suggestions` is non-empty**, so nothing is stolen when there is nothing to step through. Moving between chosen frames — not between raw frames — is what annotating actually consists of.

The dialog's key filter therefore sits on **three** objects (`_install_key_filter`): the dialog, the video canvas (owned by the main window, so clicking to label moves focus out of the dialog) and **the main window itself** — key events propagate up to it from any widget inside, and without that target Shift+arrows pressed while looking at the video hit the global window-stepping shortcut instead. `_owned_key(event, main_window=...)` yields **everything except the arrows** for events that came from the main window; check `event.type()` before anything else, since every main-window event now passes through this filter.

**Otherwise the tab carries no status text.** Which mode is armed shows in the buttons and the active `(individual, keypoint)` is highlighted in the Keypoints tree, so a status line only restated them; the strategy tip, the labelled/recommended counter, the last-edit label, "Go to last" and a dedicated Undo button were all removed as duplicated or shortcut-covered. Tooltips carry the detail. The points table is multi-select with a right-click "Delete labels on N frames" menu (`_delete_table_rows` → `store.clear_individual`).

**The dialog is the hierarchy**: one `QTreeWidget` (branch per individual, leaf per keypoint *of that individual*, per-frame `●`/`·` column, `k/n` count — a leaf's row is NOT the schema index, always resolve via its `(individual, keypoint)` `UserRole`). Schema edits **all** go through `_apply_schema(keypoints=, individuals=, shared=, individual_keypoints=)`. There is no separate flat keypoint list, and no "seed from loaded pose" button (a fresh store still seeds `keypoint_names` from `app_state.keypoints`). When keypoints are not shared, the Add/Remove keypoint buttons act on the selected individual only.

**The points table is a `QTableView` over a virtual model** (`PointTableModel`), NOT a `QTableWidget` — once a fill exists there is a row per `(frame, individual)` for the **whole video**, which no item grid can hold, and a model reading `store.positions()` on demand cannot disagree with the store (the old diffing item table could). Rows pivot **wide**: `Frame | Individual | Source | Confidence` then an `x`/`y` pair per keypoint, so every keypoint on a frame is visible at once — a long one-row-per-point table hides exactly that. **Layout: before a fill, rows are the labelled `(frame, individual)` and columns only keypoints carrying ≥1 label** (an unlabelled 20-keypoint schema must not bury the ones in use); **after a fill, every frame gets a row and every keypoint a column.** `_layout_signature()` is the cheap check that gates rebuilding — never enumerate the dense row set on a drag; a refresh otherwise repaints one frame (`refresh_frame`, rows of a frame are contiguous) or everything (`refresh_all`, after a fill).

**Provenance shows twice**: the `Source` cell (`Human`/`Fill`, by the `is_human` row rule) and dimmed text (`QPalette.Disabled`) on any `x`/`y` that came from the fill, so a mixed row still reads correctly. A fourth fixed column, **`Confidence`**, shows `nanmean` of the row's `store.confidence` (same reduction as `frame_confidence` — never `min`, an absent point sits at NaN forever), is **numerically filterable** ("show me everything under 0.4"), carries the per-row worst keypoint as a cell tooltip, and explains the whole scoring scheme in its **header tooltip**. Human points are displayed as `1.00` even when the stored array predates them: `store.confidence` is a snapshot of the last fill, and `set_point` deliberately does not write into it. Right-click gives up to three actions, **each shown only when the selection holds something for it to act on** — "Delete labels on N frames" (`store.clear_individual`) when any row is human, plus "Delete filled points" (`store.clear_fill_for`, blanks `filled`/`confidence` to NaN and keeps the labels — for frames where the animal is occluded and the backend placed a point anyway) and "Pin filled points as labels" (`store.promote_fill`) when any row has fill. Offering "Delete labels" on an all-prediction row did nothing and read as a broken menu. A row stripped of both shows an **empty** `Source` cell, never "Fill".

The header is **two rows** (`PairedHeaderView`, subclassing `FilterHeaderView` — Qt has no multi-level header): the keypoint name, in its keypoint colour, spans its pair, with `x`/`y` beneath. Both sections of a pair paint the *same* name across their union rect, which is idempotent and so survives a partial repaint; the `DisplayRole` is empty (tooltips carry "beak x") and `sectionSizeFromContents` reserves half the name's width per column, plus `FILTER_ZONE_W` on filterable ones, so ResizeToContents can't elide either. `setResizeContentsPrecision(50)` — ResizeToContents measures rows, and there can be a hundred thousand.

It scrolls inside `TABLE_MAX_HEIGHT`. Clicking a cell seeks the playhead and makes the clicked keypoint (or, in the fixed columns, just the individual) active; the current frame's row is selected and scrolled to. Use the `clicked` signal, never the selection signal: the table also selects itself when the playhead moves. Selection and row lookups always go **through the proxy** (`mapToSource`/`mapFromSource`) — a filtered-out row has no view index. **`store.last_labelled_frame` is gone** (with its bold row and its sidecar key): the bold either duplicated the selection or sat off-screen with nothing to scroll to it. A narrow repaint targets `_current_frame()` instead, and `store.undo()` **returns the frame it changed** so an undo landing on another frame still repaints the right row.

**Column filters live in `gui/table_filter.py`** (`MultiColumnFilterProxy`, `CategoryFilterDialog`, `NumericFilterDialog`, `FilterHeaderView`, `SORT_ROLE`), lifted out of `widgets_ephys.py` so the Kilosort cluster table and the points table share one interaction: a funnel in a reserved zone at the right of each filterable header section, criteria ANDed across columns, numeric comparisons on `SORT_ROLE` (never the formatted text). An empty allowed-set means *no* filter, never "hide everything". The proxy reads through `QModelIndex`, so it works over an item model and a virtual one alike; ephys keeps its probe-channel restriction in a `_ChannelFilterProxy` subclass. The points table filters `Frame` (numeric), `Individual` and `Source`; schema edits clear every filter, since a filter naming a deleted individual just looks like a broken table.

**Which frames to label** (`pose_suggest.py`): labelling consecutive frames is near-wasted effort. `suggest_frames(method, count, n_frames, frames, exclude, min_gap)` offers `uniform` (evenly spaced, no decode), `diverse` (DeepLabCut's `KmeansbasedFrameselection`: downscaled grayscale thumbnails -> mean-centre -> MiniBatchKMeans with `n_clusters = count` -> the frame nearest each centroid) and `motion` (mean |diff| from the previous frame — the same signal as `extract_video_motion`). Candidates are strided to `MAX_CANDIDATES` so long videos stay tractable, and **every method passes through `enforce_min_gap()`** — SLEAP's velocity method returns every frame over a threshold, so one fast bout can supply a run of neighbours; the min-gap pass keeps the best frame per neighbourhood instead. `min_gap` is derived from `n_frames / count` (never a hardcoded frame or second count). Already-labelled frames are excluded, mirroring SLEAP's `filter_unique_suggestions`. **`uncertain` is the method that actually fits a tracker**, but it is listed **last** and is only the opening choice once a fill exists: CoTracker takes `(t, x, y)` queries — ONE query frame per point — so it is never trained here and extra labels exist only to reset drift. The useful anchors are therefore where tracking *fails* (occlusion, blur), not where images look diverse — that is DeepLabCut's criterion, which serves model training. It ranks by `store.confidence` (forward/backward disagreement × visibility, already computed by `_GapBackend`), needs no video decode, and closes the label → fill → correct-worst → fill loop; it is the analogue of SLEAP's `prediction_score`. `frame_confidence()` reduces the store's `(frames, individuals, keypoints)` array with nanmean — NOT min, since a structurally absent point in an asymmetric schema sits at 0 forever and would pin every frame. The dialog's "Which frames to label" group runs it and gives Previous/Next navigation (and `Shift+←`/`Shift+→`) that seeks via `app_state.video.seek_to_frame()`.

**The combo is ordered by when in the workflow each method applies** — `uniform`, `motion`, `diverse` need nothing but the video, so a new project starts there; `uncertain` needs a fill and comes last. `_build_suggest_group` opens on `uncertain` only when `store.confidence` exists, else on `_SUGGEST_METHODS[0]`, so the first press of the button can never be a bare warning. **How many frames is asked as a share of the video** (`suggest_percent_spin`, floor `MIN_SUGGEST_PERCENT`), with `_suggest_count()` resolving it and a label spelling out "40 of 200 frames" — an absolute count means very different things on a 200-frame clip and a 60k-frame recording. The default share is `RECOMMENDED_ANCHORS` converted through the clip length (`_default_suggest_percent`), never a fixed percentage.

**y-flip on the plot-bound paths only.** Anchors are image coordinates — y grows *downward* from the top-left — but plots are y-up, so an unflipped trajectory renders vertically mirrored in `plots_space.py` (which, being generic, has no `invertY` unlike `dialog_skeleton_editor`). `store_to_movement_ds(store, fps, image_height=...)` applies `y_out = height - y`, driven by the dialog's **default-on** "Flip y" checkbox. It applies to *Load into the GUI* and the NetCDF export **only**: `_push_pose_override` must stay in image coordinates (the overlay does its own `y_world = img_height - y`, so flipping first mirrors the points off the animal) and DeepLabCut expects raw image coordinates. Kinematics are computed after the flip, so velocity's y sign matches the drawn trajectory.

**"Load into the GUI"** (`store_to_kinematics` + `DataWidget.add_keypoint_features`) merges the keypoints and, per tick-box, `velocity`/`speed`/`acceleration` from `movement.kinematics` into the **current trial** as ordinary features (`keypoints_position`, `keypoints_speed`, …), so a fill is plottable without writing and reloading a `.nc`. **Filled frames are included** — inspecting what the fill did is the whole point. The time axis is renamed to `FEATURE_TIME_DIM` (`time_keypoints`): keypoints run at video fps, the trial's `time` almost never does, and merging under one name would outer-join the two and pad every other feature with NaN. `add_keypoint_features` drops the previous load's vars *and dims* first, so re-running with a changed schema cannot outer-join old and new axes; it then follows the established path — `dt.update_trial` → `app_state.ds` → `store.update_ds` → `_register_feature`. **It does NOT require a dataset to already exist**: the keypoints *are* one, their time axis being the video's own frame grid (frame index × 1/fps) with names from the dialog. When the session has no xarray behind it — the common case after dropping a bare video, which loads via pynapple — `_install_keypoint_dataset()` builds a single-trial `TrialTree` from the arrays and installs a fresh catalog + `XarrayLoader`. That replacement only happens when nothing else is serving features, so it can never strand another source's panels; a pynapple session that *does* have features is told to export instead, since serving both at once needs a composite loader.

**Fill results render through the normal pose overlay** — `PoseDisplayManager.set_pose_override(pr)` swaps in an in-memory `PoseRenderData`, so the confidence spinbox filters filled points for free. `movement_ds_to_pose_render()` is the inverse of `pose_render_to_movement_ds()`.

**CoTracker is CC-BY-NC (not OSI open-source) — never in `[gui]`, and there is deliberately **no `[co-tracker]` extra**. cotracker has no PyPI release and declares no dependencies of its own (`install_requires=[]`, not even torch), while PyPI rejects metadata containing a direct URL reference — so neither half of the requirement can live in an extra. It is **one command**, `pose_fill.COTRACKER_INSTALL_HINT` (single source of truth for the GUI hint, the docs and pyproject's comment): `uv pip install --torch-backend=auto torch "cotracker @ git+…@{COTRACKER_COMMIT}"`. `--torch-backend=auto` is not optional in practice: PyPI's Windows torch wheels are CPU-only, so a GPU is otherwise silently ignored. The commit is pinned; an unpinned branch is the moving target we avoid `torch.hub` for. The ~97 MB weights auto-download on first fill (`download_cotracker_checkpoint()`, `.part` + rename so an interrupted fetch never loads as garbage) — so missing weights are a note, not an unavailability. Never pass `checkpoint=None` to `CoTrackerPredictor`: it builds an unloaded network that returns confident nonsense with no error.

### Skeleton visualization

`ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses Dataset into a Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. Only the config layer is ethograph-specific: `nwb_skeleton_to_config(nodes, edges)` converts an ndx-pose `Skeleton` (read by `_read_skeleton_config()` → `PoseRenderData.skeleton_config`), rendered by `_display_skeleton_direct()` behind "Show skeleton". Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. A NaN endpoint automatically drops any edge touching it.

`dialog_skeleton_editor.py` draws a skeleton on real pose data → `skeleton_config_override`. **Anchored shapes** (`skeleton/shapes.py`): square/triangle/circle templates deforming to follow the pose; bind ≥2 control points (`ShapeAnchorDialog`) → per-frame transform (2 anchors = similarity, 3+ = affine), rendered as a Shapes layer with the frame index as first vertex coord. Shapes live under `"shapes"` in the skeleton config.

### Panels are layout instances — no per-plot-type toggles

Panels are instances created via the layout, never on/off toggles. **There is NO saved per-panel yes/no state, no panel checkboxes, no Shift+A/S/E/F/C toggles** — never assume a boolean visibility toggle per plot type. **Duplicates are never prevented**: dropping an already-shown source always creates another instance (the user removes extras via ✕) — never add "already shown → just reveal it" dedup.

- Created via the add-panel popup (`SourcePopup`, bottom bar ➕ / Ctrl+N): drag a source onto the plot area, or Enter for the default spot. Every panel has a ✕. Templates define layouts in the same terms.
- Initial visibility derives purely from data availability (`DataWidget._setup_panel_controls`): audio → trace + spectrogram, features → feature plot, neo/neurons → neural panels.
- **Panels are dock widgets** (pynaviz-style): `UnifiedPanelContainer` hosts a nested `QMainWindow` with nesting enabled; each panel is a `QDockWidget` with a slim title bar. Default is a vertical stack in `_PANEL_ORDER`, line plots at the bottom.
- **The media/plots separator drags across the whole window (~10/90 either way).** Qt clamps a separator drag at the minimum size of the widgets on each side, so every minimum in the split is deliberately a sliver: `PLOT_CONTAINER_MIN_HEIGHT`, `PANEL_MIN_HEIGHT` (applied to every panel widget in `_make_dock`, plus the nested dock host, whose layout minimum is otherwise the sum of all open panels) and `MEDIA_VIEW_MIN_WIDTH`/`_HEIGHT` (camera views, space plots). **Never raise a minimum to get a default proportion** — defaults come from sizeHints and `resizeDocks`. Covered by `tests/test_integration/test_split_ratio.py`.
- **Layout persistence is automatic; NO JSON layout files exist.** `app_state.panel_layout` (open panels + `panel_settings()` + `dock_state_b64`) is SCOPE_LOCAL → dataset `local_settings.yaml`; `app_state.window_state` (outer geometry only, machine-local) → `gui_settings.yaml`. Refreshed by `MetaWidget._snapshot_layouts` (10s auto-save + close). No Save/Load layout actions. Applied via `shell.apply_dock_state_b64()`, **deferred to show when hidden** — `restoreState` must run on a visible window or Qt evicts docks created in between; later docks go through `shell.restoreDockWidget()`.
- **Audio panels are instances**: `audio_trace_plots`/`spectrogram_plots`, `add_audio_panel("audiotrace"|"spectrogram", mic_name=None)` / `remove_audio_panel(plot)`; `audio_trace_plot`/`spectrogram_plot` are compat properties (first or None). `plot.mic_name` pins an `audio_source_map` key (`None` follows the global Mic combo); the popup lists one entry per mic, a channel picker pins file+channel on drop. `_create_default_audio_panels()` makes one pair per mic when several exist. Spectrogram settings apply to all instances. When removing, stop `plot._td`.
- **Extra camera views are instances, each in its OWN closable dock** (`CameraViewDock {key}`); only the primary lives in the "Video" dock. `VideoArea._extras` is keyed by instance key (`"cam"`, `"cam (2)"`…) with the real name on `view.camera_name` (always read via `getattr(view, "camera_name", key)`). Closing defers `remove_extra(key)` → `camera_view_removed`; pose layers/combos reset only when the LAST view of a camera closes. `add_camera(..., duplicate=True)` always creates a new view.
- **Static images are camera-like media** (`IMAGE_EXTENSIONS` in `io/validation.py`): stored in `app_state.image_paths` (SCOPE_LOCAL), listed as `Image (name.png)` plus a permanent `IMAGE_BROWSE` entry; each drop creates a view via `add_image_view()`. Static views are timeless, but the primary camera's pose/skeleton overlay animates via `_display_pose_on_image()` + `CameraView.set_overlay_time(t)`. Image + pose with no video works (dialog asks pose fps, image written as a static `video_cam-N` stream); image-only drops are rejected ("no time axis").
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

**Reloading the same file must never rebuild the `PlotVideo`.** `CameraView.set_video()` reuses the loaded plot when `_video_path` is unchanged (trial change, camera re-apply) and only re-clips `start_frame`/`end_frame`/`time_offset`; only a genuinely different decode path (proxy swap, other camera) goes through `clear()`. Each `PlotVideo` owns a **spawned** pynaviz worker process that must re-import `av`/`pygfx`/`pynapple` (~1.5–2 s) before attaching to the parent's shared memory, while `PlotVideo.close()` waits only `join(timeout=2)` before dropping the parent's handle — which is what destroys the mapping on Windows — so a close-then-create cycle kills the new worker with `FileNotFoundError: [WinError 2] … 'wnsm_…'` on slower machines. `update_video()` therefore drops only the `VideoSync` (`_teardown_primary_sync`); `_cleanup_primary_video` (= teardown + `view.clear()`) is for genuinely unloading the video, and any aborted setup must clear the view itself. On the reuse path the renderer handlers and the `_update_extra_objects` overlay hook are **not** re-registered (they survive with the plot); `_detach_load_state()` drops the per-load state — labelling mode and pose overlay — that `clear()` would have. Covered by `tests/test_integration/test_video_reload.py`.

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
