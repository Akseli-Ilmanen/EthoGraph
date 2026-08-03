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
    pose_fill.py              # Fill backends: Spline, OpticalFlow (+ private CoTracker3 tracking)
    pose_refine.py            # PosePAL backend: CoTracker3 + test-time refinement, GPU only
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

Label a few frames by clicking the video, let a point tracker fill the rest — no training, no GPU for the default backend. Single video, 2D, one or more individuals. **Tools ▸ Keypoint labelling…** or the Pose sidebar → `DataWidget.open_keypoint_labelling()`, one non-modal dialog.

#### Store (`pose_annotate.py`)

- **SLEAP-style hierarchy**: one keypoint schema, each individual an instance of it; the target is an `(individual, keypoint)` pair (`individual=None` = the first). **`n_individuals == 0` is legal** — the store never resurrects a deleted individual. **Naming is singular** for dims/columns/keys (`keypoint`, `individual`), plural only for Python containers and counts.
- **Asymmetric schemas** (`shared_keypoints=False`): per-individual subsets in `keypoint_sets`, while `keypoint_names` stays the **union** so arrays stay rectangular and backends/exports/overlay are untouched. `keypoints_for()` / `has_keypoint()` / `keypoint_mask()` are the only readers of the split; `set_keypoints_for()` edits one set, `set_shared_keypoints()` toggles non-destructively. Pairs outside a set are permanently `NaN` (`set_point` refuses, `set_fill` blanks). `n_points` = flat grid (what backends see), `n_schema_points` = what can be labelled.
- **`KeypointStore` owns all state** — the GUI never mutates the arrays. Anchors `frame → (n_individuals, n_keypoints, 2)` in **`(x, y)` source-video pixels**, NaN unlabelled; `filled` `(n_frames, …, 2)`, `confidence` `(n_frames, …)` = 1.0 on anchors. `poses_ds_to_points` emits `(track_id, frame, y, x)` — **the axis swap lives in `store_to_movement_ds()` and nowhere else**. A frame is an anchor if *any* point is labelled, so backends use per-point sets (`anchor_frames_for`), never one shared frame list. Schema edits carry points over by name and invalidate the fill.
- **Provenance is the anchors/filled split — there is no third state.** A fill never feeds the next: `flat_anchors()` reads anchors only, so re-filling is a pure function of the labels (`test_refilling_never_feeds_on_the_previous_fill`). `human_mask(frame)` per point, `is_anchor()` one point, `is_human(frame, ind)` the **row rule** — one hand-placed point makes the whole row the user's, which is what `Source` shows. A prediction is accepted only by becoming a label: `promote_fill(frame, individual=None)` copies filled → anchors (never overwriting, one undo step each); `nearest(..., include_fill=True)` lets a click do it for one point. Deleting stays anchors-only.
- **Anchors are project data, not settings** — `<video>.keypoints.json` beside the video (keys `keypoint`, `individual`, `shared_keypoints`, `keypoint_set`, `n_frames`, `anchors`; unknown keys ignored, missing `individual` = legacy sidecar, empty list = the user deleted them all). app_state holds only `labelling_keypoints` / `labelling_individuals` (SCOPE_LOCAL), `labelling_backend`, `labelling_disagreement_px`, `labelling_cotracker_checkpoint`.

#### Fill backends (`pose_fill.py`)

One protocol (`name`, `requires_video`, `fill(anchors, n_frames, frames, progress)`; `progress` returns `False` on cancel), **no knowledge of the hierarchy** — flat `(n_points, 2)` rows via `store.flat_anchors()` / `set_fill_from_flat()`.

- **A fill covers `anchor_span()` and nothing else** — first labelled frame to last, NaN (position *and* confidence) outside, since past the outermost label there is only an extrapolation asserted as confidently as a bracketed frame. `_restrict_to_span` enforces it in `SplineBackend`, which every gap backend seeds from, so the rule lives in one place. Inside the span a point labelled on only *some* frames holds its nearest value — that seeds a gap endpoint for a keypoint labelled once.
- `store.fill_range` is the span **measured from the result** (`set_fill` / `clear_fill`); never re-derive it from the anchors.
- `SplineBackend` (PCHIP, no new deps) is the default and the yardstick; `OpticalFlowBackend` (cv2) and `_CoTrackerTracking` subclass `_GapBackend` (each gap tracked forward + backward and crossfaded; spline pre-pass seeds keypoints missing on an endpoint). **`_CoTrackerTracking` is private and never offered as a backend** — unrefined tracking is the same method made worse; `PosePALBackend` subclasses it and is the only learned backend the user sees.
- **Invariant asserted in tests: anchor frames come back exactly as labelled.**
- Confidence = `min(vis_fwd, vis_bwd) × exp(−‖p_fwd − p_bwd‖ / disagreement_px)` in **source pixels** (rescale before subtracting). `disagreement_px` is a constructor argument threaded through `build_backend` to the "Disagreement tolerance" spin (`_TRACKING_BACKENDS` only — the spline scores by distance from the nearest anchor); it depends on the footage, so it is not a constant. CoTracker's visibility is **boolean** (0.9 threshold, not configurable), so the graded part is entirely the disagreement term.
- `build_backend(key)` imports lazily; `available_backends()` reports install hints. `VideoFrameSource` decodes lazily via PyAV with `max_side` downscaling (`scale` back to source pixels) and `start_frame`.

#### PosePAL: CoTracker3 + test-time refinement (`pose_refine.py`)

`PosePALBackend`, key `POSEPAL_BACKEND` (`"posepal"`), label `POSEPAL_LABEL`. Pan et al. 2025 (arXiv:2506.03868): freeze CoTracker3 and optimise **only the query-point features** against the labelled frames (49.6 → 67.5 δ_avg, beating tuning the CNN or the whole network). `QueryFeatureRefinement` holds them per `(level, support, point, channel)`; `applied()` wraps `model.get_track_feat` so they reach training *and* inference, and `_patched_track_feat` deletes the shadowing attribute rather than reassigning the bound method. **Never vendor PosePAL's fork** — it patches `forward` to take precomputed pyramids, which upstream does not. Departures forced by real footage: fit over **windows** (`training_windows`, ≥2 anchors each) not one clip; `valids` masks the loss to the pairs a human placed; `_CachedEncoder` caches `fnet` per window. Levels are keyed by **fmaps spatial shape**, not call order.

**The features are absolute and substituted, never a residual added to what the call sampled** (`applied()` → `torch.where`). A residual's base moves with the query frame — a window's first labelled frame during the fit, a gap endpoint at inference — so the correction would land on an appearance it was never fitted against; the paper's object is a *frame-independent* template. `_initialise()` seeds it as PosePAL's `get_kp_feats` does: the **mean support window over every frame the user labelled that point on** (each `(point, frame)` pair counted once, from the first window holding it), which is also what the L1 term pulls back towards. Per level, from that level's own fmaps — PosePAL computes level 0 and reuses it for all four, which upstream's `forward` never does. `_learned` masks rows with no label at all (a mean of nothing). `REFINE_LR` matches PosePAL's `1e-4`: features come off an L2-normalised map, and a step an order larger walks them off the unit sphere.

**Checked, not argued: `tests/_test_posepal_parity.py`** runs PosePAL's own `tto` and our `fit` on one clip, one checkpoint and one set of labels, against **their fork** (`PosePAL/dependencies/cotracker3`, i.e. `Zhuoyang-Pan/co-tracker` — their optimisation cannot run on upstream at all, which has no `fmaps_pyramid=` / `track_feat_support_pyramid=` on `forward`). GPU + their checkpoint, so it is `_test_`-prefixed and pytest skips it. Findings it pins: our initial features equal their `get_kp_feats` to **~1.5e-5** on features of magnitude 5e-2; the two refinements agree with each other (0.5 px median) more closely than either departs from stock CoTracker (0.4–0.7 px). **One divergence is deliberate and theirs is the wrong one**: their `extract_features` runs the CNN on `process_video`'s raw 0..255 tensor, while `forward` (our path, and everyone else's) normalises to `[-1, 1]` first — that accounts for the entire residual gap against their code as written, and building the pyramid inside `forward` both avoids copying their 40-line pyramid builder and feeds the network its trained input range.

**Inference stays per-gap, and a gap tracks only the rows it has a seed for** — so `_GapBackend.fill` announces the compressed mapping via `_on_rows(rows)` (no-op hook, overridden to `refinement.set_rows`) and `applied()` gathers those rows. Without it every point after a never-labelled one was tracked by another keypoint's feature — silent, and only where some point is unlabelled (asymmetric schemas, a keypoint not yet reached).

**The fit is state, and that is the whole UI difference.** ~2 min per 500 steps on a 3080, so it is reused: `fill` refits only when `signature` mismatches, the dialog keeps the backend alive across fills (`_refined_backend`, rebuilt only when the checkpoint or `n_points` changes — hence `disagreement_px` is a settable property), and it caches to `<video>.posepal.pt` (`refinement_path`, project data beside the anchors, `weights_only=True`; unreadable = cache miss, not an error, as is a sidecar with no `features` key). A kept fit is cleared and re-loaded when the **video** changes (`_refined_video`) — the signature is made of labels and a copied sidecar can match exactly. `_refinement_signature()` hashes the schema + every anchor, so one more label marks the fit stale — and the **next fill refits by itself**. **There is therefore no fit button** (`_on_refit` existed and was removed): a refit is a fresh fit on all the labels, never a continuation, so a second verb offered only a choice about *when* to wait while reading as one about the result. What cannot be inferred is stated instead — `refinement_method` names the two phases (fit → track) and `_refinement_status_text()` says which the next fill will pay for. `_on_fill` then **discards a cancelled fill** (`cancelled` latched in the progress closure): backends answer a cancel with the spline seed, and applying that would trade a fill the user liked for an interpolation they never asked for — with PosePAL the wait being cancelled is usually the fit, and cancelling it is the only way out. Inside the backend, cancelling restores the previous fit and returns the spline (`fill` must return arrays) — never keep a half-optimised embedding; the dialog is what refuses to apply that fallback. **GPU only**: `available_backends()` hides it on CPU (500 optimisation steps, not one forward pass). `on_stage` renames the progress stage mid-run, since most of the wait is the fit.

**CoTracker is CC-BY-NC — never in `[gui]`, and deliberately no `[co-tracker]` extra**: no PyPI release, no declared dependencies (not even torch), and PyPI rejects direct URL references, so neither half of the requirement fits in an extra. One command, `pose_fill.COTRACKER_INSTALL_HINT` (single source for the GUI hint, docs and pyproject comment); `--torch-backend=auto` is not optional (PyPI's Windows torch wheels are CPU-only); the commit is pinned. The ~97 MB weights auto-download on first fill (`.part` + rename), so missing weights are a note, not an unavailability. **Never pass `checkpoint=None` to `CoTrackerPredictor`** — an unloaded network returning confident nonsense with no error. `COTRACKER_CHECKPOINT_URL` is pinned to the commit's architecture; better animal weights are a **drop-in state dict** chosen via `build_backend(checkpoint=…)` / the "Model weights" row (`app_state.labelling_cotracker_checkpoint`, global scope, empty = stock), never by editing the URL — a different architecture is a new backend. That row shows for `POSEPAL_BACKEND` only, `disagreement_row` for `_TRACKING_BACKENDS`; both via `_refresh_backend_rows()`.

#### Canvas editing (`pose_edit_mixin.py`)

`KeypointLabelMode` attaches to a `CameraView`, which gained `scene()`, `screen_to_image()` (pygfx unprojection, correct under pan/zoom), `image_units_per_pixel()`, `set_label_mode()` and `key_target()`. While attached, left-drag labels and panning moves to Shift+left-drag.

- **Three visual channels: shape = individual, colour = keypoint, fill = provenance.** `MARKER_SHAPES` / `marker_for_individual()` / `glyph_for_individual()` / `keypoint_colors()` are the single source for the canvas and the dialog tree alike — colouring both axes the same way cannot show which beak belongs to which animal. Labels are solid, predictions the same marker drawn **hollow** (`_FILL_INTERIOR` alpha 0 + `edge_color_mode="vertex"`), so the pixels being judged stay visible. Shape is a **material** property, so `AnchorOverlay` holds two layers *per individual* (`_layers`, `_fill_layers`) fed by `store.positions()` + `store.human_mask()`; never collapse them into one object. Inactive individuals dim; the active point's white outline is a **transparent-fill `circle` with a hairline edge**, never `ring` (its fixed-thickness donut swamps the keypoint).
- **There is NO separate edit mode**: clicking an existing point always selects and drags it, `Ctrl+Z` undoes — correcting must never require switching mode first. **Filled points are grabbable** (`nearest(..., include_fill=True)`); grabbing one pins it where the backend put it (`_drag_recorded`, so a drag is one undo step). `on_released` fires once per pointer release so the dialog rebuilds the pose override then (`mode.dragging` gates the mid-drag path) — per-mouse-move rebuilds are not viable.
- The two modes differ only in what follows a placement: `SEQUENTIAL_MODE` advances to the **first** keypoint this individual lacks on this frame (schema order = the points table's column order, so it fills from the left) and **never navigates**; `LOOP_MODE` keeps the keypoint and calls `on_advance_frame`, the dialog owning navigation. Armed by their buttons only, no shortcuts.
- **"Lock" suspends the pointer, not the mode** (`set_locked()`, `CameraView.set_label_locked()` → `_bind_pan_to_shift()`): left-drag pans again and clicks place/grab/pin nothing, while the overlay, active pair and armed mode survive; `_attach_mode` re-reads the tick so a schema-driven restart keeps it. Keyboard editing is deliberately untouched.

#### Keys

The dialog's filter (`_install_key_filter`) sits on **three** objects: the dialog, the video canvas (owned by the main window) and **the main window itself** — events propagate up to it from any widget inside, and without that target the keys pressed while looking at the video did nothing or hit a global shortcut. `_owned_key(event, main_window=...)` claims only `Backspace`/`Delete`, `Ctrl+Z`, `Shift+H` and `N` for main-window events, and checks `event.type()` first since every main-window event now passes through. Installed for the dialog's **whole lifetime** (`__init__` → `closeEvent`).

| Key | Mode? | Behaviour |
|---|---|---|
| `Backspace`/`Delete` | no | with a mode, `delete_selected()` = the **active** point, falling back to `delete_under_cursor()` when the active pair is unlabelled here; without one, the tree's selected pair on the current frame |
| `Ctrl+Z` | no | undo; the main window binds neither this nor Backspace |
| `Shift+H` | no | `_approve_frame`: `promote_fill(frame)` for **every individual at once**, then `_advance_frame()`. The frame is `_current_frame()` — the playhead, which a table click moves; approving something not on screen is what a review action must not do. Promoting zero still advances; a frame with neither fill nor labels warns and stays |
| `Tab`/`Shift+Tab` | yes | cycle keypoints |
| `1`–`9` | yes | select individual — claimed from the main window's behaviour labels via `ShortcutOverride` |
| `N` | no | `_next_suggestion()` — the next suggested frame, wrapping. **One direction only**: the suggestions are a queue to work down, and any frame is one click away in the points table, so a "previous" key bought nothing. Both arrow keys, plain and Shift, now belong entirely to the main window |

`_typing()` hands Backspace, `Shift+H` and `N` back to a focused spin box or line edit. The no-mode keys reach across from the main window because they are pressed right after clicking the video.

- **`Tab`, `Shift+H` and `N` must be `QShortcut`s** (`_bind_shortcuts()`, `Qt.WindowShortcut`), because the item views eat them before a dialog-level filter can see them: Qt turns Tab into focus navigation inside the focus widget, and `QAbstractItemView` turns any **printable** key into a type-ahead keyboard search and accepts it — so `Shift+H` did nothing whenever the tree or the table had focus, which is exactly where you are when picking the frame to review. `eventFilter` must **decline** their `ShortcutOverride` (all of `Key_Tab`, `Key_Backtab`, `Key_H`, `Key_N`), which would otherwise suppress our own shortcut; the `KeyPress` branch still handles them for the canvas and the main window, where a window-context shortcut cannot fire. Exactly one of the two paths acts per press (`test_shift_h_approves_once_per_press`, `test_n_steps_once_per_press_with_the_table_focused` — a double fire skips a frame). Tests must use `QTest.keyClick` on the real focus widget (`sendEvent` bypasses the shortcut map): `test_tab_cycles_when_the_tree_has_focus`, `test_shift_h_works_when_the_tree_has_focus`.
- **The canvas filter target is `CameraView.key_target()`, never `canvas_widget()`**: `RenderCanvas` is a wrapper whose inner `QRenderWidget.keyPressEvent` neither ignores the event nor calls super, so keys pressed over the video die there. `key_target()` returns the first focusable descendant (the wrapper when there is none, e.g. static images); `canvas_widget()` still means the laid-out widget, which `widgets_labels.py` parents an overlay to. Re-installed on `video_path_changed` (deferred a tick). Test: `test_backspace_works_when_the_render_widget_has_focus`, whose fake view nests a key-swallowing render widget.

#### The dialog (`dialog_pose_labelling.py`)

A `QTabWidget`, one tab per stage: "Keypoints" (schema), "Label" (modes + points table + suggestions), "Fill and export". Arming a mode jumps to the Label tab; disarming leaves the tab alone.

- **The dialog is the hierarchy**: one `QTreeWidget` (branch per individual, leaf per keypoint *of that individual*, per-frame `●`/`·`, `k/n` count) — a leaf's row is NOT the schema index, always resolve via its `(individual, keypoint)` `UserRole`. Schema edits **all** go through `_apply_schema(keypoints=, individuals=, shared=, individual_keypoints=)`. No flat keypoint list and no "seed from loaded pose" button (a fresh store still seeds from `app_state.keypoints`). Unshared keypoints → Add/Remove act on the selected individual only.
- **Label tab header is one compact row** — `[Sequential] [Loop] [Lock] [individual ▼] [keypoint ▼]` — over a status chip showing the marker the next click drops (glyph + keypoint name in the keypoint's colour; individuals are shape-coded, so an individual colour would teach a mapping the canvas lacks), or "Locked — clicks pan the video". Keypoint picker: **Loop only**. **"Then go to:"** row: Loop mode **or once a fill exists**, since it also says where `Shift+H` lands. **"Approve frame"**: on `store.filled is not None` alone (`_refresh_approve_button`), no mode needed. The combos are inputs *and* readouts, so `_refresh_target_combos()` syncs them under `_blocked()` whenever a key, Tab or a click moves the target. **"Then go to" is explicit, not inferred** (`_AFTER_CLICK_CHOICES` → `AFTER_CLICK_FRAME` / `_SUGGESTION` / `_STAY`, read by `_advance_frame`); there is no `+step` spin box.
- **Otherwise the tab carries no status text** — the armed mode shows in the buttons and the active pair in the tree; tooltips carry the detail.

#### Points table

A `QTableView` over a virtual model (`PointTableModel`), NOT a `QTableWidget` — a fill gives a row per `(frame, individual)` for every frame it covers, and a model reading `store.positions()` on demand cannot disagree with the store. Rows pivot **wide**: `Frame | Individual | Source | Confidence` then an `x`/`y` pair per keypoint, so a whole frame is visible at once.

- **Layout**: before a fill, rows are the labelled `(frame, individual)` and columns only keypoints with ≥1 label; after, every frame of `store.fill_range` and every keypoint (never the whole video). `_layout_signature()` gates rebuilds carrying `fill_range` as two numbers standing in for the dense row set — never enumerate it on a drag; otherwise a refresh repaints one frame (`refresh_frame`, a frame's rows are contiguous) or everything (`refresh_all`).
- **Provenance shows twice**: the `Source` cell (`Human`/`Fill`, by the row rule) and dimmed `x`/`y` (`QPalette.Disabled`) for filled values. A row stripped of both shows an **empty** `Source`, never "Fill".
- **`Confidence`** = `nanmean` of the row (as in `frame_confidence` — never `min`, an absent point sits at NaN forever), numerically filterable, worst keypoint in the cell tooltip, whole scoring scheme in the header tooltip. Human points show `1.00` even against an older array: `store.confidence` is a snapshot of the last fill and `set_point` deliberately does not write into it.
- Right-click offers up to three actions, **each only when the selection holds something for it**: "Delete labels on N frames" (`clear_individual`) for human rows; "Delete filled points" (`clear_fill_for`, keeps the labels — for occluded frames the backend guessed at) and "Pin filled points as labels" (`promote_fill`) for rows with fill.
- The header is **two rows** (`PairedHeaderView` ⊂ `FilterHeaderView` — Qt has no multi-level header): the keypoint name in its colour spans its pair, `x`/`y` beneath. Both sections paint the *same* name across their union rect (idempotent, so it survives a partial repaint); `DisplayRole` is empty (tooltips carry "beak x") and `sectionSizeFromContents` reserves half the name per column plus `FILTER_ZONE_W` where filterable. `setResizeContentsPrecision(50)` — ResizeToContents measures rows and there can be 100k.
- Scrolls inside `TABLE_MAX_HEIGHT`. A click seeks the playhead and activates that keypoint (in the fixed columns, just the individual); the current frame's row is selected and scrolled to. Use the `clicked` signal, never the selection signal — the table also selects itself when the playhead moves. Selection and lookups go **through the proxy** (`mapToSource`/`mapFromSource`). A narrow repaint targets `_current_frame()`, and `store.undo()` **returns the frame it changed** so an undo landing elsewhere repaints the right row.
- **Column filters live in `gui/table_filter.py`** (`MultiColumnFilterProxy`, `CategoryFilterDialog`, `NumericFilterDialog`, `FilterHeaderView`, `SORT_ROLE`), shared with the Kilosort cluster table: a funnel in a reserved zone per filterable header section, criteria ANDed, numeric comparisons on `SORT_ROLE` (never the formatted text). An empty allowed-set means *no* filter, never "hide everything". The proxy reads through `QModelIndex`, so item and virtual models both work; ephys subclasses it as `_ChannelFilterProxy`. The points table filters `Individual`, `Source`, `Confidence` — **not `Frame`** (already the row order, and suggestion navigation covers it). `NumericFilterDialog(..., default_op=)` opens `<=` for `Confidence`, `>=` elsewhere. Schema edits clear every filter.

#### Which frames to label (`pose_suggest.py`)

Labelling consecutive frames is near-wasted effort. `suggest_frames(method, count, n_frames, frames, exclude, min_gap)`: `uniform` (no decode), `motion` (mean |diff| from the previous frame, the same signal as `extract_video_motion`), `diverse` (DeepLabCut's k-means over grayscale thumbnails, the frame nearest each centroid) and `uncertain`. Candidates are strided to `MAX_CANDIDATES`, and **every method passes through `enforce_min_gap()`** — a threshold method returns a run of neighbours from one fast bout, so keep the best per neighbourhood instead. `min_gap` derives from `n_frames / count` (never a hardcoded frame or second count). Already-labelled frames are excluded.

**`uncertain` is the analogue of SLEAP's `prediction_score`** and closes the label → fill → correct-worst → fill loop: it ranks by `store.confidence` and needs no decode. Listed **last**, and `_build_suggest_group` opens on it only when `store.confidence` exists, so the first press is never a bare warning. It is the right criterion because no detector is trained here — CoTracker takes ONE query frame per point, so extra labels mostly reset drift and the useful anchors are where tracking *fails* (occlusion, blur), not where images look diverse; `diverse` earns its place for PosePAL alone, which does fit an embedding to the labels. `frame_confidence()` reduces `(frames, individuals, keypoints)` with **nanmean, not min** (an absent point would pin every frame), written as sum/count rather than `np.nanmean` because frames outside the span are all-NaN and the warnings are normal. Those frames score NaN and `suggest_uncertain` **drops them** (`argsort` would rank NaN best): a NaN is no prediction, not a bad one, and the unlabelled tail is usually most of the video.

**Count is asked as a share of the video** (`suggest_percent_spin`, floor `MIN_SUGGEST_PERCENT`, resolved by `_suggest_count()` beside a "40 of 200 frames" label) — an absolute count means different things at 200 and 60k frames. Default `RECOMMENDED_LABEL_SHARE` = **10%, roughly every 10th frame**: a *spacing*, since what the backends bridge is a gap measured in frames; the figure follows the density CoTracker3 is evaluated at (Pan et al. 2025: 6 of 60). "Next suggested frame" (and `N`) seeks via `app_state.video.seek_to_frame()`.

#### Getting results out

- **y-flip on the plot-bound paths only.** Anchors are image coordinates (y down from the top-left), plots are y-up, so an unflipped trajectory renders mirrored in the generic `plots_space.py`. `store_to_movement_ds(store, fps, image_height=...)` applies `y_out = height - y` behind the **default-on** "Flip y" checkbox, for *Load into the GUI* and the NetCDF export **only** — `_push_pose_override` stays in image coordinates, since the overlay does its own `y_world = img_height - y`. Kinematics are computed after the flip.
- **"Load into the GUI"** (`store_to_kinematics` + `DataWidget.add_keypoint_features`) merges the keypoints and the ticked kinematics into the **current trial** as ordinary features (`keypoints_position`, `keypoints_speed`, …), filled frames included. The time axis is renamed to `FEATURE_TIME_DIM` (`time_keypoints`): keypoints run at video fps, and merging under the trial's `time` would outer-join the two and pad every other feature with NaN. `add_keypoint_features` drops the previous load's vars *and dims* first, then follows `dt.update_trial` → `app_state.ds` → `store.update_ds` → `_register_feature`. **No existing dataset required**: the keypoints *are* one (frame index × 1/fps). With no xarray behind the session (common after dropping a bare video, which loads via pynapple), `_install_keypoint_dataset()` builds a single-trial `TrialTree` plus a fresh catalog + `XarrayLoader` — only when nothing else is serving features, so it cannot strand another source's panels; a pynapple session that *does* have features is told to export instead.
- **Fill results render through the normal pose overlay** — `set_pose_override(pr)` swaps in an in-memory `PoseRenderData`, so the confidence spinbox filters filled points for free; `movement_ds_to_pose_render()` inverts `pose_render_to_movement_ds()`. **But not while a mode is armed**: the anchor overlay draws every point itself, so `_push_pose_override()` clears the override instead (tracked by `_override_pushed`, restored on `_detach_mode`), and `_on_fill` must call `_mode.refresh()`.

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
