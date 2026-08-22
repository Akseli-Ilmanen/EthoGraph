# Video features: what is done, and what phases 4–5 still need

Status note, 2026-08-22. Phases 1–3 of the `ethograph/video_features` refactor are
implemented and tested; phases 4–5 are **not**, and are described here as
requirements, not decisions.

> **Caveat for whoever picks this up.** Phases 4–5 are about how the dataset build,
> the model config and the user-facing functions are shaped. Those choices depend on
> how the owner wants the config side of EthoGraph to look (one config object? the
> existing `model_config_*.py` scripts? something driven from the GUI?) and on which
> functions are meant to be public. An agent with that context should make those
> calls; nothing below should be read as settled design.

## What phases 1–3 delivered

| Piece | Where | What it guarantees |
|---|---|---|
| Config in seconds | `video_features/plan.py: S3DConfig` | `analysis_fps`, `stack_s`, `mode`, `truncate_at`, `batch`, `chunk`, `precision`. No frame counts anywhere in user-facing config. |
| Plan per video | `plan_s3d(video_fps, cfg) → S3DPlan` | `step` (only ever skips frames), odd `stack_frames ≥ MIN_STACK` (13). Refuses, naming the shortest `stack_s` that works, instead of clamping. |
| Rate from the file | `io/video_probe.py: probe_video` | Moved out of the GUI so extractors and GUI read the same probe; `gui/video_manager.py` re-exports it. |
| Streaming decode | `video_features/frames.py` | PyAV, every `step`-th frame, chunked; nothing reads a video whole. |
| Windows mode (default) | `extract.py: window_features` | Today's sliding-stack scheme, batched through `unfold` with a rolling carry; identical maths (bit-exact in `precision="fp32"`). |
| Dense mode (ablation) | `extract.py: dense_positions`, `dense_to_frames`; `s3d.py: S3D_STAGES`, `truncated_base` | Trunk once over the video; stage geometry (stride / offset / receptive field) encoded for `Mixed_3c`, `Mixed_4f`, `Mixed_5c`. |
| Output | `extract_s3d() → xr.DataArray (time_s3d, s3d_dims)` | Time in seconds of the video clock at the effective rate; the plan and mode in `attrs`. |
| CLI | `scripts/model/s3d_features.py` | `VIDEO... --out DIR [--analysis-fps] [--stack-s] [--mode] [--truncate-at] [--legacy-npy]` → `{stem}_s3d.nc` (+ `.npy`). |
| Tests | `tests/test_unit/test_s3d_plan.py`, `test_s3d_extract.py` | Plan table + refusal; window batching/chunking/padding equivalence; dense geometry; end-to-end on a synthetic clip. |

Removed: `base_extractor.py`, `utils.py`, `transforms.py`, `s3d.yml`, `extract_s3d.py`
(the `video_features` fork baggage), and the `omegaconf` dependency.

## Phase 4 — dataset side (requirements)

1. **Attach S3D to a trial on the trial's own grid.** Today `examples/create_dataset_*.ipynb`
   does `np.load("{stem}_s3d.npy")` and assumes one row per video frame. With the
   `.nc` output it becomes `xr.open_dataarray(path).interp(time_s3d=ds.time)` after
   applying the video stream offset from the alignment (`stream_offset_for_trial`).
   Interpolating to 200 Hz is information-neutral (the feature is a temporal average
   already); the `--legacy-npy` flag exists only to keep the current notebook working.
   *Open question:* does this live in a public helper (`eto.attach_s3d(ds, path)`?), in
   the dataset-creation notebook, or in `model/dataset.py` at training time?
2. **Cache keyed by plan.** Re-extraction should be skipped when an existing `.nc` carries
   the same `attrs` (plan + checkpoint + mode). The CLI currently only checks existence.
3. **Keypoint-free mode.** `model/dataset.py: extract_features_per_trial` assumes a
   `keypoint` dim via `feat_kwargs` / `cp_kwargs`, and `get_trial_dict` filters inference
   trials on a hardcoded keypoint. For "video + accelerometer only": make the kinematic
   block optional (select only dims a variable has — the catalog's `select()` already
   does this), let changepoints target any signal (accelerometer magnitude, or
   `features/movement.py: extract_video_motion`), and filter trials on frame count /
   non-NaN samples. The existing ablation `condition` strings
   (`no_kinematic`, `no_s3d`, …) should probably become a `feature_sources` list.
4. **`good_s3d_feats`** (hand-picked S3D columns from `find_best_s3d.ipynb`) stays a
   dataset-side choice; nothing in phases 1–3 touches it.

## Phase 5 — config, user functions, docs (requirements)

1. **Config surface.** `S3DConfig` is a dataclass; how it is exposed (CLI flags only, a
   section of the model JSON, a GUI dialog under Tools) is the owner's call. The one
   rule to keep: *no frame counts or rates in config* — `plan_s3d` derives them.
2. **`scripts/model/model_config_*.py`** still carry `"fps": 200` and hardcoded personal
   paths; once S3D is on its own time axis that key has no reader and should go.
3. **Docs page** (`docs/source/advanced/video_features.md`, not written): the knobs in
   seconds; the ≥ 13-frame constraint and what it means at 30 fps (shortest window
   0.43 s); why interpolation to the trial rate is fair; the ablation recipe
   (windows vs dense vs `truncate_at`, `analysis_fps` None vs 25). `installation.md`
   already no longer mentions `omegaconf`.
4. **Decide on `scripts/model/s3d_features.py` vs `python -m ethograph.video_features`**
   (an `__main__.py`) vs an `eto` CLI subcommand — depends on how the other model
   scripts are meant to be invoked.

## Ablations the new code makes cheap (not run)

- `mode="windows"` (default) vs `mode="dense"` vs `dense + truncate_at="Mixed_3c"`.
- `analysis_fps=None` vs `25` on 200 fps footage (matches Kinetics' temporal statistics,
  but forces `stack_s ≥ 0.52 s`).
- `precision="fp16"` vs `"fp32"` — expected to be identical in downstream metrics.
