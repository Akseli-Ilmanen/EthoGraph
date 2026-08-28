# Config reference

One YAML file per project. Relative paths resolve against the file's own
folder. `base: other.yaml` deep-merges this file over another.

Any key can be overridden without editing the file, with the same dotted
spelling the YAML has — passed to `Project(...)` or accumulated with
`update()`. Values are parsed as YAML, so `train.augment.stretch=[0.8,1.2]`
works:

```python
import ethograph as eto

project = eto.segment.Project("project.yaml", "model.architecture=mstcn")
project.update("train.run_name=mstcn", "train.augment.stretch=[0.8,1.2]")
```

```{important}
An unknown key is an error, in the file and in an override alike — a typo
must not silently become a default. The message names the valid keys of the
section it failed in.
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `root` | the config's folder | Project directory: `data/` and `runs/` live here. |
| `sessions` | required | List of sessions: `{source, labels_path, video_dir}`. |
| `trials.where` | `{}` | Metadata column → allowed values. The one trial filter; applied in every stage. |

```{important}
A session has **no role**. Every session you list is material for the model,
and what a trial is *used for* comes from `train.split` — three ratios, drawn
by whole trial. Holding a whole session out is a cross-validation fold, which
`project.cross_validate()` writes per fold (`train.split.holdout_sessions`);
it is not something you write per session. See `train.split` below.
```

```{important}
Every session names its own `labels_path` explicitly — there is no
`{stem}_labels.tsv` sidecar guess. `video_dir`, when the session has video, is
the one folder searched for it (no project-level list to fall through).
```

## `features`

| Key | Default | Meaning |
|---|---|---|
| `name` | `default` | Materialised dataset name → `{root}/data/{name}`. |
| `columns` | required | `feature → dim → values`. Every dim of a feature must be pinned except the individual dim (pinned per sample). A second individual dim is spelled `other: "*"` — the remaining individuals in dataset order. |
| `sin_cos` | `[]` | Features in `columns` that are **angles** — see below. |
| `individuals` | dataset's individual coord | Which individuals become samples. Required when the dataset has no individual dim but the labels name individuals. |
| `labels.mapping` | required | `mapping.txt` (`id name [branch] [event_type]`). |
| `labels.branch` | `0` | The one branch this model predicts. |
| `labels.classes` | all state classes of the branch | Subset of label ids to predict. |

### `features.sin_cos`

An angle read as a plain number lies about its own geometry: 359° and 1° are
two degrees apart and the column says they are the furthest apart it ever
gets, and no amount of z-scoring repairs that jump. Name the feature here and
each of its columns is replaced by the two components of its angle:

```yaml
features:
  columns:
    angles: {keypoint: [beakTip, stickTip]}
  sin_cos: [angles]
```

gives `angles|keypoint=beakTip|sin`, `angles|keypoint=beakTip|cos`, and the
same pair for `stickTip` — the raw column is gone, not supplemented. The
units are the variable's own `units` attr (`rad` / `deg`, either spelling,
which is what {mod}`ethograph.features.geometry` writes); a variable that
declares none has them read off its values, logged at INFO, since a full turn
is 6.28 one way and 360 the other. A `units` that is not angular at all is an
error — it says the feature is not an angle.

The components live in `[-1, 1]` and mean what they say there, so they are
never z-scored or percentile-clipped, exactly like a column carrying
`attrs["normalise"] = 0`. Naming a feature that `columns` does not select is
an error.

### `features.changepoint_features`

Optional. Expands named raw changepoint masks into
{func}`~ethograph.features.changepoints.more_changepoint_features` once per
session, at `materialise`/`infer` time, and **merges the generated columns
straight into `features.columns`** — you never spell out a name like
`speed_troughs_cp_sigma2_weighted` yourself. See {doc}`../../api/changepoints`
and `examples/segment_changepoint_features.ipynb` for what each output column
looks like. This is the one exception to "features are built with the
session, never by the pipeline": it is a deterministic expansion of a mask
already in the file, not a new modelling choice.

| Key | Default | Meaning |
|---|---|---|
| `sigmas` | required | Kernel widths (in samples) for the Laplacian/Gaussian proximity curves, e.g. `[2.0, 3.0, 5.0]`. |
| `distribution` | `laplacian` | `laplacian` or `gaussian`. |
| `inputs` | required | `feature → dim → values` — which raw changepoint masks to expand, dims pinned the same way as `features.columns` (the individual dim is still pinned per sample). |
| `transforms` | all four | Subset of `binary`, `proximity`, `proximity_weighted`, `segment_id` — which of `more_changepoint_features`'s column groups to keep. `binary` duplicates the raw mask (just marked `normalise=0`), so drop it if you already select the mask itself. |
| `merge` | `false` | OR every mask named in `inputs` into one `changepoints` mask before expanding, so the whole section costs **one** block of columns instead of one per mask × pinned dim. All merged masks must share a `target_feature`. |

Use `merge: true` when what matters is *that* something changed, not which
detector noticed. Without it you get one block of columns per mask per
keypoint, which can easily outnumber your kinematic features; with it you get
a single block named `changepoints_cp_*`. Each animal keeps its own
changepoints either way.

Xarray sessions only — pynapple changepoints are event times, not a dense
mask, so a pynapple session with this set raises immediately.

```yaml
features:
  changepoint_features:
    sigmas: [2.0, 3.0, 5.0]
    transforms: [proximity, proximity_weighted, segment_id]
    inputs:
      speed_troughs: {keypoint: [beakTip, stickTip]}
      speed_turning_points: {keypoint: [beakTip, stickTip]}
```

This generates every `speed_troughs_cp_sigma*`/`speed_turning_points_cp_sigma*`
(etc.) column for both keypoints and merges them into `features.columns` —
naming `speed_troughs`/`speed_turning_points` there too, or under `inputs`
again elsewhere, is a config error (`config.features.columns already names
[...], which config.features.changepoint_features also generates`).

The generated columns are already in `[0, 1]`, so `preprocess` leaves them
alone — you do not need a `zscore_exclude` entry for any of them. The one
exception is the raw mask itself (`speed_troughs`): if you select it directly
in `features.columns`, rather than taking its `_cp_binary` twin from here, add
it to `zscore_exclude`.

### `features.preprocess`

All five keys live under `features.preprocess` in the YAML, but they run at
two different stages — the first four bake into the materialised `.npy`
files at `materialise` time; `zscore`/`zscore_exclude` are deferred to
`train`/`infer`, because mean/std can only be computed once a train split
exists. **If you inspect `data/{name}/features/*.npy` directly, expect it to
be un-z-scored** — that normalisation happens later, per run, and lands in
`runs/{run}/stats.npz` (see below).

Baked in at `materialise` (session-level, order below):

| Key | Default | Meaning |
|---|---|---|
| `likelihood_threshold` | `null` | Keypoint columns whose `likelihood_feature` is below this become NaN. |
| `likelihood_feature` | `confidence` | The per-keypoint confidence feature. |
| `interpolate` | `true` | Linear interpolation over NaNs. |
| `clip_percentiles` | `[2, 98]` | Pull outliers in to this percentile range (`null` = off). Columns already on a fixed scale — unit vectors, angles, flags, changepoint features — are left alone. |

Applied at `train`/`infer` (run-level, not materialise):

| Key | Default | Meaning |
|---|---|---|
| `zscore` | `true` | Z-score each column, using statistics from the training trials only. The same statistics are reused at inference (`runs/{run}/stats.npz`). Columns already on a fixed scale are left alone. |
| `zscore_exclude` | `[]` | Extra feature names to leave un-z-scored, on top of those detected automatically. |

## `model`

| Key | Default | Meaning |
|---|---|---|
| `architecture` | `c2f_tcn` | Which network to train. `eto.segment.architectures()` lists the names. |
| `params` | `{}` | Change individual hyperparameters of that network. Keys you leave out keep their default. |

```yaml
model:
  architecture: mstcn
  params: {num_f_maps: 64}     # only this one changes; the rest keep their defaults
```

### What you can put in `params`

Anything the architecture accepts. To see the full list of keys for a given
architecture, with a comment on each and its default value, open its file:

```
ethograph/segment/dlc2action/config/model/{architecture}.yaml
```

One name differs from its file: `mstcn` reads `ms_tcn3.yaml`. Every other
architecture matches.

The two skeleton-graph architectures (`specscalpel`, `lady`) are the
exception to "params are architecture hyperparameters only": their `params`
also carry the **joint layout** — `keypoints` (the ordered keypoint names) and
`skeleton` (a skeleton-config YAML, an ndx-pose `.nwb`, or `[a, b]` pairs) — and,
for `lady`, the root-frame landmarks `root`/`spine`/`left`/`right`. These are
structural, not tunable, so `eto.segment.tunable_params(name)` lists only the
network numbers. Their defaults live in
`ethograph/segment/{specscalpel,lady}/config/defaults.yaml`.

An unknown key is an error naming the valid ones, so a typo cannot silently do
nothing — and it is raised *before* training starts, not by the constructor
half-way into a search.

Or ask, which is what a script sweeping several architectures wants:

```python
for name in eto.segment.architectures():
    print(name, eto.segment.tunable_params(name))
```

```{important}
The architectures share almost no hyperparameter names — `mlp` takes
`f_maps_list`, `mstcn` takes `num_f_maps`, `edtcn` takes `kernel_size`. So
`model.params` and any `search.params` entry under `model.params.*` are
**per architecture**: a sweep needs one search space each, not one shared
space. See {doc}`index` for the loop.
```

For what each architecture is good at, see {doc}`index`.

## `train`

| Key | Default | Meaning |
|---|---|---|
| `run_name` | `{architecture}_{features.name}` | Base run name. Each `train()` call gets its own directory, `runs/{run_name}_{YYYYmmdd-HHMM}/` — never overwrites a previous run. |
| `epochs` | `50` | How long to train. Every run trains its full budget. |
| `batch_size` | `1` | Trials per step. One whole trial at a time is the tested setting; raising it pads every trial in the batch out to the longest, which costs memory and changes what the C2F models' BatchNorm sees. |
| `learning_rate` | `1e-3` | Adam, held constant for the whole run. |
| `weight_decay` | `0` | Adam weight decay. |
| `grad_clip` | `1.0` | Clip the gradient norm to this. `0` turns clipping off. |
| `eval_every` | `5` | Score the validation trials every N epochs. Lower it to place the best checkpoint more precisely, at the cost of time. |
| `select_on` | `f1@50` | Which validation metric decides the kept checkpoint: `acc`, `edit`, `frame_f1`, `f1@50`, `f1@75`, `f1@90`. Pick the one that matches what you need from the model — `f1@50` for "did it find the behaviour", `f1@90` for "are the boundaries right", `edit` for "is the sequence of behaviours right". |
| `f1_thresholds` | `[0.5, 0.75, 0.9]` | IoU thresholds of the segmental F1 scores. |
| `seed`, `device` | `0`, auto | `device` = `cuda`, `mps`, `cpu`; auto picks the best available. |
| `drop_kinds` | `[]` | Feature categories to leave out of this run — the ablation axis (see {doc}`../variable_schema`). `[video_feature]` trains the same model without S3D. Applied to the materialised dataset's columns, so an ablation costs a run rather than a re-materialisation; columns whose `kind` is undeclared are always kept. |
| `frame_weight` | `1.0` | Weight of `train.loss` in the total. `0` leaves `train.circle` as the only thing training. |
| `subsample` | `1` | Train and predict at `fs / subsample` — the temporal-resolution axis, run-level like `drop_kinds`, so one materialised dataset serves every rate. Every frame count the run reports (its metrics, its `_probs.npz`) is then in *its* frames, so runs at different rates are only comparable once their predictions are scored back on one grid (`scripts/experiment2_smoothing.py` does that). Striding, with no anti-alias filter. |

### Losses

The total objective is a sum of up to two terms, each independently
switched on by its own weight — `Objective` in
{mod}`ethograph.segment.losses` computes and itemises them, and every
weighted term's value lands in `metrics.tsv`/the console log by the name
below, so a loss that stops moving can be traced to the term that stopped
moving.

| Term | Weight key | Default | Config section | Needs |
|---|---|---|---|---|
| frame (CE + consistency) | `train.frame_weight` | `1.0` | `train.loss` | any architecture |
| circle (metric-learning) | `train.circle.weight` | `0` | `train.circle` | any architecture |

`frame_weight: 0` with `circle.weight` left at `0` is a `ValueError`
("nothing to train on") — at least one term must be active.

### `train.loss`

Cross-entropy per frame, plus a consistency term that penalises the
prediction changing from one frame to the next — that second term is what
stops the output flickering between classes mid-behaviour.

| Key | Default | Meaning |
|---|---|---|
| `alpha` | `0.001` | Weight of the consistency term. Raise it if predictions flicker; lower it if short behaviours are being swallowed by their neighbours. The default is DLC2Action's, at which the term barely registers; MS-TCN's published value is `0.15`, which is what the earlier CETNet training script used and what `scripts/bench.py` pins when it asks whether the term helps at all. |
| `tau` | `4` | How large a frame-to-frame jump in log-probability that term still penalises; beyond `tau` it is truncated, so a genuine class change is not punished without limit. Ours, not a key of a config file upstream: DLC2Action writes MS-TCN's `tau` of 4 into the arithmetic as `clamp(..., max=16)`. Both it and `alpha` were tuned in the literature at 15–30 fps, so at a high sampling rate they are worth re-tuning together — that is what `scripts/experiment2_smoothing.py` sweeps. |
| `focal` | `true` | Focus the loss on frames the model still gets wrong, instead of ones it already has right. |
| `gamma` | `2` | How sharply `focal` does that. Higher = more focus on hard frames. No effect when `focal: false`. |
| `weights` | `null` | Per-class multipliers on the cross-entropy, as a list one entry per class (background first). `null` treats every class alike.
| `hard_negative_weight` | `1` | Multi-label only; no effect at the default `exclusive: true`. |
| `exclusive` | `true` | One class per frame. `false` (multi-label sigmoid) is not wired up — the targets and metrics assume one class. |

```yaml
train:
  loss: {alpha: 0.01}          # only this one changes; the rest keep their defaults
```

An unknown key is an error naming the valid ones. The defaults, with a comment
on each, live in `ethograph/segment/dlc2action/config/losses.yaml`.


**TODO**: See if inverse_frequency weights loss is detrimental

### `train.circle`

A deep metric-learning term over the finest-stage logits (Sun et al. 2020,
circle loss) — pulls same-class frames' logit vectors together and pushes
different-class ones apart, independent of the frame cross-entropy above.
Architecture-agnostic — every registered model produces logits.

| Key | Default | Meaning |
|---|---|---|
| `weight` | `0` | Weight of the circle loss in the total. `0` leaves it untrained — this is the default, so circle loss is off unless you set it. |
| `m` | `0.25` | Margin: how far inside the unit circle a pair may sit before it counts against the loss. |
| `gamma` | `128` | Scale applied to the (margin-weighted) similarities before the softplus. |
| `max_frames` | `2048` | Subsample the batch's frames to at most this many before forming pairs (`O(n^2)` pairs otherwise); `null` = no cap. |

```yaml
train:
  circle: {weight: 0.001}   # the weighting it was ported with; 0 (the default) switches it off
```

#### Choosing a weight

Start at **`0.001`**, with `m` and `gamma` at their defaults. The term is a
softplus of a log-sum-exp over every same-class and different-class pair,
scaled by `gamma` = 128, so its raw value runs to tens where the frame
cross-entropy sits below 1 — the weight is what brings the two onto one
scale, and `0.001` is exactly the weighting the CETNet training script this
was ported from used (`0.001 * CircleLoss(m=0.25, gamma=128)`). One
difference to keep in mind: that script applied it to the encoder's feature
map, whereas here it reads the class logits, a vector only as wide as the
number of classes, so the same weight is a starting point rather than a
tuned answer. Whether the term earns its place is what `scripts/bench.py`
measures — every architecture, with and without it, cross-validated per
individual.

### `train.augment`

| Key | Default | Meaning |
|---|---|---|
| `noise_std` | `0` | Gaussian noise, as a fraction of each column's std (`normalise=0` columns untouched). |
| `stretch` | `null` | Random temporal stretch range, e.g. `[0.8, 1.2]` (labels follow by nearest frame). |
| `mirror` | `false` | Negate the first component of every vector group with probability ½. |
| `rotate_deg` | `0` | Rotate every vector group's (x, y) by a random angle within ±this. |

Vector groups are columns spanning the `space` dim of one vector (position,
velocity, …); the layout records them, so a dataset without coordinates
silently gets no geometric augmentation.

### `train.split`

Three ratios. Your trials — every trial of every session, after
`trials.where` — are pooled, shuffled once and cut into the three roles.

| Role | What it is for |
|---|---|
| **train** | The model learns from these. |
| **val** | The model never learns from these. They decide **which epoch's weights you keep**, and they are the objective a `search` maximises. |
| **test** | Touched exactly once, at the very end, to report a number you can trust. |

| Key | Default | Meaning |
|---|---|---|
| `train_fraction` | `0.6` | Fraction of trials the model learns from. |
| `val_fraction` | `0.2` | Fraction held back to choose settings and checkpoints. `0` turns validation off. |
| `test_fraction` | `0.2` | Fraction read once, at the end. |
| `seed` | `0` | Change it to re-draw the split. |
| `holdout_sessions` | `[]` | Sessions held out *whole* as `test` — a cross-validation fold. Written per fold by `project.cross_validate()`; see below. |

The three fractions must sum to **1**, and an override that breaks that is an
error rather than a silent renormalisation. Splitting is by **whole trial**,
never mid-trial, so no trial ever appears in two roles.

```{warning}
The random split depends on the full list of trials, so **adding a session
reshuffles the existing ones**, same `seed` or not. Two runs across a growing
dataset are therefore not strictly comparable — some of what was training data
in the first run is validation data in the second.

When you need runs to be comparable, pin the split instead of drawing it:
`holdout_sessions` names sessions held out whole, which is exactly what a
cross-validation fold does. Each run records the split it used in
`runs/{run}/splits/*.bundle`, so you can always check after the fact which
trials went where.
```

#### `holdout_sessions` — one fold

Name one or more sessions and **all** of their trials become `test`, whatever
the fractions say; the sessions that remain are split train/val by
`val_fraction` renormalised against `train_fraction`. That is one
leave-one-session-out fold, and `project.cross_validate()` writes it once per
session rather than asking you to. You rarely set this key by hand — it is
documented because you will see it in a fold's `runs/{run}/config.yaml`.

#### What validation actually buys you

Every `eval_every` epochs the run scores the validation trials and writes a row
to `metrics.tsv`. Three things then happen:

- **The best epoch is saved as `best.pt`.** Training a segmentation model past
  its best is normal — it keeps fitting the training trials while getting worse
  on new ones. Validation is what notices, so you keep the good weights instead
  of whatever the last epoch happened to produce. `best.pt` is what
  `project.inference()` uses.
- **You get the metric curve** in `metrics.tsv` — one row per validation —
  which is how you find the right `epochs` for a later run. Each row also
  carries a test readout (raw and post-processed, `test_raw_*`/`test_post_*`)
  computed on the current epoch's weights — a training-time diagnostic only;
  it never influences `best.pt`, which stays keyed on validation.
- **`search` reads the best of that curve** as its objective, and can abandon a
  trial whose curve is already behind the others (`search.prune`).

Runs are never cut short: `epochs` is the budget and every run trains it out.
Set `val_fraction: 0` and you lose all three — `best.pt` becomes the last
epoch, no `metrics.tsv` is written, and a search refuses to start. The log says
so:

```
No validation samples — no metrics curve, and best.pt is the last epoch.
```

#### Choosing values

- **Leave it at 60/20/20 while you are developing**, which is what a search
  needs: a validation set big enough that the score it hands Optuna means
  something, and a test set that stays untouched underneath it.
- **Drop `val_fraction` to `0` once the settings are settled.** That is the
  cross-validation default: the hyperparameters, `epochs` included, came out of
  the search, so every remaining trial is worth training on.
- **Set it to `0` when you have very few labelled trials.** The count is
  rounded, so `0.2` gives you **1** validation trial anywhere between 3 and 7
  trials, and **0** at 2 — a score from one trial is noise, and will pick a
  checkpoint more or less at random. Below roughly 15 trials, prefer a fixed
  `epochs` over a validation set you cannot trust. Check the count in the log:

  ```
  Samples — train: 24, val: 6, test: 8
  ```

Validation trials come out of the pool the model would otherwise learn from, so
raising `val_fraction` gives it less to learn from. `0.2` of a reasonable
number of trials is a sensible balance.

#### Getting a number you can trust

`val` is used repeatedly to make choices — every search trial reads it — so its
score is optimistic: it is the best of many peeks, not an unbiased estimate.
`test` is what you report, and nothing selects on it.

The strongest version is a session the model has never seen at all — a
different recording day or animal — since trials from the same session share
lighting, camera position and the animal's mood, and a random trial split
flatters the model accordingly. That is cross-validation, and it is the second
stage of the workflow rather than a setting: see {doc}`index`.

## `search`

Stage 1 of the workflow: [Optuna](https://optuna.org) over the config, every
trial a full training run scored by `train.select_on` on the **validation**
trials. Run it with `project.search()`.

| Key | Default | Meaning |
|---|---|---|
| `params` | `{}` | dotted config key → search space. Required to search. |
| `n_trials` | `20` | How many configurations to try. `project.search(n_trials=...)` overrides it for one call. |
| `timeout` | `null` | Stop the study after this many seconds, however many trials are left. |
| `name` | derived from the run name | Study name → `{root}/searches/{name}`, and `runs/{name}/trial000_…`. |
| `seed` | `0` | The sampler's seed. |
| `prune` | `true` | Abandon a trial whose validation curve is behind the running median at the same epoch. |
| `keep_weights` | `false` | Keep every trial's `best.pt`/`last.pt`. Off by default — a study is dozens of runs, and only the winner's weights are worth the disk. Each trial's config, split and metrics are kept either way. |

A **search space** is one entry of `params`, keyed by the same dotted path an
override uses, so there is exactly one spelling for a setting:

| Key | Applies to | Meaning |
|---|---|---|
| `type` | all | `float`, `int` or `categorical`. |
| `low`, `high` | float, int | The range, inclusive. |
| `step` | float, int | Quantise the range. |
| `log` | float, int | Sample on a log scale (needs `low > 0`) — the right choice for a learning rate. |
| `choices` | categorical | The list to pick from. |

```yaml
search:
  n_trials: 30
  params:
    train.learning_rate:     {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
    train.loss.alpha:        {type: float, low: 0.0, high: 0.5}
    model.params.num_f_maps: {type: int, low: 32, high: 256, step: 32}
    train.augment.mirror:    {type: categorical, choices: [true, false]}
```

The study is stored in `searches/{name}/study.db`, so calling `search()` again
**adds** trials to it rather than starting over. The winning draw is written to
`searches/{name}/best.yaml` as a config that inherits yours:

```yaml
base: ../../project.yaml
train:
  learning_rate: 0.00043
```

which is what stage 2 reads: `eto.segment.Project(result.config_path)`.

```{important}
A search tunes the **model**, not the feature engineering. The materialised
dataset is built once, before the study starts, and every trial reads it — so
`features.*` keys have no business in `search.params`.
```

## `video_features`

The two settings that change the *features*, plus the camera. In seconds —
frame counts come from each video's own rate. See {doc}`video_features`.

| Key | Default | Meaning |
|---|---|---|
| `stack_s` | `0.5` | Temporal extent of one S3D window — how much motion context each frame's feature sees. |
| `analysis_fps` | `null` | Rate S3D sees; frames are skipped to reach it, never interpolated up, so halving this roughly halves the cost. `null` = every frame. |
| `camera` | `null` | Which camera's video to take, when the alignment holds several. |

`stack_s` must be at least 13 frames at the effective rate. The 0.5 s default
works down to 26 fps; if it does not, the error names the shortest window
that does.

```{note}
Everything else about the extraction — batch size, decode chunk, `fp16`,
device, the `dense` ablation mode — is a performance detail with one sensible
answer, so it is not a project setting and naming it here is an error. Build
a {class}`~ethograph.video_features.S3DConfig` yourself in the rare case you
need one.
```

Sidecars go to `{root}/video_features/`.

## `infer`

| Key | Default | Meaning |
|---|---|---|
| `run` | `train.run_name` | Run name (exact or base) or a run directory under `runs/`; a base name resolves to its most recently trained timestamped run (`project.inference(run=…)` overrides it for one call). |

### `infer.postprocess`

Purge → stitch → snap → purge, through the same functions as the GUI's
changepoint correction. Also used for the *post-processed* numbers in
`test_metrics.yaml`.

The interval steps are the GUI's *CP Correction* section under other names,
and the default way to fill them is to **take the GUI's numbers**:

```yaml
infer:
  postprocess:
    gui_settings: true          # ~/.ethograph/gui_settings.yaml (or a path)
    max_shrink_s: 0.1           # anything spelled beside it still wins
```

`gui_settings` reads the file every time the config is loaded, so the
pipeline stays in step with what you tune in the GUI; a saved run config
carries the resolved values explicitly (plus the path they came from), so a
finished run does not change when the GUI does. The GUI's step checkboxes
read as zeroed parameters (purge off → `min_duration_s: 0`, stitch off →
`stitch_gap_s: 0`, snap off → `changepoint_correction: false`). Spell the
values instead when one project needs settings the GUI does not hold.
`changepoints` has no GUI counterpart and is always the config's. See
`docs/adr/0006-postprocess-from-gui-settings.md`.

| Key | Default | Meaning |
|---|---|---|
| `gui_settings` | `null` | `true` or a path: read the interval-step values below from the GUI's `gui_settings.yaml`; explicit keys override. |
| `min_duration_s` | `0` | Drop predicted labels shorter than this (`0` = off). |
| `label_thresholds` | `{}` | Per-label-id minimum durations overriding `min_duration_s`. |
| `stitch_gap_s` | `0` | Merge same-label predictions separated by less than this. |
| `changepoint_correction` | `false` | Snap onsets/offsets to the session's changepoint masks (xarray sessions only; see {doc}`../variable_schema`). |
| `changepoints` | `{}` | Selections pinning those variables (e.g. `{keypoint: beakTip}`); the individual is pinned per sample. |
| `max_expansion_s`, `max_shrink_s` | `0.05`, `0.05` | How far an interval edge may move outwards / inwards when snapping. |

## What a run writes

```
runs/{run}/
  config.yaml        the resolved config (absolute paths) this run was trained with
  columns.yaml       input layout, copied from the materialised dataset
  classes.yaml       class index ↔ label id
  stats.npz          normalisation statistics of the training samples
  splits/            train.bundle / val.bundle / test.bundle — sample keys per role
  metrics.tsv        one row per validation: epoch, loss, val metrics, + test_raw_*/test_post_* diagnostic
  best.pt / last.pt  weights selected on validation / at the end
  test_metrics.yaml  test evaluation of best.pt: raw and post-processed, overall and class-wise
  eval.pdf           overall + class-wise F1, onset/offset |Δ| histograms
  train.log          everything logged during this run (always written, on top of the console)
  infer.log          everything logged by every `project.inference()` call against this run (appended)
runs/compare.tsv     written by `project.compare()`, which also returns it as a DataFrame
```

A search's trials and a cross-validation's folds are ordinary runs, nested one
level deeper (`runs/{search or cv name}/trial000_…`, `runs/{cv name}/fold-…`)
so they do not bury the runs you trained by hand — and so `project.compare()`,
which reads only the top level, keeps showing those.

## What a search and a cross-validation write

```
searches/{name}/
  study.db           the Optuna storage — calling search() again resumes it
  trials.tsv         one row per trial: number, state, value, best_epoch, run_dir, parameters
  best.yaml          `base:` your config + the winning parameters — what stage 2 reads
  search.log         everything logged during the study
cross_validation/{name}/
  folds.tsv          one row per fold: session, run, run_dir, best_epoch, held-out metrics, predictions
  crossval.log       everything logged during the folds
```
