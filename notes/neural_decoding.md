# Neural decoding with the segment pipeline — plan and open questions (2026-09-02)

Single-trial decoding of the curated wake behaviours from spike trains, with
the same models, folds and prediction sets the behaviour work uses. Design
in ADR 0009; config in `configs/neural/decoding.yaml`; sweeps in `run.py`;
sleep windows in `run_sleep.py`. Nothing below is measured yet beyond one
fold of one variant — this note records what was decided and why, so the
numbers can be read against it when they arrive.

## What exists

- `features.neural`: spikes → `TsdFrame` by pynapple expressions at session
  open; unit columns resolved at materialise; `kind: neural_feature`, so
  `train.drop_kinds: [neural_feature]` is the ablation.
- `cross_validate(n_folds=k)`: trial folds, every trial predicted once by
  a model that never saw it, merged into one prediction set per session.
- `sessions[].alignment` + `segment/windows.py`: a sleep epoch tiled into
  one-minute windows as a second alignment on the same `units.npz`.
- `run.py`: models × transforms, all pairs on the same folds, summary in
  `configs/neural/model_sweep.tsv`.

## Decisions

### Transform
1. **Sweep the binning, do not fix it.** Nineteen transforms in `run.py`:
   binning only (5/10/20 ms, count vs rate), boxcar width (25–200 ms),
   gaussian std (10–50 ms), coarser bins with matched smoothing, counts per
   window, sqrt / log1p before and after smoothing. Four representative ones
   are the default so the model axis is not paid for nineteen times.
2. **No percentile clipping** for rates (`clip_percentiles: null`): a rate's
   tail is the signal. Z-scoring stays, per unit, training trials only.
3. **Each transform is its own materialised dataset**; the raw spikes are
   never written out as a feature.

### Loss
4. **Keep upstream's focal loss on** (gamma 2). It was switched off on a
   literal reading of "basic loss"; that made the neural runs incomparable
   with every behaviour run, and 27 imbalanced classes is the case focal
   was made for. The one argument against — with 9 units many frames are
   hard because they are unpredictable, not rare, and focal chases them —
   is an empirical question: one extra cross-validation with
   `train.loss.focal=false` on the winning transform.

### Architecture
5. **Run `mlp`, `mstcn` at several receptive fields, and `c2f_tcn`.**
   Expect C2F-TCN to win on F1 (multi-scale pooling is a denoiser for a
   low-SNR input, and using whole-trial context is legitimate within wake).
   MS-TCN's receptive field is one knob (`2**num_layers` frames: 8 layers ≈
   1.3 s, 10 ≈ 5 s, upstream's 15 ≈ 3 min at 200 Hz), which gives the
   "how much history does decoding need" curve, interpretable in a way the
   pooling ladder is not. The MLP is the no-context baseline, close to a
   linear decoder once the transform has smoothed the rates.
6. **Always run the prior-only control.** The trials follow a stereotyped
   order, so a model with global context scores well above chance from
   time-in-trial alone. Same model, same folds, each unit's spike train
   circularly shifted by a random offset per trial before binning (one
   pynapple expression on the `TsGroup`). The gap to the real run is the
   decoding; without it neither architecture's number means much.
7. Neither vendored model is causal. A "prediction" claim needs the future
   masked, which the vendored code does not do.

### Cross-validation
8. **Folds by trial, seeded**, so every transform and model is scored on the
   same trial groups and the numbers are paired. With several sessions in
   one config `n_folds` folds by trial *id* across all of them; fine for the
   one-session case this exists for.

## Sleep / replay — the plan, not yet run

- **Windows, not trials.** One-minute contiguous windows over the sleep
  epoch, written as a second alignment (`state: sleep`). Overlap is not
  possible in one alignment (pynapple merges, the loader clips); a
  half-shifted second tiling covers every frame interior to one pass.
- **Event-triggered windows first** when the question is replay: candidate
  population bursts on the smoothed summed count, one trial-sized window
  per burst. Hundreds of windows with a physiological reason, not
  thousands.
- **Receptive field must fit the training trials.** MS-TCN's default sees
  minutes; trained on 5 s trials its long-dilation taps only ever saw zero
  padding, and a long window feeds them real spikes for the first time —
  undefined, not "local". Cut the stages to ~10 layers (≈ one trial) or
  train on padded / long wake chunks. C2F-TCN stays out of the sleep pass.
- **Time compression is a parameter to sweep** (rodent replay 5–20×, songbird
  ~1×; crows unknown): stretch the sleep clock by k — spike times and window
  bounds — and the wake-trained transform sees replay at wake speed. No
  pipeline setting involved.
- **Evidence is relative to a null.** The model labels every frame no matter
  what. Controls in order of strength: wake inter-trial intervals through
  the same windows; per-unit circular shuffles within sleep (rates survive,
  co-firing does not); a *sequence* score against the wake template
  (`sequence` column of the labels TSV, edit distance weighted by
  confidence) rather than any single confident label.
- **Read the dense output, not the TSV**, for replay: the TSV is what
  survives the wake post-processing (50 ms minimum duration, per-class
  thresholds), and compressed replay may be shorter than that.
  `_probs.npz` holds every window at every frame.
- Later, if it earns it: `infer.windows` (length, stride) generating windows
  directly and bypassing the trials table; a per-session transform override
  for stretched time; a sequence-score step over the predictions TSV.

### Training may have to change shape before sleep inference means anything

The wake training set and the sleep windows are different kinds of input,
and a model trained on one has no reason to respond to a replay inside the
other. Every wake trial starts at a behaviour onset, lasts a few seconds,
is filled with labelled behaviour end to end, and follows the stereotyped
sequence; a sleep window starts anywhere, lasts a minute, is mostly
background, and a replay — if there is one — sits at an unknown position,
lasts an unknown time, and may be time-compressed. A decoder that has only
ever seen the first kind has learned to expect behaviour at frame 0, has
never produced a long stretch of background, and has never seen a
behaviour-shaped burst embedded in silence. On a sleep window it may
therefore predict the wake sequence from the start regardless, or nothing
at all — neither is evidence. Before trusting any absence in the sleep
predictions, make the training input look like the inference input:

- **Jitter the trial windows.** Cut each wake sample not at the label onset
  but at a random offset before it (and past its end), so the behaviour
  sits at a random position inside the window with real inter-trial
  activity on both sides. The alignment can express this (a widened trials
  table), or the pipeline could draw the offset per epoch as augmentation.
- **Match the window length.** Train on windows the length of the sleep
  windows, or predict sleep in windows the length of the wake trials — the
  two must not differ by an order of magnitude. Long wake windows also
  give MS-TCN's long-dilation taps real context instead of zero padding.
- **Give the model background.** Include inter-trial stretches (and, if the
  recording allows, quiet wake) as all-background samples, so "nothing is
  happening" is a state it has produced under training, not a default it
  falls into.
- **Consider time-scale augmentation** (`train.augment.stretch` exists) so
  a compressed replay is not the first stretched behaviour the model sees;
  this complements sweeping the sleep clock at inference.
- **Check the pipeline's edges on sleep-shaped wake data first.** Take wake
  inter-trial windows shaped exactly like the sleep windows, run the
  decoder, and confirm it predicts background there and still finds the
  behaviours when a real trial is planted inside such a window at a random
  position. That is the positive control for "if something is there the
  model could pick it up"; without it a null sleep result is
  uninterpretable.

The prediction sets written so far come from a model trained the old way
and should be read with that in mind.

## First results — model × transform sweep (2026-09-03, `configs/neural/model_sweep.tsv`)

63 units, 172 trials, 16 state classes, **2 folds** (each model trained on
half the trials), focal off, 50 epochs. Post-processed F1@50, mean over
folds; differences under ~2 points are within fold-to-fold spread.

| model | boxcar 25 ms | gauss 25 ms | sqrt + boxcar 25 | 10 ms / boxcar 50 |
|---|---|---|---|---|
| mstcn_rf5s (≈5 s) | 80.6 | **81.8** | 80.1 | 81.0 |
| mstcn_rf5s_64 | 80.9 | 80.9 | 80.8 | 81.0 |
| mstcn_rf1s (≈1.3 s) | 81.1 | 80.1 | 79.3 | 81.1 |
| mstcn_default (≈3 min) | 77.8 | 79.6 | 78.2 | 80.5 |
| c2f_tcn_64 | 66.8 | 67.4 | 68.1 | 73.2 |
| c2f_tcn_default | 66.8 | 66.7 | 65.5 | 68.4 |
| mlp_256x3 | 20.2 | 65.2 | 20.0 | 50.1 |
| mlp_128x2 | 17.5 | 64.9 | 17.6 | 49.8 |

What it says:

1. **Architecture is the lever; MS-TCN wins by ~12 points over C2F-TCN**,
   the reverse of the prior in decision 5. C2F-TCN's whole-trial pooling
   does not help a noisy 63-unit input; local dilated convolutions do.
2. **A second of history is enough.** rf1s ≈ rf5s ≈ rf5s_64; upstream's
   minutes-long default is the worst MS-TCN. 64 maps equal 128 —
   data-limited, so the small local model is the one to keep. This is the
   best possible news for the sleep pass: the local model is also the
   best model.
3. **The transform barely matters to a TCN** (≤ 2 points), but it decides
   how much post-processing does: gaussian input gives raw F1 77 vs 61 for
   the boxcar, i.e. smoother predictions before purge/stitch. It is
   everything to the MLP, whose only temporal context *is* the smoothing:
   boxcar 25 ms → 17, boxcar 50 → 50, gauss std 25 → 65. The instantaneous
   rate vector at 25 ms is nearly useless (raw F1 2.5); ~100 ms of
   integration is needed. Default transform from here: gauss 25 ms.
4. **Boundaries are poor everywhere**: F1@75 ≈ 50, F1@90 ≈ 15 for the best
   model, against ~60 F1@90 for the kinematic behaviour models. Neural
   decoding says *which* behaviour, not *when* to within a few tens of ms.
5. **Per class** (mstcn_rf5s, gauss 25): > 90 for pullOutStick, snapPellet,
   reachRightCorner, diagonalToBox, right, reachLeftCorner, lookToPellet;
   < 55 for swoop, beakToDisp, stickToDisp, stickInDispTwo. The same
   ranking holds across all three architectures, so it is the classes
   (short, rare, or near-duplicates of each other) rather than the model.
6. **The MLP is a floor on real neural signal**: it has no temporal
   context at all, so its 93 on snapPellet and 85 on pullOutStick cannot
   come from the sequence prior. The prior-only control (decision 6) is
   still owed for the TCN numbers.

Sleep so far: one fold of `mstcn_default` on the wake trials, then on the
post-wake epoch — no control, default receptive field; not a result.

## Housekeeping

- `configs/neural/runs/` is tracked by git, unlike `configs/paper/runs`; fold
  runs from the sweep show up as changes.
- `configs/neural/decoding.yaml` still carries `focal: false` (decision 4).
- mypy is not installed in the `ethograph` env; ruff is what gets run.
