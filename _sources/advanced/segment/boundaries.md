# Boundaries: heads that predict *where*, not just *what*

Frame-wise accuracy and frame-wise F1 saturate long before the boundaries are
right. The metric that still moves is **F1@90** — a predicted segment counts
only if it overlaps the true one by 90% — and at 200 Hz a 0.2 s syllable has to
be placed within about ±2 frames to clear it.

A frame-wise cross-entropy cannot be asked to do that. Whatever you weight it
by, the question it puts to the model is *"is frame t class c?"*, never *"where
is the transition?"*. Reweighting the frames near a boundary changes how loudly
that question is asked; it does not change the question. This page is about the
two heads that do.

Both are **additions to the encoder you already trained**, not replacements for
it. The class branch of `asrf` is bit-identical to the plain backbone's, and
`baformer` reuses the same ASFormer encoder layer for layer — so a comparison
against the encoder-only baseline is a comparison of heads.

```{note}
Nothing here is on by default. `train.boundary.weight` is `0` and
`infer.postprocess.boundary_refinement` is `none`, so an existing project
trains and post-processes exactly as it did.
```

## Everything is spelled in seconds

Every setting on this page that has a temporal extent is a **duration**, and it
is resolved against the materialised dataset's own sampling rate.

This is not house style; it is the whole reason to re-run these experiments.
Every hyperparameter inherited from the temporal action segmentation literature
was tuned at 15–30 fps and published in *frames*. ASRF dilates its boundary
target by ±4 frames — at 25 fps that is ±160 ms, a real tolerance for a human
annotator. Copy the number to 200 Hz and it becomes ±20 ms, which is a
different instruction entirely. So `train.boundary.tolerance_s` takes seconds,
and `ethograph.segment.boundary.tolerance_frames` is the only place the
conversion happens.

## The boundary branch (`asrf`)

One extra output channel, a 1×1 convolution off the shared trunk, trained to
predict a binary target: 1 at every class transition of the dense labels,
dilated by `tolerance_s`. Onsets and offsets both count — an offset is a
transition *to* background — and frame 0 is not a boundary, because the trial
merely starts there.

```yaml
model:
  architecture: asrf
  params:
    backbone: asformer              # any registered architecture that keeps the time axis
    backbone_params: {num_decoders: 0}   # exactly what the baseline trains with
    brb_stages: 1                   # boundary refinement stages; >1 adds a TCN each

train:
  boundary:
    weight: 0.5                     # w_b; 0 builds the head and never trains it
    tolerance_s: 0.025              # ±5 frames at 200 Hz
    pos_weight: null                # null recomputes n_neg / n_pos per batch
    focal: false                    # the alternative to pos_weight, not a stack
```

Boundaries are about 1% of frames, so the positive class needs weighting. The
default recomputes `n_negative / n_positive` per batch — honest for a channel
whose density changes from trial to trial — and whichever weight was used is
written into the run's `test_metrics.yaml` under `objective`.

The run also writes **`boundary.pdf`**: the predicted boundary probability of a
few test samples with the true transitions and the detected peaks marked. It is
the figure that says whether the head learnt *where* rather than *what*.

`asrf` wraps any vendored architecture that keeps the time axis (`asformer`,
`mstcn`, `edtcn`, `mlp`). The C2F U-Nets pool the timeline and are refused when
the model is built, not on the first forward pass.

## Four ways to read one prediction

The boundary channel changes what post-processing can do, and the four options
are worth comparing on **the same trained model** — they are all inference, so
retraining for each would confound the comparison with the seed.

```{list-table}
:header-rows: 1
:widths: 16 44 40

* - Mode
  - What it does
  - Config
* - **raw**
  - Nothing. The frame argmax, as-is.
  - all post-processing off
* - **existing**
  - Purge short segments, stitch same-class gaps, snap interval edges onto
    *detected* changepoints.
  - `boundary_refinement: none` + `changepoint_correction: true`
* - **predicted**
  - Cut the timeline at the peaks of the model's own boundary probability and
    give each span its majority class, then the existing interval steps.
  - `boundary_refinement: predicted`
* - **hybrid**
  - The predicted peaks, restricted to those within `boundary_snap_s` of a
    detected changepoint and moved onto it. A peak with no changepoint nearby
    is dropped.
  - `boundary_refinement: hybrid` (requires `changepoint_correction: true`)
```

**Hybrid is the one this project is shaped for.** The physical prior — a
syllable boundary sits at a speed minimum — stays a hard constraint, and the
network only has to *select* which of the overspecified changepoints are real.
That is the learned version of the hand rule in `correct_changepoints`, and it
is why the changepoint features are worth having as model inputs as well as
post-processing candidates.

Note the difference in kind between **existing** and **predicted**: snapping
moves an interval *edge*, so a span the model got wrong stays wrong. Re-cutting
happens on the dense prediction *before* it becomes intervals, so a span can
change class outright.

```yaml
infer:
  postprocess:
    boundary_refinement: hybrid
    boundary_threshold: 0.5      # below this, a local maximum is not a peak
    boundary_snap_s: 0.05        # how far a peak may move onto a changepoint
    changepoint_correction: true
    changepoints: {keypoint: beakTip}
```

A run whose architecture has no boundary head ignores the mode rather than
failing, so a config can carry it across a whole benchmark.

## The query head (`baformer`)

A frame-wise loss optimises a different thing from the segmental metric being
reported. BaFormer's answer is to predict **segments as instances**: a set of
queries, each emitting a class and a soft mask over the timeline, matched
one-to-one against the true segments — so the objective is IoU-shaped from the
start. A global class-agnostic boundary query cuts the timeline into spans, and
each span is classified by letting the queries vote.

Structurally that is this project's pipeline (cut at changepoints, then
classify) learned end to end, on the same ASFormer encoder.

```yaml
model:
  architecture: baformer
  params:
    num_queries: 100        # start near 2× the worst trial's segment count
    num_f_maps: 64          # the encoder's, must divide by nheads
    nheads: 4
    boundary_threshold: 0.3

train:
  frame_weight: 0           # upstream trains on the set objective alone
  queries:
    class_weight: 2.0
    mask_weight: 5.0
    dice_weight: 5.0
    boundary_weight: 1.0
    eos_coef: 0.1           # the weight of the "no segment" class
```

Three things to know before running it.

**The query budget is a hard constraint.** A trial with more segments than the
head has queries cannot be matched one-to-one, and the set criterion raises
rather than silently dropping the overflow — the error names the number to set.
`scripts/experiment4_queries.py` reads the worst case off the materialised
dataset instead of guessing.

**`model.eval()` is not optional.** In training the dense logits are the soft
query composition, which is what the gradients flow through; in eval they are
the hard boundary-aware vote. A soft composition averages the queries and blurs
exactly the edges F1@90 measures, so evaluating a model left in train mode
silently measures the blurred version. The pipeline switches modes for you; a
notebook that calls the model directly must too.

**Watch the repetitive classes.** Query-based methods assume a reasonable
number of segments per sequence, and a trial with fifteen consecutive
identical syllables is exactly what stresses one-to-many matching. Read the
per-class F1@90, not the aggregate — the aggregate hides it.

## Running the experiments

Two scripts, both resumable (`results.tsv` is append-only, keyed by cell and
fold) and both leave-one-session-out:

```bash
python scripts/experiment3_boundary.py    # w_b × dilation, then the four modes
python scripts/experiment4_queries.py     # query budget × frame-loss weight
```

Experiment 3 trains one model per cell and re-scores it under all four
refinement modes, so the mode comparison costs nothing beyond the training that
was happening anyway. Both screen on one fold before paying for the rest;
`SCREEN_FOLDS` and `CONFIRM_TOP` at the top of each script control that.

Report mean ± sd across folds with the individual folds as dots, raw **and**
post-processed, F1@50 / F1@75 / F1@90 and edit distance — with frame accuracy
only as a saturation check, since it will not move and that is the point.

## What is deliberately not here

**Boundary-weighted cross-entropy** and the **Circle loss** were both ablated
and neither improved F1. Neither is re-added: a head that is asked where the
transition is supersedes reweighting a head that is only ever asked what class
a frame is.

**Deeper decoders** (DiffAct, CETNet). Decoders in this literature exist to
repair over-segmentation caused by I3D features being unreliable *at*
boundaries. Here the most informative feature is maximally informative exactly
at the boundary, so the decoder's job is already done by the input layer.
