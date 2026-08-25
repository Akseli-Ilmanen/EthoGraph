# Decisions to revisit

Things the first version settled simply, with the alternative written down
so the next pass does not have to rediscover it.

## Splits

Today: three ratios drawn by whole trial (`train.split`), and
leave-one-session-out via `Project.cross_validate()`, which pins
`split.holdout_sessions` per fold.

**Stratification by a metadata column** is the missing piece. The draw is
uniform over trials, so a condition that appears in a third of them can land
entirely in `train` by chance, and a class that appears in two trials can end
up with none in `val`. `trials.where` already reads the metadata table, so the
stratifying key exists; what is missing is drawing per group rather than over
the pool, and saying in the log which groups came out unbalanced.

**Repeated k-fold over trials** (rather than leave-one-session-out) is the
other shape worth having, for a project with one long session where holding
out a whole session is not an option. It is the same fold loop with a
different assignment function.

## Comparing runs in the GUI

A prediction set is a labels TSV; the GUI already draws **one** second label
set (`app_state.pred_labels_df`, rendered in the free `top1`/`top2` strip).
Comparing runs A/B/C against the curated labels needs that to become a list
of read-only prediction sets, each in its own lane with its run name, loaded
from `predictions/{run}/` beside the session, plus a per-lane confidence
threshold. The file contract is done; this is GUI work only.

`cross_validate()` makes this the obvious next thing to build: one fold per
parameter set leaves several prediction sets side by side in the same
session's `predictions/`, and comparing them is exactly the lane problem
above.

## Active learning

DLC2Action scores uncertainty per frame (least-confidence, entropy, or BALD —
MC-dropout disagreement across ~10 stochastic passes, Gaussian-smoothed) and
turns the highest-scoring stretches into suggested intervals. In EthoGraph
the natural home is the **curation scope**: a run's `_probs.npz` already
holds per-frame probabilities, so an uncertainty curve per sample is a few
lines; the label grid's confidence threshold and the video grid's
"mark low-confidence as uncurated" are the existing surfaces that would
consume it. BALD needs the model in train mode at inference (dropout on) —
~30 lines in `infer.py` behind a `mc_dropout: N` key. Nothing of this is
built yet.

## Multi-animal beyond actor labels

Samples are (trial, individual) with the label as *actor*; the recipient
column (`individual_rec`) is carried but not predicted. Predicting
actor–recipient pairs is a second head (or a class per pair) on top of the
same samples — a `target: actor_recipient` switch, not a new pipeline.
Likewise `target: multilabel` (one binary head per class, branches
coexisting) is a loss + label-encoding change behind one key.

## Pair features with a varying cast

`other: "*"` enumerates the remaining individuals in dataset order, so every
session must have the same number of individuals. When the cast varies,
`other: nearest` (rank by distance per frame) is the option to add.

## Video features for pynapple / NWB sessions

`merge_video_features` writes the `s3d` variable into an xarray session. For
pynapple and NWB sessions the sidecar `.nc` exists but the merge is by hand
(a `TsdFrame` on the trial's time axis, or an NWB `TimeSeries`).
