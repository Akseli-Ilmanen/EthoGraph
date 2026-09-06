# Conclusions

## Experiment 2 — smoothing loss retune (`scripts/experiment2_smoothing.py`)

**Lowering τ (tau) did not improve F1@90.** The τ sweep (4 / 16 / 48 at fixed α) showed no
consistent gain from a tighter truncation on the consistency term.

This argues against the theory that the consistency term's log-probability-transition penalty
is a major driver of F1@90 loss — if it were actively smearing boundaries hard enough to matter,
tightening τ (penalising smaller log-prob jumps between frames) should have recovered some of
that F1@90, and it didn't move it appreciably. The bigger lever, if there is one, likely lives
elsewhere (e.g. α / whether the term is on at all, or the rate ablation) rather than in how far
the term truncates.

## Experiment 3 — disabling the consistency term entirely (α = 0)

**Setting α = 0 (the consistency term's weight, not just its truncation τ) also had negligible
impact on F1@90 relative to the searched baseline.**

| run | α | epoch | test_post f1@50 | f1@75 | **f1@90** | acc | edit |
|---|---|---|---|---|---|---|---|
| `search_asformer_enc_kin_v1/trial001` (baseline, α searched) | 0.446 | 45 | 91.58 | 82.36 | **60.08** | 96.05 | 90.52 |
| `sweep_asformer_enc_alpha_kin_v1/alpha=0.0_1647` (full 50-epoch run, best-epoch eval) | 0.0 | 45 | 89.75 | 79.81 | **59.92** | 95.84 | 91.15 |
| `sweep_asformer_enc_alpha_kin_v1/alpha=0.0_1135` (repeat, last-epoch eval) | 0.0 | 45 | 90.38 | 80.59 | **60.37** | 95.91 | 91.27 |

Two independent α = 0 runs land within ~0.5 points of F1@90 of the α = 0.446 baseline (and
straddle it, one slightly above, one slightly below) — well inside run-to-run noise, not a
regression. Note: `search_asformer_enc_alpha_kin_v1` and `cv_asformer_enc_alpha_kin_v1`
(the runs this experiment was originally checked against) were both incomplete when inspected —
the search trial had only reached epoch 25/50, and the LOSO cross-val fold only epoch 3/50 with
no metrics logged yet — so the comparison above uses the completed sibling sweep runs instead.

**Consequence: the smoothing/consistency term (MS-TCN's `alpha`-weighted, `tau`-truncated
log-probability transition penalty) is not a meaningful lever on F1@90 at this dataset's rate,
whether tuned via τ (Experiment 2) or switched off entirely via α (this experiment).** It can be
left at α = 0 (i.e. dropped) with no measurable cost, which simplifies the loss by one
hyperparameter. It also means the F1@90 ceiling seen across these runs (~55–60% postprocessed,
with large classwise spread — e.g. class 4 and 15 sit at ~30% f1@90 vs. class 10 at ~97%) is not
being capped by this term, so the remaining gap has to be chased elsewhere: the boundary head
(`boundary_refinement`), postprocessing/changepoint correction, the rate/subsample ablation, or
per-class imbalance — not further tuning of the consistency term.

## E2E-Spot stride ladder — temporal aperture at 200 fps (`scripts/spot_ladder.sh`, 2026-08-26)

**A pixel model can spot the stick/pellet point events on a held-out session, and what was
blocking it was temporal aperture, not capacity.** The first full run collapsed to background
(11/54 misses on val, and by epoch 9 it emitted no candidate at all on 22 of 27 trials). The
cause: every temporal hyperparameter in E2E-Spot is expressed in *frames* and was tuned at
25 fps, and we ran the identical numbers at 200 fps — `clip_len 100` is 4.0 s upstream and 1.0 s
here, the GSM shift is ±40 ms upstream and ±5 ms here. The model was being asked to say *the
reach is now* from one second of a mostly-resting animal.

`--stride k` widens clip, GRU and GSM aperture together, at the price of k-frame label
quantisation. Five runs, 8 epochs each, only stride and clip length varying, with `dilate_len`
chosen per stride so the positive window stays ±10 ms in real time and dilation is not a
confound. Each read at its own best epoch, on val (54 events, 1 frame = 5 ms):

| run | ep | stride | `clip_len` | context | quantisation | miss | median | ≤2 (10 ms) | ≤4 (20 ms) | ≤10 (50 ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| `overnight2` (baseline) | 3 | 1 | 200 | 1.0 s | 5 ms | 11/54 | 12.0 | 13 % | 22 % | 35 % |
| `A0` | 2 | 1 | 200 | 1.0 s | 5 ms | 1/54 | 7.0 | 30 % | 39 % | 57 % |
| `A1` | 3 | 2 | 100 | 1.0 s | 10 ms | 2/54 | 12.0 | 22 % | 31 % | 44 % |
| **`A2`** | **1** | **2** | **200** | **2.0 s** | 10 ms | **0/54** | **3.0** | **46 %** | **57 %** | **72 %** |
| `A3` | 1 | 4 | 200 | 4.0 s | 20 ms | 0/54 | 9.0 | 24 % | 37 % | 56 % |
| `A4` | 3 | 8 | 200 | 8.0 s | 40 ms | 0/54 | 6.0 | 19 % | 39 % | 72 % |

**Context fixes recall; quantisation caps precision — so the ladder has an interior optimum.**
Every run with ≥ 2.0 s of context reaches 0 misses at its best epoch and neither 1.0 s run does.
But `A2`'s median error is 3 frames = 15 ms, *finer than the label grid* `A3` (10 ms) and `A4`
(20 ms) are trained on; past stride 2 the model cannot represent the answer it is asked for, and
more context makes it worse. `A0` vs `A1` prices resolution alone (1 extra miss, 8 points of
≤20 ms at fixed 1.0 s context); `A1` vs `A2` buys context at that same resolution (2 misses and
26 points back). Context is worth several times what the resolution costs.

**Confirmed once on the held-out session** (`A2`, epoch 1, `test.json`, 36 trials / 72 events,
0 misses):

| class | median | ≤2 (10 ms) | ≤4 (20 ms) | ≤10 (50 ms) | ≤20 (100 ms) |
|---|---|---|---|---|---|
| `label_31` (stick) | 3.0 | 42 % | 72 % | 83 % | 89 % |
| `label_32` (pellet) | 5.0 | 36 % | 47 % | 58 % | 72 % |

Val and test agree, so the numbers are the model's and not the split's. Against the 3-4 frame
(15-20 ms) target this is a qualified yes for the stick event and a not-yet for the pellet event.

**Every run still collapses, and fast.** Misses per epoch:

```
A0  3  1  3 10 11 11 14      A2  0  1  2  3  4  5  9
A1  3 27  2  7  7  6  4      A3  0  6  3  9  7  9  7
                             A4  0  0  0  2  3  3  3
```

Best epochs are 1-3, i.e. **4-10 passes over the training frames** (`1.24 × stride` passes per
epoch at `--epoch_num_frames 250000`); the rest of the 8-epoch budget is spent watching the model
come apart. Validate from epoch 0 and stop by epoch 3. **`A4` is the exception** — it never
collapses and is still improving at epoch 7, which is the strongest evidence that collapse tracks
how sparse the positives are *on the model's own clock* (0.152 % positive rate at stride 1,
8× that at stride 8) rather than anything about the optimiser.

**Consequence: the next lever is a displacement head, not more context and not MSAGSM.** The
ladder's own shape says the remaining error is quantisation-bound, and `A4` says stability comes
from denser positives — a displacement head (regress the sub-bin offset instead of dilating the
label) buys both at once: `A4`'s stability at `A2`'s resolution, and it is the principled
replacement for `dilate_len`, which currently trains a ~16-frame-wide plateau on purpose. MSAGSM
attacks the same axis stride already moved, so it is not the first thing to try. A two-stage
temporal-ROI model is not needed here — the trials are already the ROI — and if it is ever needed
for long unsegmented recordings it should be a predicted **state** event feeding the point model,
composing two shipped defaults rather than adding a stage.

**Two measurement traps found on the way, both worth remembering.** (1) Rank on the `score`
sweep, never on `val_mAP` or exact-frame F1 — `val_mAP` is AP over un-NMS'd candidates at 0/5/10/
20 ms and swung 0.074 → 0.054 → 0.078 → 0.021 across consecutive epochs of one run, and
exact-frame F1 read `TP=30 FP=1126` on a model whose events were all correctly located, because
1126/72 ≈ 15.6 is simply the width of each correct bump. (2) A strided run must be scored back to
the bin's **centre** (`bin × k + (k-1)/2`); mapping to `bin × k` reads every strided run as early
by half a stride — 7.5 ms at stride 4, against a 20 ms budget.
