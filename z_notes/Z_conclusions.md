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
