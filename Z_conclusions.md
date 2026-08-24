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
