(target-onset-model)=
# Predicting point events (LightGBM)

EthoGraph includes a lightweight model for predicting **point events**, built
on scikit-learn's histogram-based gradient boosting — its implementation of
Microsoft's LightGBM. It is a small classifier that detects the onset of an
event from a local window of hand-crafted features: the first time a mouse
touches a lever, the frame a bird lands, the moment a beak opens. You label
the moment in a handful of trials, tick the features it should look at, and it
fills in the rest.

**Model ▸ LightGBM: Train…** and **Model ▸ LightGBM: Predict…**

```{admonition} Scope — read this first
:class: important

**Point events only**, and **at most one event per class per trial**.
Inference is an `argmax` over the trial's smoothed probability curve, so it
returns exactly one time per class per trial and cannot return two.

**State events are out of scope**: a state event has two boundaries ("start" and "stop") whose
order and non-overlap have to be respected, which is a different problem from
"when did this happen?". Use an external action-segmentation model for those.
```

---

## How it works

For each frame, the model sees a **window** of your chosen features centred on
that frame and answers one question: *is the event here?*

* **Targets.** Frames within `tolerance_s` of your labelled event count as
  positive, weighted by a Gaussian bump peaking at the event, so a frame one
  tick off counts less than the exact frame. Far negatives are subsampled.
* **Inference** smooths the per-frame probability with the same tolerance — a
  plateau of near-hits beats one spurious spike — and takes the argmax.

Each class gets **its own binary classifier** over the same features, window
and tolerance. The design matrix is built once per trial and reused, so
predicting five classes costs barely more than predicting one.

---

## Training

1. **Name the model** — leave the combo on *New model…* and type a name, or
   pick an existing model to add more training data to it.
2. **Tick the point events to predict.** Only classes marked as point events
   in {doc}`mapping.txt <mapping>` are listed; tick as many as you like.
3. **Tick the features.** Ticking `speed ▸ keypoints ▸ beak, head` gives two
   input columns. **Every dim has to be pinned to explicit values** — that
   frozen list *is* the model's input layout, which is what lets the model run
   on another session.
4. **Set the parameters.** `Window size` is how much context the classifier
   sees around each frame; `Tolerance` is how precisely you believe your own
   labels.
5. **Add current session's events**, then **Train**.

Only trials **visible in the trials table** contribute, so the table's filters
double as a training-set selector. A trial carrying none of the ticked events
is skipped, and one carrying only some contributes only to those — an
unlabelled trial is not evidence that the event never happened, so it is never
used as a negative example for that class.

Once a model exists its targets, features, window and tolerance are
**read-only**: they define the classifier's input columns, so editing them
would invalidate every training trial already stored. To change them, make a
new model. To add more sessions, open the dialog there, pick the model, and
press **Add current session's events**.

```{warning}
**Every chosen feature must share one sampling rate.** Windows are
index-based, so mixing a 30 Hz pose feature with a 44.1 kHz audio feature
would silently misalign them. EthoGraph refuses instead — resample first, or
pick features from one stream.
```

### On disk

```
~/.ethograph/models/{name}/
├── config.yaml                 # frozen: targets, features, window, tolerance
├── model.joblib                # one trained classifier per target (+ CRF)
└── train_data/
    └── {session}-{hash}/
        ├── meta.yaml           # source path, columns, trial count
        └── trial_7.npz         # time, features, the events' times
```

The features are stored, not the source data — training data from a session
survives that session moving or going offline.

---

(target-onset-model-sequence)=
## Modelling the order of the events

When the classes always run in the same order, tick **Model the order of the
events (CRF)** and EthoGraph fits a linear-chain CRF
([sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io)) over the
per-class probabilities. Every frame is tagged with the class of the most
recent event, so the CRF's transitions *are* the sequence dependencies, and
only orders seen in training can be decoded — a class whose signal has a decoy
early in the trial can no longer be predicted before the class it always
follows. The whole trial is decoded at once, which also means a class the
decoded order leaves out gets no prediction there.

Needs at least 2 training trials; training reports the orders it saw
(`peck→land ×18`) and takes minutes rather than seconds, since each class is
refitted once per cross-fitting fold. Untick **Use the sequence model** when
predicting to fall back to independent per-class argmaxes.

---

## Predicting

Pick a trained model, choose which **individual** the predicted labels belong
to, and press **Predict missing onsets**. Two things are never touched:

* **Trials that already carry a class** keep what they have — the model fills
  gaps, it never overrides. A trial that already has *one* class can still
  receive the others.
* **Trials excluded by the filters.** The trials table's filters apply as
  always, and the dialog adds one filter per metadata column of its own. Those
  **combine**: setting `genotype = wt` and `stimulus = tone` predicts only the
  trials that are both. A column left on *All* constrains nothing, and the
  dialog says how many trials survive before you run anything.

Predictions land in memory like any other label — review them, correct them,
and save with `Ctrl+S`. **Review predictions…** at the bottom of the dialog
opens the {ref}`label-frames grid <target-onset-model-confidence>` on exactly
what the run just wrote — those classes, those trials — so you can check the
video frame at each one and click through to fix the doubtful ones.

---

(target-onset-model-confidence)=
## Confidence

Every label row carries a **`confidence`** column in the
{ref}`labels TSV <target-exporting-labels>`: a label you placed by hand is
**1.0**, and a predicted label carries the model's own score in `[0, 1]`.

The score combines two readings of the probability curve:

| | Question | Low when |
|---|---|---|
| **peak** | How strongly does the model believe? | the best frame still only scores 0.2 |
| **sharpness** | How localised is that belief? | the curve is a broad smear over the trial |

*Sharpness* is one minus the normalised entropy of the curve read as a
distribution over the trial's frames — a single-frame spike scores 1, a flat
curve scores 0. It is the point-event counterpart of the frame-wise
`1 - normalised entropy` used for {doc}`imported dense predictions
<importing>`: there the distribution is over classes, here over time. The
reported `confidence` is their **geometric mean**, so a prediction has to hold
up on both counts. With the sequence model the same two readings are taken on
the CRF's marginals instead.

**Min confidence** in the Predict dialog is a floor on what gets written: a
prediction below it produces **no label at all**, leaving the trial unlabelled
for that class rather than giving it a doubtful label. Leave it at 0 to write
everything and triage afterwards — the confidence is on every label either
way, so nothing is lost by writing first and filtering later.

### Reviewing by confidence

**Tools ▸ Labels: Show frames as Grid/PDF…** puts each label's confidence on
its tile and takes a **Flag confidence below** threshold: every tile under it
gets a red outline, in the grid and in the exported PDF. Set it to `0.6`, scan
the sheet, and click straight through to the ones worth fixing.

**Histogram…** next to the threshold shows where the scores actually sit
before you commit to a number: one histogram per label class — split per
individual as well, when more than one animal is labelled — with everything
below the threshold drawn in the same red. A class whose scores pile up near
1.0 with a thin low tail wants a high threshold; one spread across the range
is a class the model has not learnt, and the whole class is worth reviewing.
Its threshold spin is the grid's, so dragging it there recolours the tiles
behind it.

**Tick flagged** ticks exactly the outlined tiles, **Tick their whole trials**
every event of every trial holding one — the second is the honest default for a
model review, since a trial with one bad event rarely has only one.

Ticking tiles and pressing **Refine ticked frame-by-frame…** hands them
to {ref}`frame-by-frame refinement <target-refining-labels>` as a queue of just
those boundaries: `Enter` moves an event onto the right frame, `Backspace`
deletes one that never happened. Either way the row stops being a prediction —
a corrected event is stamped back to `confidence = 1.0`.
