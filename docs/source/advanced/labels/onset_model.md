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
Inference takes the tallest peak of the trial's smoothed probability curve, so
it returns exactly one time per class per trial and cannot return two.

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
  plateau of near-hits beats one spurious spike — and takes the tallest peak.

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

   Ticking a feature's **d/dt** box adds its rate of change beside every column
   it produces (`np.gradient`: central differences, centred on the frame, so a
   turn in the signal shows up at the frame it happened). The classifier sees
   each tap of the window on its own and cannot difference them, so *how fast
   is this changing* has to be handed to it as its own input — worth a tick
   when the event is a change of speed or direction rather than a level.
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
├── config.yaml                 # frozen: targets, features (+ d/dt), window, tolerance
├── model.joblib                # one trained classifier per target
└── train_data/
    └── {session}-{hash}/
        ├── meta.yaml           # source path, columns, trial count
        └── trial_7.npz         # time, features, the events' times
```

The features are stored, not the source data — training data from a session
survives that session moving or going offline.

---

## Predicting

Pick a trained model, choose which **individual** the predicted labels belong
to, and press **Predict missing onsets**. Two things are never touched:

* **Trials that already carry a class** keep what they have — the model fills
  gaps, it never overrides. A trial that already has *one* class can still
  receive the others.
* **Trials the trials table hides.** Training and prediction both run over
  exactly the trials the {doc}`trials table <../metadata>` shows — its filters
  are the one trial filter in EthoGraph, so filtering `genotype = wt` there
  trains on and predicts into wild-type trials only. The dialog has no
  filters of its own; it says how many trials it will run over, read off the
  table.

Predictions land in memory like any other label, stamped
`labeling_method = automated` — they draw dotted on the plots until you
{doc}`curate <curation>` them, and a trial holding one stays red in the trial
list. **Review predictions…** at the bottom of the dialog opens the
{ref}`label grid view <target-onset-model-confidence>` on exactly what the run
just wrote — those classes, those trials — so you can check the video frame at
each one and either click through to fix it or mark it curated.

---

(target-onset-model-confidence)=
## Confidence

A predicted label's **`confidence`** is the height of the tallest peak of that
class's probability curve — the model's per-frame belief that the event is
here (a label you placed by hand is `1.0`). That curve is the **dotted line**
{ref}`frame-by-frame review <target-curation-frame>` draws under the label it
is on, one per class in the class's own colour, against a fixed 0–1 right-hand
axis. The peak's frame is where the label sits, so the number and the label
point at the same place on the same curve.

Set a threshold by looking: open a few reviews, see what a good curve peaks at
and what a bad one peaks at, and put the threshold between them. Training
separately reports what the model scored on trials it did not see (*peck: 6/8
within 0.05 s*) — a verdict on the model, not on any one label, and folded
into no confidence.

### Reviewing by confidence

**Label grid view…** (Labels tab ▸ Curation) puts each label's confidence and
`labeling_method` on its tile and outlines everything below **Flag confidence
below** in red, in the grid and in the exported PDF. The threshold is typed in
full rather than stepped, so a model whose scores sit at the bottom of the
range can be flagged at `0.0002` as easily as at `0.6`; **Histogram…** beside
it shows where the scores actually sit, per class, before you commit.

In the default *Click = uncurated, rest = curated* mode, **Mark low-confidence
as uncurated** pre-clicks exactly the outlined tiles; click any other tile that
looks wrong, and **Done** curates everything else in one go. With the Curation
section in {ref}`frame-by-frame review <target-curation-frame>`, a tile click
drops straight into that boundary instead: `Enter` moves the event onto the
right frame, `Backspace` deletes one that never happened, `N` marks it curated.
