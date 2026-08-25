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
4. **Tick the existing labels to read as inputs** (optional) — see
   {ref}`below <target-onset-model-label-inputs>`.
5. **Set the parameters.** `Window size` is how much context the classifier
   sees around each frame; `Tolerance` is how precisely you believe your own
   labels.
6. **Add current session's events**, then **Train**.

Only trials **visible in the trials table** contribute, so the table's filters
double as a training-set selector. A trial carrying none of the ticked events
is skipped, and one carrying only some contributes only to those — an
unlabelled trial is not evidence that the event never happened, so it is never
used as a negative example for that class.

Once a model exists its targets, features, label inputs, window and tolerance
are **read-only**: they define the classifier's input columns, so editing them
would invalidate every training trial already stored. To change them, make a
new model. To add more sessions, open the dialog there, pick the model, and
press **Add current session's events**.

(target-onset-model-label-inputs)=
### Existing labels as inputs

What a session already knows about *when* is often the strongest evidence
there is: a peck rarely happens before the head has turned, a landing never
before the approach. **3 — Existing labels as inputs** lets the classifier read
those classes as ordinary input columns.

Each class is one row, ticked like a feature; its children are the
individuals, so ticking the class row is that class's "all individuals"
toggle, and **Select all** / **Clear** do the same for the whole list. A
single-individual session draws no children at all — there is nothing to
choose. How a class is rendered follows its type in {doc}`mapping.txt
<mapping>`, frozen into the model at creation so a later edit to the mapping
cannot change the model's input layout:

* a **state** class becomes its **on/off indicator** — `1` inside every
  interval of that class, `0` outside. That is the whole of what a state says.
* a **point** class becomes a **Laplacian bump** centred on the event, at two
  hard-coded widths (0.1 s and 1 s), one column each — the same kernel
  EthoGraph puts on {doc}`changepoints <../changepoints>`, for the same reason:
  the narrow peak points straight at the moment while the long tails stay
  readable from far away, so one column carries both *it is here* and *it was
  a while ago*.

A class the trial does not carry renders as zeros, which is the honest reading
— the column says "no such label here", exactly the state a trial is in when
the model runs on it.

```{important}
**A class the model predicts cannot be one of its own inputs.** Its row is
greyed out the moment you tick it as a target. At training the label is there
and at inference it is not — prediction only ever runs on trials that lack the
target — so such a column would mean opposite things on the two sides.
```

Predicted (`automated`) labels count as inputs like any other, so a model that
reads a class you have not curated yet is learning from whatever the last run
wrote. Curate first if that matters.

```{warning}
**Every chosen feature must share one sampling rate.** Windows are
index-based, so mixing a 30 Hz pose feature with a 44.1 kHz audio feature
would silently misalign them. EthoGraph refuses instead — resample first, or
pick features from one stream.
```

### On disk

```
~/.ethograph/models/{name}/
├── config.yaml                 # frozen: targets, features (+ d/dt), label inputs, window, tolerance
├── model.joblib                # one trained classifier per target, and a copy
│                               #   of the config they were fitted with
└── train_data/
    └── {session}-{hash}/
        ├── meta.yaml           # source path, columns, trial count (provenance)
        └── trial_7.npz         # time, features, the events' times
```

The features are stored, not the source data — training data from a session
survives that session moving or going offline.

**Predicting reads `model.joblib` and nothing else.** The bundle carries its
own copy of the config because that is the layout the classifiers were fitted
on: `config.yaml` is what *Train* reads next time, and `train_data/` is what it
fits from. Neither is consulted at prediction time — a trained model is
self-contained, and `train_data/` can be deleted once you are sure you will not
extend or retrain that model (it is the bulk of the folder). Editing
`config.yaml` by hand therefore changes nothing about what a trained model
reads, and the Predict dialog says so when the two have drifted apart.

### Using one animal's model on another animal

A classifier is fitted on numbers. The individual in `config.yaml` is only the
key that selects those numbers out of a session, so a model trained on one
animal runs on another's — same rig, same feature layout — with nothing copied
and nothing retrained: **pick the model, pick the other individual, predict.**
The model's individual pinning is re-pointed at whoever the *Individual* combo
names, giving the classifier the same columns in the same order from the other
animal's data, and the events are written for that individual too. Its
{ref}`label inputs <target-onset-model-label-inputs>` are re-pointed by the
same rule, so a model that times a peck off the approach reads *this* animal's
approach. The dialog says which animal the model was trained reading.

A model whose features read *several* individuals at once — an actor and a
partner — is a different thing, and it is left alone: asked for either animal
it reads, it keeps reading both, and the combo only decides whose labels the
events are. It is refused only for an animal it does **not** read, because
collapsing two columns onto one would hand the classifier the same data in the
slots it learned as two different animals. Train a model on that session
instead, or build the features so they carry no individual at all — a distance
or an egocentric feature travels between animals for free.

---

## Predicting

Pick a trained model, choose the **individual** — whose data is read *and*
whose labels these are — and press **Predict missing onsets**. Two things are
never touched:

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
