# EthoGraph — modelling

The vocabulary of the model pipelines (`ethograph.segment`, `ethograph.spot`)
and of the pose that feeds them. It exists because the same physical thing —
a box corner, a moment in a trial, a model's belief — reaches the code by
several routes and must be called one thing.

## Language

### Pose

**Keypoint**:
A named point on the animal, its tool or its object whose position is
tracked frame by frame.
_Avoid_: joint, node, marker

**Static keypoint**:
A keypoint that does not move — a box corner, a dispenser. Labelled once, or
measured once, and present on every frame.
_Avoid_: landmark (ambiguous), constant keypoint, anchor

**Calibration landmark**:
A physical point with known real-world coordinates, used only to fit the
pixel-to-cm mapping.
_Avoid_: landmark (bare), reference point

**Clip**:
One trial's video file. A session cut into trials has one clip per trial.
_Avoid_: video (when the trial is meant), segment

**Fill**:
Positions produced by a tracker between the frames a person labelled.
_Avoid_: interpolation, prediction (that is the model's word), track

### Point events

**Point event**:
A behaviour that happens at one moment — a contact, a peck, a call onset — as
opposed to a state, which spans an interval.
_Avoid_: onset (that is one of its two edges), spike, hit

**Spotting**:
Placing a point event at a moment in a trial, to within a tolerance measured
in milliseconds.
_Avoid_: detection, localisation, action spotting

**Tolerance**:
The distance from the labelled moment within which a spotted event counts as
correct. Always a duration.
_Avoid_: delta, window (that is the training target's word)

**Positive window**:
The stretch either side of a labelled moment that training treats as the
event. A duration.
_Avoid_: dilation, dilate_len

**Context**:
How much of a trial a model sees at once when judging one moment. A duration.
_Avoid_: clip length, receptive field

**Resolution**:
How finely a model can place an event — the grid its answer lands on. A
duration.
_Avoid_: stride, downsampling

### The pose side

**Feature (pose)**:
A variable in the session file a config lists under `features:` — a
position, a velocity, a distance the user computed and can plot. The whole
of what any model here reads from the pose; there is no graph.
_Avoid_: node, edge, keypoint feature (the graph vocabulary is gone)

**Block**:
The listed features z-scored on the training split, on the strided clock —
the pixel model's second input, beside the CNN features, before the GRU.
_Avoid_: fusion, side input

### Models

**Teacher**:
The pose-only model that reads the listed features. It exists to be learned
from, not deployed.
_Avoid_: pose model (that is its input, not its role), graph model

**Student**:
The model that reads video alone and is trained to match the teacher, so it
runs where no pose exists.
_Avoid_: pixel model (that is the untaught baseline), RGB model

**Distillation**:
Teaching the student the teacher's per-frame representation on every trial
that has both pose and video, then adapting its head to the labelled ones.
One act with two steps.
_Avoid_: transfer, fine-tuning (that is only its second step)

**Baseline**:
The video model trained on labels alone — what the student must beat.
_Avoid_: E2E-Spot (the architecture), pixel model

**Confidence**:
A number written beside a predicted event that says how much to trust it,
readable off the curve the review draws.
_Avoid_: probability, score (that is the raw per-frame output), certainty

**Curve**:
A model's per-frame belief that a class's event is at that frame, over a
whole trial. What a confidence is read off, and what review draws.
_Avoid_: heatmap, probabilities, scores
