# Keypoint labelling

**Tools ▸ Keypoint labelling…** (or the Pose sidebar) opens a dialog where you
label a few frames by clicking the video and let a point tracker fill in the
rest. No training, no annotated dataset, no GPU for the default backend.

```{important}
**This is a tracker, not a pose detector.** It follows points you have already
placed; nothing in it *finds* the animal in a video it has not been pointed at.
Everything on this page — including the model choice — follows from that. See
{ref}`target-tracking-vs-training` for when this is the right tool and when
DeepLabCut or SLEAP is.
```

(target-tracking-vs-training)=
## Tracking vs. training: which tool for which problem

Established pose tools (DeepLabCut, SLEAP) **train a detector**: you label
several hundred frames, train a network, and it then finds the keypoints in
videos it has never seen. The cost is paid once and amortised over the whole
project — provided your videos are similar enough that one model covers them.

EthoGraph's keypoint labelling **tracks instead of trains**. You label a handful
of frames in *one* video and the tracker propagates them through *that* video.
Nothing is learned that transfers to the next recording.

| | Point tracking (this dialog) | Detector training (DLC / SLEAP) |
|---|---|---|
| Frames to label | roughly every 10th, per video | several hundred, once |
| Setup cost | none — works immediately | training run, tuning, GPU time |
| Generalises to new videos | **no** | yes, within its domain |
| Best when | few videos, or every video looks different | many videos that look alike |
| Fails when | you have 500 similar videos | your videos are all different |

**This is the tool for heterogeneous footage.** If every recording has a
different animal, camera angle, background or lighting, a trained detector
generalises worst exactly where your data varies most. Tracking never had to
generalise — it is only ever asked about the video in front of it.[^pan]

## In the classroom

Ethographs keypoint labelling mode can also be used in an animal behaviour class. Students meet in groups, use
their phones to film a short video of one or multiple animals doing a behaviour. Using their laptops (no GPU required), they
can label a a few keypoints and get started with pose-estimation within an hour.

Once a few frames are labelled and the rest filled in, the video can be loaded
into the GUI with velocity, speed and acceleration ticked. The traces sit on the
same time axis as everything else in the recording, so students can read them
against the video, the audio waveform or the spectrogram and check they match
what the animal was actually doing. {doc}`../examples/create_dataset_cricket`
does exactly this, putting leg kinematics next to the sound they produce.

Confidence and the {ref}`target-correction-loop` also make the measurement itself
discussable — where an automated number is trustworthy and where a human still
has to look.

## The workflow

The dialog is one tab per stage.

### 1. Define keypoints

A single shared schema — a list of keypoint names — plus one or more
**individuals**, each an instance of that schema (the SLEAP arrangement). The
tree shows a branch per individual and a leaf per keypoint, with a per-frame
mark and a `labelled/total` count.

Untick **"Individuals share the same keypoints"** to give each individual its
own subset (a chick with no visible tail, say). Names outside an individual's
set can never be labelled for it and stay empty in every export.

Two visual channels run through the whole dialog and the video overlay:
**shape = individual, colour = keypoint**. Colouring both the same way could not
show you which beak belongs to which bird.

### 2. Label & Edit

Arm **Sequential** or **Loop** and click the video:

- **Sequential** — label every keypoint on one frame. Each click places the
  active keypoint and advances to the next one this individual still lacks.
  It never navigates.
- **Loop** — sweep one keypoint across frames. Each click places it, then does
  whatever **Then go to** says: step one frame, jump to the next suggested
  frame, or stay put. The same dropdown says where `Shift+H` lands.

`Tab` cycles keypoints, `1`–`9` pick the individual, `Backspace` deletes the
active point, `Ctrl+Z` undoes, `Shift+H` approves this frame's predictions (see
{ref}`target-correction-loop`). Clicking an existing point always selects and
drags it — correcting never requires switching mode first.

The overlay shows **both** your labels and the fill's predictions, so you can
judge a prediction before accepting it: a label is a **solid** marker, a
prediction is the same shape and colour drawn **hollow**, left empty so you can
see the pixels underneath. Clicking a hollow marker pins it as a label and it
turns solid; dragging one corrects it first.

While a mode is armed, left-drag labels and panning moves to `Shift`+left-drag.
Tick **Lock** to look around instead: left-drag pans again and clicks no longer
place, move or pin anything. The labels stay on screen and the active keypoint
is kept, so unticking carries on where you were — unlike stopping the mode,
which takes the anchor overlay with it.

#### Which frames to label

Labelling consecutive frames is close to wasted effort: neighbouring frames look
almost identical, so the second one tells the tracker nothing the first did not.
The **Which frames to label** group proposes a spread instead. Ask for a *share*
of the video — the default is **10%, roughly every 10th frame**, the density
CoTracker3 is evaluated at for this task (6 labelled frames of 60[^pan]).

| Method | What it picks | Use it |
|---|---|---|
| **Evenly spaced** | equally spaced frames, no video scan | first pass; hard to beat on a short clip of one behaviour |
| **Biggest pixel change** | frames that differ most from their predecessor | fast motion, where a spline cuts the corner |
| **Most different frames** | k-means over frame thumbnails, one per cluster (DeepLabCut's method) | long, varied recordings |
| **Lowest fill confidence** | frames the last fill scored worst, within the span it covered | **after a fill** — the correction loop |

`N` — or **Next suggested frame** — jumps to the next suggestion, wrapping at
the end; plain `←` / `→` still move one frame at a time. There is no key for the
*previous* suggestion: the list is a queue to work down, and clicking a row of
the points table seeks to any frame, suggested or not.

### 3. Fill and export

Pick a backend, press **Fill frames between labels**, and every frame from your
first label to your last gets a position. Your labelled frames always come back
exactly as you placed them — a fill never overwrites a label, and never feeds on
the previous fill.

Frames outside that span are left empty. Before the first label and after the
last there is no second label to interpolate towards and no gap to track across,
so anything put there would be a guess extended from one end — asserted with the
same confidence as a properly bracketed frame. Label a frame further out and the
next fill reaches it: on a 1000-frame video labelled at frames 100 and 500, you
get frames 100–500, and labelling frame 900 extends that to 100–900.

(target-labelled-span)=
#### Everything happens between your labels

The span between your outermost labels is the boundary for the whole workflow,
not just for the fill:

- **Filling** runs in the gaps *between* labelled frames and nowhere else.
- **Confidence** is only scored where something was filled.
- **Lowest fill confidence** only suggests frames inside that span. Outside it
  there is no prediction to be unsure about, so there is nothing to correct and
  nothing to rank — an unlabelled frame is not a bad prediction. Extending the
  span is something you do on purpose, by labelling a frame further out, not
  something the correction loop drifts into.
- **Export and Load into the GUI** cover every frame of the video, with
  `NaN` position *and* `NaN` confidence outside the span — the same way
  [movement](https://movement.neuroinformatics.dev) represents a missing point.
  Nothing downstream (plots, kinematics, NWB) invents a value there.

This is what makes partial labelling well-defined. If only part of a recording
interests you — one bout, one trial, the minute the animal is actually in frame
— label its start and its end and work inside it; the rest of the video stays
`NaN` rather than being padded with extrapolated positions you would then have
to detect and discard. The trade is the obvious one: you get results exactly
where you put labels, so a stretch you never bracketed produces nothing at all,
however long you spend filling.

## Choosing a fill backend

| Backend | How it works | Needs | Cost |
|---|---|---|---|
| **Spline** (default) | Monotone cubic (PCHIP) interpolation per keypoint over its own labelled frames[^pchip] — geometry only, the pixels are never read | nothing extra | instant |
| **Optical flow** | Pyramidal Lucas–Kanade, tracked forwards and backwards across each gap[^lk] | `opencv-python-headless` | ~video speed, CPU |
| **PosePAL (CoTracker3 + refinement)** | A transformer point tracker[^cotracker] whose per-keypoint appearance features are first fitted to *your* labels on *this* video[^pan], then tracked forwards and backwards across each gap | torch + cotracker, **GPU** | a few minutes to fit, reused by every fill made from the same labels |

Installation for the optional backends is covered in
{ref}`target-keypoint-fill`.

**Without a GPU**, start with the spline — it costs nothing and is hard to beat when motion is smooth and your labels are dense — and switch to optical flow when the path between two labels isn't a smooth curve, like a fast turn or a wingbeat.

**With a GPU**, use PosePAL.[^pan] It handles longer gaps and larger
displacements than either of the others, and because it learns what *your*
keypoints look like in *this* recording, it stays on the right leg and the right
animal.

### Fit and track

PosePAL fills in two phases, and it is worth knowing which one you are paying
for:

- **Fit** — optimise CoTracker3's per-keypoint appearance features against the
  frames you labelled. Minutes on a GPU. It depends on nothing but your labels
  and the video.
- **Track** — run the fitted tracker forwards and backwards across each gap.
  This is what produces the filled frames, and it is fast.

**Fill** always does both, but it skips the fit while the fit it already has was
made from exactly the labels you have now — held in memory and cached beside the
video as `<video>.posepal.pt`. Correct a point, approve a frame, edit the schema,
and that no longer holds: the next **Fill** refits by itself before tracking.
There is no fit button, because there is no decision to make; the Fill tab simply
says which phases the next fill will pay for, so a three-minute wait is never a
surprise. Cancelling a fill leaves your labels, your current fill and the current
fit exactly as they were.

So there is no "carry on from the old fit" anywhere: **a refit is a fresh fit**,
started from scratch on all your labels — the reference implementation
re-optimises on every press for the same reason. The only thing the cache buys
you is not repeating a fit that would come out the same, which is what keeps the
correction loop below quick.


(target-correction-loop)=
## The correction loop

The fill is a first draft. The loop that makes it accurate:

1. **Fill.**
2. Sort or filter the points table by **Confidence** — or set the suggestion
   method to **Lowest fill confidence** and press **Suggest frames**. Both rank
   by the same number.
3. Correct the worst frames. Clicking a filled point pins it where it is (that
   is how you *accept* a prediction); dragging it corrects it.
4. Approve the rest. Reviewing a fill is mostly *agreeing* with it, so agreeing
   is one key: **Approve frame** (`Shift+H`) keeps every predicted point on the
   current frame as your own label — all individuals at once — and then moves on
   as **Then go to** says. Points you had already labelled are untouched, so
   correcting one keypoint and approving the frame is two actions, not a choice
   between them.
5. **Fill again.** Re-filling is a pure function of your labels, so nothing you
   have not touched drifts further — and everything you approved is now ground
   truth the next fill tracks from rather than re-derives. With PosePAL this
   round of corrections is also more training data, so the fill refits on it
   first: your corrections improve the tracker, not only the frames you fixed.

Confidence means different things per backend, which the column's header tooltip
also spells out: the spline decays it with distance from the nearest labelled
frame; the tracking backends track each gap twice — forwards from the label on
its left, backwards from the one on its right — and score by how far the two
tracks disagree, falling to zero where either direction reports the point lost.
**Disagreement tolerance** sets how many source pixels of disagreement costs a
factor of 1/e; raise it for large or fast animals. It changes only the scores,
never the positions.

```{note}
Your labels are project data, saved to `<video>.keypoints.json` next to the
video — not to app settings. A cached PosePAL fit sits beside it as
`<video>.posepal.pt`.
```

## Getting the result out

- **Load into the GUI** — adds the keypoints, and whichever kinematics are
  ticked (velocity, speed, acceleration), to the current trial as ordinary
  features, so you can plot them straight away. Filled frames are included:
  seeing what the fill did is the point. No file is written, and this works even
  when the session has no dataset behind it — the keypoints *are* one.
- **Export poses (NetCDF)…** — a [movement](https://movement.neuroinformatics.dev)-compatible
  poses dataset covering every frame of the video; frames outside the filled
  span carry `NaN` position and `NaN` confidence, which is how movement
  represents a missing point (see {ref}`target-labelled-span`).


## References

[^pchip]: Fritsch, F. N. & Carlson, R. E. (1980). [Monotone Piecewise Cubic Interpolation](https://doi.org/10.1137/0717021). *SIAM Journal on Numerical Analysis*, 17(2), 238–246. Implemented by [`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

[^lk]: Lucas, B. D. & Kanade, T. (1981). [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). *IJCAI*, 674–679. The pyramidal form used here is Bouguet, J.-Y. (2001), [Pyramidal Implementation of the Lucas Kanade Feature Tracker](https://robots.stanford.edu/cs223b04/algo_tracking.pdf), via [`cv2.calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323).

[^cotracker]: Karaev, N., Makarov, I., Wang, J., Neverova, N., Vedaldi, A. & Rupprecht, C. (2024). [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831). [Project page](https://cotracker3.github.io/) · [GitHub](https://github.com/facebookresearch/co-tracker)

[^pan]: Pan, Z., Pan, B., Yang, G., Harley, A. W. & Guibas, L. (2025). [Animal Pose Labeling Using General-Purpose Point Trackers](https://arxiv.org/abs/2506.03868). *arXiv:2506.03868*. [Project page](https://zhuoyang-pan.github.io/animal-labeling) · [Reference implementation](https://github.com/Zhuoyang-Pan/PosePAL)
