(target-correction-loop)=
# 5. The correction loop

The fill is a first draft. The loop that makes it accurate:

1. **Fill.**
2. Set the suggestion method to **Lowest fill confidence** and press **Suggest
   frames**. It ranks each frame by its *worst* keypoint, so one point the
   tracker lost is enough to bring the frame forward — which is what the points
   table's per-keypoint **conf** columns then show you when you get there.
3. Correct the worst frames. Clicking a filled point pins it where it is (that
   is how you *accept* a prediction); dragging it corrects it.
4. Approve the rest. Reviewing a fill is mostly *agreeing* with it, so agreeing
   is one key: **Approve frame** (`Shift+H`) keeps every predicted point on the
   current frame as your own label.
5. **Fill again.** Re-filling is a pure function of your labels, so nothing you
   have not touched drifts further — and everything you approved is now ground
   truth the next fill tracks from rather than re-derives. With PosePAL this
   round of corrections is also more training data, so the fill refits on it
   first: your corrections improve the tracker, not only the frames you fixed.

Confidence is scored **per keypoint**, and the points table shows it that way —
a `conf` column beside each keypoint's `x` and `y`, rather than one figure for
the whole row. A row-level average is exactly what hides the problem: nine
well-tracked points pull a frame with one lost point back into the middle of the
ranking, when that one point is the only reason to go back at all.

Confidence means different things per backend, which each `conf` header tooltip
also spells out. A labelled point always scores 1, and both backends decay from
there.

**Spline** never reads the pixels, so all it can report is how far it had to
reach — $d$ frames to your nearest label of that keypoint:

$$c = e^{-d/10}$$

**Optical flow and PosePAL** track each gap twice, forwards from the label on its
left and backwards from the one on its right, and score by how far apart the two
answers land — $\Delta$ pixels:

$$c = e^{-\Delta/D}$$

Two independent passes agreeing is the evidence that the point is right; an
occlusion sends them to different places. A point either pass reports as lost
scores 0. $D$ is the **Disagreement tolerance** spin box, in source-video pixels:
how far the two may drift before the point counts as unreliable. Raise it for
large or fast animals, lower it to be strict. It changes only the scores, never
the positions.

With markers, the same loop starts one step earlier: **Detect**, then set the
suggestion method to **Where the detector saw nothing** and label those frames by
hand — the detector is not *unsure* there, it is *absent*, so its failures are a
set of frames rather than a low score.

```{note}
Your labels are project data, saved to `<video>.keypoints.json` next to the
video — not to app settings, and neither is what each detector label means,
which is saved in the same file. A cached PosePAL fit sits beside it as
`<video>.posepal.pt` and a cached detector run as `<video>.detections.npz`;
both are derived data you can delete at the cost of recomputing them.
```
