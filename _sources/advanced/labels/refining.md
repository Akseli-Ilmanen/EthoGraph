(target-refining-labels)=
# Refining labels frame-by-frame

EthoGraph's default labelling is **time-series based**: you place boundaries by
clicking on the plots, because scanning a line plot or spectrogram is much
faster than hunting for the exact video frame. But some boundaries are defined
by what happens *in the video* — a foot lifting, a beak opening — and for those
a frame-accurate pass is worth the extra care.

**Tools ▸ Labels: Refine via frame-by-frame labelling…** turns the labels you
already placed into *seeds* and walks you through them:

<!-- TODO: image comparing frame-by-frame vs time-series based labelling
     (side by side: clicking a boundary on the line plot vs stepping video
     frames around a seed with the refine dialog open).
![refine_vs_timeseries](../../_static/media/refine_vs_timeseries.png) -->

1. Tick the label classes to refine (and optionally one individual), then
   **Start refining**. Each point event becomes one seed; each state event
   becomes a start seed then an end seed. Seeds are visited in time order,
   trial by trial, so every trial is visited once.
2. Each seed opens in a small window centred on the boundary, with the label
   (and START/END/POINT) named in large coloured text.
3. Nudge the video with `Left` / `Right`, press `Enter` — the frame on screen
   becomes the new boundary and the dialog jumps to the next seed. **Skip**
   and **Back** move without committing.
4. If the event should not be there at all — a prediction of something that
   never happened in this trial — press `Backspace` instead. The label is
   deleted (both boundaries of a state event go together) and the queue moves
   straight on. Nothing reaches disk until you save with `Ctrl+S`.

A boundary you confirm becomes a **hand-made label**: its `confidence` is set
to `1.0`, whatever produced it first (see
{ref}`confidence <target-onset-model-confidence>`). Reviewing a model's output
this way is how predictions turn into ground truth.

## Refining only the events you picked

Walking every instance of a class is the right pass for a first careful
labelling round, but a review pass is usually about the handful that look
wrong. **Tools ▸ Labels: Show frames as Grid/PDF…** is where you spot them:
each tile shows the video frame at a label's time and its confidence.

1. Pick the labels on the **Setup** tab and press **Generate**: the grid fills
   the **Frames** tab of the same window. Set **Flag confidence below** so the
   doubtful tiles are outlined in red, and scan the sheet — maximise the window
   to spread the tiles over the whole screen, they refit to the new width.
2. Tick the tiles that need work. **Tick flagged** ticks every outlined one at
   once; **Tick their whole trials** widens that to every event of every trial
   holding a flagged one — a trial the model got one event wrong in is worth
   reading end to end, because its other events may score high and still sit on
   the wrong frame. Then press **Refine ticked frame-by-frame…**.
3. The refinement dialog opens with a queue holding exactly those boundaries
   and nothing else. Walk it with `Enter` / `Backspace` as above.

Two cameras showing the same label are two tiles but one boundary: the queue
stops at it once. When the queue runs out, EthoGraph suggests saving and then
regenerating the grid — the tiles you just corrected still show the old frames,
and a fresh grid is the check that the pass worked.

For corrections that belong far from the current label, untick **Locked around
initial label**: you can then pan and zoom the whole trial freely and `Enter`
still commits whatever frame is on screen. Navigating trials the normal way
(trial combo, `Up`/`Down`) pulls the session along to that trial's first seed.

Seeds don't have to be hand-placed. Because the queue is built from the labels
TSV, you can generate first-guess labels **programmatically from a time-series
criterion** — say, the first frame where beak opening exceeds a threshold
width — write them into the `{name}_labels.tsv` file (see
{ref}`the column reference <target-exporting-labels>`), load it into the GUI,
and use this dialog to walk the automatic guesses and correct each one to the
true frame. A rough automatic pass plus a fast frame-accurate review is often
far quicker than either alone.

Progress is remembered per dataset: **Resume last session** returns to the
exact seed you stopped at, and **History…** lists every refined boundary
(filterable by trial/label/individual/boundary) with an export that writes
`_prerefined.tsv` / `_postrefined.tsv` — the original and refined values, row
for row.
