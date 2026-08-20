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
