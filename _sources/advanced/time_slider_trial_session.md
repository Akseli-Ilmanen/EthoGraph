# Time slider & trial/session

EthoGraph shows one recording through two lenses: a **trial** lens, where the
plot axis restarts at 0 for every trial, and a **session** lens, where one
absolute clock spans the whole recording. The **Slider scope** combo decides
which lens is active, and that choice — called the *display basis* — governs
every time you see or click in the app.

## The one rule

> The slider scope decides the clock of the plot axis. Everything —
> features, labels, the time marker, video seeking, audio playback, spike
> rasters — speaks that clock, and exactly one converter translates between
> it and per-trial storage.

| Slider scope | Axis clock | Trial 28 (e.g. 347–353 s into the session) appears at |
|---|---|---|
| Trial start → Trial end / Trial start → Trial start (i+1) | trial-relative (starts at 0) | 0 – 6.1 s |
| Session start → Session end | session-absolute | 347 – 353 s |

Label and Sequence navigation windows are built from label onsets, which are
stored trial-relative — those views are always trial-basis, even when the
slider scope is set to session.

## What session scope shows

- **Features** — pynapple sources are natively session-absolute and render
  across the whole session. Multi-trial `.nc` (xarray) datasets are natively
  trial-local: the **current trial** renders at its true session position;
  other trials' features are absent (stitching every trial onto one axis is
  deliberately not supported — use a pynapple export for session-wide
  analysis).
- **Labels** — *every* trial's labels appear at their session positions, not
  just the current trial's.
- **Video** — per-trial video files follow the slider: rest the time marker
  inside another trial's span and, after a short settle (~0.3 s), that
  trial's video loads and shows the frame under the marker. Crossing a trial
  boundary re-opens a video file, which takes a moment (the plots stay
  live). In inter-trial gaps the last frame holds.
- **Labelling works across trials** — click any trial's span to label it:
  the trial under the click becomes current, and the label is stored in that
  trial (trial-relative). Clicks in the gaps between trials are refused, as
  is an interval whose two clicks fall in different trials — a label belongs
  to exactly one trial.
- **Ephys** — rasters, traces and firing rates convert between the recording
  clock and the axis via one shared rule; the scalar *Ephys offset* setting
  applies identically to trace, raster and PSTH.

## What trial scope shows

The classic per-trial view: axis 0-based, one trial's data, video and labels.
This is unchanged — and it is the coherent mode for both backends, including
pynapple sources (whose absolute times are rebased onto the 0-based axis).

## Where label times live on disk

Label TSVs store **trial-relative** onsets/offsets plus a `trial` column —
regardless of the scope you labelled in. Saved files carry a header line:

```text
# time_basis: trial
```

When you import a TSV without that header, EthoGraph infers the basis per
trial (onsets inside `[0, trial duration]` → trial time; inside the trial's
session window → session time) and rebases session-absolute files
automatically. If the file is genuinely ambiguous you are asked once —
trial-relative or session-absolute — and the answer is written into the
header on the next save, so the question never repeats.

Exports add derived `onset_global` / `offset_global` columns (trial start +
trial-relative onset) for analysis in session time; the internal columns stay
trial-relative.

## Backend cheat-sheet

| | trial scope | session scope |
|---|---|---|
| pynapple (`.npz`, folders, NWB) | axis 0-based; data rebased per trial | native — full session renders |
| xarray (`.nc` TrialTree) | native — full trial renders | current trial at its true position; other trials absent |
| labels (TSV) | current trial's rows | all trials' rows, shifted |
| video (per-trial files) | current trial's file | follows the marker across trials |

Trial timing itself comes from the alignment NWB's trials table (see
{doc}`metadata`): the same `start_time`/`stop_time` values drive the session
axis, the trial switcher, and the conversion between the two clocks.
