(target-curation)=
# Curating labels

Every label carries a `labeling_method` — the vocabulary of
[ndx-ethogram](https://github.com/catalystneuro/ndx-ethogram):

| Value | Meaning | Drawn as |
|-------|---------|----------|
| `manual` | A human placed or last edited it | solid outline |
| `automated` | A model produced it and nobody has looked at it | **dotted** outline |
| `curated` | Automated output a human looked at and let stand | solid outline |

Curation is the move *automated → curated*. It never touches a manual label
(a human already vouched for it), and it never runs backwards: editing a
label makes it manual, re-running a model over a trial can add new automated
labels, and nothing else changes a method.

A **trial is curated** when none of its labels is still automated. That
verdict is everywhere you navigate — the trial combo in the Navigation section
and the `Trial 12 (12/173)` counter in the bottom bar are green for a curated
trial and red for one with automated labels left — and it is written to the
metadata table's `curated` column (`1` / `0`), refreshed every few seconds
while you work (see {doc}`../metadata`). Predicting new labels into a curated
trial turns it red again until those are curated too.

Everything about curation lives in one place: the **Curation** section at the
bottom of the **Labels** tab.

## Scope: which labels

Curation acts on a *scope* of label classes. Drag rows out of the label tables
above into the drop area — a multi-selection drags as one — and their ids are
listed there. Empty (or **All**) means every class; **Reset** empties the area
so other labels can be dragged in. The scope is remembered per dataset.

## Modes: how a label gets curated

**Manual (trial level)** — the default. Placing, moving or deleting a label
makes it manual, as always. Press `Ctrl+C` (or **Curate trial**) and every
automated label in scope of the current trial becomes curated; manual labels
stay manual.

**Inspect is enough (trial level)** — merely opening a trial curates its
automated labels in scope. Use it when a model is good enough that looking at
the trial is the review. The mode is per dataset, so it never follows you
silently into another one.

(target-curation-frame)=
**Frame-by-frame review** — the labels in scope become a queue of boundaries
(one per point event, a start then an end per state event, in time order,
trial by trial) walked one at a time, each centred in a small **View window**
(untick **Locked around label** to pan the whole trial). The boundary being
reviewed is named in large coloured text; the keys are drawn in the section,
and **Shortcuts…** spells them out:

| Key | Action |
|-----|--------|
| `←` / `→` | Step the video one frame |
| `Enter` | Confirm: the frame on screen becomes the boundary. A boundary that moved makes the label **manual** (`confidence = 1.0`); one confirmed where it stood becomes **curated** |
| `Backspace` / `Delete` | The event should not exist — delete it (both boundaries of a state event) and move on |
| `N` | Next boundary. With **N (next) = seen, mark curated** ticked (the default) the boundary you leave is curated |
| `B` | Back to the previous boundary |
| `Space` | Play / pause |

Navigating trials the normal way (trial combo, `Up`/`Down`) pulls the review
along to that trial's first boundary. Nothing reaches disk until you save with
`Ctrl+S`.

Seeds don't have to be hand-placed. Because the queue is built from the labels
TSV, you can generate first-guess labels **programmatically from a time-series
criterion** — say, the first frame where beak opening exceeds a threshold
width — write them into the `{name}_labels.tsv` file with
`labeling_method = automated` (see {ref}`the column reference
<target-exporting-labels>`), load it into the GUI, and walk the guesses here.
A rough automatic pass plus a fast frame-accurate review is often far quicker
than either alone.

## The grids

Two buttons in the section open review grids on the scope; both come with the
same **mode** combo and a **Done** button. Their *Setup* tab lists the labels
in scope for clarity but cannot change them — the scope area is the one place
labels are chosen, so close the grid, drag other rows in, and open it again.

**Label grid view…** shows the video frame at every boundary in scope — one
tile per point event, a start and an end tile per state event, per camera —
titled with label, trial, time, confidence and method, with **Flag confidence
below** outlining doubtful tiles in red and **Histogram…** showing where the
scores pile up (see {ref}`target-onset-model-confidence`). The grid exports to
a paginated PDF.

* *Click = navigate* — a tile click jumps the main GUI to that trial and time.
  In frame-by-frame mode it drops straight into the review at that boundary.
* *Click = curated* — click the tiles that are right (green); **Done** curates
  those labels.
* *Click = uncurated, rest = curated* — for a batch that is mostly right:
  click only the bad ones (orange), **Mark low-confidence as uncurated**
  pre-clicks what the threshold outlines (it exists only in this mode — a low
  score is a reason to doubt a label, never to approve it), and **Done**
  curates every other label.

**Video grid…** plays the labels instead of freezing them — a state event's
whole span, a point event's window (**Window around point events**) with a
red marker in the corner on the frame the event falls on. It is built for
comparison: only clips of **one label class** are on screen at a time
(**Previous / Next label** switch the class, greyed out when there is only
one), they are **sorted by duration** so clips of similar length share a
screen (**Clips on screen** sets how many; **Previous / Next clips** step
through the rest), and the view never scrolls — one **Play** button and one
slider spanning the longest clip on screen drive every tile at once, played
once and stopped, shorter clips holding their last frame; **←/→** pause and
step every tile one frame back or forward. The **speed** field
next to Play opens at the GUI's current playback speed and can be changed
for the grid alone, as a percentage of real time. The layout choices —
window around point events (0.5 s by default), clips on screen, columns —
are remembered across sessions and datasets, like the label grid's column
count. Each tile's caption says where in the
trial the label sits (`at 0.03 s` for a point event, `1.20–1.85 s` for a
state) and whether the clip had to be cut at the video's start or end — so a
point event that seems to show "the start of the trial" can be told apart
from one whose window was clipped. Clips decode a screenful at a time at a
reduced size, so opening long events takes a moment. Clicks mean the same as
in the label grid.
