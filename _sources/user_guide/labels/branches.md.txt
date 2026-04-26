(target-label-branches)=
# Label branches

Within a single branch, each timepoint can only belong to **one label**.
When you place a new label that overlaps an existing one in the same branch,
the existing label is trimmed or split automatically. This is by design:
downstream action-segmentation models require a single class per frame, and
the plot overlays assume non-overlapping intervals.

Sometimes, however, you need **overlapping annotations** at different
timescales --- for example, brief transient events like syllables or
movements (`song`, `peck`, `jump`) on one tier, and longer behavioural
states like arousal or passivity (`active`, `resting`, `sleep`) on another.
A bird can be `active` and producing a `song` at the same time, and those
annotations need to coexist. Label branches solve this.

---

## How branches work

Each label in `mapping.txt` belongs to a branch (default `0`). Labels in
**different branches are independent**: they can overlap freely without
trimming each other. Labels in the **same branch** remain mutually exclusive.

Exactly one branch is the **Main labels** branch (the editing branch). When
a branch is the Main slot:

- Keyboard shortcuts and click-to-label only create intervals for labels in
  that branch.
- Existing labels in other branches are protected --- they are never trimmed
  or overwritten.
- Up to two additional branches (or the imported predictions) can be shown
  alongside the Main labels as thin top strips --- see
  {ref}`label-display-slots` below for how to configure them.

```{raw} html
<video autoplay loop muted playsinline style="width:100%">
  <source src="../../_static/videos/branch.mp4" type="video/mp4">
</video>
```

---

## Assigning branches in `mapping.txt`

Add a third column to any line in `mapping.txt`:

```
0 background
1 song 0
2 peck 0
3 jump 0
4 active 1
5 resting 1
6 sleep 1
```

Labels without an explicit branch number default to branch `0`.
In this example, transient events (`song` / `peck` / `jump`) live on
branch 0 and behavioural states (`active` / `resting` / `sleep`) on
branch 1 --- a `song` interval and an `active` interval can overlap
because they belong to different branches.

See {doc}`mapping` for the full mapping file format.

---

## Switching branches in the GUI

Each branch appears as a section in the Labels panel. The branch shown in
the **Main** slot of the *Overlays* row (see below) is the editing branch.
Press **Shift+B** to toggle the Main slot between the current and the
previously-selected branch.

You can also **drag labels between branches**: grab a label row from one
branch table and drop it onto another. The mapping file is updated
automatically.

---

(label-display-slots)=
## Choosing what to display: Main / Top 1 / Top 2

The **Overlays** row in the Data panel has three dropdowns that decide
which annotations are drawn on the plots:

| Slot | Where it draws | Allowed values |
|------|----------------|----------------|
| **Main**  | Full plot height (or a single bottom strip on spectrogram-style plots) | Any branch, or `(none)` |
| **Top 1** | Thin strip at the top of the plot                                      | Any branch, the imported `Predictions`, or `(none)` |
| **Top 2** | Thin strip just below Top 1                                            | Same options as Top 1 |

Defaults and auto-fill behaviour:

- On startup, Main is set to **Branch 0** and both top slots are empty.
- When you **import predictions**, they are placed into Top 1 if it is
  empty, otherwise into Top 2. If both are taken, the slots are left
  unchanged --- pick a slot manually.
- When you **add a new branch** with the **+** button, it is placed into
  Top 1 if it is empty, otherwise into Top 2. Same fallback as above.
- Press **Ctrl+Y** to toggle Predictions in or out of the top slots.

Changing a dropdown immediately redraws every plot.

---

## Adding and removing branches

- Click the **+** button below the branch tables to add a new empty branch.
  The new branch is auto-assigned to the first free top slot.
- Click the **x** button on a branch header to delete it. A branch can only
  be deleted when it contains no labels --- move all labels out first.
  Slots that referenced the deleted branch are reset to `(none)`.
