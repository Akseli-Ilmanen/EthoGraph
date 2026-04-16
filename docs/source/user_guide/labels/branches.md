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

Only **one branch is active** at a time. When a branch is active:

- Keyboard shortcuts and click-to-label only create intervals for labels in
  that branch.
- Existing labels in other branches are protected --- they are never trimmed
  or overwritten.
- Plot overlays show labels from all branches, but only the active branch's
  labels are clickable / editable.

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

Each branch appears as a collapsible section with a checkbox in the Labels
panel. Click a branch checkbox to make it active (radio behaviour --- only
one branch active at a time). Press **Shift+B** to toggle between the
current and previous branch.

You can also **drag labels between branches**: grab a label row from one
branch table and drop it onto another. The mapping file is updated
automatically.

---

## Adding and removing branches

- Click the **+** button below the branch tables to add a new empty branch.
- Click the **x** button on a branch header to delete it. A branch can only
  be deleted when it contains no labels --- move all labels out first.
