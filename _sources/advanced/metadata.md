(target-metadata)=
# Trial metadata

Attach per-trial conditions (e.g. stimulus, reward outcome) via a TSV file. The
trials table in the GUI turns those columns into filters, restricting navigation
and analysis to a subset of trials.

```{important}
**The trials table's filters are the one trial filter in EthoGraph, and they
apply to everything.** Filter, say, `num_pellets` to `1, 2` (not `0`) in the
trials table, and every operation from then on sees only those trials:
navigation, label and sequence jumps, changepoint correction, purging short
labels, curation (Ctrl+C, inspect mode, frame-by-frame review, the label and
video grids), model **training** and model **inference**. A label in a
filtered-out trial is not visited, not curated, not trained on and not
predicted over — it simply does not exist for those operations until you
widen the filter again. No dialog has a trial filter of its own; each one says
how many trials it will run over, read off the table.
```

---

## The metadata file

Tab-separated (`.tsv`); `.csv`, `.xlsx` and `.xls` also work. One row per trial.
The only required column is **`trial`**, matching the trial IDs in your dataset.

```
trial   food_pellet_side   rewarded
1       left               yes
2       right              yes
3       left               no
4       right              yes
```

Add any columns you like — categorical (pellet side, protocol variant) or
numeric (stimulus intensity). Values may be strings, ints or floats; missing
values are allowed.

Some names are reserved for structure rather than conditions: `start_time` /
`stop_time`, plus `video_*`, `audio_*`, `pose_*`, `ephys_*` and `*_start`, are
alignment-NWB trials-table columns (timing, media paths, offsets) and are
ignored when they appear in a metadata table — **trial timing always comes
from the alignment NWB, never from metadata**. A metadata table contributes
only its condition columns, joined on `trial`.

---

## Loading metadata

Loading a dataset auto-detects a sidecar `{stem}_metadata.tsv` beside it — e.g.
`session.nc` picks up `session_metadata.tsv`.

To use a different file, set the **Metadata:** field in the loader form on the
start page (*Custom set-up* card) before clicking **Load**. It accepts a
tabular file (`.tsv` / `.csv` / `.xlsx`) with a `trial` column; other file
types are ignored. The path is saved with the project.

For pynapple folders whose trial timing lives in a `trials.npz` IntervalSet:
the loader never reads timing from it. When no alignment NWB exists, the start
page offers — once — to convert the IntervalSet into
`.ethograph/alignment.nwb` (its metadata columns travel into the trials
table); after that, the alignment file is the single per-trial record. The **Template** button next to it writes a
`{stem}_metadata.tsv` pre-filled with all trial IDs, ready to edit in a
spreadsheet.

Sources are tried in this order:

1. The **Metadata:** field (or `metadata_path` in the API).
2. The **NWB trials table**, when the data source is a `.nwb` file.
3. The **sidecar TSV** next to the data file.
4. **Pynapple `IntervalSet` metadata**, for `.npz` or folder sources.

With none of these, trials carry no conditions and no filtering UI appears.
Drag & drop loading never uses a metadata table.

---

## The trials table

Top of the **Navigation** section in the right sidebar, above the navigation
controls. Lists every trial in the session; shown only when the metadata has at
least one column besides `trial`.

### Filtering

Click the funnel icon at the right edge of a column header (the rest of the
header sorts). Categorical columns give a checkbox list with an **(All)**
toggle; numeric columns give a `≥` / `≤` threshold with a **Remove filter**
button. Filters are AND-combined, and the funnel turns yellow while one is
active.

Filtered-out trials disappear from the table and from the trial navigator — the
*Previous / Next trial* buttons and the trial slider skip them. A combination
matching no trials is ignored rather than emptying the navigator.

And not just the navigator: as the box at the top says, **every** operation —
changepoint correction, purging short labels, curation, model training and
inference — runs over the filtered trials only.

---

(target-label-filter)=
### Filtering by what the labels do

The column filters ask about metadata. **Tools ▸ Find label inconsistencies…**
asks about the labels themselves — which trials have an event without its
partner, which carry a label twice, which ran the classes in an order they
should not have, which are missing a sequence altogether. Type the label ids the way the Sequence
navigator takes them (`1-2-6-8`) and pick the question:

| | |
|---|---|
| **All of them occur** | the classes are all somewhere in the trial, any order |
| **Some but not all occur** | one event without its partner — the uncoupled case |
| **Any of them occurs more than once** | a class that should happen once per trial happens twice — a doubled click, a prediction that fired twice |
| **In this order** | in that order, other labels allowed in between (`1-2-6-6-8` matches `1-2-6-8`) |
| **In this order, one straight after another** | the same, contiguously (`1-2-6-6-8` does *not* match) |

**Invert** turns any of them into "find the trials where this is *not* true",
which is how you ask which trials are missing the sequence. With more than one
animal labelled, pick whose labels to read — two animals' events interleave,
and an order across both means nothing.

The count updates as you type. **Filter trials to these** puts the answer into
the table's own **label filter**, a slot that sits *on top of* the column
filters — so "wild-type trials where the order broke" is one question, and
asking it does not throw the genotype filter away. The status line says when
it is on, **Clear label filter** takes it off, and the column filters are
untouched either way. Nothing about the labels is ever modified.

---

### The `curated` column

EthoGraph maintains one column itself: **`curated`** is `1` when every label
of the trial is `manual` or `curated` and `0` while any is still a model's
unreviewed `automated` output (see {doc}`labels/curation`). It is refreshed
every few seconds while you curate rather than on every edit, so labelling
never waits on a file write, and it flips back to `0` whenever new predictions
land in a trial. Filter on it like any other column to walk only the trials
that still need a look.

## Editing metadata as you watch

Some metadata is only knowable once you have watched the trial — whether the
animal engaged, whether the recording is usable. Tick **Edit values on
double-click**, above the trials table, and the table becomes editable.

Double-click a cell in the current trial's row (the tinted one) to change its
value; the editor offers the values that column already uses, and accepts
anything else you type. **Add column…** starts a new column to fill in.

### Saving is automatic — Ctrl+S is only for labels

Every edit is written to disk on its own: about a second after you stop typing,
and again whenever you change trial or close the app. **There is no save button
and no Ctrl+S for metadata** — `Ctrl+S` / *Save labels* writes the label TSV and
nothing else. Since edits overwrite the source file with no undo, keep a backup
copy of your metadata before you start editing.

### Where the edits go

Straight back into the source the metadata was read from — the tabular file, or
the NWB trials table (edited columns only). Anything else (pynapple
`IntervalSet`, no metadata yet) gets a sidecar `{stem}_metadata.tsv`, which
outranks it on the next load. There is no undo.

**One limit:** an NWB column keeps the dtype it was written with, so text
cannot go into a numeric NWB column. Use a new column (or a TSV metadata file)
for free-text values.

---

## Export

Metadata is merged into exported label DataFrames by `enrich_labels_df()`, so
every label row carries its trial's condition columns.

---

## References

- {ref}`NWB alignment <target-nwb-alignment>` — trial timing metadata in NWB
- {doc}`labels/index` — label export and the enriched labels DataFrame
