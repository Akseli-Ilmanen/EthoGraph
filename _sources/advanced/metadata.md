(target-metadata)=
# Trial metadata

Attach per-trial conditions (e.g. stimulus, reward outcome) via a TSV file. The
trials table in the GUI turns those columns into filters, restricting navigation
and analysis to a subset of trials.

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

**Note:** operations across trials, such as changepoint correction or purging
short labels, apply only to the filtered trials.

---

## Export

Metadata is merged into exported label DataFrames by `enrich_labels_df()`, so
every label row carries its trial's condition columns.

---

## References

- {ref}`NWB alignment <target-nwb-alignment>` — trial timing metadata in NWB
- {doc}`labels/index` — label export and the enriched labels DataFrame
