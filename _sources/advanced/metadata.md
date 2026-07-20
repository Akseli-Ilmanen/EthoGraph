(target-metadata)=
# Trial metadata

EthoGraph lets you attach per-trial condition metadata (e.g. stimulus conditions) via a plain TSV file. Once loaded, the **Trials tab** in the GUI turns these columns into interactive filters so you
can restrict navigation and analysis to any subset of trials.

---

## The metadata TSV file

### File format

The file must be tab-separated (`.tsv`). Comma-separated (`.csv`) and Excel
(`.xlsx`) are also accepted.

One row per trial. The only required column is **`trial`**, which must match
the trial identifiers in your dataset.

```
trial   food_pellet_side   rewarded
1       left               yes
2       right              yes
3       left               no
4       right              yes
5       left               yes
6       right              no
```

- Column names are arbitrary — add as many as you need.
- Values can be strings, integers, or floats.
- Missing values are allowed.

### What columns to include

Any variable that differs across trials:

- **Categorical:** pellet side (left/right), reward outcome, protocol variant, recording site, …
- **Numeric:** trial number within session, stimulus_intensity, …

Columns named `start_time` and `stop_time` (in seconds) are treated as trial
timing and used to set navigation boundaries rather than shown as filterable
conditions. Columns following the `video_*`, `audio_*`, `pose_*`, `ephys_*`
naming convention are treated as media paths and are also hidden from the
filter UI.

---

## Loading metadata

### Auto-detection

When you load a dataset, EthoGraph automatically looks for a sidecar file named
`{data_stem}_metadata.tsv` in the same folder. For example, loading `session.nc`
will pick up `session_metadata.tsv` if it exists.

The full priority order is:

1. **Explicit path** set in the **Metadata:** field of the I/O panel (or via
   `metadata_path` in the API).
2. **Sidecar TSV** — `{stem}_metadata.tsv` next to the data file (auto-detected).
3. **NWB trials table** — extra columns in `nwb.trials` (for `.nwb` sources).
4. **Pynapple IntervalSet metadata** — metadata columns from `.npz` or folder
   sources.

### Using a custom path or filename

If your file has a different name or location, point to it in the
**Metadata:** field in the I/O panel (the browse button next to the field) before
loading the dataset. The path is saved in the project settings and restored on
next open.

### Generating a template

Click the **Template** button next to the Metadata field. EthoGraph writes a
`{stem}_metadata.tsv` pre-populated with all trial IDs and opens it in the
field — edit it in a spreadsheet editor, then reload the dataset.

---

## The Trials tab

The **Trials tab** (collapsible panel in the sidebar) shows all trials in the
loaded session as a table and lets you filter which trials are active in the
navigator.

### Filtering trials

Click the filter icon (≡) in any column header. Categorical columns show a
checkbox list of unique values; numeric columns offer a `≥` / `≤` threshold.
All active filters are AND-combined. A yellow icon indicates an active filter;
click it again and choose **Remove filter** to clear it.

### Effect on navigation

Filtered-out trials are hidden in the table and removed from the trial
navigator — the *Previous / Next trial* buttons and the trial slider skip them.
The set of visible trials is updated live whenever a filter changes. **Note**:
Operations such as changepoint correction across trials, or purging short labels
are applied only to filtered trials.


---

## Export labels

Metadata is also merged into exported label DataFrames automatically via
`enrich_labels_df()`, so every label row carries the trial's condition columns.

---

## References

- {ref}`NWB alignment <target-nwb-alignment>` — trial timing metadata in NWB
- {doc}`labels/index` — label export and the enriched labels DataFrame
