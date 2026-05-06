(target-label-mapping)=
# Label mapping (`mapping.txt`)

EthoGraph uses **integer label IDs** , e.g. in the TSV file (`labels`
column). The `mapping.txt` contains a mapping from these integers to the label names. This is the same format as used in the [action-segmentation literature](https://github.com/nus-cvml/awesome-temporal-action-segmentation) where models predict a per-frame class index and a dataset-level `mapping.txt` lists the corresponding action names. 

One major benefit of this format is that if you rename a behaviour once in `mapping.txt`, it propagates everywhere (old backups, predictions, exports).


---

## Format

Whitespace-delimited, one label per line.  The full schema is
`<id> <name> [<branch>] [<event_type>]` — both trailing fields are
optional with defaults (`branch=0`, `event_type=state`):

```
0 background
1 pullOutStick
2 diagonalToBox
3 toss 0
4 nod 1
5 reachLeftCorner 0
11 peck 0 point
12 call 1 point
```

Rules:

- **ID `0` is always `background`**. It is excluded from display, export,
  and model loss.
- IDs don't have to be contiguous, but they must be unique.
- Names should be valid identifiers (no spaces) so they round-trip through
  Crowsetta text formats cleanly.
- Trailing whitespace is ignored; blank lines are skipped.
- `branch` is an optional integer grouping labels for independent labeling
  (see {doc}`branches`).  Defaults to `0`.
- `event_type` is `"state"` (interval, default) or `"point"` (instantaneous
  marker — see {ref}`target-point-events` below).  Lines without it are
  treated as state events, so existing mapping files keep working unchanged.

To declare a *point* class without a custom branch you must still write the
default branch explicitly, because the columns are positional:
`11 peck 0 point` — not `11 peck point`.

Read / write programmatically:

```python
from ethograph.labels.intervals import load_mapping
from ethograph.labels.converters import write_mapping_file

class_to_idx, idx_to_class = load_mapping("mapping.txt")
class_to_idx["pullOutStick"]  # 1
idx_to_class[1]               # "pullOutStick"

write_mapping_file("mapping.txt", {"background": 0, "walk": 1, "run": 2})
```

See {func}`~ethograph.labels.intervals.load_mapping` and
{func}`~ethograph.labels.converters.write_mapping_file`.

---

## Resolution order

When the GUI needs a mapping, it searches with
{func}`~ethograph.utils.paths.find_mapping_file`:

1. Walk up from the loaded data directory looking for
   `.ethograph/mapping.txt` in each ancestor. This lets a shared
   `.ethograph/` in a parent folder serve many sessions, while a
   per-session override wins.
2. Fall back to `~/.ethograph/mapping.txt` (global user default).

Typical layouts:

```
~/.ethograph/mapping.txt                          # global default
project/.ethograph/mapping.txt                    # project-wide (shared across sessions)
project/session_01/.ethograph/mapping.txt         # per-session override
```

---

## Auto-generated mappings

When you import labels from an external source whose classes don't exist
in the active `mapping.txt`, the GUI auto-creates a new mapping file
alongside the data so no IDs clash:

| Import source | Generated file |
|---------------|----------------|
| Crowsetta formats (aud-seq, textgrid, ...) | `mapping_{format}.txt` |
| Pynapple / NWB `IntervalSet` labels | `mapping_pynapple.txt` |
| NWB epoch imports | `mapping_nwb_epochs.txt` |

The mapping path field in the Label controls updates to point at the new
file. You can rename it to `mapping.txt` to make it the default for this
project.

See {func}`~ethograph.labels.converters.resolve_crowsetta_mapping` and
{func}`~ethograph.labels.converters.build_mapping_from_labels` for the
auto-generation logic.

---

(target-point-events)=
## State vs point events

Every label is one of two kinds:

| Kind     | Stored as                   | Use for                                   |
|----------|-----------------------------|-------------------------------------------|
| `state`  | interval (`onset_s`, `offset_s`) | Behaviours with a duration: walk, syllable  |
| `point`  | instant (`onset_s` only, `offset_s` is `NaN`) | Instantaneous events: peck, brief call |

The kind is declared per class in `mapping.txt` (4th column, see above) and
stored per row in the TSV (`event_type` column).  When a class is declared
`point` in `mapping.txt`, the labelling shortcut drops a marker at the
playhead instead of starting an interval drag.

Point events pass through every interval operation (purge, stitch, snap,
changepoint correction) untouched — they have no duration, so concepts like
"too short" or "stitch the gap" don't apply to them.  Internally this is
enforced by `split_by_kind()` in `ethograph.labels.intervals`.

---

