(target-label-mapping)=
# Label mapping (`mapping.txt`)

EthoGraph uses **integer label IDs** everywhere — in the TSV (`labels`
column), in predictions (`argmax` over softmax), and in on-screen number-key
shortcuts. String names are kept in a separate `mapping.txt` file and
resolved only at the edges (display, Crowsetta export, plotting).

This matches the convention used in the [action-segmentation literature](https://github.com/nus-cvml/awesome-temporal-action-segmentation) where models predict a per-frame class index and a dataset-level `mapping.txt` lists the
corresponding action names. Keeping string↔int separate means:

- Models and EthoGraph agree on the same integer IDs without re-encoding.
- Rename a behaviour once in `mapping.txt` and it propagates everywhere
  (old backups, predictions, exports).
- The `.tsv` files stay compact and language-neutral.

---

## Format

Whitespace-delimited, one label per line, `<id> <name>`:

```
0 background
1 pullOutStick
2 diagonalToBox
3 toss
4 nod
5 reachLeftCorner
...
```

Rules:

- **ID `0` is always `background`**. It is excluded from display, export,
  and model loss.
- IDs don't have to be contiguous, but they must be unique.
- Names should be valid identifiers (no spaces) so they round-trip through
  Crowsetta text formats cleanly.
- Trailing whitespace is ignored; blank lines are skipped.

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

