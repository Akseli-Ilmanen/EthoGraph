(target-importing-labels)=
# Importing labels

In the **Import labels** tab, the **Labels format** combo offers:

| Option | Source | Converter |
|--------|--------|-----------|
| **`.tsv`** | EthoGraph TSV (backup, colleague's labels, manual edit) | (native) |
| **`pynapple (.npz)`** | Pynapple file with {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **`pynapple (.nwb)`** | NWB file loaded via {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **Crowsetta formats** (aud-seq, simple-seq, textgrid, notmat, timit, yarden, ...) | [crowsetta](https://crowsetta.readthedocs.io/)-supported annotation tools (Audacity, Praat, Raven, ...) | {class}`~ethograph.labels.converters.CrowsettaLabelConverter` |


---


## Pynapple / NWB IntervalSets

Selecting **`pynapple (.npz)`** or **`pynapple (.nwb)`** loads the file with
{func}`pynapple.load_file` and extracts every
{class}`~pynapple.IntervalSet` in the data dict **except** those named
`"trials"` or `"epochs"` (those are treated as trial boundaries, not labels).
Each `IntervalSet` name becomes a label class.

The GUI auto-generates a `mapping_pynapple.txt` file with integer IDs for
each label name (see {doc}`mapping` for the file format and resolution
order) and writes the result to the canonical `_labels.tsv` alongside the
`.nc`.

Global-time intervals are split across trials using the `trials` /
`epochs` `IntervalSet` (or the session's trial table). See
{class}`~ethograph.labels.converters.PynappleLabelConverter` for the
conversion logic.

---

## Crowsetta interop

EthoGraph registers an `ethograph-seq`
[crowsetta](https://crowsetta.readthedocs.io/) format for sharing labels with
string names (resolved via `mapping.txt`):

```python
from ethograph.labels.crowsetta_format import EthographSeq

# Export: int labels -> string labels via mapping
ethoseq = EthographSeq.from_intervals_df(df, id_to_name={1: "Head bob", 2: "Song"})
ethoseq.to_file("labels_for_sharing.tsv")

# Import via crowsetta
import crowsetta
scribe = crowsetta.Transcriber(format="ethograph-seq")
annot = scribe.from_file("labels_for_sharing.tsv").to_annot()
```

On import, the GUI checks the active `mapping.txt` against the labels found
in the file. If some labels are missing from the mapping, it auto-generates
a new `mapping_{format}.txt` alongside the data file and warns about
unmatched labels. See
{func}`~ethograph.labels.converters.resolve_crowsetta_mapping` for details.

---

## Programmatic usage

All converters expose the same `resolve_labels(...)` contract, which falls
back through `existing TSV → extract from source → empty`:

```python
from pathlib import Path
from ethograph.labels.converters import PynappleLabelConverter
import pynapple as nap

data = nap.load_file("session.nwb")
trials_ep = data["trials"] if "trials" in data.keys() else None

converter = PynappleLabelConverter(data, trials_ep=trials_ep)
df = converter.resolve_labels(
    source_path=Path("session.nwb"),
    trial_ids=[1, 2, 3],
)
```

See {doc}`exporting` for the full TSV column reference.
