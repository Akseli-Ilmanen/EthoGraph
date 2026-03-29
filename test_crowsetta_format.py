"""Quick test for the EthographSeq Crowsetta format."""
import tempfile
import os
import pandas as pd
import crowsetta
from ethograph.labels.crowsetta_format import EthographSeq

df = pd.DataFrame({
    "onset_s": [0.41, 0.51, 0.77],
    "offset_s": [0.505, 0.62, 0.885],
    "label": ["Head bob", "Wing flap", "Song"],
    "individual": ["Poppy", "Poppy", "Poppy"],
    "trial": [1, 1, 1],
})

tmp = tempfile.mktemp(suffix=".tsv")
df.to_csv(tmp, sep="\t", index=False)

scribe = crowsetta.Transcriber(format="ethograph-seq")
ethoseq = scribe.from_file(tmp)
print("Loaded:", type(ethoseq).__name__)
print("Labels:", ethoseq.labels)
print("Individuals:", ethoseq.individuals)

annot = ethoseq.to_annot()
print("Segments:", len(annot.seq.segments))
seg0 = annot.seq.segments[0]
print(f"First: {seg0.onset_s} -> {seg0.offset_s} '{seg0.label}'")

out = tempfile.mktemp(suffix=".tsv")
ethoseq.to_file(out)
result = pd.read_csv(out, sep="\t")
print("Round-trip columns:", list(result.columns))
print(result)

os.unlink(tmp)
os.unlink(out)
print("\nAll tests passed!")
