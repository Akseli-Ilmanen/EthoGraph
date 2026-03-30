import json
with open("examples/convert_dataset_pair24.ipynb") as f:
    nb = json.load(f)
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"Cell {i} ({c['cell_type']}): {src[:150]}")
    print()
