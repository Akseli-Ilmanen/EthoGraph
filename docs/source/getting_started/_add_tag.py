"""Add hide-input tag to cell-11 metadata."""

import json
from pathlib import Path

nb_path = Path(__file__).parent / "user_manual.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("id") == "cell-11":
        tags = cell.setdefault("metadata", {}).setdefault("tags", [])
        if "hide-input" not in tags:
            tags.append("hide-input")
        print(f"cell-11 tags: {tags}")

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
