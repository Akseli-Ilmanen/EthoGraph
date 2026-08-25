"""Score every epoch of every ladder run and print one comparison table.

Reads the same high-recall prediction files ``spot_point_events.py score``
reads and applies the same tallest-peak rule, but reports each run at its
own best epoch so runs that peak at different points are comparable.

Ranked on misses first — a run that predicts nothing cannot be precise —
then on hits inside the tolerance budget. Hit rates are over *all* truth
events, not only the ones a run chose to emit, so a miss counts against
them. Usage::

    python scripts/spot_compare_ladder.py                 # the whole ladder
    python scripts/spot_compare_ladder.py A0 A3 overnight2
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spot_point_events import SPOT_ROOT, TARGET_LABELS, best_per_class  # noqa: E402

#: A run lives directly under ``runs/``, or under ``runs/old/`` once retired.
RUN_ROOTS = (SPOT_ROOT / "runs", SPOT_ROOT / "runs" / "old")
DEFAULT_RUNS = ("A0", "A1", "A2", "A3", "A4")
EPOCH_RE = re.compile(r"pred-(?:val|test)\.(\d+)\.")


def find_run(name: str) -> Path | None:
    for root in RUN_ROOTS:
        if (root / name / "config.json").exists():
            return root / name
    return None


def read_predictions(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(path: Path, truth: dict[str, dict], n_events: int) -> dict:
    """Absolute frame errors and misses for one prediction file."""
    errors: list[int] = []
    misses = 0
    for entry in read_predictions(path):
        ground = truth.get(entry["video"])
        if ground is None:
            continue
        stride = int(round(float(ground["fps"]) / float(entry["fps"])))
        best = {
            label: int(round(frame * stride + (stride - 1) / 2))
            for label, frame in best_per_class(entry).items()
        }
        for event in ground["events"]:
            if event["label"] in best:
                errors.append(abs(best[event["label"]] - int(event["frame"])))
            else:
                misses += 1
    return {
        "miss": misses,
        "median": float(np.median(errors)) if errors else float("nan"),
        "le2": sum(e <= 2 for e in errors) / n_events,
        "le4": sum(e <= 4 for e in errors) / n_events,
        "le10": sum(e <= 10 for e in errors) / n_events,
    }


def main() -> None:
    names = sys.argv[1:] or list(DEFAULT_RUNS)
    truth_list = json.loads((SPOT_ROOT / "data" / "crow_pellet" / "val.json").read_text(encoding="utf-8"))
    truth = {video["video"]: video for video in truth_list}
    n_events = sum(len(video["events"]) for video in truth_list)

    print(f"\n{'run':<12} {'ep':>3} {'stride':>6} {'clip':>5} {'context':>8} "
          f"{'miss':>7} {'med':>6} {'<=2':>6} {'<=4':>6} {'<=10':>6}")
    print("-" * 78)
    for name in names:
        run_dir = find_run(name)
        if run_dir is None:
            print(f"{name:<12} {'-':>3}  (not started)")
            continue
        files = sorted(run_dir.glob("pred-val.*.recall.json.gz"),
                       key=lambda p: int(EPOCH_RE.search(p.name).group(1)))
        if not files:
            print(f"{name:<12} {'-':>3}  (running; no val predictions yet)")
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        stride, clip = config.get("stride", 1), config["clip_len"]
        scored = [(int(EPOCH_RE.search(p.name).group(1)), evaluate(p, truth, n_events)) for p in files]
        epoch, row = min(scored, key=lambda item: (item[1]["miss"], -item[1]["le4"]))
        fps = float(truth_list[0]["fps"])
        print(f"{name:<12} {epoch:>3} {stride:>6} {clip:>5} {clip * stride / fps:>7.1f}s "
              f"{row['miss']:>3}/{n_events:<3} {row['median']:>6.1f} "
              f"{row['le2']:>5.0%} {row['le4']:>5.0%} {row['le10']:>5.0%}")

    frame_ms = 1000 / float(truth_list[0]["fps"])
    print(f"\nBest epoch per run, fewest misses first. 1 frame = {frame_ms:.0f} ms, so "
          f"<=2 is {2 * frame_ms:.0f} ms and <=4 is {4 * frame_ms:.0f} ms.")
    print(f"Hit rates are over all {n_events} truth events across "
          f"{len(TARGET_LABELS)} classes, so misses count against them.")


if __name__ == "__main__":
    main()
