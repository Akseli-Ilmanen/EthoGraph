"""Sweep the spike → feature transform of a neural decoding project.

Every variant below is one materialised dataset (``data/{name}/``) and one
5-fold trial cross-validation (``cross_validation/cv_{name}/``), all on the
same folds (``train.split.seed``), so the numbers are paired. The summary —
mean ± std over folds of every metric, one row per variant — is written to
``configs/neural/transform_sweep.tsv`` and printed sorted by post-processed
F1@50.

    python run.py            # everything in VARIANTS
    python run.py sqrt       # only variants whose name contains "sqrt"

A variant that fails (a transform pynapple cannot evaluate, a trial too
short for the architecture at a coarse bin) is logged and skipped so the
sweep finishes; the failures are listed at the end.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

import ethograph as eto
from ethograph.segment import as_overrides

CONFIG = Path(r"configs\neural\decoding.yaml")
N_FOLDS = 5
OVERRIDES = [
    # "train.epochs=20",   # uncomment for a quick pass before committing a night
]

RATE_5 = "x.count(0.005) / 0.005"
RATE_10 = "x.count(0.01) / 0.01"
RATE_20 = "x.count(0.02) / 0.02"

#: name → pynapple expressions applied in order to `x` (the TsGroup first).
#: `nap`, `np` and `sliding_window` are in scope; the last step must leave a TsdFrame.
VARIANTS: dict[str, list[str]] = {
    # --- nothing but binning: is smoothing even needed? -------------------
    "count_5ms_raw": ["x.count(0.005)"],
    "rate_5ms_raw": [RATE_5],
    "rate_10ms_raw": [RATE_10],
    "rate_20ms_raw": [RATE_20],
    # --- boxcar smoothing at 5 ms: window width -------------------------
    "rate_5ms_boxcar25ms": [RATE_5, "sliding_window(x, window_size=0.025)"],
    "rate_5ms_boxcar50ms": [RATE_5, "sliding_window(x, window_size=0.05)"],
    "rate_5ms_boxcar100ms": [RATE_5, "sliding_window(x, window_size=0.1)"],
    "rate_5ms_boxcar200ms": [RATE_5, "sliding_window(x, window_size=0.2)"],
    # --- gaussian smoothing (pynapple's own): std ------------------------
    "rate_5ms_gauss10ms": [RATE_5, "x.smooth(std=0.01)"],
    "rate_5ms_gauss25ms": [RATE_5, "x.smooth(std=0.025)"],
    "rate_5ms_gauss50ms": [RATE_5, "x.smooth(std=0.05)"],
    # --- coarser bins with matched smoothing -----------------------------
    "rate_10ms_boxcar50ms": [RATE_10, "sliding_window(x, window_size=0.05)"],
    "rate_20ms_boxcar100ms": [RATE_20, "sliding_window(x, window_size=0.1)"],
    # --- counts per window instead of a rate (sum, not mean) -------------
    "count_5ms_sum50ms": ["x.count(0.005)", "sliding_window(x, window_size=0.05, reduction='sum')"],
    # --- variance stabilisation ------------------------------------------
    "sqrt_count_5ms_raw": ["x.count(0.005)", "np.sqrt(x)"],
    "sqrt_count_5ms_boxcar25ms": ["x.count(0.005)", "np.sqrt(x)", "sliding_window(x, window_size=0.025)"],
    "sqrt_rate_5ms_boxcar50ms": [RATE_5, "sliding_window(x, window_size=0.05)", "np.sqrt(x)"],
    "log1p_count_5ms_boxcar50ms": ["x.count(0.005)", "sliding_window(x, window_size=0.05)", "np.log1p(x)"],
    # --- the kitchen sink: sqrt, wide gaussian, then log ------------------
    "sqrt_gauss50ms_log1p": ["x.count(0.005)", "np.sqrt(x)", "x.smooth(std=0.05)", "np.log1p(x)"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sweep")


def run_variant(name: str, steps: list[str]) -> pd.DataFrame:
    project = eto.segment.Project(
        str(CONFIG),
        f"features.name={name}",
        f"train.run_name={name}",
        *as_overrides({"features.neural.transform": steps}),
        *OVERRIDES,
    )
    return project.cross_validate(n_folds=N_FOLDS)


def summarise(name: str, steps: list[str], folds: pd.DataFrame) -> dict:
    row: dict = {"variant": name, "transform": " | ".join(steps), "n_folds": len(folds)}
    metrics = folds.select_dtypes("number").drop(columns=["best_epoch"], errors="ignore")
    for column in metrics.columns:
        row[f"{column} mean"] = float(metrics[column].mean())
        row[f"{column} std"] = float(metrics[column].std(ddof=0))
    return row


def main(selector: str | None) -> None:
    chosen = {n: s for n, s in VARIANTS.items() if selector is None or selector in n}
    if not chosen:
        raise SystemExit(f"No variant matches {selector!r}; have {list(VARIANTS)}")
    out_path = CONFIG.parent / "transform_sweep.tsv"
    rows: list[dict] = []
    failed: dict[str, str] = {}
    for i, (name, steps) in enumerate(chosen.items(), start=1):
        logger.info("[%d/%d] %s: %s", i, len(chosen), name, " | ".join(steps))
        try:
            folds = run_variant(name, steps)
        except Exception as exc:  # the sweep's outer boundary: one bad variant must not lose the night
            logger.exception("%s failed", name)
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        rows.append(summarise(name, steps, folds))
        # written after every variant, so a killed sweep still leaves what it finished
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    if rows:
        table = pd.DataFrame(rows)
        key = "postprocessed.f1@50 mean"
        show = [
            c
            for c in ("variant", key, "postprocessed.f1@50 std", "raw.f1@50 mean", "postprocessed.edit mean")
            if c in table
        ]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(table.sort_values(key, ascending=False)[show].to_string(index=False) if key in table else table)
        print(f"\nWrote {out_path}")
    if failed:
        print("\nFailed variants:")
        for name, why in failed.items():
            print(f"  {name}: {why}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
