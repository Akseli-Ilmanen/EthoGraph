"""Sweep architectures × spike transforms of a neural decoding project.

Two axes. A **transform** is one materialised dataset (``data/{transform}/``,
built once and shared by every model that reads it); a **model** is an
architecture plus its ``model.params``. Every (model, transform) pair is
one 5-fold trial cross-validation (``cross_validation/cv_{model}__{transform}/``),
all on the same folds (``train.split.seed``), so every number in the summary
is paired with every other. The summary — mean ± std over folds of every
metric, one row per pair — is written to ``configs/neural/model_sweep.tsv``
and printed sorted by post-processed F1@50.

    python run.py                          # every MODEL × DEFAULT_TRANSFORMS
    python run.py mstcn                    # models whose name contains "mstcn" × DEFAULT_TRANSFORMS
    python run.py mstcn sqrt               # ... × transforms whose name contains "sqrt"
    python run.py . all                    # every MODEL × every TRANSFORM ("." matches everything)

A pair that fails (a transform pynapple cannot evaluate, a trial too short
for the architecture at a coarse bin) is logged and skipped so the sweep
finishes; the failures are listed at the end.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

import ethograph as eto
from ethograph.segment import as_overrides

CONFIG = Path(r"configs\neural\decoding.yaml")
N_FOLDS = 2
OVERRIDES = [
    # "train.epochs=20",   # uncomment for a quick pass before committing a night
]

# ---------------------------------------------------------------------------
# Models: architecture + params. Params not named keep upstream's defaults
# (ethograph/segment/dlc2action/config/model/{architecture}.yaml).
# ---------------------------------------------------------------------------

#: name → dotted overrides. At 200 Hz a wake trial is ~1000 frames; an MS-TCN
#: stage of L layers sees 2**L frames, so 8 layers ≈ 1.3 s, 10 ≈ 5 s (a trial),
#: upstream's 15 ≈ 3 min — the "how much history does decoding need" axis.
MODELS: dict[str, dict] = {
    # frame-wise decoder on the instantaneous rate vector: the no-context baseline
    "mlp_128x2": {"model.architecture": "mlp", "model.params.f_maps_list": [128, 128]},
    "mlp_256x3": {"model.architecture": "mlp", "model.params.f_maps_list": [256, 256, 256]},
    # MS-TCN++ at three receptive fields; refinement stages sized to match
    "mstcn_rf1s": {
        "model.architecture": "mstcn",
        "model.params.num_layers_PG": 8,
        "model.params.num_layers_R": 8,
    },
    "mstcn_rf5s": {
        "model.architecture": "mstcn",
        "model.params.num_layers_PG": 10,
        "model.params.num_layers_R": 10,
    },
    "mstcn_default": {"model.architecture": "mstcn"},
    # a narrower MS-TCN, for a 9-unit input 128 maps may be more than the data supports
    "mstcn_rf5s_64": {
        "model.architecture": "mstcn",
        "model.params.num_layers_PG": 10,
        "model.params.num_layers_R": 10,
        "model.params.num_f_maps": 64,
    },
    # C2F-TCN: multi-scale pooling over the whole trial, the behaviour paper's model
    "c2f_tcn_default": {"model.architecture": "c2f_tcn"},
    "c2f_tcn_64": {"model.architecture": "c2f_tcn", "model.params.num_f_maps": 64},
}

# ---------------------------------------------------------------------------
# Transforms: pynapple expressions applied in order to `x` (the TsGroup first).
# `nap`, `np` and `sliding_window` are in scope; the last step must leave a TsdFrame.
# ---------------------------------------------------------------------------

RATE_5 = "x.count(0.005) / 0.005"
RATE_10 = "x.count(0.01) / 0.01"
RATE_20 = "x.count(0.02) / 0.02"

TRANSFORMS: dict[str, list[str]] = {
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

#: The transforms every model runs on when none is selected — one per idea,
#: so the model axis is not paid for nineteen times over.
DEFAULT_TRANSFORMS = ["rate_5ms_boxcar25ms", "rate_5ms_gauss25ms", "sqrt_count_5ms_boxcar25ms", "rate_10ms_boxcar50ms"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sweep")


def run_pair(model: str, transform: str) -> pd.DataFrame:
    project = eto.segment.Project(
        str(CONFIG),
        f"features.name={transform}",
        f"train.run_name={model}__{transform}",
        *as_overrides({"features.neural.transform": TRANSFORMS[transform], **MODELS[model]}),
        *OVERRIDES,
    )
    return project.cross_validate(n_folds=N_FOLDS)


def summarise(model: str, transform: str, folds: pd.DataFrame) -> dict:
    row: dict = {
        "model": model,
        "transform": transform,
        "architecture": MODELS[model]["model.architecture"],
        "params": " ".join(f"{k.split('.')[-1]}={v}" for k, v in MODELS[model].items() if k != "model.architecture"),
        "steps": " | ".join(TRANSFORMS[transform]),
        "n_folds": len(folds),
    }
    metrics = folds.select_dtypes("number").drop(columns=["best_epoch"], errors="ignore")
    for column in metrics.columns:
        row[f"{column} mean"] = float(metrics[column].mean())
        row[f"{column} std"] = float(metrics[column].std(ddof=0))
    return row


def select(names: list[str], selector: str | None, default: list[str]) -> list[str]:
    """*names* containing *selector*; ``None`` = *default*, ``"."``/``"all"`` = every name."""
    if selector is None:
        return default
    if selector in (".", "all"):
        return list(names)
    chosen = [n for n in names if selector in n]
    if not chosen:
        raise SystemExit(f"Nothing matches {selector!r}; have {names}")
    return chosen


def main(model_selector: str | None, transform_selector: str | None) -> None:
    models = select(list(MODELS), model_selector, list(MODELS))
    transforms = select(list(TRANSFORMS), transform_selector, DEFAULT_TRANSFORMS)
    pairs = [(m, t) for t in transforms for m in models]  # transform-major: one dataset materialised, then every model
    out_path = CONFIG.parent / "model_sweep.tsv"
    rows: list[dict] = []
    failed: dict[str, str] = {}
    for i, (model, transform) in enumerate(pairs, start=1):
        logger.info("[%d/%d] %s on %s", i, len(pairs), model, transform)
        try:
            folds = run_pair(model, transform)
        except Exception as exc:  # the sweep's outer boundary: one bad pair must not lose the night
            logger.exception("%s on %s failed", model, transform)
            failed[f"{model} on {transform}"] = f"{type(exc).__name__}: {exc}"
            continue
        rows.append(summarise(model, transform, folds))
        # written after every pair, so a killed sweep still leaves what it finished
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    if rows:
        table = pd.DataFrame(rows)
        key = "postprocessed.f1@50 mean"
        show = [
            c
            for c in ("model", "transform", key, "postprocessed.f1@50 std", "raw.f1@50 mean", "postprocessed.edit mean")
            if c in table
        ]
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(table.sort_values(key, ascending=False)[show].to_string(index=False) if key in table else table)
        print(f"\nWrote {out_path}")
    if failed:
        print("\nFailed:")
        for name, why in failed.items():
            print(f"  {name}: {why}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
