"""Try every architecture variant, then cross-validate the winner.

Stage 1 runs once per **variant**, because a hyperparameter space is
per-architecture: the models share almost no names (``mlp`` takes
``f_maps_list``, ``mstcn`` takes ``num_f_maps``), and
``eto.segment.tunable_params(name)`` prints what each one accepts. A variant
may also be one architecture set up two ways — ``asformer_enc`` pins
``num_decoders: 0`` while ``asformer_dec`` searches over ``1..3``, which is a
question about the architecture that a single study cannot answer cleanly.

**A variant is swept or searched, decided by its space.** ``is_exhaustible()``
checks every entry of ``{**SHARED_SPACE, **spec["space"]}`` — the variant's
own space *plus* ``SHARED_SPACE``, merged, not either one alone. All-categorical
(or empty) → *sweep*: ``cells()`` enumerates every combination and trains each
exactly once, no Optuna. Any ``float``/``int`` entry, in either half of that
merge → *search*: Optuna's TPE draws ``N_TRIALS`` configurations from the
whole space. Because the check runs on the merge, one continuous entry left in
``SHARED_SPACE`` pushes *every* variant into search mode, even one whose own
``space`` is empty or all-categorical — the per-variant comments about "3
values of tau, 3 runs" only hold while ``SHARED_SPACE`` is empty/categorical
too; check both before trusting a variant's own comment.

Search over an exhaustible space is also the wrong tool even when nothing
forces it: TPE samples *with replacement* and ``train.seed`` is fixed, so
``n_trials`` above the size of the grid buys the same runs again — 12 trials
over 3 values of ``tau`` is 3 answers and 9 repeats, at a full training run
each. A sweep is also the thing you can resume after editing the grid, which
an Optuna study is not: its ``study.db`` pins the choices a parameter was
created with and refuses a new set ("CategoricalDistribution does not support
dynamic value space").

Both modes score the same number — ``train.select_on`` on the **val** split —
so the variants stay comparable however each one was run.

**Stage 1 watches test to decide when to stop, stage 2 never does.** A fixed
epoch guess either wastes epochs on a variant that plateaus early or cuts one
off that is still climbing — and that trade-off is architecture-dependent
(``mstcn`` epochs are cheap and plentiful; others train slower but need
fewer). ``train.patience_on="test"`` (below) lets each cell stop once its own
test curve stops moving, which is fine here: this run's test split is never
the number that gets reported anywhere — only ``val_score`` decides the
winner, and the number a user sees is a fresh cross-validation on
session-held-out data. Stage 2 forces ``patience_on="val"`` — its test *is*
the reported number, so it stays untouched.

Stage 2 runs once, on the variant that scored best: a cross-validation costs
one training run per session, so it is not something to spend on the variants
that already lost.

Both stages resume. A swept cell already in ``bench_cells.tsv`` is skipped and
its score read back; a search resumes its study.

    python bench.py
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import ethograph as eto

CONFIG = Path(r"C:\Users\aksel\Documents\Code\ethograph\data\model\project.yaml")

#: Append-only, one row per swept cell — what makes a sweep resumable.
CELLS_FILE = CONFIG.with_name("bench_cells.tsv")

#: Varied for every architecture — these keys mean the same thing to all of them.
SHARED_SPACE: dict[str, dict] = {
    "train.learning_rate": {"type": "float", "low": 1.0e-5, "high": 1.0e-2, "log": True},
    # "train.loss.alpha": {"type": "categorical", "choices": [0.0]},
    # "train.augment.noise_std": {"type": "float", "low": 0.0, "high": 0.1},
}

#: One entry per **variant**, not per architecture — two variants may share an
#: architecture and differ only in what is pinned versus varied (the two
#: asformers below). The variant name becomes the run name and the study name.
#:
#: `params` is what stays fixed; `space` is what varies — enumerated when every
#: entry is categorical, sampled by Optuna otherwise. Every `model.params.*`
#: key must come from `eto.segment.tunable_params(arch)` — anything else is
#: refused before training starts, naming what that architecture does take.
VARIANTS: dict[str, dict] = {
    # ASFormer, encoder only: no refinement decoders, one output stage.
    # An all-categorical space, so this one is swept: 3 values of tau, 3 runs.
    "asformer_enc_alpha": {
        "architecture": "asformer",
        "params": {"num_decoders": 0},
        "space": {
            # "model.params.num_f_maps": {"type": "categorical", "choices": [64, 128, 256]},
            # "model.params.num_layers": {"type": "int", "low": 6, "high": 12},
            # "model.params.channel_masking_rate": {"type": "float", "low": 0.0, "high": 0.5},
            # "train.loss.tau": {"type": "categorical", "choices": [0.1, 1.0]},
        },
    },
    # # ASFormer as published: the encoder's prediction refined by `num_decoders`
    # # decoders (S = num_decoders + 1 stages, the last one read). Searching it
    # # asks "does refinement earn its cost here?" — each decoder is another full
    # # pass, and this architecture already runs one sample at a time, so this is
    # # the most expensive variant in the sweep by some way.
    # "asformer_dec": {
    #     "architecture": "asformer",
    #     "params": {},
    #     "space": {
    #         "model.params.num_decoders": {"type": "int", "low": 1, "high": 3},
    #         "model.params.num_f_maps": {"type": "categorical", "choices": [64, 128, 256]},
    #         "model.params.num_layers": {"type": "int", "low": 6, "high": 12},
    #         "model.params.channel_masking_rate": {"type": "float", "low": 0.0, "high": 0.5},
    #     },
    # },
    # "mstcn": {
    #     "architecture": "mstcn",
    #     "params": {},
    #     "space": {
    #         "model.params.num_f_maps": {"type": "categorical", "choices": [64, 128, 256]},
    #         "model.params.num_R": {"type": "int", "low": 1, "high": 3},
    #         "model.params.dropout_rate": {"type": "float", "low": 0.1, "high": 0.6},
    #     },
    # },
    # "c2f_tcn": {
    #     "architecture": "c2f_tcn",
    #     "params": {},
    #     "space": {"model.params.num_f_maps": {"type": "categorical", "choices": [64, 128, 256]}},
    # },
    # "edtcn": {
    #     "architecture": "edtcn",
    #     "params": {},
    #     "space": {"model.params.kernel_size": {"type": "int", "low": 9, "high": 33, "step": 4}},
    # },
    # "mlp": {  # the floor: no temporal context at all
    #     "architecture": "mlp",
    #     "params": {},
    #     "space": {"model.params.dropout_rates": {"type": "float", "low": 0.1, "high": 0.7}},
    # },
}

#: How many configurations Optuna draws — for a *searched* variant only. A
#: swept variant trains its whole grid, however large that is.
N_TRIALS = 1

#: Stage-1 epochs of no test-``select_on`` improvement before a cell stops.
#: Stage 2 (cross-validation) never uses this — see the module docstring.
PATIENCE = 10

#: Appended to every variant name, so one feature set's runs stay together.
SUFFIX = "kin_v1"

logger = logging.getLogger("bench")


def run_name(variant: str) -> str:
    """The base run name, which also names the study or the sweep's runs."""
    return f"{variant}_{SUFFIX}"


def base_overrides(variant: str, spec: dict) -> dict[str, Any]:
    """What every run of *variant* shares: its architecture, its pinned params, its name.

    A distinct run name per variant, which also names the study
    (``searches/search_{run_name}/``) — otherwise two variants of one
    architecture would pool incomparable trials into a single ``study.db``.
    """
    return {
        "model.architecture": spec["architecture"],
        "model.params": spec["params"],
        "train.run_name": run_name(variant),
        "train.patience": PATIENCE,
        "train.patience_on": "test",
    }


def is_exhaustible(space: dict[str, dict]) -> bool:
    """True when the space is a finite grid — enumerate it rather than sample it."""
    return all(entry["type"] == "categorical" for entry in space.values())


def cells(space: dict[str, dict]) -> list[dict[str, Any]]:
    """Every combination of an exhaustible *space* — the sweep's work list."""
    grid = {key: list(entry["choices"]) for key, entry in space.items()}
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())] or [{}]


def tag_for(params: dict[str, Any]) -> str:
    """``{"train.loss.tau": 4.0}`` → ``tau=4.0`` — the cell's name, in its run dir and its row."""
    return ",".join(f"{key.rsplit('.', 1)[-1]}={value}" for key, value in sorted(params.items())) or "default"


def trained_cells(variant: str) -> pd.DataFrame:
    """The cells of *variant* already trained, so an interrupted sweep resumes."""
    if not CELLS_FILE.is_file():
        return pd.DataFrame(columns=["variant", "cell", "val_score"])
    done = pd.read_csv(CELLS_FILE, sep="\t")
    return done[done["variant"] == variant]


def append_cell(row: dict[str, Any]) -> None:
    """One row, appended as its cell finishes — never rewritten in bulk."""
    pd.DataFrame([row]).to_csv(CELLS_FILE, sep="\t", mode="a", header=not CELLS_FILE.is_file(), index=False)


def sweep(variant: str, spec: dict, space: dict[str, dict], select_on: str) -> tuple[dict[str, Any], float]:
    """Train every cell of an exhaustible space once, and return the best draw.

    No Optuna and no ``study.db``: the work list is the grid, the objective is
    the same number a search trial returns (``train.select_on`` on the val
    split), and the only state is ``bench_cells.tsv``.
    """
    work = cells(space)
    done = trained_cells(variant)
    logger.info("Sweep %r: %d cell(s) over %s, maximising val %s", variant, len(work), sorted(space), select_on)

    scored: list[tuple[dict[str, Any], float]] = []
    for params in work:
        tag = tag_for(params)
        previous = done[done["cell"] == tag] if not done.empty else done
        if not previous.empty:
            score = float(previous["val_score"].iloc[-1])
            logger.info("[%s] %s — trained already, val %s = %.4f", variant, tag, select_on, score)
            scored.append((params, score))
            continue

        logger.info("[%s] %s", variant, tag)
        overrides = eto.segment.as_overrides(
            {
                **base_overrides(variant, spec),
                # Nested one level, like a study's trials, so `compare_runs` —
                # which reads only the top level of runs/ — keeps showing the
                # runs trained by hand.
                "train.run_name": f"sweep_{run_name(variant)}/{tag}",
                **params,
            }
        )
        result = eto.segment.Project(CONFIG, *overrides).train()
        append_cell(
            {
                "variant": variant,
                "cell": tag,
                "val_score": result.best_score,
                "best_epoch": result.best_epoch,
                "run_dir": str(result.run_dir),
                **params,
            }
        )
        logger.info(
            "[%s] %s val %s = %.4f at epoch %d", variant, tag, select_on, result.best_score, result.best_epoch
        )
        scored.append((params, result.best_score))

    return max(scored, key=lambda item: item[1])


def search(variant: str, spec: dict, space: dict[str, dict]) -> tuple[dict[str, Any], float]:
    """Optuna over a space that cannot be enumerated: ``N_TRIALS`` draws, scored on val."""
    overrides = eto.segment.as_overrides(
        {**base_overrides(variant, spec), "search.params": space, "search.n_trials": N_TRIALS}
    )
    result = eto.segment.Project(CONFIG, *overrides).search()
    return result.best_params, result.best_score


def main() -> None:
    # Materialise once: every architecture and every cell reads the same
    # features, so this is not part of what stage 1 varies.
    # eto.segment.Project(CONFIG).materialise()

    select_on = eto.segment.Project(CONFIG).config.train.select_on
    rows = []
    winners: dict[str, dict[str, Any]] = {}
    for variant, spec in VARIANTS.items():
        space = {**SHARED_SPACE, **spec["space"]}
        exhaustible = is_exhaustible(space)
        if exhaustible:
            params, score = sweep(variant, spec, space, select_on)
        else:
            params, score = search(variant, spec, space)
        winners[variant] = {**base_overrides(variant, spec), **params}
        rows.append(
            {
                "variant": variant,
                "architecture": spec["architecture"],
                "mode": "sweep" if exhaustible else "search",
                "val_score": score,
                **params,
            }
        )
        logger.info("%s: best val %s = %.4f %s", variant, select_on, score, params)

    table = pd.DataFrame(rows).sort_values("val_score", ascending=False)
    table.to_csv(CONFIG.with_name("architecture_search.tsv"), sep="\t", index=False)
    print(table.to_string(index=False))

    # Stage 2 on the winner only — one fold per session, each predicting the
    # session it never saw, for the GUI. Rebuilt from the variant's own
    # overrides plus what won: a search's `best.yaml` inherits the *project*
    # config, so on its own it would not carry the architecture the variant pinned.
    winner = str(table.iloc[0]["variant"])
    logger.info("Cross-validating %s (val %.4f) %s", winner, table.iloc[0]["val_score"], winners[winner])
    folds = eto.segment.Project(CONFIG, *eto.segment.as_overrides(winners[winner])).cross_validate()
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
