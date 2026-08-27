# Vendored SpecScalpel

A partial copy of **[SpecScalpel](https://github.com/HaoyuJi/SpecScalpel)**
(CVPR 2026, *Spectral Scalpel: Amplifying Adjacent Action Discrepancy via
Frequency-Selective Filtering for Skeleton-Based Action Segmentation*) at commit
`01adb1e` (`main`, 2025-07-10). MIT License, © 2025 Haoyu Ji — see `LICENSE`.

Kept in upstream's own `models/` layout so a refresh is "copy these files at
commit X". The adapter that puts it under this project's architecture-registry
contract is `../models/skeleton_graph.py` (registry name `specscalpel`); the
architecture's hyperparameters live in `config/defaults.yaml`, read there and
never restated in code.

## What is vendored

| Path | Upstream origin |
|---|---|
| `models/SpecScalpel.py` | `libs/models/SpecScalpel.py` |
| `models/SP.py` | `libs/models/SP.py` |
| `models/tcn.py` | `libs/models/tcn.py` |
| `models/graph/tools.py` | `libs/models/graph/tools.py` |
| `LICENSE` | `LICENSE` |

`config/defaults.yaml` is **ours**: upstream's `config/PKU-subject/config.yaml`
minus everything dataset-, loss- and optimiser-specific, leaving only the
architecture hyperparameters. The keypoint count, channels-per-keypoint and the
skeleton adjacency are resolved from the run's feature layout and pose source by
the adapter, not read from a config file.

## Edits made to the copies

Every `.py` keeps its upstream body, prefixed with two lines
(`# Vendored from HaoyuJi/SpecScalpel (MIT) — see NOTICE.md` and `# ruff: noqa`);
the linter and mypy skip this directory (see `pyproject.toml`).

- **`models/SP.py`** and **`models/SpecScalpel.py`**: upstream selects the joint
  graph by a hardcoded dataset name (`Graph(layout=dataset)`, one of PKU-MMD /
  LARa / TCG's human skeletons) and switches a TCG-only branch on it. Both are
  removed: `MultiScale_GraphConv` and `Model` now take the binary adjacency
  `A_binary` and the node count `node` directly, which the adapter builds from
  this project's own skeleton config or the pose source's ndx-pose `Skeleton`.
  No skeleton is hardcoded anywhere. Diff is minimal and confined to those two
  concerns.

## Not vendored

Upstream's training harness (`libs/config.py`, `train.py`, `evaluate.py`), its
`graph/graph.py` (hardcoded human layouts + a module-level `matplotlib` import),
the BERT `text_embeddings/` and the action-text contrastive loss. The registry
adapter, this project's config system and its skeleton config replace them, the
same way the DLC2Action vendor drops upstream's `options.py`.
