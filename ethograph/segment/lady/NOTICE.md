# Vendored LaDy

A partial copy of **[LaDy](https://github.com/HaoyuJi/LaDy)** (CVPR 2026,
*LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action
Segmentation via Spatial-Temporal Modulation*) at commit `bdba27d`
(`main`, 2025-12-02). MIT License, © 2025 Haoyu Ji — see `LICENSE`.

Kept in upstream's own layout. The adapter that puts it under this project's
architecture-registry contract is `../models/skeleton_graph.py` (registry name
`lady`); the architecture's hyperparameters live in `config/defaults.yaml`.

LaDy is SpecScalpel's spatial-temporal chassis with a second stream: a learned
Lagrangian dynamics model (`M(q)q̈ + C(q,q̇)q̇ + G(q) + F(q,q̇)`) over the
skeleton's generalised coordinates `q`, whose torque and power modulate the
temporal stream and whose energy consistency is an auxiliary loss.

## What is vendored

| Path | Upstream origin |
|---|---|
| `models/LaDy.py` | `libs/models/LaDy.py` |
| `models/SP.py` | `libs/models/SP.py` |
| `models/tcn.py` | `libs/models/tcn.py` |
| `models/dynamics.py` | `libs/models/dynamics.py` |
| `models/graph/tools.py` | `libs/models/graph/tools.py` |
| `generalized_coordinates.py` | `libs/generalized_coordinates.py` |
| `loss_fn/ECloss.py` | `libs/loss_fn/ECloss.py` |
| `LICENSE` | `LICENSE` |

`config/defaults.yaml` is **ours** (see `../specscalpel/NOTICE.md` for the
rationale). The keypoint count, channels-per-keypoint and the
generalised-coordinate dimensionality `dof` are resolved from the run's feature
layout and skeleton by the adapter.

## Edits made to the copies

Every `.py` carries the two vendor header lines; the linter and mypy skip this
directory (`pyproject.toml`).

- **`models/SP.py`**: the hardcoded `Graph(layout=dataset)` skeleton selection
  is removed — `MultiScale_GraphConv` takes the binary adjacency directly, as in
  the SpecScalpel vendor.
- **`models/LaDy.py`**: `Model` takes the adjacency `A_binary` instead of a
  dataset name; the `q`-reshaping view uses `-1` rather than a hardcoded
  `2 * D` (this project's sample is one individual, `num_people = 1`, so the
  two-people reshape never runs); the `get_derivatives` import is repointed to
  this package. Nothing in the dynamics maths changed.

## `generalized_coordinates.py` — kept for provenance, only `get_derivatives` used

`models/LaDy.py` imports `get_derivatives` (generic finite differences) from
here, and that is the only function used. Upstream's
`generate_generalized_coordinates` hardcodes a human joint map and kinematic
tree per dataset; this project computes `q` in a species-agnostic way instead —
`pose_to_generalised_coordinates` in `../models/skeleton_graph.py`, driven by
the skeleton config's rooted tree and named root-frame landmarks
(`model.params.root` / `spine` / `left` / `right`). The upstream function is
left in place, unused, so the provenance of the maths it mirrors stays with the
vendored file.

## Not vendored

Upstream's training harness, `graph/graph.py`, the BERT `text_embeddings/`, the
action-text contrastive loss, and its `boundary`/contrastive loss wiring beyond
`ECloss.py`.
