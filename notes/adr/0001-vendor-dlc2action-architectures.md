# 1. Vendor DLC2Action's model files instead of depending on the package

Date: 2026-08-22

## Status

Accepted

## Context

The segmentation pipeline wants the action-segmentation architectures
DLC2Action ships (MS-TCN++, ASFormer with decoders, C2F-TCN, C2F-Transformer,
EDTCN, MLP). DLC2Action is AGPLv3; EthoGraph is GPL-3.0-or-later, so the two
are licence-compatible and importing is allowed. The question is mechanics:

* `pip install dlc2action` pulls ~27 unpinned dependencies (optuna, plotly,
  tika, pdfplumber, pyinquirer, opencv, ipykernel, …) onto every user who
  wants to train, with no torch pin.
* Its model modules import only `base_model.Model` (plus `ms_tcn_modules`
  and einops for MotionBERT) — about 3.7k lines that stand alone.
* Its feature extraction, augmentation and normalisation code is welded to
  its input-store / transformer / dataset classes and cannot be imported in
  isolation.

## Decision

Copy the standalone model files verbatim into
`ethograph/segment/dlc2action/` with their AGPL headers, the licence
text and a `NOTICE.md` naming the upstream commit and every edit (import
paths, the removed global `device`, print → logging). Register each as an
architecture through a thin adapter. **Reimplement the ideas** of the
feature-engineering, preprocessing and augmentation code against xarray and
this project's feature attrs rather than carrying the originals.

## Consequences

* Six literature architectures for ~2k lines and zero new dependencies.
* The combined work is distributed under GPLv3 terms as before; the vendored
  files stay AGPL-headed and attributed.
* Upstream fixes do not arrive automatically; re-vendoring is a diff against
  the commit recorded in `NOTICE.md`.
* The vendored directory is excluded from ruff and mypy; it is not held to
  this repository's style.
