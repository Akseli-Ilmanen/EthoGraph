# ADR 0009 — Video feature extractors are pip-installed, never vendored

**Status:** accepted (2026-09-06). Qualifies ADR 0001, which vendored
DLC2Action's architectures; that decision stands for the segmentation heads
and does not extend to the networks that read video.

## Context

Video features entered the repo as one network: S3D, adapted from a research
checkout, with its Kinetics-400 weights shipped inside the wheel and its own
config (`S3DConfig`), dims (`time_s3d`, `s3d_dims`), sidecar (`{stem}_s3d.nc`)
and variable name spelled through `segment/video_features.py`. Two more
extractors are now wanted — frame-wise ImageNet/DINO-family backbones, and
FERAL's V-JEPA2 — and the multi-animal work will want per-individual crops
through all of them.

Every candidate is on PyPI under a permissive licence (timm: Apache 2.0;
feral: MIT) with a Python entry point. Vendoring would copy their model code
and weights into our tree, as ADR 0001 did for DLC2Action, where the reasons
were the AGPL boundary and a toolbox layer we did not want. Neither reason
applies here: the licences are permissive and the packages are libraries.

The cost of pip is the dependency surface. feral 1.0.0 pins exact versions of
libraries the GUI also depends on (`timm==1.0.26`, `pandas==2.3.3`,
`transformers==5.5.3`), so it cannot be installed into the GUI environment
without downgrading it.

## Decision

- **An extractor is a name in one registry** (`ethograph.video_features`),
  and every extractor writes the same sidecar: a `(time, dims)` DataArray on
  the video's clock, dims `time_video` / `{name}_dims`, file
  `{stem}_{name}.nc`, variable `{name}` once merged. `segment.VideoFeaturesConfig`
  selects it by `extractor:`; only the settings that change the features are
  project settings.
- **Extractor code is imported from its package, never copied.** A new
  extractor is a pip extra (`ethograph[timm]`) plus an adapter of tens of
  lines that turns the package's own preprocessing config into our streaming
  decode. An extractor whose package cannot share the GUI environment runs
  in its own environment by subprocess, exchanging files — the pattern the
  vendored E2E-Spot clone already uses — until its pins loosen.
- **The S3D checkpoint in the package is grandfathered**, not a precedent.
  It stays because it works on the data this was built for and because there
  is no pip package for it; it is the last network whose weights ship in the
  wheel.
- **The default extractor is `s3d`**: measured on this project's data, its
  weights ship with the package (no download, no extra), on ten clips of
  that data it ran ~35 % faster than DINOv2 ViT-B at its native 518 px, and
  it is a *motion* feature — each frame's vector embeds the `stack_s` window
  around it, where a frame-wise backbone embeds the still image and leaves
  motion for the temporal model downstream to recover.
  **`timm` is the second choice, by name**, with
  `vit_base_patch14_reg4_dinov2.lvd142m` as its default backbone and timm's
  own pooling (CLS for a ViT, global average for a convnet), so a different
  backbone is a string swap and no per-model code exists on our side.

## Consequences

- One default per slot holds: the extractor slot has one default (`s3d`)
  and alternatives selectable by name (`timm`, later FERAL). The model-comparison tools (`compare_runs`,
  Cohen's-d ranking) see all three as the same kind of feature.
- Old `_s3d.nc` sidecars keep loading: the merge reads the time coordinate by
  name prefix and the feature dim as the other one, so a file written with
  `time_s3d` / `s3d_dims` is a valid `s3d` sidecar.
- Per-individual features need no new format: the sidecar records the
  individual in `attrs` today (one per video) and gains it in its file name
  when a per-individual crop exists. The variable's dims are unchanged either
  way, so `features.columns` keeps its spelling.
- FERAL enters through this registry when the interop discussion
  (`notes/Discuss_feral-interop.md`) settles the environment question; until
  then nothing FERAL-specific is in the tree.
