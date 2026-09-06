# Vendored vocalseg

A partial copy of Tim Sainburg's
[vocalization-segmentation](https://github.com/timsainb/vocalization-segmentation)
(`vocalseg`) at commit `8bc85ee` (`master`, 2021-04-12 — the repository's
latest; the three files below were last changed at `01ad8b0`). MIT License,
© 2019 Tim Sainburg — see `LICENSE`.

Vendored 2026-07-19. Used by `ethograph/features/audio_changepoints.py` for
the audio changepoint candidates (dynamic-threshold and continuity
segmentation of a spectrogram).

## What is vendored

| Path | Upstream origin |
|---|---|
| `dynamic_thresholding.py` | `vocalseg/dynamic_thresholding.py` |
| `continuity_filtering.py` | `vocalseg/continuity_filtering.py` |
| `utils.py` | `vocalseg/utils.py` |
| `LICENSE` | `LICENSE` |

Only the two segmentation entry points ethograph calls
(`dynamic_threshold_segmentation`, `continuity_segmentation`) and their shared
spectrogram helpers.

## Edits made to the copies

- **All files**: `from vocalseg.X import ...` rewritten to package-relative
  imports; trailing whitespace stripped.
- **`__init__.py`** is ours: aliases `np.product` (removed in NumPy 2.0) to
  `np.prod` before the upstream modules import.
- **`continuity_filtering.py`**: the module-level colormap uses
  `ListedColormap(...).with_extremes(bad=...)` instead of the `set_bad` call
  matplotlib deprecated; `make_continuity_filter`'s signature is on one line.
- **Dropped**: upstream's `plot_labelled_elements`, `plot_segmentations` and
  `plot_segmented_spec` (matplotlib/seaborn figures ethograph never draws).
  `utils.plot_spec` is kept.

The linter and mypy skip this directory (see `pyproject.toml`).
