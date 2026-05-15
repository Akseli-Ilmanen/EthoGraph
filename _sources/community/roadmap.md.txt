# Roadmap

Currently this is still more of a collection of notes, than a roadmap.


## Improved interop. with segmentation models:

- Movement community call (April 2026): [PPT slides](https://neuroinformatics.zulipchat.com/user_uploads/58792/jV4lyzfLheHU4Gj4qCQkKzwy/2026_04-Ethograph-demo.pptx)
- Discussion on schema for segmentation feature data: https://github.com/EthoML/VAME/issues/189#issuecomment-4425518919

### Import predictions from action segmentation models
Import predictions from action segmentation models (DLC2Action, ASFormer, MS-TCN) directly in the GUI. Per-trial prediction files (`.npy`/`.pickle`) with shape `(T, n_classes)` or `(T,)` will be converted to label intervals with confidence overlay (1 - entropy of classwise softmax).

---

## Easier alignment of video and data streams via `.nwb` files

- https://github.com/catalystneuro/nwb-video-widgets/issues/34
- https://github.com/NeurodataWithoutBorders/nwb-schema/issues/677

---

## Changepoints


### More sophisticated changepoint detection

Current methods are fast (gradient based, RMS-based, etc.), but could also use ML for detection. Important that it's easily reproducible, so it represents a reliable feature in feature space. Sometimes the changepoint correction post-model output makes things worse, transformer learns better representation than simple gradient based methods.

### Changepoint features

Using {func}`~ethograph.features.changepoints.more_changepoint_features` massively improved fine-grained accuracy for [ASFormer](https://github.com/ChinaYi/ASFormer), would be cool if this could be exported generally to segmentation models (DlC2Action, etc.)


## Other

- Audio changepoints
- Interactive PSTH `ethograph.gui.widgets_psth`
- Single-trial neural dimensionality reduction techniques, visualize label segments in latent spaces
