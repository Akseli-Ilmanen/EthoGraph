# Roadmap



## Model predictions import

Import predictions from action segmentation models (DLC2Action, ASFormer, MS-TCN) directly in the GUI. Per-trial prediction files (`.npy`/`.pickle`) with shape `(T, n_classes)` or `(T,)` will be converted to label intervals with confidence overlay (1 - entropy of classwise softmax).

## More sosphiticated changepoint detection

Current methods are fast (gradient based, RMS-based, etc.), but could also use ML for detection. Important that it's easily reproducable, so it represents a reliable feature in feature space. Sometimes the changepoint correction post-model output makes things worse, transformer learns better representation than simple gradient based methods. 

## Changepoint features

Using {func}`~ethograph.features.changepoints.more_changepoint_features` massively improved fine-grained accuracy for [ASFormer](https://github.com/ChinaYi/ASFormer), would be cool if this could be exported generally to segmentation models (DlC2Action, etc.)


## Other

- Audio changepoints
- Interactive PSTH `ethograph.gui.widgets_psth`
- Single-trial neural dimensionality reduction techniques, visualize label segments in latent spaces
