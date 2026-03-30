# Roadmap



## Model predictions import

Import predictions from action segmentation models (DLC2Action, ASFormer, MS-TCN) directly in the GUI. Per-trial prediction files (`.npy`/`.pickle`) with shape `(T, n_classes)` or `(T,)` will be converted to label intervals with confidence overlay (1 - entropy of classwise softmax).

## Changepoint features

Changepoint features to aid action and audio segmentation models.


