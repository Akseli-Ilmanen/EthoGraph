# Roadmap



## Model predictions import

Import predictions from action segmentation models (DLC2Action, ASFormer, MS-TCN) directly in the GUI. Per-trial prediction files (`.npy`/`.pickle`) with shape `(T, n_classes)` or `(T,)` will be converted to label intervals with confidence overlay (1 - entropy of classwise softmax).

## More sosphiticated changepoitn detection

Current methods are fast (gradient based, RMS-based, etc.), but could also use ML for detection. Important that it's easily reproducable, so it represents a reliable feature in feature space.
Sometimes the changepoint correction post-model output makes things works, transformer learns better rpresentation than simple gradient based methods. 

## Changepoint features

Changepoint features to aid action and audio segmentation models.


