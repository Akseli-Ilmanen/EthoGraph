(target-changepoints-overview)=
# Overview

Changepoint detection methods (Xu et al., 2025) allow finding transitions in
time series data. This can be leveraged to identify **candidates** of action
boundaries in various data formats (kinematic, audio, spectral).

![Changepoints overview](../../_static/media/changepoints0.gif)

---

Ethograph uses changepoints in two ways:

1. The GUI can compute changepoints and mark them visually in a time series
   plot (e.g. as circles or vertical lines). When a user labels the
   onset/offset of a behaviour by clicking on the time series, their click is
   refined by jumping to the nearest changepoint. For example, when labelling
   the onset of a movement, the user may click close to a speed minima, and
   the GUI will refine the selection to jump exactly to that minima. This
   increases the accuracy and consistency of human labelling.

2. Changepoint detection methods often have a false-positive problem (Cohen,
   2022), where a large subset of the detections are not at real behavioural
   boundaries but false positives. Similarly, the changepoint algorithms
   often **overspecify** by providing too many changepoints. The
   human-in-the-loop through labelling can then specify which of these
   detections are **good candidates**. Next, these changepoint times are
   converted into learnable changepoint features, and exported along with
   human behavioural labels to supervised segmentation models (e.g.
   transformers). By receiving all changepoints along with human labels,
   these models can learn when certain changepoint features co-occur with
   behavioural boundaries, thus also refine their segmentation accuracy on
   unseen data.

---

## Next

- {doc}`kinematic` — troughs, turning points
- {doc}`audio` — VocalPy and VocalSeg methods for acoustic signals
- {doc}`ruptures` — general-purpose changepoint detection
- {doc}`correction` — label-boundary correction pipeline

---

## References

- Cohen, Y., Nicholson, D. A., Sanchioni, A., Mallaber, E. K., Skidanova, V., & Gardner, T. J. (2022). Automated annotation of birdsong with a neural network that segments spectrograms. eLife, 11, e63853. <https://doi.org/10.7554/eLife.63853>
- Gu, N., Lee, K., Basha, M., Kumar Ram, S., You, G., & Hahnloser, R. H. R. (2024). Positive Transfer of the Whisper Speech Transformer to Human and Animal Voice Activity Detection. ICASSP 2024. <https://doi.org/10.1109/ICASSP48485.2024.10447620>
- Kozlova, E., Bonnetto, A., & Mathis, A. (2025). DLC2Action: A Deep Learning-based Toolbox for Automated Behavior Segmentation. bioRxiv. <https://doi.org/10.1101/2025.09.27.678941>
- Xu, R., Song, Z., Wu, J., Wang, C., & Zhou, S. (2025). Change-point detection with deep learning: A review. Frontiers of Engineering Management, 12(1), 154-176. <https://doi.org/10.1007/s42524-025-4109-z>
