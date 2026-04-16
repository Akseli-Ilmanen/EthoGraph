(target-ruptures)=
# Ruptures

Reference: [Ruptures (Truong et al., 2020)](https://centre-borelli.github.io/ruptures-docs)

General-purpose changepoint detection using the `ruptures` library.

---

## Methods

Five search methods are available:

| Method | Description |
|--------|-------------|
| **Pelt** | Penalty-based, fast. Good when the number of changepoints is unknown. |
| **Binseg** | Binary segmentation. Fast recursive splitting. |
| **BottomUp** | Bottom-up merging of segments. |
| **Window** | Sliding window approach. |
| **Dynp** | Dynamic programming. Optimal but slow on long signals. |

Each method supports a cost model (`l2`, `l1`, `rbf`, etc.) and parameters
like `min_size`, `jump`, and either `pen` (penalty) or `n_bkps` (fixed number
of breakpoints).

```{note}
Ruptures detection has not been tested as extensively as the kinematic and
audio methods. Results may vary — check visually and adjust parameters as
needed.
```

---

## Usage

1. Select a feature in Data Controls.
2. Open the **Ruptures** panel.
3. Choose a method and click **Configure...**.
4. Click **Detect**.

---

## References

- Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing, 167, 107299. <https://doi.org/10.1016/j.sigpro.2019.107299>
