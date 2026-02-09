"""CURRENTLY NOT USED: Rolling autocorrelation periodicity detection for behavioral signals."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import argrelextrema


def rolling_autocorrelation_1d(
    signal: np.ndarray,
    window_size: int,
    max_lag: int,
    metric: str = "peak_lag",
    step: int = 1,
) -> np.ndarray:
    """Compute rolling autocorrelation of a 1D signal.

    For each sliding window, normalizes the segment, computes the
    autocorrelation via ``np.correlate``, and extracts a periodicity
    metric from the positive-lag portion (excluding lag 0).

    Parameters
    ----------
    signal : np.ndarray
        1D input signal of length T.
    window_size : int
        Number of samples in each sliding window.
    max_lag : int
        Maximum lag (in samples) to consider.  Must be < window_size.
    metric : str
        ``"peak_lag"`` returns the lag of the maximum autocorrelation.
        ``"peak_value"`` returns the autocorrelation value at that lag.
    step : int
        Subsampling factor — compute every *step*-th window and
        interpolate back to full length.

    Returns
    -------
    np.ndarray
        1D array of length T.  Leading/trailing positions that cannot
        form a full window are filled with NaN.
    """
    if metric not in ("peak_lag", "peak_value"):
        raise ValueError(f"metric must be 'peak_lag' or 'peak_value', got '{metric}'")
    if max_lag >= window_size:
        raise ValueError(f"max_lag ({max_lag}) must be < window_size ({window_size})")
    if window_size > len(signal):
        raise ValueError(f"window_size ({window_size}) must be <= signal length ({len(signal)})")

    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    windows = sliding_window_view(signal, window_size)
    n_windows = len(windows)

    indices = np.arange(0, n_windows, step)
    results = np.empty(len(indices), dtype=np.float64)

    for i, idx in enumerate(indices):
        w = windows[idx]
        w_mean = w.mean()
        w_centered = w - w_mean
        norm = np.dot(w_centered, w_centered)
        if norm < 1e-12:
            results[i] = np.nan
            continue

        acf_full = np.correlate(w_centered, w_centered, mode="full")
        acf_full /= norm
        mid = len(acf_full) // 2
        acf_positive = acf_full[mid + 1 : mid + 1 + max_lag]

        if len(acf_positive) < 3:
            results[i] = np.nan
            continue

        local_max_indices = argrelextrema(acf_positive, np.greater, order=1)[0]
        if len(local_max_indices) == 0:
            peak_idx = np.argmax(acf_positive)
        else:
            peak_idx = local_max_indices[np.argmax(acf_positive[local_max_indices])]

        if metric == "peak_lag":
            results[i] = peak_idx + 1
        else:
            results[i] = acf_positive[peak_idx]

    if step > 1 and len(indices) > 1:
        results = np.interp(np.arange(n_windows), indices, results)
    elif step > 1:
        results = np.full(n_windows, results[0] if len(results) else np.nan)

    pad_before = window_size // 2
    pad_after = n - n_windows - pad_before
    out = np.full(n, np.nan, dtype=np.float64)
    out[pad_before : pad_before + n_windows] = results
    return out


def rolling_autocorrelation_nd(
    data: np.ndarray,
    window_size: int,
    max_lag: int,
    metric: str = "peak_lag",
    step: int = 1,
) -> np.ndarray:
    """Apply :func:`rolling_autocorrelation_1d` to each channel of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape ``(T, C)`` where *T* is time and *C* is channels.
    window_size, max_lag, metric, step
        Forwarded to :func:`rolling_autocorrelation_1d`.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, C)``.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (T, C), got {data.ndim}D")

    t, c = data.shape
    out = np.empty_like(data)
    for ch in range(c):
        out[:, ch] = rolling_autocorrelation_1d(
            data[:, ch], window_size, max_lag, metric, step
        )
    return out


def summarize_multichannel(result_2d: np.ndarray, method: str = "mean") -> np.ndarray:
    """Reduce a ``(T, C)`` periodicity result to ``(T,)``.

    Parameters
    ----------
    result_2d : np.ndarray
        Shape ``(T, C)`` — output of :func:`rolling_autocorrelation_nd`.
    method : str
        One of ``"mean"``, ``"median"``, ``"max"``.

    Returns
    -------
    np.ndarray
        1D array of length T.
    """
    funcs = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax}
    if method not in funcs:
        raise ValueError(f"method must be one of {list(funcs)}, got '{method}'")
    return funcs[method](result_2d, axis=1)
