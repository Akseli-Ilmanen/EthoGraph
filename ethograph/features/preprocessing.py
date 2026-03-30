<<<<<<< HEAD
from typing import Tuple

=======
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.interpolate import interp1d
import ethograph as eto




def downsample_with_antialiasing(time: np.ndarray, data: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
<<<<<<< HEAD
    """Downsample (T,) or (T, D) array using moving-average low-pass filter before subsampling to prevent aliasing."""
=======
    """Downsample array using moving-average low-pass filter before subsampling.

    Parameters
    ----------
    time : numpy.ndarray
        Time values of shape ``(T,)``.
    data : numpy.ndarray
        Data of shape ``(T,)`` or ``(T, D)``.
    factor : int
        Downsample factor.

    Returns
    -------
    time_ds : numpy.ndarray
        Downsampled time values.
    data_ds : numpy.ndarray
        Downsampled data.
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    smoothed = uniform_filter1d(data, size=factor, axis=0)
    return time[::factor], smoothed[::factor]


def resample_to_frames(
    data: np.ndarray,
    time_original: np.ndarray,
    time_target: np.ndarray,
) -> np.ndarray:
<<<<<<< HEAD
    """Resample data to target time points using moving-average anti-aliasing filter before interpolation."""
=======
    """Resample data to target time points with anti-aliasing before interpolation.

    Parameters
    ----------
    data : numpy.ndarray
        Source data array.
    time_original : numpy.ndarray
        Time axis of the source data.
    time_target : numpy.ndarray
        Desired output time axis.

    Returns
    -------
    numpy.ndarray
        Resampled data at ``time_target`` points.
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    factor = len(time_original) / len(time_target)
    if factor > 1:
        data = uniform_filter1d(data, size=int(round(factor)), axis=0)
    f = interp1d(time_original, data, axis=0, fill_value='extrapolate')
    return f(time_target)


def interpolate_nans(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Interpolate NaNs using NumPy's interp, with leading/trailing NaNs set to zero.

<<<<<<< HEAD
    Args:
        arr: Input array possibly containing NaNs.
        axis: Axis along which to interpolate. Defaults to 0 (interpolate across rows, each column is treated independently).

    Returns:
        np.ndarray with NaNs interpolated and leading/trailing NaNs set to zero.

    1D Example:
        >>> arr = np.array([np.NaN, np.NaN, 2.0, np.NaN, 8.0, np.NaN, 10.0, np.NaN])
        >>> interpolate_nans(arr)
        array([ 0.,  0.,  2.,  5.,  8.,  9., 10.,  0.])
=======
    Parameters
    ----------
    arr : numpy.ndarray
        Input array possibly containing NaNs.
    axis : int
        Axis along which to interpolate. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Array with NaNs interpolated and leading/trailing NaNs set to zero.

    Examples
    --------
    >>> arr = np.array([np.NaN, np.NaN, 2.0, np.NaN, 8.0, np.NaN, 10.0, np.NaN])
    >>> interpolate_nans(arr)
    array([ 0.,  0.,  2.,  5.,  8.,  9., 10.,  0.])
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    """
    arr = np.asarray(arr)

    def interpolate_1d(x: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(x)
        if not mask.any():
            return np.zeros_like(x)
        indices = np.arange(len(x))
        result = np.interp(indices, indices[mask], x[mask])
        # Set leading/trailing NaNs to zero
        first_valid, last_valid = np.where(mask)[0][[0, -1]]
        result[:first_valid] = 0
        result[last_valid + 1:] = 0
        return result

    if arr.ndim == 1:
        return interpolate_1d(arr)
    return np.apply_along_axis(interpolate_1d, axis, arr)






def z_normalize(data: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
    """Apply z-score normalization to each feature (column) independently."""
=======
    """Apply z-score normalization to each feature (column) independently.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(T, F)`` where F is the number of features.

    Returns
    -------
    numpy.ndarray
        Z-normalized array with zero mean and unit variance per column.
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1
    return (data - np.nanmean(data, axis=0)) / std

def clip_by_percentiles(
    features: np.ndarray,
<<<<<<< HEAD
    percentile_range: Tuple[float, float] = (1, 99)
) -> np.ndarray:
    """Clip to percentiles and z-normalize a single trial.
    
    Args:
        features: (T, F) array of features for one trial
        percentile_range: (lower, upper) percentiles for clipping
        eps: Small constant for numerical stability
        
    Returns:
        Processed features (T, F)
=======
    percentile_range: tuple[float, float] = (1, 99)
) -> np.ndarray:
    """Clip feature values to percentile bounds.

    Parameters
    ----------
    features : numpy.ndarray
        Array of shape ``(T, F)`` — T time samples, F features.
    percentile_range : tuple[float, float]
        ``(lower, upper)`` percentiles for clipping.

    Returns
    -------
    numpy.ndarray
        Clipped features of shape ``(T, F)``.
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    """
    # Compute percentiles per feature dimension
    lower = np.nanpercentile(features, percentile_range[0], axis=0, keepdims=True)
    upper = np.nanpercentile(features, percentile_range[1], axis=0, keepdims=True)

    features_clipped = np.clip(features, lower, upper)
        
    return features_clipped

def gaussian_smoothing(da, **smoothing_params):
    """
    Apply Gaussian smoothing to position data across DataArray dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with position data.
    **smoothing_params
        Keyword arguments for gaussian_filter1d.

    Returns
    -------
    xarray.DataArray
        Smoothed position data.
    """

    
    def nan_smooth(data):
        data = interpolate_nans(data)
        return gaussian_filter1d(data, **smoothing_params)
    
    
    time_dim = eto.get_time_coord(da).dims[0]

    smoothed = xr.apply_ufunc(
        nan_smooth,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64]
    )
    smoothed = smoothed.transpose(time_dim, ...)
    
    return smoothed
