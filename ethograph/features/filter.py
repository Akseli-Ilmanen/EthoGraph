"""Taken from https://github.com/bendalab/thunderhopper."""

import numpy as np
from scipy.interpolate import interpn
from scipy.signal import butter, sosfiltfilt, sosfilt, sosfilt_zi
from ethograph.utils.arraytools import array_slice



# DATA TRANSFORMATION:


def downsampling(data, rate, new_rate, axis=0):
    """ Resamples time series data of given sampling rate to a lower new rate.
        Uses slicing along the specified axis where applicable. Otherwise, uses
        np.interp() for 1D data and scipy.interpolate.interpn() for ND data.
        Interpolation method is linear in both cases.

    Parameters
    ----------
    data : ND-array of floats (arbitrary shape)
        Data to be downsampled. Non-arrays are converted, if possible. Only the
        temporal array dimension specified by axis is resampled.
    rate : float or int
        Current sampling rate of data in Hz. Must be same for all time series.
    new_rate : float or int
        New sampling rate of data in Hz. Must be smaller than rate.
    axis : int, optional
        Time axis of data to be resampled. The default is 0.

    Returns
    -------
    downsampled : ND array of floats (data.shape except shape[axis])
        Downsampled data with the given new rate. Returns data unchanged if
        new_rate is not smaller than rate. Input dimensionality is preserved,
        but the size of axis is reduced.
    """

    
    # Rate conflict early exit:
    if new_rate >= rate:
        return data
    
    
    # Sampling rate ratio:
    n = rate / new_rate
    
    if abs(n - np.round(n)) < 0.01:
        # Clean ratio early exit (nth-entry selection along axis):
        return array_slice(data, axis, step=int(np.round(n)))

    # Interpolation for non-integer ratios:
    t = np.arange(data.shape[0]) / rate
    new_t = np.arange(0, t[-1], 1 / new_rate)
    if data.ndim == 1:
        # 1D interpolation early exit:
        return np.interp(new_t, t, data)

    # Prepare for ND-interpolation along axis:
    data_coords = [np.arange(i) for i in data.shape]
    sample_coords = data_coords.copy()
    data_coords[axis], sample_coords[axis] = t, new_t
    # Expand from dimension-wise to point-wise coordinates:
    sample_coords = np.meshgrid(*sample_coords, indexing='ij')
    sample_coords = np.vstack([grid.ravel() for grid in sample_coords]).T
    # Interpolate from current onto new point grid:
    downsampled = interpn(data_coords, data, sample_coords)
    # Restore initial dimensionality:
    new_shape = list(data.shape)
    new_shape[axis] = len(new_t)
    return downsampled.reshape(new_shape)


# SPECTRAL FILTERS:
def sosfilter(data, rate, cutoff, mode='lp', order=1, axis=0, refilter=True,
              padtype='even', padlen=None, padval=0., sanitize_padlen=True,
              zi=None, init=None):
    """ Applies a digital Butterworth filter in second-order sections format.
        Includes both the sosfilt() and sosfiltfilt() function of scipy.signal.
        Data can be filtered once (forward) or twice (forward-backward, which
        doubles the order of the filter and centers the phase of the output).
        Provides low-pass, high-pass, and band-pass filter types.

        Forward filters by sosfilt() can be set to a given initial state to
        control the start value. Forward-backward filters by sosfiltfilt() can
        be used with all built-in and a custom padding method. The refilter
        argument can be used to switch between the two functions.

    Parameters
    ----------
    data : ND array (any shape) of floats or ints
        Data to be filtered. Filter is applied along the specified time axis.
    rate : float or int
        Sampling rate of the time axis of data in Hz.
    cutoff : float or int or tuple (2,) of floats or ints
        Cut-off frequency of the applied filter in Hz. If mode is 'lp' or 'hp',
        must be a single value. If mode is 'bp', must be a tuple of two values.
    mode : str, optional
        Type of the applied filter. Options are 'lp' (low-pass), 'hp'
        (high-pass), and 'bp' (band-pass). The default is 'lp'.
    order : int, optional
        Order of the applied filter. If refilter is True, actual filter order
        is twice the specified order. The default is 1.
    axis : int, optional
        Time axis of data along which to apply the filter. The default is 0.
    refilter : bool, optional
        If True, uses the forward-backward filtering method of sosfiltfilt(),
        which enables use of padtype, padlen, and padval to control padding. If
        False, uses the forward filtering method of sosfilt(), which enables
        use of zi and init to control the start value. The default is True.
    padtype : str, optional
        Method used to pad the data before forward-backward filtering. Can be:
        # 'constant': Pads with first and last value of data, respectively.
        -> For signals with assessable endpoints (e.g. bound to some baseline).
        # 'even': Mirrors the data around each endpoint.
        -> For (noisy) signals whose statistics do not change much over time.
        # 'odd': Mirrors data, then turns it 180° around the endpoint.
        -> For oscillatory signals or where stable phase and smoothness is key.
        # 'fixed': Pads with padval (managed externally, not by scipy).
        -> For signals that are meant to be seen in a certain temporal context.
        # None: No padding.
        Ignored if refilter is False. The default is 'even'.
    padlen : int, optional
        Number of points added to each side of data. Applies for any padtype.
        Calculated by sosfiltfilt() as a very small number of points if None.
        Ignored if refilter is False or padtype is None or padtype is 'fixed'
        with padval being an array. If sanitize_padlen is True and padtype is
        a built-in method, may be reduced to avoid errors. The default is None.
    padval : float or int or ND array (any shape) of floats or ints, optional
        If specified and padtype is 'fixed', used as custom padding. If scalar,
        creates a constant padding of size padlen with the given value. If
        array, must have the same shape as data except along axis, so that it
        can be concatenated to data. Ignored if refilter is False or padtype is
        not 'fixed'. The default is 0.0.
    sanitize_padlen : bool, optional
        If True and padtype is a built-in method with padlen specified, clips
        padlen to be less than the size of the time axis of data, avoiding an
        internal sosfiltfilt() error. Ignored if refilter is False or padlen is
        None or padtype is 'fixed'. The default is True.
    zi : ND array of floats, optional
        If specified, sets the initial state for each second-order section of
        the applied forward filter, which in turn determines the start value(s)
        for filtering. Either returned by sosfilt() after a previous filtering
        step, or constructed manually. Shape must be (n_sections, ..., 2, ...),
        where n_sections equals sos.shape[0] and ..., 2, ... is the shape of
        data with data.shape[axis] replaced by 2. Use sosfilt_zi() to generate
        the initial state for a single section. If None, creates a filter state
        array of matching shape and adapts the inital state to the start value
        specified by init. Ignored if refilter is True. The default is None.
    init : float or int or ND array (any shape) of floats or ints, optional
        If specified and zi is None, adapts the initial filter state to this
        start values(s). If scalar, determines the start value for all filtered
        slices in data. If array, must have the same shape as data (except that
        init.shape[axis] must be 1) to set the start value for each slice. If
        None, uses the values of the first slice of data along axis. Ignored if
        refilter is True or zi is specified. The default is None.

    Returns
    -------
    filtered : ND array (data.shape) of floats
        Filtered data along the given time axis. If refilter is True, output of
        sosfiltfilt(), else sosfilt(). If mode is 'lp' and cutoff is above the
        Nyquist frequency (rate / 2), returns unchanged data. If mode is 'bp'
        and cutoff[1] is above Nyquist, falls back to pure high-pass filtering.
    next_state : ND array (zi.shape) of floats
        New filter state array after the applied forward filtering step to pass
        on with the next function call. Only returned if refilter is False. 
    """
    # Nyquist low-pass early exit:
    if mode == 'lp' and cutoff > rate / 2:
        return data
    # Nyquist band-pass fallback to high-pass:
    elif mode == 'bp' and cutoff[1] > rate / 2:
        mode, cutoff = 'hp', cutoff[0]

    # Initialize filter as second-order sections:
    sos = butter(order, cutoff, mode, fs=rate, output='sos')

    # FORWARDS:
    if not refilter:
        if zi is None:
            # Initialize filter state array:
            data_shape = list(data.shape)
            data_shape[axis] = 2
            # Shape must be (n_sections, ..., 2, ...):
            zi = np.zeros([sos.shape[0]] + data_shape)

            # Construct initial state:
            shape = [1] * data.ndim
            shape[axis] = 2
            # Shape must be (..., 2, ...):
            init_state = sosfilt_zi(sos).reshape(shape)
            if init is None:
                # Take values of 1st slice along axis:
                init = array_slice(data, axis, 0, 1)

            # Adapt filter state per section:
            for i in range(sos.shape[0]):
                zi[i] = init_state * init

        # Apply filter once with given state and start value:
        filtered, next_state = sosfilt(sos, data, axis, zi)
        return filtered, next_state

    # FORWARDS-BACKWARDS:
    if padtype == 'fixed':
        # Manage custom padding options:
        if isinstance(padval, np.ndarray):
            # Individual values:
            if padval.ndim != data.ndim:
                msg = 'If padval is an array, must have the same dimensions'\
                      'as data and be of matching shape except along axis.'
                raise ValueError(msg)
            data = np.concatenate((padval, data, padval), axis=axis)
            padlen = padval.shape[axis]
        else:
            # Constant value: 
            if padlen is None:
                # Auto-generated as per scipy default:
                padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                                     (sos[:, 5] == 0).sum()))
            padding = [(0, 0)] * data.ndim
            padding[axis] = (padlen, padlen)
            data = np.pad(data, padding, constant_values=padval)

    # Clip to maximum allowed padding length to avoid scipy error:    
    elif sanitize_padlen and padlen is not None and padlen >= data.shape[axis]:
        padlen = data.shape[axis] - 1

    # Apply filter twice with given padding method:
    filtered = sosfiltfilt(sos, data, axis, padlen=padlen,
                           padtype=None if padtype == 'fixed' else padtype)
    # Return options:
    if padtype == 'fixed':
        # Remove custom padding manually:
        start, stop = padlen, data.shape[axis] - padlen
        return array_slice(filtered, axis, start, stop)
    return filtered


def envelope(data, rate, cutoff=500., env_rate=2000., **kwargs):
    """ Extracts the signal envelope by low-pass filtering the rectified data.
        Envelope can be resampled to a lower rate to reduce memory load.

    Parameters
    ----------
    data : ND array of floats or ints
        Data to be filtered. Non-arrays are converted, if possible. If 1D,
        assumes a single time series. If 2D, assumes that each column is a
        separate time series and performs filtering along the first axis.
    rate : float or int
        Sampling rate of data in Hz.
    cutoff : float or int, optional
        Cut-off frequency of the low-pass filter in Hz. The default is 500.0.
    env_rate : float or int, optional
        Sampling rate of the resampled envelope in Hz. Skips downsampling if
        env_rate >= rate. The default is 2000.0.
    **kwargs : dict, optional
        Additional keyword arguments passed to sosfilter(). Can be used to
        modify properties of the applied low-pass filter.

    Returns
    -------
    env : 1D array (p,) or 2D array (p, n) of floats
        Extracted signal envelope with given sampling rate for each time series
        in data. Returns 1D if input was 1D, else 2D.
    """
    filtered = sosfilter(np.abs(data), rate, cutoff, mode='lp', **kwargs)
    env = downsampling(filtered, rate, env_rate)
    return env



