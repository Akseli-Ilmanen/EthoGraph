"""Taken from https://github.com/bendalab/thunderhopper"""

import numpy as np


def ensure_sequence(*vars, skip_None=True, unwrap=True):
    """ Ensures that all passed variables are iterable and sequence-like.
        Allows functions expecting iterable inputs (e.g in single for-loops) to
        handle scalar inputs, as well. "Sequence-like" refers to tuples, lists,
        and ND arrays with at least one dimension (np.ndim(var) > 0). Variables
        of these types are returned unchanged. Scalars (ints/floats/bools) or
        other iterables (dicts/strings/sets) are converted to single-element
        tuples. 0D arrays are expanded to 1D arrays. Nones can be treated as
        scalar variables or be passed through as None on request. 

    Parameters
    ----------
    *vars : arbitrary types (m,)
        One or multiple variables to be checked and converted if necessary.
    skip_None : bool, optional
        If True, returns Nones as Nones, else as single-element tuples (None,).
        The default is True.
    unwrap : bool, optional
        If True and only a single variable is passed, returns the converted
        variable without enclosing tuple. If False, output is always wrapped in
        a tuple, even if only one variable is passed. The default is True.

    Returns
    -------
    vars : tuple of arbitrary types (m,)
        Checked and converted input variable or variables.
    """    
    # Delegate type-checking and sequence enforcement to helpers:
    skip_var = lambda v: np.ndim(v) > 0 or (skip_None and v is None)
    make_sequence = lambda v: v[None] if isinstance(v, np.ndarray) else (v,)
    # Check each input variable and modify it if necessary:
    vars = tuple(v if skip_var(v) else make_sequence(v) for v in vars)
    return vars[0] if unwrap and len(vars) == 1 else vars

def equal_sequences(*vars, skip_None=True):
    """ Ensures that all passed variables are sequence-likes of same length.
        "Sequence-like" refers to tuples, lists, and ND arrays with at least
        one dimension (np.ndim(var) > 0). Calls ensure_sequence() to convert
        any other variables to single-element tuples or 1D arrays. Nones can
        either be tuple-wrapped or passed through as None. Single-element
        sequences are repeated to match the length of the longest sequence,
        converting all arrays to lists beforehand.

    Parameters
    ----------
    *vars : arbitrary types (m,)
        Multiple variables to be type-checked and equalized, if possible.
    skip_None : bool, optional
        If True, returns Nones as Nones. If False, treats Nones as single-
        element tuples (None,) to be repeated. The default is True.

    Returns
    -------
    vars : tuple of arbitrary types (m,)
        Checked and equalized input variables.

    Raises
    ------
    ValueError
        Breaks if any sequence size is neither 1 nor the maximum across vars.
    """  
    # Enforce tuples, letting lists and arrays >0D pass through:
    vars = ensure_sequence(*vars, skip_None=skip_None, unwrap=False)

    # Count number of elements in each sequence:
    sizes = [len(v) if v is not None else (1 - skip_None) for v in vars]
    target = max(sizes)

    # Validate compatibility of element counts:
    if not all(n in (0, 1, target) for n in sizes):
        msg = f'With a maximum sequence length of {target}, variables can '\
              f'only have length {target} or 1 or be None: {sizes}'
        raise ValueError(msg)

    # Equalize sequence length across variables:
    convert_var = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    zipped = zip(vars, sizes)
    return tuple(convert_var(v) * target if l == 1 else v for v, l in zipped)


## CHECKS & VALIDATION ##

def is_valid_numpy_index(ind):
    """ Checks if the given variable is valid for indexing numpy arrays.
        Valid array indices can be integers, bools, arrays of ints or bools,
        slice objects, ellipsis, None, or tuples of a combination of those.

    Parameters
    ----------
    ind : any type (any shape)
        The variable to validate as numpy array index.

    Returns
    -------
    valid : bool
        Returns True if ind can be used to index numpy arrays, False otherwise.
    """
    # Force dimension specificity:    
    if not isinstance(ind, tuple):
        ind = (ind,)

    for dim in ind:
        # Validate indices along each array dimension separately:
        data_type = type(dim) if not isinstance(dim, np.ndarray) else dim.dtype
        is_int = np.issubdtype(data_type, np.integer)
        is_bool = np.issubdtype(data_type, np.bool_)
        is_slice = isinstance(dim, slice)
        # Type violation early exit:
        if not any([is_int, is_bool, is_slice, dim is Ellipsis, dim is None]):
            return False
    return True



## ARRAY MANIPULATION (VIEW) ##


def reduce_array(array, squeeze=False, unpack=False):
    """ Wrapper to quickly squeeze and itemize numpy arrays, if possible.
        Allows to remove singleton dimensions (squeeze, maintains at least 1D)
        and to convert single-entry arrays into scalar output (unpack). 

    Parameters
    ----------
    array : ND-array of arbitrary type and shape
        Array to be shape-checked and reduced to simpler formats, if possible.
        Must be at least 1D. Array is not copied or modified by this function.
    squeeze : bool, optional
        If True, calls np.squeeze() on array to remove dimensions of size 1. If 
        array is size 1, retains a single dimension to prevent 0D output. The
        default is False.
    unpack : bool, optional
        If True and output would be an array of size 1, converts the array into
        a single scalar. Applied after squeezing. The default is False. 

    Returns
    -------
    array : 1D...ND-array or scalar of array.dtype
        View of input array after dimensional pruning and scalar conversion.
        If squeeze is True, may have fewer dimensions (down to 1D). If unpack
        is True, may be converted to a scalar. If input array has no singleton
        dimensions and is not size 1, the output is the same as the input. If
        squeeze and unpack are both False, immediately returns the input array.
    """
    # Optional dimensional pruning:
    if squeeze and array.ndim > 1:
        # Remove all singleton dimensions, retaining at least 1D:
        axes = (ind for ind, size in enumerate(array.shape)[1:] if size == 1)
        array = np.squeeze(array, axis=axes)
    # Optional scalar conversion of single-entry 1D arrays:
    return array.item() if (unpack and array.size == 1) else array


def slice_index(dims, axis=0, start=None, stop=None, step=1, include=False):
    """ Indices to slice an N-dimensional array along one or multiple axes.
        Shamelessly stolen and adapted from scipy.signal._arraytools.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target array.
    axis : int or tuple/list/1D array of ints (m,), optional
        Target axis or axes along which to slice the array. If several, enables
        multi-dimensional slicing, also accepting sequence-like inputs for each
        of start, stop, and step. Single elements apply to all target axes.
        The default is 0.
    start : int or tuple/list/1D array of ints (m,), optional
        Inclusive start index of the slice. The default is None.
    stop : int or tuple/list/1D array of ints (m,), optional
        Stop index of the slice. Inclusive if include is True, otherwise
        exclusive as by standard python doctrine. The default is None.
    step : int or tuple/list/1D array of ints (m,), optional
        Step size between indices in the slice. The default is 1.
    include : bool, optional
        If True, makes the specified stop the last index of the slice.
        The default is False.

    Returns
    -------
    slice_inds : tuple of slice objects (dims,)
        Indices for slicing along the given target axes.
    """
    # Ensure 1D sequence-likes of equal length:
    axis, start, stop, step = equal_sequences(axis, start, stop, step,
                                              skip_None=False)

    # Full slice along each dimension:
    slice_inds = [slice(None)] * dims
    for ax, first, last, interval in zip(axis, start, stop, step):
        if include and last is not None:
            # Ensure inclusive stop:
            step_sign = np.sign(interval)
            # Sanitize edge cases (specific combinations of stop and step):
            if any(np.isin([-1, 0], last) & np.isin([1, -1], step_sign)):
                last = None
            else:
                # One more or less:
                last += step_sign
        # Adjust slice along each target axis:
        slice_inds[ax] = slice(first, last, interval)
    return tuple(slice_inds)


def array_slice(array, axis=0, start=None, stop=None, step=1,
                include=False, squeeze=False, unpack=False):
    """ Slices an N-dimensional array along one or multiple target axes.
        Returns a view of the sliced input array, preserving dimensionality.
        Sliced arrays can optionally be squeezed to remove excess singleton
        dimensions. Single-entry arrays can further be converted to scalars.
        Shamelessly stolen and adapted from scipy.signal._arraytools.

    Parameters
    ----------
    array : ND-array of arbitrary type and shape
        Input array to be sliced.
    axis : int or list of ints (m,), optional
        Target axis or axes along which to slice the array. If several, enables
        multi-dimensional slicing, also accepting list inputs for start, stop,
        and step (integer inputs apply to all target axes). The default is 0.
    start : int or list of ints (m,), optional
        Inclusive start index of the slice. The default is None.
    stop : int or list of ints (m,), optional
        Stop index of the slice. Inclusive if include is True, otherwise
        exclusive as by standard python doctrine. The default is None.
    step : int or list of ints (m,), optional
        Step size between indices in the slice. The default is 1.
    include : bool, optional
        If True, makes the specified stop the last index of the slice.
        The default is False.
    squeeze : bool, optional
        If True, calls np.squeeze() on the sliced array to remove dimensions of
        size 1, retaining at least a single dimension. The default is False.
    unpack : bool, optional
        If True and output would be an array of size 1, converts the array into
        a single scalar. Applied after squeezing. The default is False.

    Returns
    -------
    sliced : array of array.dtype or scalar of array.dtype
        View of the sliced array with as many dimensions as the input array.
        If squeeze is True, may have less dimensions (but will be at least 1D).
        If unpack is True, may be a scalar instead of an array.
    """
    slice_inds = slice_index(array.ndim, axis, start, stop, step, include)
    return reduce_array(array[slice_inds], squeeze, unpack)





## ARRAY INSPECTION ##


def edge_along_axis(array, axis=0, which=-1, validate=True, reduce=False):
    """ In a 2D array, finds first or last non-zero entry per column or row.
        Designed for multiple distributions or multi-channel time series data,
        but can handle 1D arrays by a separate processing branch. By default,
        identified non-zero entries are returned as a tuple of 1D arrays that
        can be used to index the original array (np.nonzero format). Output can
        be reduced to a single 1D array of indices along the given axis.

    Parameters
    ----------
    array : 2D array (m, n) or 1D array (m,) of arbitrary type 
        Array in which to identify the requested edge indices.
    axis : int, optional
        Array dimension along which to find the requested edge indices. Options
        are 0 or 1. If axis is 0, searches along each array column. If axis
        is 1, searches along each array row. Must be 0 if array is 1D. The
        default is 0.
    which : int, optional
        Specifies whether to find the first or last non-zero entry along the
        given axis. Options are 0 (first) and -1 (last). The default is -1.
    validate : bool, optional
        If True and array contains all-zero columns/rows, omits corresponding
        indices from the output. If False, returns indices for all slices (non-
        sensical integers in case of all-zero slices). The default is True.
    reduce : bool, optional
        If False, returns a tuple of two 1D arrays (one if input array is 1D),
        which provide the full index pair for each column/row. This format is
        valid for indexing both 1D and 2D input arrays, producing 1D output. If
        True, returns a single 1D array of indices along the given array axis,
        with one entry per slice. This format is good to access the indices
        themselves but may be unsuitable for indexing. The default is False.

    Returns
    -------
    inds : 1D array (p,) or tuple (2, or 1,) of 1D arrays (p,)
        Indices of the first or last non-zero entry per column or row in array.
        If reduce is True, returns a single 1D array of indices along axis,
        else a tuple of one or two 1D index arrays (depending on array.ndim).
        Can at most hold as many entries as specified slices in array (less if
        validate is True and and array contains all-zero slices).

    Raises
    ------
    ValueError
        Breaks if array is neither 1D nor 2D. Breaks immediately if array
        contains no non-zero entries. If validate is True, breaks if no entries
        remain after omitting all-zero slices.
    """
    # All-zero early exit:
    if not np.any(array):
        raise ValueError('No non-zero entries found in the given array.') 

    # Input interpretation:
    if which == 0:
        # Find first indices:
        mask_value = np.inf
        func = np.argmin
    elif which == -1:
        # Find last indices:
        mask_value = -np.inf
        func = np.argmax

    # Univariate early exit:
    if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
        # Get index coordinates of entry:
        inds = np.argwhere(array)[which]
        if reduce:
            # Slice coordinates to single dimension (1D array):
            return array_slice(inds, start=axis, stop=axis, include=True)
        # Ensure index use (tuple of 1D arrays):
        return tuple(i for i in inds[:, None])
    # Shape validation:
    elif array.ndim != 2:
        raise ValueError('Input array must be 1D or 2D.')

    # Draw parallel index gridlines along axis:
    grid_inds = np.arange(array.shape[axis])
    grid_inds = np.expand_dims(grid_inds, axis=1 - axis)
    # Set non-zero entries to indices, hide others:
    masked_array = np.where(array, grid_inds, mask_value)
    # Find smallest/largest index per column/row:
    inds = func(masked_array, axis=axis).astype(int)

    # Return options:
    if validate:
        # Omit all-zero columns/rows:
        is_valid = np.any(array, axis=axis)
        inds = inds[is_valid]
        if inds.size == 0:
            # Break instead of returning empty index array:
            raise ValueError('No non-zero entries found in the given array.')
    if reduce:
        # 1D array:
        return inds
    # Complete index pairs by each row/column:
    other_inds = np.arange(array.shape[1 - axis])
    if validate:
        # Match to valid counterparts:
        other_inds = other_inds[is_valid]
    # Return sorted to enable index use (tuple of 1D arrays):
    return (other_inds, inds) if axis else (inds, other_inds)


