"""
Helpers and utilities

Masked arrays:
 Helper functions to work with masked arrays for the control of sample lables
 during the active learning loop.
 It has `True` in masked indices and `False` in unmasked indices.

"""

import numpy as np


def _check_1d(arr):
    arr = np.array(arr)

    # if the conversion to array didn't succeeded, it could be an array of objects.
    if arr.dtype == 'O':
        raise ValueError('Expected `arr` to be an iterable or array of class labels, got {}.'.format(arr))

    if arr.ndim > 1:
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.flatten()
        else:
            raise ValueError('Expected `arr` to be an one-dimensional array, got {}.'.format(arr))

    return arr


def _check_masked_labels(arr):
    """ Raise an exception if `arr` is not a 1d masked array

    arr: numpy.ma.masked_array
        A 1-dimensional masked array

    Raises
    ------
    ValueError
    """
    if not isinstance(arr, np.ma.masked_array):
        raise TypeError('Expected `arr` to be an masked array, got {}.'.format(type(arr)))

    if arr.ndim > 1:
        raise TypeError('Expected `arr` to be an one-dimensional array, got {}.'.format(arr))

    return arr


def _create_mask(arr, known_idx):
    """ Check and create a mask for `y` where it is `True` in `known_idx`s and `False` in the rest.

    Parameters
    ----------
    arr: numpy.ma.masked_array
        A 1-dimensional masked array

    known_idx:

    Returns
    -------
    labels: np.array
        The checked `arr`.

    mask: np.array
        A vector of booleans of the same shape as `arr` and `labels`.
        It has `True` in masked indices and `False` in unmasked indices.
    """
    labels = _check_1d(arr)
    knowns = _check_1d(known_idx)

    mask = np.zeros(len(labels), dtype=bool)
    if len(knowns) > 0:
        mask[known_idx] = True

    return labels, mask


def _mask_indices(arr, indices):
    """ Return a masked array where the mask is `True` (excluded values) in
    the array indices included in `indices`.

    Parameters
    ----------
    arr: numpy.ma.masked_array
        A 1-dimensional masked array

    indices:

    Returns
    -------

    Notes
    -----
    I would name this function as `mask_indices` but Numpy already has
    `np.mask_indices` which does something different.
    """
    labels, mask = _create_mask(arr, indices)
    return np.ma.masked_array(labels, mask)


def mask_unknowns(y, unknown_idx=None):
    """ Return a masked array where the mask is `True` (excluded values) in
    the indices included in `unknown_idx`.

    Parameters
    ----------
    y: numpy.array or list
        Vector of sample labels

    unknown_idx: numpy.array or list
        Indices of the labels in `y` that are unknown.
        If None, will leave all elements unmasked.

    Returns
    -------
    masked_y: numpy.masked_array
        Masked array with the values of `y` and a mask where is `True` if the component
        index is in `unknown_idx`, `False` otherwise.
    """
    if unknown_idx is None:
        unknown_idx = []

    return _mask_indices(y, unknown_idx)


def unmasked_indices(arr):
    """ Return the indices where `arr` is unmasked.

    Parameters
    ----------
    arr: numpy.ma.masked_array
        A 1-dimensional masked array

    Returns
    -------
    indices: np.array
        Array of indices
    """
    return np.where(arr.mask == False)[0]


def masked_indices(arr):
    """ Return the indices where `arr` is unmasked.

    Parameters
    ----------
    arr: numpy.ma.masked_array
        A 1-dimensional masked array

    Returns
    -------
    indices: np.array
        Array of indices
    """
    return np.where(arr.mask == True)[0]


# def mask_knowns(y, known_idx):
#     """ Return a masked array where the mask is `True` (excluded values) in
#     the indices NOT included in `known_idx`.
#
#     Parameters
#     ----------
#     y: numpy.array or list
#         Vector of sample labels
#
#     known_idx: numpy.array or list
#         Indices of the labels in `y` that are known.
#
#     Returns
#     -------
#     obscured_y: numpy.ndarray
#         Vector of sample labels with `-1` in the indices not included in `known_idx`.
#     """
#     labels, mask = _create_mask(y, known_idx)
#
#     return np.ma.masked_array(labels, np.invert(mask))