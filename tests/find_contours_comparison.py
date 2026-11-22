"""Assertions for the ordered, ragged result of measure.find_contours."""

import numpy as np


def assert_contours_equal(actual, expected):
    """Compare an ordered list of float64 coordinate arrays word for word."""
    assert type(actual) is type(expected) is list, 'contour result must be a list'
    assert len(actual) == len(expected), 'contour count differs'
    for index, (left, right) in enumerate(zip(actual, expected)):
        message = f'contour {index}'
        assert type(left) is type(right) is np.ndarray, f'{message} must be an ndarray'
        assert left.shape == right.shape, f'{message} shape differs'
        assert left.ndim == 2 and left.shape[1] == 2, f'{message} must contain coordinate pairs'
        assert left.dtype == right.dtype == np.dtype('float64'), f'{message} must use native float64'
        assert left.dtype.metadata is right.dtype.metadata is None, f'{message} must use an untagged dtype'
        np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64),
                                      err_msg=f'{message} coordinate words differ')
