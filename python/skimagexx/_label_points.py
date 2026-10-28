"""Literal coordinate preparation followed by owned label-mask generation."""
import numpy as np
from . import _core


def label_points(coords, output_shape):
    # scikit-image0.26 _label.py, BSD-3-Clause; see the retained license in
    # third_party/scikit-image-LICENSE.txt. Keep observable prelude order.
    # Object callbacks can mutate the original shape during rounding/casting;
    # retain their literal late np.arange evaluation in the provider tail.
    native_coords = type(coords) is np.ndarray and coords.dtype.kind != 'O'
    if coords.shape[1] != len(output_shape):
        raise ValueError('Dimensionality of points should match the output shape')
    if np.any(coords < 0):
        raise ValueError('Coordinates should be positive and start from 0')
    np_indices = tuple(np.transpose(np.round(coords).astype(int, copy=False)))
    labels = np.zeros(output_shape, dtype=np.uint64)
    native_result = None
    if (native_coords and type(output_shape) in (tuple, list)
            and output_shape and all(type(size) is int for size in output_shape)
            and coords.shape[0] == np_indices[0].size):
        native_result = _core.label_points_apply(
            np_indices, labels, _core.get_num_threads())
        if native_result is not None:
            return native_result
    # Reuse already prepared indices after refusal; do not repeat protocols,
    # warning-producing casts, rounding or the negative-coordinate check.
    labels[np_indices] = np.arange(1, coords.shape[0] + 1)
    return labels
