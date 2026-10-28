"""Literal map preparation followed by owned typed lookup and output writes."""
import numpy as np
from . import _core


def map_array(input_arr, input_vals, output_vals, out=None):
    # scikit-image0.26 _map_array.py, BSD-3-Clause; license attribution is in
    # third_party/scikit-image-LICENSE.txt. Preserve the source prelude: its
    # actual shape setter and casts supply observable warnings/callbacks.
    if not np.issubdtype(input_arr.dtype, np.integer):
        raise TypeError('The dtype of an array to be remapped should be integer.')
    orig_shape = input_arr.shape
    input_arr = input_arr.reshape(-1)
    if out is None:
        out = np.empty(orig_shape, dtype=output_vals.dtype)
    elif out.shape != orig_shape:
        raise ValueError(
            'If out array is provided, it should have the same shape as '
            f'the input array. Input array has shape {orig_shape}, provided '
            f'output array has shape {out.shape}.'
        )
    try:
        out_view = out.view()
        out_view.shape = (-1,)
    except AttributeError:
        raise ValueError(
            'If out array is provided, it should be either contiguous '
            f'or 1-dimensional. Got array with shape {out.shape} and '
            f'strides {out.strides}.'
        )
    input_vals = input_vals.astype(input_arr.dtype, copy=False)
    output_vals = output_vals.astype(out.dtype, copy=False)
    native_result = _core.map_array_apply(
        input_arr, out_view, input_vals, output_vals, _core.get_num_threads())
    return out
