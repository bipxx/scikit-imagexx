"""Byte-row sorting in C++, with literal NumPy storage preparation/gather."""
import numpy as np
from . import _core


def unique_rows(ar):
    # scikit-image0.26 unique.py, BSD-3-Clause; see the retained upstream
    # license. Preserve rank/protocol/copy/view errors before native sorting.
    if ar.ndim != 2:
        raise ValueError(
            'unique_rows() only makes sense for 2D arrays, ' f'got {ar.ndim}')
    ar = np.ascontiguousarray(ar)
    ar_row_view = ar.view(f'|S{ar.itemsize * ar.shape[1]}')
    native_result = _core.unique_rows_apply(ar, _core.get_num_threads())
    if native_result is not None:
        unique_row_indices = native_result
    else:
        _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    # Keep source dtype transfers, metadata and output ownership in the literal
    # advanced-index gather. Native admission claims sorting/equality only.
    ar_out = ar[unique_row_indices]
    return ar_out
