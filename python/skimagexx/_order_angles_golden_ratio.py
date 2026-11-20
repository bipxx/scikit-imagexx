"""Owned finite POD ordering; NumPy represents arrays and int64 index scalars.

No upstream delegate. Source0.26 lazy recurrence/order preserved; provider
protocol/nonfinite/large duplicate-sort/extended accuracy domains stay pending.
"""
import numpy as np
from . import _core


def order_angles_golden_ratio(theta):
    # This is a true generator: even conversion and worker snapshot wait until
    # first resume. Plain list/tuple conversion is the argsort input's explicit
    # NumPy representation provider, not a replacement of the live theta.
    if type(theta) not in (np.ndarray,list,tuple):
        raise NotImplementedError('golden-angle arbitrary protocol frontier')
    array=np.asarray(theta)
    if array.ndim!=1 or array.dtype.kind not in 'biuf' or array.dtype.itemsize not in (1,2,4,8):
        raise NotImplementedError('golden-angle shape/dtype frontier')
    state=_core._order_angles_begin(array,int(np.__version__.split('.')[0])>=2)
    yield np.int64(state.first)
    while _core._order_angles_resume(state):
        if type(theta) in (list,tuple):
            # Source fancy-indexes the original carrier only AFTER firstyield
            # and increment construction. Let CPython own exact lazy error.
            theta[state.remaining_indices]
        yield np.int64(_core._order_angles_next(state,theta))
