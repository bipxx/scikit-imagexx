"""Serial native shape arithmetic with explicit scalar/metadata authorities.

Source phase order follows scikit-image0.26 util/_regular_grid.py (BSD-3-Clause;
third_party/scikit-image-LICENSE.txt). NumPy owns array construction, scalar FP
callbacks and the platform default-int slice scalar type. Nonvector shape
prefixes retain their literal NumPy metadata provider, an open native domain.
"""
import math
import numpy as np
from . import _core
from ._compat import fallback


def _nonvector_metadata(shape,points):
    # Invalid-shape broadcasting/cast errors have provider-specific precedence.
    # Keep this bounded metadata partition explicit; it is not native ownership.
    ndim=len(shape);inverse=np.argsort(np.argsort(shape));dimensions=np.sort(shape)
    space=float(np.prod(shape))
    if space<=points:return (slice(None),)*ndim
    steps=np.full(ndim,(space/points)**(1./ndim),dtype='float64')
    if (dimensions<steps).any():
        for dim in range(ndim):
            steps[dim]=dimensions[dim]
            space=float(np.prod(dimensions[dim+1:]))
            steps[dim+1:]=(space/points)**(1./(ndim-dim-1))
            if (dimensions>=steps).all():break
    starts=(steps//2).astype(int);steps=np.round(steps).astype(int)
    values=[slice(start,None,step) for start,step in zip(starts,steps)]
    return tuple(values[i] for i in inverse)


def regular_grid(ar_shape,n_points):
    """Return source-typed slices locating approximately n_points grid points."""
    shape=np.asanyarray(ar_shape);ndim=len(shape)
    if type(shape) is not np.ndarray:
        return fallback('util','regular_grid',ar_shape,n_points)
    if shape.ndim!=1:return _nonvector_metadata(shape,n_points)
    if shape.dtype.kind=='O' and all(type(v)is int for v in shape):
        # Exact CPython integer product for object-integer early branches.
        space=float(math.prod(shape.tolist()))
        if space<=n_points:return (slice(None),)*ndim
        return fallback('util','regular_grid',ar_shape,n_points)
    if shape.dtype.kind not in 'biuf' or shape.dtype.itemsize not in (1,2,4,8) or shape.dtype.kind=='f' and shape.dtype.itemsize not in (4,8):
        return fallback('util','regular_grid',ar_shape,n_points)
    prepared=_core.regular_grid_prepare(shape,np.dtype(int).itemsize)
    if prepared.space_size<=n_points:return (slice(None),)*ndim
    if type(n_points) not in (int,float,bool,np.int64,np.float64):
        return fallback('util','regular_grid',ar_shape,n_points)
    if isinstance(n_points,np.generic) and n_points<0:
        # NumPy negative-base scalar power has a different warning/NaN policy
        # from Python float power. This unadmitted partition stays explicit.
        return fallback('util','regular_grid',ar_shape,n_points)
    if isinstance(n_points,np.generic) and n_points==0:
        # Actual NumPy scalar division owns warn/raise/call and FENV. Its zero
        # path necessarily reaches the source's final Python exponent error.
        quotient=prepared.space_size/n_points
        quotient**(1./ndim)
        for dim in range(ndim):
            quotient=_core.regular_grid_suffix_product(prepared,dim+1)/n_points
            quotient**(1./(ndim-dim-1))
        raise AssertionError('unreachable NumPy zero-count source phase')
    points=float(n_points)
    if math.isnan(points):return fallback('util','regular_grid',ar_shape,n_points)
    fields=_core.regular_grid_finish(prepared,points)
    scalar=np.dtype(int).type
    return tuple(slice(None) if start is None else slice(scalar(start),None,scalar(step)) for start,step in fields)
