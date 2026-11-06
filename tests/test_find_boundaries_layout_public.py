"""Native-only loss-layout validation; no original functions or measurements."""
import itertools
import numpy as np
import pytest
from test_find_boundaries_public import owned


@pytest.mark.parametrize('kind',('F-inner','reverse-outer','3d-outer','bool-reverse-inner'))
@pytest.mark.parametrize('workers',(1,4))
def test_find_boundaries_layout_native_public(kind,workers,owned):
    shape=(17,19,23)if kind=='3d-outer'else(257,509)
    dtype=np.uint16 if kind in('F-inner','3d-outer')else np.int32
    a=((np.arange(np.prod(shape),dtype=np.uint32)//7)%4).astype(dtype).reshape(shape)
    if kind=='F-inner':a=np.asfortranarray(a)
    elif kind=='bool-reverse-inner':a=np.where(a%3==0,0,np.where(a%3==1,2,255)).astype(np.uint8).view(np.bool_)
    if 'reverse'in kind:a=a[...,::-1]
    a.flags.writeable=False
    mode='inner'if 'inner'in kind else'outer';connectivity=1 if kind=='bool-reverse-inner'else a.ndim
    # Independent bounded clamped-neighbor oracle, integer NumPy primitives only.
    labels=(a.view(np.uint8)!=0).astype(np.uint8)if a.dtype.kind=='b'else a
    low=np.full(shape,np.iinfo(labels.dtype).max,dtype=labels.dtype);high=np.full(shape,np.iinfo(labels.dtype).min,dtype=labels.dtype)
    full_high=high.copy();inverted=low.copy()
    for delta in itertools.product((-1,0,1),repeat=a.ndim):
        neighbor=labels
        for axis,shift in enumerate(delta):neighbor=np.take(neighbor,np.clip(np.arange(shape[axis])+shift,0,shape[axis]-1),axis=axis)
        full_high=np.maximum(full_high,neighbor)
        inverted=np.minimum(inverted,np.where(neighbor==0,np.iinfo(labels.dtype).max,neighbor))
        if sum(d!=0 for d in delta)<=connectivity:low=np.minimum(low,neighbor);high=np.maximum(high,neighbor)
    expected=(high!=low)&((labels!=0)if mode=='inner'else((labels==0)|(full_high!=inverted)))
    before=a.tobytes(),a.shape,a.strides,a.ctypes.data,a.flags.writeable
    policy=np.geterr(),np.geterrcall(),np.getbufsize();old=owned._core.get_num_threads();owned._core.set_num_threads(workers)
    try:
        actual=owned.find_boundaries(a,connectivity=connectivity,mode=mode)
        assert type(actual)is np.ndarray and actual.dtype==np.dtype(bool)and actual.shape==a.shape
        assert actual.tobytes()==expected.tobytes()and not np.shares_memory(actual,a)
        assert (a.tobytes(),a.shape,a.strides,a.ctypes.data,a.flags.writeable)==before
        assert owned._core.get_num_threads()==workers and (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    finally:owned._core.set_num_threads(old)
