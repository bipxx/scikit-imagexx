"""Affected positive F/F public route with all original algorithms refused."""
import itertools
import numpy as np
import pytest
from test_find_boundaries_public import owned


@pytest.mark.parametrize('dtype',('bool','uint8','int8','uint16','int16','uint32','int32'))
@pytest.mark.parametrize('workers',(1,4))
def test_find_boundaries_positive_F_rows(dtype,workers,owned):
    labels=((np.arange(17*33,dtype=np.int32).reshape(17,33)//5)%4)
    a=np.asfortranarray(labels.astype(dtype))
    if dtype=='bool':a=np.asfortranarray(np.where(labels==0,0,np.where(labels==1,2,255)).astype(np.uint8)).view(np.bool_)
    a.flags.writeable=False
    values=(a.view(np.uint8)!=0).astype(np.uint8)if dtype=='bool'else a
    before=a.tobytes(),a.strides,a.flags.writeable,a.ctypes.data
    old=owned._core.get_num_threads();owned._core.set_num_threads(workers)
    policy=np.geterr(),np.geterrcall(),np.getbufsize()
    try:
        for connectivity in (1,2):
            low=np.full(a.shape,np.iinfo(values.dtype).max,dtype=values.dtype)
            high=np.full(a.shape,np.iinfo(values.dtype).min,dtype=values.dtype)
            for dy,dx in itertools.product((-1,0,1),repeat=2):
                if (dy!=0)+(dx!=0)>connectivity:continue
                v=np.take(np.take(values,np.clip(np.arange(17)+dy,0,16),axis=0),np.clip(np.arange(33)+dx,0,32),axis=1)
                low=np.minimum(low,v);high=np.maximum(high,v)
            for mode in ('thick','inner'):
                for background in (0,2,-1):
                    expected=(high!=low)&((values!=background)if mode=='inner'else True)
                    result=owned.find_boundaries(a,connectivity=connectivity,mode=mode,background=background)
                    assert type(result)is np.ndarray and result.dtype==np.dtype(bool)
                    assert result.flags.f_contiguous and result.strides==(1,17)and not np.shares_memory(result,a)
                    assert result.tobytes()==expected.tobytes()
                    assert owned._core.get_num_threads()==workers
                    assert (a.tobytes(),a.strides,a.flags.writeable,a.ctypes.data)==before
                    assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    finally:owned._core.set_num_threads(old)
