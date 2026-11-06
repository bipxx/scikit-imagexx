"""Genuine shared-core ownership and source-bound saved metadata smoke.

Private component extensions cannot satisfy these tests. Platform/raw FENV and
provider/full-domain limits remain explicit in the component contracts.
"""
import importlib
import inspect
import sys
from pathlib import Path
import numpy as np
import pytest
from util_metadata_reference import reference
DTYPES={'crop':('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128'),
        'regular_grid':('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float32','float64')}


@pytest.fixture
def genuine():
    import skimagexx as xx
    crop=importlib.import_module('skimagexx._crop');grid=importlib.import_module('skimagexx._regular_grid')
    assert crop._core is grid._core is xx._core
    assert xx.util.crop is crop.crop and xx.util.regular_grid is grid.regular_grid
    assert Path(xx._core.__file__).resolve().parent==Path(xx.__file__).resolve().parent
    for name in ('crop_view','crop_copy_into','regular_grid_prepare','regular_grid_finish','regular_grid_suffix_product','get_num_threads','set_num_threads'):
        entry=getattr(xx._core,name);assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    assert xx.get_num_threads()==1
    previous=xx.set_fallback_enabled(False)
    try:yield xx
    finally:xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def traced_native(xx,call):
    # Observe the actual installed builtin return, preserving caller profiling.
    entries={name:getattr(xx._core,name)for name in ('crop_view','crop_copy_into','regular_grid_prepare','regular_grid_finish')}
    observed=[];previous=sys.getprofile()
    def trace(frame,event,value):
        if event=='c_return':
            observed.extend(name for name,entry in entries.items()if value is entry)
    sys.setprofile(trace)
    try:return call(),observed
    finally:sys.setprofile(previous)


@pytest.mark.parametrize('name,dtype',[(n,d)for n,ds in DTYPES.items()for d in ds])
@pytest.mark.parametrize('workers',(1,4))
def test_util_metadata_shared_core_binary_dtype_and_swap(name,dtype,workers,genuine,monkeypatch):
    xx=genuine
    import skimage.util as original
    from skimagexx.coverage import default_call,assert_default_parity
    from crop_observer import encode as crop_encode
    from regular_grid_observer import encode as grid_encode
    api='util.'+name;source=inspect.getsourcefile(getattr(original,name))
    def refused(*args,**kwargs):raise AssertionError('ORIGINAL_BODY_REFUSED')
    monkeypatch.setattr(original,name,refused)
    args,options=default_call(api,np.empty((1,1),dtype));encode=crop_encode if name=='crop'else grid_encode
    before=encode(args,np),dict(options);state=np.geterr(),np.geterrcall(),np.getbufsize()
    expected,scope,row=reference(api,args,options,source,np)
    xx.set_num_threads(workers)
    from skimagexx.util import crop as imported_crop,regular_grid as imported_grid
    imported=imported_crop if name=='crop'else imported_grid
    assert imported is getattr(xx.util,name)
    def pipeline(backend):return getattr(backend.util,name)(*args,**options)
    actual,entered=traced_native(xx,lambda:pipeline(xx))
    assert entered==(['crop_view']if name=='crop'else['regular_grid_prepare']if dtype=='bool'else['regular_grid_prepare','regular_grid_finish'])
    assert_default_parity(api,np.dtype(dtype),actual,expected)
    if name=='crop':assert np.shares_memory(actual,args[0])and actual.base is args[0].base and actual is not args[0]
    assert (encode(args,np),dict(options))==before
    assert state==(np.geterr(),np.geterrcall(),np.getbufsize())and xx.get_num_threads()==workers
    with pytest.raises(TypeError):imported(args[0])


def test_util_metadata_strict_provider_frontier_and_no_worker_snapshot(genuine):
    xx=genuine
    # Provider partitions are disclosed; strict unsupported grid never forwards.
    with pytest.raises(NotImplementedError):xx.util.regular_grid(np.array([5,7],np.complex64),6)
    assert xx.get_num_threads()==1
    # Copy alloc/order are NumPy metadata; native transport must return disjoint POD.
    a=np.arange(20,dtype=np.int16).reshape(4,5)
    for order in ('C','F','A','K'):
        out,entered=traced_native(xx,lambda:xx.util.crop(a,1,copy=True,order=order))
        assert entered==['crop_view','crop_copy_into']
        assert out.shape==(2,3)and not np.shares_memory(out,a)and out.flags.owndata
        assert out.tobytes()==bytes.fromhex('0600070008000b000c000d00')
    assert xx.get_num_threads()==1
