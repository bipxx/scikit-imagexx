"""Genuine whole-core util view namespace/default14dtype contracts; saved source only."""
import importlib
import inspect
from pathlib import Path
import sys
import numpy as np
import pytest
from util_views_reference import APIS,reference
DTYPES=('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128')


@pytest.fixture
def genuine():
    import skimagexx as xx
    arrays=importlib.import_module('skimagexx._array_views');slicing=importlib.import_module('skimagexx._slice_along_axes')
    assert arrays._core is slicing._core is xx._core
    assert Path(xx._core.__file__).resolve().parent==Path(xx.__file__).resolve().parent
    for name in ('util_view_as_blocks','util_view_as_windows','util_slice_view','util_slice_copy_into','get_num_threads','set_num_threads'):
        entry=getattr(xx._core,name);assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    assert xx.get_num_threads()==1
    previous=xx.set_fallback_enabled(False)
    try:yield xx
    finally:xx.set_num_threads(1);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('tagged',(False,True))
def test_view_default_dtype_namespace_and_native_entry(api,dtype,workers,tagged,genuine,monkeypatch):
    xx=genuine
    import skimage.util as original
    from skimagexx.coverage import default_call,assert_default_parity
    from crop_observer import encode
    name=api.split('.')[-1];source=inspect.getsourcefile(getattr(original,name))
    def refused(*args,**options):raise AssertionError('ORIGINAL_BODY_REFUSED')
    monkeypatch.setattr(original,name,refused)
    carrier=np.empty((1,1),dtype=np.dtype(dtype,metadata={'units':'linear','source':'parity'})if tagged else dtype)
    args,options=default_call(api,carrier);before=encode((args,options),np)
    expected,scope,row=reference(api,args,options,source,np,tagged=tagged)
    module=importlib.import_module('skimagexx._slice_along_axes'if api==APIS[0]else'skimagexx._array_views')
    imported=getattr(importlib.import_module('skimagexx.util'),name)
    assert imported is getattr(module,name)is getattr(xx.util,name)is getattr(xx,name)
    entry=getattr(xx._core,{'slice_along_axes':'util_slice_view','view_as_blocks':'util_view_as_blocks','view_as_windows':'util_view_as_windows'}[name])
    entered=[];previous=sys.getprofile();state=np.geterr(),np.geterrcall(),np.getbufsize();xx.set_num_threads(workers)
    def trace(frame,event,value):
        if event=='c_return'and value is entry:entered.append(name)
        if event=='call'and frame.f_globals.get('__name__','').startswith('skimage.'):
            raise AssertionError('ORIGINAL_OR_HELPER_BODY_REFUSED')
    def pipeline(backend):return getattr(backend.util,name)(*args,**options)
    try:sys.setprofile(trace);actual=pipeline(xx)
    finally:sys.setprofile(previous)
    assert entered==[name]
    assert_default_parity(api,np.dtype(dtype),actual,expected)
    assert np.shares_memory(actual,args[0])and actual is not args[0]
    assert encode((args,options),np)==before and state==(np.geterr(),np.geterrcall(),np.getbufsize())
    assert xx.get_num_threads()==workers


def test_view_serial_metadata_and_real_copy_have_no_worker_lookup(genuine,monkeypatch):
    xx=genuine
    def refused(*args,**options):raise AssertionError('SERIAL_OPERATION_QUERIED_GLOBAL_WORKERS')
    monkeypatch.setattr(xx._core,'get_num_threads',refused)
    a=np.arange(24,dtype=np.int16).reshape(4,6)
    assert xx.util.view_as_blocks(a,(2,2)).shape==(2,3,2,2)
    assert xx.util.view_as_windows(a,(2,2)).shape==(3,5,2,2)
    result=xx.util.slice_along_axes(a,[(1,3)],copy=True)
    assert result.shape==(2,6)and result.flags.owndata and not np.shares_memory(result,a)
    assert result.tobytes()==a[1:3].tobytes()
