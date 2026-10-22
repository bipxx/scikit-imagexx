"""Adaptive threshold API, storage, borders and precision against upstream."""
import inspect
import re
import warnings
import numpy as np
import pytest
from skimage.filters import threshold_local as reference
import skimagexx as xx

REAL=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def settings(request):
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(previous)

def sample(dtype,shape):
    rng=np.random.default_rng(1502)
    values=rng.integers(0,127,shape)
    if dtype==np.bool_:return (values>64)
    if np.dtype(dtype).kind=='f':return (values/32-1).astype(dtype)
    return values.astype(dtype)

def compare(image,**options):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:expected=reference(image,**options)
        except Exception as error:
            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter('always')
                with pytest.raises(type(error),match=re.escape(str(error))):xx.filters.threshold_local(image,**options)
            assert [(type(w.message),str(w.message)) for w in actual_warnings]==[(type(w.message),str(w.message)) for w in expected_warnings]
            return
    before=image.copy()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=xx.filters.threshold_local(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in expected_warnings]
    assert type(actual)==type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
    assert actual.dtype.metadata==expected.dtype.metadata
    tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
    np.testing.assert_array_equal(image,before)

def test_signature():assert inspect.signature(xx.filters.threshold_local)==inspect.signature(reference)

@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('shape',[(),(0,),(0,7),(1,),(1,1),(7,),(8,),(9,),(17,19),(3,5,7),(1,5,1,7)])
@pytest.mark.parametrize('method',['gaussian','mean','median'])
def test_shapes_types(dtype,shape,method):compare(sample(dtype,shape),method=method)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('mode',['reflect','constant','nearest','mirror','wrap','grid-mirror','grid-constant','grid-wrap'])
@pytest.mark.parametrize('block',[1,3,5,(1,7),(7,3)])
def test_borders_windows(dtype,method,mode,block):compare(sample(dtype,(11,13)),method=method,mode=mode,block_size=block,cval=.31,offset=.17)

@pytest.mark.parametrize('dtype',[np.uint16,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('layout',['f','reverse','slice','unaligned','broadcast','readonly','endian','metadata'])
def test_storage(dtype,method,layout):
    a=sample(dtype,(13,17))
    if layout=='f':a=np.asfortranarray(a)
    if layout=='reverse':a=a[::-1,::-1]
    if layout=='slice':a=a[:,::2]
    if layout=='unaligned':
        storage=np.empty(a.nbytes+1,np.uint8);b=np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1);b[:]=a;a=b
    if layout=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    if layout=='readonly':a.flags.writeable=False
    if layout=='endian':a=a.astype(a.dtype.newbyteorder('S'))
    if layout=='metadata':a=a.view(np.dtype(dtype,metadata={'adaptive':'input'}))
    compare(a,method=method,offset=np.float32(.25))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('block',[-3,-1,0,1,2,3.,2.5,3.5,(),(3,),(3,4),np.int8(3),True])
def test_validation(dtype,method,block):compare(sample(dtype,(7,9)),method=method,block_size=block)

@pytest.mark.parametrize('method',['gaussian','mean','median','invalid'])
@pytest.mark.parametrize('mode',['bad',(),('reflect',),('wrap','mirror')])
@pytest.mark.parametrize('block',[1,3])
def test_modes_and_error_order(method,mode,block):compare(sample(np.float32,(7,9)),method=method,mode=mode,block_size=block)

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('offset',[0,-0.,.13,-2,True,np.float16(.2),np.float32(.2),np.float64(.2),np.int64(1)])
@pytest.mark.parametrize('shape',[(),(13,17)])
def test_offset_promotion(dtype,method,offset,shape):compare(sample(dtype,shape),method=method,offset=offset)

@pytest.mark.parametrize('param',[0,.5,1.,np.float16(.5),np.float32(.5),np.float64(.5),(0,.8),(1.,2.),np.nan,-1])
def test_gaussian_parameter(param):compare(sample(np.float32,(7,9)),param=param)

@pytest.mark.parametrize('method',['mean','median'])
def test_other_methods_ignore_param(method):compare(sample(np.float64,(9,13)),method=method,param=object())

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_parallel_reduction(dtype):compare(sample(dtype,(513,513)),method='mean',block_size=15,offset=.02)

def test_generic_explicit_fallback():
    a=sample(np.float32,(7,9));options={'method':'generic','param':np.mean}
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.threshold_local(a,**options)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.threshold_local(a,**options)
    np.testing.assert_array_equal(actual,reference(a,**options))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(2,),(3,),(1,5),(4,5),(2,3,4)])
@pytest.mark.parametrize('block',[3,5])
@pytest.mark.parametrize('mode',['reflect','mirror','nearest','wrap','constant'])
@pytest.mark.parametrize('cval',[-3.,0.,4.])
def test_zero_stride_offset_table(dtype,shape,block,mode,cval):
    a=np.broadcast_to(dtype(1.25),shape)
    compare(a,method='median',block_size=block,mode=mode,cval=cval)

@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_explicit_fallback(method,value):
    a=sample(np.float32,(7,9));a[2,3]=value
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.threshold_local(a,method=method)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.threshold_local(a,method=method)
    np.testing.assert_array_equal(actual,reference(a,method=method))

@pytest.mark.parametrize('method',['gaussian','mean','median'])
@pytest.mark.parametrize('dtype',[np.int64,np.uint64,np.float32,np.float64])
def test_full_range_and_continuous_values(method,dtype):
    if np.dtype(dtype).kind in 'iu':
        info=np.iinfo(dtype);data=np.array([info.min,info.max,0,1,info.max//2],dtype=dtype)
        a=np.resize(data,(17,19))
    else:a=np.random.default_rng(1029).normal(-2,7,(17,19)).astype(dtype)
    compare(a,method=method,block_size=7,offset=.123)
