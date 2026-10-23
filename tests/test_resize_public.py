"""Registered resize/rescale calls in interchangeable public pipelines."""
import inspect
import warnings
import numpy as np
import pytest
import skimage as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def observe(function,*args,**kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=function(*args,**kwargs),None
        except Exception as caught:result,error=None,caught
    return result,error,[(type(w.message),str(w.message)) for w in notices]


def equivalent(expected,actual):
    result,error,notices=expected;value,failure,actual_notices=actual
    assert actual_notices==notices
    if error is not None:
        assert type(failure) is type(error) and str(failure)==str(error)
        return
    if failure is not None:raise failure
    assert type(value) is type(result) is np.ndarray
    assert value.shape==result.shape and value.strides==result.strides
    assert value.dtype==result.dtype and value.dtype.metadata==result.dtype.metadata
    if result.dtype.kind in 'biu':assert value.tobytes()==result.tobytes()
    else:
        tolerance=3e-6 if result.dtype==np.float32 else 3e-13
        np.testing.assert_allclose(value,result,rtol=tolerance,atol=tolerance,equal_nan=True)


def test_public_registry_and_signatures():
    from skimagexx._resize import resize,rescale
    assert xx.transform.resize is resize and xx.transform.rescale is rescale
    for name in ('resize','rescale'):
        assert 'transform.'+name in native_functions()
        assert inspect.signature(getattr(xx.transform,name))==inspect.signature(getattr(reference.transform,name))


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('endian',[False,True])
@pytest.mark.parametrize('api,target',[('resize',(7,11)),('rescale',1.)])
def test_memmap_identity_and_dtype(tmp_path,dtype,endian,api,target):
    descriptor=np.dtype(dtype).newbyteorder('>') if endian else np.dtype(dtype)
    image=np.memmap(tmp_path/'image.dat',mode='w+',dtype=descriptor,shape=(7,11))
    image[...]=np.arange(77).reshape(7,11)%17
    before=image.tobytes()
    options=dict(order=0,anti_aliasing=False)
    expected=observe(getattr(reference.transform,api),image,target,**options)
    actual=observe(getattr(xx.transform,api),image,target,**options)
    equivalent(expected,actual);assert image.tobytes()==before
    if actual[1] is None:assert not np.shares_memory(image,actual[0])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('pattern',[[0.,-0.,np.nan],[np.nan],[0.],[-0.]])
def test_memmap_runtime_bounds(tmp_path,dtype,pattern):
    image=np.memmap(tmp_path/'bounds.dat',mode='w+',dtype=dtype,shape=(7,11))
    image[...]=np.resize(np.array(pattern,dtype),image.shape)
    expected=observe(reference.transform.resize,image,image.shape,anti_aliasing=False)
    actual=observe(xx.transform.resize,image,image.shape,anti_aliasing=False)
    equivalent(expected,actual)
    if actual[1] is None:assert actual[0].tobytes()==expected[0].tobytes()


@pytest.mark.parametrize('dtype',[np.uint8,np.float16,np.float32,np.float64])
def test_complete_pipeline_can_swap_backend(dtype):
    image=(np.arange(17*19).reshape(17,19)%251).astype(dtype)
    def pipeline(backend):
        first=backend.transform.resize(image,(9,13))
        second=backend.filters.gaussian(first,.7,mode='mirror')
        return backend.transform.rescale(second,.7,mode='symmetric')
    equivalent(observe(pipeline,reference),observe(pipeline,xx))
