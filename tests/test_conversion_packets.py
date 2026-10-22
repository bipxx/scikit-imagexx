"""Exact boolean and uint16-to-int16 packet/tail contracts."""
import warnings
import numpy as np
import pytest
from skimage import util as reference
import skimagexx as xx

DTYPES=(np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64,np.float32,np.float64)

@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)

def checked(name,image):
    saved=image.copy()
    def observed(function):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            result=function(image)
        return result,[(n.category,str(n.message)) for n in notices]
    expected,notices=observed(getattr(reference,name))
    actual,actual_notices=observed(getattr(xx.util,name))
    assert actual_notices==notices
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and type(actual)==type(expected)
    assert (actual is image)==(expected is image)
    assert np.shares_memory(actual,image)==np.shares_memory(expected,image)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(image,saved)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('count',range(1,66))
def test_boolean_packet_order_and_tail(dtype,count):
    if np.dtype(dtype).kind=='f':
        half=dtype(.5)
        pattern=np.array([0.,1.,-1.,half,np.nextafter(half,dtype(0)),np.nextafter(half,dtype(1)),np.nan,np.inf,-np.inf,-0.],dtype=dtype)
    else:
        limits=np.iinfo(dtype);half=limits.max//2
        pattern=np.array([0,limits.max,half,half+1,half+2,limits.min,1],dtype=dtype)
    image=np.resize(pattern,count)
    image.flags.writeable=False
    checked('img_as_bool',image)

@pytest.mark.parametrize('count',range(1,66))
@pytest.mark.parametrize('maximum',[32766,32767,32768,65535])
def test_uint16_signed_packet_and_downcast_boundary(count,maximum):
    image=np.resize(np.array([0,1,2,maximum,127,255,maximum//2],dtype=np.uint16),count)
    checked('img_as_int',image)

@pytest.mark.parametrize('dtype',[*DTYPES,np.bool_])
@pytest.mark.parametrize('shape',[(17,19),(3,7,11)])
@pytest.mark.parametrize('unaligned',[False,True])
def test_fortran_boolean_output_layout(dtype,shape,unaligned):
    image=np.asfortranarray(np.resize(np.array([0,1,0,1,1],dtype=dtype),np.prod(shape)).reshape(shape))
    if unaligned:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        raw=np.ndarray(shape,dtype=dtype,buffer=storage,offset=1,order='F')
        raw[:]=image;image=raw
    image.flags.writeable=False
    checked('img_as_bool',image)
    expected=reference.img_as_bool(image)
    actual=xx.util.img_as_bool(image)
    assert actual.strides==expected.strides

@pytest.mark.parametrize('maximum',[32766,65535])
@pytest.mark.parametrize('unaligned',[False,True])
def test_fortran_signed_conversion_output_layout(maximum,unaligned):
    image=np.asfortranarray(np.resize(np.array([0,1,maximum],dtype=np.uint16),17*19).reshape(17,19))
    if unaligned:
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        raw=np.ndarray(image.shape,dtype=np.uint16,buffer=storage,offset=1,order='F')
        raw[:]=image;image=raw
    checked('img_as_int',image)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=reference.img_as_int(image);actual=xx.util.img_as_int(image)
    assert actual.strides==expected.strides
