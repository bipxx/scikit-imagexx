import inspect,warnings
import numpy as np
import pytest
from skimage.feature import corner_kitchen_rosenfeld as reference
import skimagexx as xx


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def image(dtype,shape=(17,19)):
    rng=np.random.default_rng(1931)
    if np.dtype(dtype).kind in 'bu':return rng.integers(0,2 if dtype is np.bool_ else 255,size=shape).astype(dtype)
    if np.dtype(dtype).kind=='i':return rng.integers(-100,100,size=shape).astype(dtype)
    return rng.normal(size=shape).astype(dtype)


def compare(a,**options):
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=function(a,**options)
            except Exception as error:result=error
        return result,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    expected,ew=outcome(reference);actual,aw=outcome(xx.feature.corner_kitchen_rosenfeld);assert aw==ew
    if isinstance(expected,Exception):assert (type(actual),str(actual))==(type(expected),str(expected));return
    assert type(actual) is type(expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.strides==expected.strides and actual.flags.owndata==expected.flags.owndata
    assert actual.flags.c_contiguous==expected.flags.c_contiguous and actual.flags.f_contiguous==expected.flags.f_contiguous
    tolerance=2e-6 if expected.dtype.type is np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
    return actual


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mode',['constant','reflect','nearest','mirror','wrap','grid-constant','grid-mirror','grid-wrap'])
@pytest.mark.parametrize('cval',[0,-.3,1.7])
def test_types_and_borders(dtype,mode,cval):compare(image(dtype),mode=mode,cval=cval)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,3),(3,0),(1,1),(1,7),(7,1),(7,9)])
def test_empty_and_singletons(dtype,shape):compare(image(dtype,shape))


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16])
@pytest.mark.parametrize('layout',['reverse','fortran','transposed-strided','endian','tagged','tagged-endian','unaligned','readonly','broadcast'])
def test_layouts(dtype,layout):
    a=image(dtype)
    if layout=='reverse':a=a[::-1,::2]
    elif layout=='fortran':a=np.asfortranarray(a)
    elif layout=='transposed-strided':a=a[::-2,::2].T
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder())
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'units':'raw'}))
    elif layout=='tagged-endian':a=a.astype(np.dtype(a.dtype.newbyteorder(),metadata={'units':'raw'}))
    elif layout=='unaligned':
        b=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);b[:]=a;a=b
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    compare(a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_zero_signs(dtype):
    a=np.zeros((7,9),dtype=dtype);a[::2,::2]=-0.
    actual=compare(a)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(reference(a)))


def test_integer_range_is_not_normalized():
    a=image(np.uint8);expected=reference(a)
    actual=compare(a)
    assert np.max(np.abs(actual))>1
    np.testing.assert_array_equal(actual,expected)
    for dtype in [np.int64,np.uint64]:compare(np.random.default_rng(72).bit_generator.random_raw(17*19).reshape(17,19).astype(dtype))
    compare(np.arange(256,dtype=np.uint8).reshape(16,16).view(np.bool_))


def test_mode_sequence_and_typed_cval():
    for constant in [True,np.bool_(False),np.float16(.7),np.float32(.7),np.float64(.7),np.int64(1)]:
        compare(image(np.float32),mode=('reflect','constant'),cval=constant)


def test_memmap(tmp_path):
    a=np.memmap(tmp_path/'image.bin',mode='w+',dtype=np.float32,shape=(17,19));a[:]=image(np.float32);compare(a)


@pytest.mark.parametrize('kind',['nan','signaling-nan','extreme','complex','longdouble','rank0','rank1','rank3','bad-mode','infinite-cval'])
def test_explicit_compatibility(kind):
    a=image(np.float32);options={}
    if kind=='nan':a[0,0]=np.nan
    elif kind=='signaling-nan':a.view(np.uint32)[0,0]=0x7f800001
    elif kind=='extreme':a[:]=np.finfo(np.float32).max
    elif kind=='complex':a=a.astype(np.complex64)
    elif kind=='longdouble':a=a.astype(np.longdouble)
    elif kind=='rank0':a=np.zeros((),dtype=np.float32)
    elif kind=='rank1':a=a[0]
    elif kind=='rank3':a=a[None]
    elif kind=='bad-mode':options['mode']='bad'
    elif kind=='infinite-cval':options['cval']=np.inf
    with pytest.raises(NotImplementedError):xx.feature.corner_kitchen_rosenfeld(a,**options)
    xx.set_fallback_enabled(True);compare(a,**options)


def test_signature():assert inspect.signature(xx.feature.corner_kitchen_rosenfeld)==inspect.signature(reference)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_parallel_without_original(dtype,monkeypatch):
    a=image(dtype,(513,519));expected=reference(a)
    def forbidden(*args,**kwargs):raise AssertionError('original derivative routine executed')
    import scipy.ndimage as ndi
    monkeypatch.setattr(ndi,'sobel',forbidden)
    xx.set_num_threads(1);serial=xx.feature.corner_kitchen_rosenfeld(a)
    xx.set_num_threads(4);parallel=xx.feature.corner_kitchen_rosenfeld(a)
    np.testing.assert_array_equal(serial,parallel)
    np.testing.assert_allclose(serial,expected,rtol=2e-6 if dtype is np.float32 else 2e-13,atol=2e-6 if dtype is np.float32 else 2e-13)
