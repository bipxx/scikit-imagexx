"""Public structure-tensor contract and strict native execution."""
import inspect
import warnings
import numpy as np
import pytest
from skimage.feature import structure_tensor as reference
import skimagexx as xx


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def image(dtype,shape=(17,19)):
    rng=np.random.default_rng(831)
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
    expected,ew=outcome(reference);actual,aw=outcome(xx.feature.structure_tensor)
    assert ew==aw
    if isinstance(expected,Exception):
        assert (type(actual),str(actual))==(type(expected),str(expected));return
    assert type(actual) is type(expected) is list and len(actual)==len(expected)
    for result,target in zip(actual,expected):
        assert type(result) is type(target) is np.ndarray
        assert result.shape==target.shape and result.dtype==target.dtype
        assert result.dtype.metadata==target.dtype.metadata and result.strides==target.strides
        assert result.flags.c_contiguous==target.flags.c_contiguous
        assert result.flags.f_contiguous==target.flags.f_contiguous
        assert result.flags.owndata==target.flags.owndata
        budget=2e-6 if result.dtype.type is np.float32 else 2e-13
        np.testing.assert_allclose(result,target,rtol=budget,atol=budget)
    return actual


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mode',['constant','reflect','nearest','mirror','wrap','grid-mirror','grid-wrap','grid-constant'])
@pytest.mark.parametrize('sigma',[0,.1,1.2])
def test_real_types_and_borders(dtype,mode,sigma):
    compare(image(dtype),sigma=sigma,mode=mode,cval=-.3)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(3,5,7),(2,3,4,5),(2,2,3,3,4)])
@pytest.mark.parametrize('sigma',[0,1,(.5,0,1.2)])
def test_nd(dtype,shape,sigma):
    if isinstance(sigma,tuple):sigma=tuple(sigma[i%3] for i in range(len(shape)))
    compare(image(dtype,shape),sigma=sigma,mode='reflect')


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16])
@pytest.mark.parametrize('layout',['reverse','fortran','endian','tagged','tagged-endian','unaligned','readonly','broadcast','squeezed'])
def test_layouts(dtype,layout):
    a=image(dtype)
    if layout=='reverse':a=a[::-1,::2]
    elif layout=='fortran':a=np.asfortranarray(a)
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder())
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'units':'linear'}))
    elif layout=='tagged-endian':a=a.astype(np.dtype(a.dtype.newbyteorder(),metadata={'units':'linear'}))
    elif layout=='unaligned':
        b=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);b[:]=a;a=b
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif layout=='squeezed':a=a[None,:,None,:]
    compare(a,sigma=.7)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('sigma',[0,1.3,(.5,1.7)])
def test_xy_and_mode_sequences(dtype,sigma):
    compare(image(dtype),sigma=sigma,order='xy',mode=('reflect','constant'),cval=.7)


@pytest.mark.parametrize('sigma',[True,np.bool_(True),np.float16(.7),np.float32(.7),np.float64(.7),np.uint8(2),np.int8(3),np.int64(1),[np.float32(.7),np.float32(1.2)]])
def test_typed_sigma(sigma):compare(image(np.float64),sigma=sigma)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('sigma',[0,1])
def test_zero_signs(dtype,sigma):
    a=np.zeros((7,9),dtype=dtype);a[::2,::2]=-0.
    actual=compare(a,sigma=sigma)
    for result,target in zip(actual,reference(a,sigma=sigma)):
        np.testing.assert_array_equal(np.signbit(result),np.signbit(target))


def test_integer_extremes_and_noncanonical_bool():
    for dtype in [np.uint64,np.int64]:
        compare(np.random.default_rng(89).bit_generator.random_raw(19*23).reshape(19,23).astype(dtype),sigma=.8)
    a=np.arange(256,dtype=np.uint8).reshape(16,16).view(np.bool_);compare(a,sigma=.8)


def test_memmap(tmp_path):
    a=np.memmap(tmp_path/'input.bin',mode='w+',dtype=np.float32,shape=(17,19));a[:]=image(np.float32)
    compare(a,sigma=.9)


def test_signature():assert inspect.signature(xx.feature.structure_tensor)==inspect.signature(reference)


@pytest.mark.parametrize('case',['nan','extreme','complex','longdouble','wrong-order','negative-sigma','nonfinite-cval','half-sigma-overflow','int-sigma-overflow','zero-dimensional-sigma'])
def test_explicit_compatibility(case):
    a=image(np.float32);options={}
    if case=='nan':a[0,0]=np.nan
    elif case=='extreme':a[:]=np.finfo(np.float32).max
    elif case=='complex':a=a.astype(np.complex64)
    elif case=='longdouble':a=a.astype(np.longdouble)
    elif case=='wrong-order':options['order']='bad'
    elif case=='negative-sigma':options['sigma']=-1
    elif case=='nonfinite-cval':options['cval']=np.inf
    elif case=='half-sigma-overflow':options['sigma']=np.float16(300)
    elif case=='int-sigma-overflow':options['sigma']=np.int8(12)
    elif case=='zero-dimensional-sigma':options['sigma']=np.array(.7)
    with pytest.raises(NotImplementedError):xx.feature.structure_tensor(a,**options)
    xx.set_fallback_enabled(True);compare(a,**options)


@pytest.mark.parametrize('shape',[(1,1),(0,3),(3,1),(1,1,1),(0,),(1,),()])
def test_invalid_shape_error(shape):compare(image(np.float32,shape))


def test_sigma_length_error():compare(image(np.float32),sigma=[1])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_parallel_without_original_image_algorithms(dtype,monkeypatch):
    a=image(dtype,(513,519));expected=reference(a,sigma=1.2)
    def forbidden(*args,**kwargs):raise AssertionError('original image algorithm executed')
    import scipy.ndimage as ndi
    monkeypatch.setattr(ndi,'sobel',forbidden);monkeypatch.setattr(ndi,'gaussian_filter',forbidden)
    xx.set_num_threads(1);serial=xx.feature.structure_tensor(a,sigma=1.2)
    xx.set_num_threads(4);parallel=xx.feature.structure_tensor(a,sigma=1.2)
    for s,p,e in zip(serial,parallel,expected):
        np.testing.assert_array_equal(s,p)
        np.testing.assert_allclose(s,e,rtol=2e-6 if dtype is np.float32 else 2e-13,atol=2e-6 if dtype is np.float32 else 2e-13)
