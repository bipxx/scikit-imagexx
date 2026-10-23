"""Fused responses share native tensors without delegating image arithmetic."""
import inspect,warnings
import numpy as np
import pytest
from skimage import feature as reference
import skimagexx as xx

APIS=('corner_harris','corner_shi_tomasi','corner_foerstner')


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def image(dtype,shape=(19,23)):
    rng=np.random.default_rng(4321)
    if np.dtype(dtype).kind in 'bu':return rng.integers(0,2 if dtype is np.bool_ else 255,size=shape).astype(dtype)
    if np.dtype(dtype).kind=='i':return rng.integers(-100,100,size=shape).astype(dtype)
    return rng.normal(size=shape).astype(dtype)


def compare(api,a,**options):
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(a,**options)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    expected,ew=outcome(getattr(reference,api));actual,aw=outcome(getattr(xx.feature,api));assert aw==ew
    if isinstance(expected,Exception):
        assert (type(actual),str(actual))==(type(expected),str(expected));return
    assert type(actual) is type(expected)
    outputs=zip(actual,expected) if isinstance(actual,tuple) else [(actual,expected)]
    for result,target in outputs:
        assert type(result) is type(target)
        assert result.dtype==target.dtype and result.dtype.metadata==target.dtype.metadata
        assert result.shape==target.shape and result.strides==target.strides
        assert result.flags.c_contiguous==target.flags.c_contiguous and result.flags.f_contiguous==target.flags.f_contiguous
        assert result.flags.owndata==target.flags.owndata
        tolerance=2e-6 if target.dtype.type is np.float32 else 2e-13
        np.testing.assert_allclose(result,target,rtol=tolerance,atol=tolerance)
    return actual


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('sigma',[0,.7,1.5,(.7,1.3)])
def test_types_and_sigma(api,dtype,sigma):compare(api,image(dtype),sigma=sigma)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('pattern',['zero','constant','impulse','signed-zero'])
def test_patterns(api,dtype,pattern):
    a=np.zeros((17,19),dtype=dtype)
    if pattern=='constant':a[:]=.7
    elif pattern=='impulse':a[8,9]=1
    elif pattern=='signed-zero':a[::2,::2]=-0.
    actual=compare(api,a,sigma=0)
    expected=getattr(reference,api)(a,sigma=0)
    for r,e in zip(actual,expected) if isinstance(actual,tuple) else [(actual,expected)]:np.testing.assert_array_equal(np.signbit(r),np.signbit(e))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16])
@pytest.mark.parametrize('layout',['reverse','fortran','transposed-strided','endian','tagged','tagged-endian','unaligned','readonly','broadcast'])
def test_layouts(api,dtype,layout):
    a=image(dtype)
    if layout=='reverse':a=a[::-1,::2]
    elif layout=='fortran':a=np.asfortranarray(a)
    elif layout=='transposed-strided':a=a[::-2,::2].T
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder())
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'units':'linear'}))
    elif layout=='tagged-endian':a=a.astype(np.dtype(a.dtype.newbyteorder(),metadata={'units':'linear'}))
    elif layout=='unaligned':
        b=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1);b[:]=a;a=b
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    compare(api,a)


@pytest.mark.parametrize('api',APIS)
def test_memmap(api,tmp_path):
    a=np.memmap(tmp_path/'image.bin',mode='w+',dtype=np.float32,shape=(19,23));a[:]=image(np.float32)
    compare(api,a)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('method',['k','eps','noble','anything'])
@pytest.mark.parametrize('parameter',[.05,0,-.2,np.float16(.1),np.float32(.1),np.float64(.1),np.int64(1)])
def test_harris_scalar_promotion(dtype,method,parameter):
    compare('corner_harris',image(dtype),method=method,**({'k':parameter} if method=='k' else {'eps':parameter}))


def test_unused_parameter_protocols():
    class Unused:
        def __float__(self):raise AssertionError('unused parameter inspected')
    compare('corner_harris',image(np.float32),method='k',eps=Unused())
    compare('corner_harris',image(np.float32),method='eps',k=Unused())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_noble_zero_denominator(dtype):
    xx.set_fallback_enabled(True)
    compare('corner_harris',np.zeros((7,9),dtype=dtype),method='eps',eps=0)


@pytest.mark.parametrize('api',APIS)
def test_error_policy(api):
    a=image(np.float32)*1e-15
    xx.set_fallback_enabled(True)
    with np.errstate(under='raise'):compare(api,a,sigma=0)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('shape',[(1,1),(0,3),(3,1),(3,4,5)])
def test_shape_errors(api,shape):compare(api,image(np.float32,shape))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('case',['nan','extreme','complex','sigma-overflow','tiny'])
def test_compatibility_warnings_once(api,case):
    a=image(np.float32);options={}
    if case=='nan':a[0,0]=np.nan
    elif case=='extreme':a[:]=np.finfo(np.float32).max
    elif case=='complex':a=a.astype(np.complex64)
    elif case=='sigma-overflow':options['sigma']=np.float16(300)
    elif case=='tiny':a*=1e-14;options['sigma']=0
    xx.set_fallback_enabled(True);compare(api,a,**options)


@pytest.mark.parametrize('api',APIS)
def test_signature(api):assert inspect.signature(getattr(xx.feature,api))==inspect.signature(getattr(reference,api))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_parallel_without_original(api,dtype,monkeypatch):
    a=image(dtype,(513,519));expected=getattr(reference,api)(a)
    def forbidden(*args,**kwargs):raise AssertionError('original image algorithm executed')
    import scipy.ndimage as ndi
    monkeypatch.setattr(ndi,'sobel',forbidden);monkeypatch.setattr(ndi,'gaussian_filter',forbidden)
    monkeypatch.setattr(reference,api,forbidden)
    xx.set_num_threads(1);serial=getattr(xx.feature,api)(a)
    xx.set_num_threads(4);parallel=getattr(xx.feature,api)(a)
    for s,p,e in zip(serial,parallel,expected) if isinstance(serial,tuple) else [(serial,parallel,expected)]:
        np.testing.assert_array_equal(s,p)
        np.testing.assert_allclose(s,e,rtol=2e-6 if dtype is np.float32 else 2e-13,atol=2e-6 if dtype is np.float32 else 2e-13)
