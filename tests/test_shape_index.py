import inspect
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.feature import shape_index as reference
from skimagexx._compat import FallbackWarning
TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
@pytest.fixture(autouse=True)
def native():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)
def image(dtype,shape=(17,19)):
    r=np.random.default_rng(8311)
    if np.dtype(dtype).kind=='f':return r.normal(size=shape).astype(dtype)
    raw=r.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)
def compare(a,**kw):
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');value=f(a,**kw)
        return value,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,aw=capture(xx.feature.shape_index);expected,ew=capture(reference)
    assert aw==ew
    assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
    assert actual.strides==expected.strides and actual.dtype.metadata==expected.dtype.metadata
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
    tol=2e-6 if actual.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
    zero=(actual==0)&(expected==0);np.testing.assert_array_equal(np.signbit(actual[zero]),np.signbit(expected[zero]))
    return actual,expected
@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('mode',('constant','reflect','mirror','nearest','wrap','grid-constant','grid-mirror','grid-wrap'))
@pytest.mark.parametrize('cval',(0,-.3,.7))
def test_types_modes(dtype,mode,cval):compare(image(dtype),sigma=.8,mode=mode,cval=cval)
@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('case',('strided','fortran','broadcast','zero_sigma','anisotropic','endian','unaligned','metadata','zero','negative_zero','constant','impulse','typed'))
def test_layouts(dtype,case):
    a=image(dtype);kw={}
    if case=='strided':a=a[::-1,::2]
    elif case=='fortran':a=np.asfortranarray(a)
    elif case=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif case=='zero_sigma':kw['sigma']=0
    elif case=='anisotropic':kw.update(sigma=(0,.8),mode=('constant','reflect'))
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype)
    elif case=='metadata':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='zero':a=np.zeros_like(a)
    elif case=='negative_zero':a=np.full_like(a,-0.)
    elif case=='constant':a=np.ones_like(a);kw['mode']='reflect'
    elif case=='impulse':a=np.zeros_like(a);a[8,9]=1;kw['sigma']=.1
    elif case=='typed':kw.update(sigma=np.float32(.8),cval=np.float32(.3))
    compare(a,**kw)
def test_signature():assert inspect.signature(xx.feature.shape_index)==inspect.signature(reference)
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
def test_workers(dtype,monkeypatch):
    a=image(dtype,(513,519));first,_=compare(a)
    def forbidden(*args,**kwargs):raise AssertionError('upstream image operation used')
    import skimage.feature.corner as corner
    import scipy.ndimage as ndi
    monkeypatch.setattr(corner,'hessian_matrix',forbidden);monkeypatch.setattr(ndi,'gaussian_filter',forbidden);monkeypatch.setattr(np,'gradient',forbidden)
    xx.set_num_threads(4);actual=xx.feature.shape_index(a);np.testing.assert_array_equal(actual,first)
def test_memmap(tmp_path):
    a=np.memmap(tmp_path/'a.bin',mode='w+',shape=(17,19),dtype='float32');a[...]=image(np.float32);compare(a)
@pytest.mark.parametrize('shape',((1,1),(0,7),(7,1)))
def test_small_axis_errors(shape):
    def capture(f):
        with pytest.raises(ValueError) as error:f(np.ones(shape))
        return str(error.value)
    assert capture(xx.feature.shape_index)==capture(reference)
@pytest.mark.parametrize('case',('nan','inf','extreme','complex','bad_sigma','bad_mode','3d','underflow'))
def test_fallback_once(case):
    xx.set_fallback_enabled(True);a=image(np.float64);kw={}
    if case=='nan':a[3,4]=np.nan
    elif case=='inf':a[3,4]=np.inf
    elif case=='extreme':a[3,4]=1e307
    elif case=='complex':a=a.astype(complex)
    elif case=='bad_sigma':kw['sigma']=-1
    elif case=='bad_mode':kw['mode']='invalid'
    elif case=='3d':a=a[None,...]
    def capture(f):
        with np.errstate(under='raise' if case=='underflow' else 'ignore'),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=f(a,**kw);error=None
            except Exception as e:result=None;error=(type(e),str(e))
        if f is xx.feature.shape_index:assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return result,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,error,aw=capture(xx.feature.shape_index);expected,expected_error,ew=capture(reference)
    assert error==expected_error and aw==ew
    if error is None:np.testing.assert_allclose(actual,expected,equal_nan=True)


@pytest.mark.parametrize('shape',((7,9),(1,1)))
def test_hessian_default_warning_origin(shape):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:xx.feature.hessian_matrix(np.ones(shape))
        except ValueError:assert shape==(1,1)
    assert len(caught)==1 and caught[0].category is FutureWarning
    assert caught[0].filename==__file__
