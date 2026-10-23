import inspect
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.feature import hessian_matrix_det as reference
from skimagexx._compat import FallbackWarning

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True)
def native():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)

def image(dtype,shape=(17,19)):
    rng=np.random.default_rng(9231)
    if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
    raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)

def compare(a,**options):
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');value=f(a,**options)
        return value,[(type(w.message),str(w.message)) for w in caught]
    actual,aw=capture(xx.feature.hessian_matrix_det);expected,ew=capture(reference)
    assert aw==ew
    assert type(actual) is type(expected) is np.ndarray
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
    tol=3e-6 if actual.dtype==np.float32 else 3e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
    zero=(actual==0)&(expected==0)
    np.testing.assert_array_equal(np.signbit(actual[zero]),np.signbit(expected[zero]))
    return actual,expected

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('sigma',(-3,-1,-.34,0,.1,.34,.8,1,1.5,3,20,np.float64(1.3)))
def test_approx_types_sigma(dtype,sigma):compare(image(dtype),sigma=sigma)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3,4,5))
@pytest.mark.parametrize('sigma',(0,.8,(0,.7)))
def test_exact(dtype,rank,sigma):
    if type(sigma) is tuple:sigma=(0,)+(.7,)*(rank-1)
    compare(image(dtype,(4,)*rank),sigma=sigma,approximate=False)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('approximate',(False,True))
@pytest.mark.parametrize('case',('strided','fortran','broadcast','endian','unaligned','metadata','zero','negative_zero','constant','impulse','readonly'))
def test_layouts(dtype,approximate,case):
    a=image(dtype)
    if case=='strided':a=a[::-1,::2]
    elif case=='fortran':a=np.asfortranarray(a)
    elif case=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype)
    elif case=='metadata':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='zero':a=np.zeros_like(a)
    elif case=='negative_zero':a=np.full_like(a,-0.)
    elif case=='constant':a=np.ones_like(a)
    elif case=='impulse':a=np.zeros_like(a);a[8,9]=1
    elif case=='readonly':a.flags.writeable=False
    compare(a,approximate=approximate)

@pytest.mark.parametrize('shape',((0,7),(7,0),(0,0),(1,1),(1,9),(9,1)))
def test_approx_small(shape):compare(np.ones(shape,dtype='f4'))

@pytest.mark.parametrize('shape',((0,7),(7,1),(1,9),(1,1)))
def test_exact_small_error(shape):
    def capture(f):
        with pytest.raises(ValueError) as error:f(np.ones(shape),approximate=False)
        return str(error.value)
    assert capture(xx.feature.hessian_matrix_det)==capture(reference)

@pytest.mark.parametrize('approximate',(False,True))
def test_memmap(tmp_path,approximate):
    a=np.memmap(tmp_path/'a.bin',mode='w+',shape=(17,19),dtype='float32');a[...]=image(np.float32)
    compare(a,approximate=approximate)

def test_signature():assert inspect.signature(xx.feature.hessian_matrix_det)==inspect.signature(reference)

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('approximate',(False,True))
@pytest.mark.parametrize('scale',(1e-30,1e-10,1e10))
def test_amplitude(dtype,approximate,scale):compare(image(dtype)*dtype(scale),sigma=.7,approximate=approximate)

@pytest.mark.parametrize('sigma',(-.1,-.3333333333333333,.3333333333333333,.6666666666666666,1.3333333333333333))
def test_size_rounding(sigma):compare(image(np.float64),sigma=sigma)

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('shape',[(h,w) for h in (1,3,7) for w in (1,2,3,4,5,6,7,8,9,12,15)])
@pytest.mark.parametrize('sigma',(-3,-1,.34,1,1.5,3,20))
def test_short_approx_rows(dtype,shape,sigma):compare(image(dtype,shape),sigma=sigma)

@pytest.mark.parametrize('position',((0,0),(1,2),(2,4)))
@pytest.mark.parametrize('value',(-1e140,1e140,np.nextafter(1e140,0)))
def test_approx_validation_limit(position,value):
    a=np.zeros((3,5));a[position]=value;compare(a)

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_volume_workers(dtype):
    a=image(dtype,(65,67,69));first,_=compare(a,approximate=False,sigma=(0,.7,1.3))
    xx.set_num_threads(4);actual=xx.feature.hessian_matrix_det(a,approximate=False,sigma=(0,.7,1.3))
    np.testing.assert_array_equal(actual,first)

@pytest.mark.parametrize('rank',(3,4,5))
def test_approximate_unused(rank):
    class Unused:
        def __bool__(self):raise AssertionError('unused argument inspected')
    compare(image(np.float32,(3,)*rank),approximate=Unused())

@pytest.mark.parametrize('approximate',(False,True))
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
def test_workers_no_upstream(dtype,approximate,monkeypatch):
    a=image(dtype,(513,519));first,_=compare(a,approximate=approximate)
    def forbidden(*args,**kwargs):raise AssertionError('upstream image operation used')
    import skimage.feature.corner as corner
    import scipy.ndimage as ndi
    monkeypatch.setattr(corner,'hessian_matrix',forbidden);monkeypatch.setattr(corner,'integral_image',forbidden)
    monkeypatch.setattr(ndi,'gaussian_filter',forbidden);monkeypatch.setattr(np,'gradient',forbidden);monkeypatch.setattr(np.linalg,'det',forbidden)
    xx.set_num_threads(4);actual=xx.feature.hessian_matrix_det(a,approximate=approximate)
    np.testing.assert_array_equal(actual,first)
    actual.flat[0]=7
    assert not np.shares_memory(actual,a)

@pytest.mark.parametrize('case',('nan','inf','extreme','complex','bad_sigma','typed_sigma','bool_sigma','bad_rank','underflow'))
@pytest.mark.parametrize('approximate',(False,True))
def test_fallback_once(case,approximate):
    xx.set_fallback_enabled(True);a=image(np.float64);options={'approximate':approximate}
    if case=='nan':a[3,4]=np.nan
    elif case=='inf':a[3,4]=np.inf
    elif case=='extreme':a[3,4]=1e307
    elif case=='complex':a=a.astype(complex)
    elif case=='bad_sigma':options['sigma']='invalid'
    elif case=='typed_sigma':
        if not approximate:pytest.skip('typed scalar supported by exact path')
        options['sigma']=np.float32(1)
    elif case=='bool_sigma':
        if not approximate:pytest.skip('boolean scalar supported by exact path')
        options['sigma']=True
    elif case=='bad_rank':a=a.reshape(-1)
    def capture(f):
        with np.errstate(under='raise' if case=='underflow' else 'ignore'),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=f(a,**options);error=None
            except Exception as e:result=None;error=(type(e),str(e))
        if f is xx.feature.hessian_matrix_det:assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return result,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,error,aw=capture(xx.feature.hessian_matrix_det);expected,expected_error,ew=capture(reference)
    assert error==expected_error and aw==ew
    if error is None:np.testing.assert_allclose(actual,expected,equal_nan=True)
