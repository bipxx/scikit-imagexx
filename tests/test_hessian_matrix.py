import inspect
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage import feature as reference
from skimagexx._compat import FallbackWarning

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
@pytest.fixture(autouse=True)
def native_only():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)

def image(dtype,shape=(13,17)):
    r=np.random.default_rng(8311)
    if np.dtype(dtype).kind=='f':return r.normal(size=shape).astype(dtype)
    raw=r.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)

def compare(a,**kwargs):
    with warnings.catch_warnings(record=True) as original_warnings:
        warnings.simplefilter('always');expected=reference.hessian_matrix(a,**kwargs)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always');actual=xx.feature.hessian_matrix(a,**kwargs)
    assert [(type(w.message),str(w.message)) for w in actual_warnings if not isinstance(w.message,FallbackWarning)]==[(type(w.message),str(w.message)) for w in original_warnings]
    assert type(actual) is type(expected) is list and len(actual)==len(expected)
    for x,y in zip(actual,expected):
        assert type(x) is type(y) and x.shape==y.shape and x.dtype==y.dtype
        assert x.strides==y.strides and x.dtype.metadata==y.dtype.metadata
        tol=2e-6 if x.dtype==np.float32 else 2e-13
        np.testing.assert_allclose(x,y,rtol=tol,atol=tol)
    return actual,expected

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('gaussian',(False,True,None))
@pytest.mark.parametrize('mode',('constant','reflect','mirror','nearest','wrap','grid-mirror','grid-constant','grid-wrap'))
def test_types_modes(dtype,gaussian,mode):
    compare(image(dtype),sigma=.8,mode=mode,cval=-.3,use_gaussian_derivatives=gaussian)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('gaussian',(False,True))
@pytest.mark.parametrize('case',('strided','fortran','broadcast','zero','anisotropic','xy','unaligned','endian','tag','typed','singleton','empty','3d','4d','5d'))
def test_layouts(dtype,gaussian,case):
    a=image(dtype);kw=dict(use_gaussian_derivatives=gaussian)
    if case=='strided':a=a[::-1,::2]
    elif case=='fortran':a=np.asfortranarray(a)
    elif case=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif case=='zero':kw['sigma']=0
    elif case=='anisotropic':kw.update(sigma=(0,.6),mode=('constant','reflect'))
    elif case=='xy':kw['order']='xy'
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype)
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='tag':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='typed':kw.update(sigma=np.float32(.8),cval=np.float32(.3))
    elif case=='singleton':
        if not gaussian:return
        a=a[:1,:]
    elif case=='empty':
        if not gaussian:return
        a=a[:0,:]
    elif case=='3d':a=image(dtype,(3,5,7))
    elif case=='4d':a=image(dtype,(2,3,4,5))
    elif case=='5d':a=image(dtype,(2,3,2,3,4))
    compare(a,**kw)

def test_signature():
    assert inspect.signature(xx.feature.hessian_matrix)==inspect.signature(reference.hessian_matrix)

@pytest.mark.parametrize('gaussian',(False,True))
@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_workers(dtype,gaussian,monkeypatch):
    a=image(dtype,(513,519))
    xx.set_num_threads(1);first=xx.feature.hessian_matrix(a,use_gaussian_derivatives=gaussian)
    def forbidden(*args,**kwargs):raise AssertionError('upstream image operation used')
    monkeypatch.setattr(reference,'hessian_matrix',forbidden)
    import scipy.ndimage as ndi
    monkeypatch.setattr(ndi,'gaussian_filter',forbidden);monkeypatch.setattr(np,'gradient',forbidden)
    xx.set_num_threads(4);other=xx.feature.hessian_matrix(a,use_gaussian_derivatives=gaussian)
    for x,y in zip(first,other):np.testing.assert_array_equal(x,y)

@pytest.mark.parametrize('gaussian',(False,True))
def test_memmap(tmp_path,gaussian):
    a=np.memmap(tmp_path/'a.bin',mode='w+',shape=(13,17),dtype='float32');a[...]=image(np.float32)
    compare(a,use_gaussian_derivatives=gaussian)


@pytest.mark.parametrize('gaussian',(False,True))
@pytest.mark.parametrize('sigma',(np.float16(.8),np.float32(.8),np.float64(.8),np.int16(2),[np.float32(.5),np.float32(.8)],np.array([.5,.8]),[0,0],[.5,0],[2,3]))
def test_sigma_protocols(gaussian,sigma):
    compare(image(np.float32),sigma=sigma,use_gaussian_derivatives=gaussian)

@pytest.mark.parametrize('kwargs',({'order':'bad'},{'order':'xy'},{'sigma':-1},{'sigma':[1]},{'mode':'invalid'},{'cval':float('inf')},{'use_gaussian_derivatives':1}))
def test_delegated_contract(kwargs):
    xx.set_fallback_enabled(True)
    a=image(np.float32,(3,4,5)) if kwargs.get('order')=='xy' else image(np.float32)
    def capture(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=function(a,**kwargs);error=None
            except Exception as e:result=None;error=(type(e),str(e))
        if function is xx.feature.hessian_matrix:assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return result,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    expected,error,warnings_expected=capture(reference.hessian_matrix)
    actual,error_actual,warnings_actual=capture(xx.feature.hessian_matrix)
    assert error_actual==error and warnings_actual==warnings_expected
    if error is None:
        for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)

@pytest.mark.parametrize('gaussian',(False,True))
@pytest.mark.parametrize('value',(np.nan,np.inf,1e307))
def test_unsafe_fallback(gaussian,value):
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        compare(np.full((5,7),value),use_gaussian_derivatives=gaussian)

@pytest.mark.parametrize('gaussian',(False,True))
def test_float32_rounding(gaussian):
    _,expected=compare(image(np.float32),sigma=.8,use_gaussian_derivatives=gaussian)
    assert all(x.dtype==np.float32 for x in expected)

@pytest.mark.parametrize('sigma',(0,.8,2))
def test_true_1d(sigma):
    compare(image(np.float64,(19,)),sigma=sigma,use_gaussian_derivatives=True)


@pytest.mark.parametrize('gaussian',(False,True))
@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('value',(0.,-0.,1.,-1.))
def test_signed_zeros(dtype,gaussian,value):
    actual,expected=compare(np.full((7,9),value,dtype=dtype),sigma=0 if not gaussian else 1,use_gaussian_derivatives=gaussian,mode='reflect')
    for x,y in zip(actual,expected):
        zero=(x==0)&(y==0)
        np.testing.assert_array_equal(np.signbit(x[zero]),np.signbit(y[zero]))


@pytest.mark.parametrize('shape',((1,1),(0,7),(7,1),(2,1,3)))
@pytest.mark.parametrize('gaussian',(False,None))
def test_finite_small_axis_errors(shape,gaussian):
    a=np.ones(shape,dtype='float32')
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(ValueError) as error:f(a,use_gaussian_derivatives=gaussian)
        return str(error.value),[(type(w.message),str(w.message)) for w in caught]
    assert capture(xx.feature.hessian_matrix)==capture(reference.hessian_matrix)


@pytest.mark.parametrize('gaussian',(None,False,True))
def test_huge_sigma_warning_order(gaussian):
    xx.set_fallback_enabled(True)
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(OverflowError) as error:f(np.ones((3,4)),sigma=2**2000,use_gaussian_derivatives=gaussian)
        return str(error.value),[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    assert capture(xx.feature.hessian_matrix)==capture(reference.hessian_matrix)
