import inspect,warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage import filters as reference

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.mark.parametrize('length',(1,3,4,7,8,9,693))
def test_double_exp_host_rounding(length):
    # Hosted AVX-512 NumPy and libm disagree on exp(x)==1 here. That single
    # bit controls ridge-Hessian's discontinuous zero-to-one finish.
    values=np.resize(np.array([float.fromhex('-0x1.ae57019c31868p-55'),
        float.fromhex('-0x1.cd8883b52286fp-55'),
        float.fromhex('-0x1.b47553d1d3d71p-55'),-0.,-1e-16,-1.,-700.]),length)
    expected=np.exp(values)
    actual=xx._core.frangi_exp_values_double(values)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
@pytest.fixture(autouse=True)
def native():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)

def image(dtype,rank=2):
    shape=(17,19) if rank==2 else (7,9,11);rng=np.random.default_rng(4691)
    if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
    raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)

def compare(name,a,**kw):
    kw.setdefault('sigmas',[1])
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');out=f(a,**kw)
        return out,[(type(w.message),str(w.message)) for w in caught]
    actual,aw=capture(getattr(xx.filters,name));expected,ew=capture(getattr(reference,name))
    assert aw==ew
    assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    tol=2e-6 if actual.dtype==np.float32 else 2e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,equal_nan=True)
    zeros=(actual==0)&(expected==0);np.testing.assert_array_equal(np.signbit(actual[zeros]),np.signbit(expected[zeros]))
    return actual,expected

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('mode',('constant','reflect','mirror','nearest','wrap'))
@pytest.mark.parametrize('black',(True,False))
def test_types_modes(name,dtype,rank,mode,black):compare(name,image(dtype,rank),mode=mode,cval=-.3,black_ridges=black)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('options',({'gamma':None},{'gamma':.7},{'gamma':np.float32(.7)},{'gamma':np.float64(.7)},{'alpha':np.float64(.3),'beta':np.float32(.8)},{'alpha':-.5,'beta':-.7,'gamma':-1.3},{'sigmas':[.7,1.3]},{'sigmas':[1.3,.7]},{'sigmas':[0]},{'sigmas':range(1,10,2)}))
def test_parameters(name,dtype,rank,options):compare(name,image(dtype,rank),**options)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('case',('fortran','strided','metadata','endian','unaligned','broadcast','zero','constant','impulse','readonly'))
def test_layouts(name,dtype,case):
    a=image(dtype)
    if case=='fortran':a=np.asfortranarray(a)
    elif case=='strided':a=a[::-1,::2]
    elif case=='metadata':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype)
    elif case=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif case=='zero':a=np.zeros_like(a)
    elif case=='constant':a=np.ones_like(a)
    elif case=='impulse':a=np.zeros_like(a);a[8,9]=1
    elif case=='readonly':a.flags.writeable=False
    compare(name,a)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_legacy(name,dtype):compare(name,image(dtype),sigmas=[999],scale_range=(1,3),scale_step=1)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('black',(True,False))
@pytest.mark.parametrize('empty',(True,False))
def test_memmap(name,black,empty,tmp_path):
    a=np.memmap(tmp_path/'a.bin',mode='w+',dtype='float32',shape=(17,19));a[...]=image(np.float32)
    compare(name,a,black_ridges=black,sigmas=[] if empty else [1])

@pytest.mark.parametrize('name',('frangi','hessian'))
def test_signature(name):assert inspect.signature(getattr(xx.filters,name))==inspect.signature(getattr(reference,name))

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('kind',('quadratic','axis-ridge','tilted-ridge','negative-zero'))
def test_degenerate(name,rank,kind):
    coords=np.indices((9,)*rank,dtype=float)
    if kind=='quadratic':a=np.sum(coords,axis=0)**2
    elif kind=='axis-ridge':a=np.exp(-((coords[0]-4)/2)**2)
    elif kind=='tilted-ridge':a=np.exp(-((np.sum(coords,axis=0)-4*rank)/2)**2)
    else:a=np.full((9,)*rank,-0.)
    compare(name,a)

def test_exp_near_one_decisions():
    a=np.linspace(-2e-6,0,10001,dtype=np.float32)
    actual=xx._core.frangi_exp_values(a);expected=np.exp(a)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual==1,expected==1)

def test_exp_negative_range():
    a=np.linspace(-103,0,10007,dtype=np.float32)
    actual=xx._core.frangi_exp_values(a);expected=np.exp(a)
    np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=1e-44)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
def test_workers_native_math(name,rank,dtype,monkeypatch):
    rng=np.random.default_rng(4691);shape=(513,519) if rank==2 else (65,67,69)
    a=rng.normal(size=shape).astype(dtype) if dtype!=np.uint64 else rng.integers(0,np.iinfo(dtype).max,shape,dtype=dtype)
    first,_=compare(name,a,sigmas=[1])
    def forbidden(*args,**kw):raise AssertionError('Python image operation used')
    import skimage.filters.ridges as ridges
    import scipy.ndimage as ndi
    import numpy.linalg._umath_linalg as native_linalg
    monkeypatch.setattr(ridges,'hessian_matrix',forbidden);monkeypatch.setattr(ridges,'hessian_matrix_eigvals',forbidden)
    monkeypatch.setattr(np.linalg,'eigvalsh',forbidden);monkeypatch.setattr(native_linalg,'eigvalsh_lo',forbidden)
    monkeypatch.setattr(ndi,'gaussian_filter',forbidden)
    xx.set_num_threads(4);actual=getattr(xx.filters,name)(a,sigmas=[1]);np.testing.assert_array_equal(actual,first)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('shape',((1,1),(1,7),(7,1),(1,3,5),(3,1,5)))
def test_singleton_axes(name,shape):compare(name,np.ones(shape,dtype='float32'))

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('case',('nan','inf','extreme','complex','negative_sigma','zero_alpha','zero_beta','zero_gamma','bad_mode','generator','underflow'))
def test_fallback_once(name,case):
    from skimagexx._compat import FallbackWarning
    xx.set_fallback_enabled(True)
    def call(f):
        a=image(np.float64);kw={'sigmas':[1]}
        if case=='nan':a[3,4]=np.nan
        elif case=='inf':a[3,4]=np.inf
        elif case=='extreme':a[3,4]=1e307
        elif case=='complex':a=a.astype(complex)
        elif case=='negative_sigma':kw['sigmas']=[-1]
        elif case=='zero_alpha':kw['alpha']=0
        elif case=='zero_beta':kw['beta']=0
        elif case=='zero_gamma':kw['gamma']=0
        elif case=='bad_mode':kw['mode']='invalid'
        elif case=='generator':kw['sigmas']=(s for s in [1,.7])
        with np.errstate(under='raise' if case=='underflow' else 'ignore'),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:out=f(a,**kw);error=None
            except Exception as e:out=None;error=(type(e),str(e))
        if f is getattr(xx.filters,name):assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return out,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,error,aw=call(getattr(xx.filters,name));expected,expected_error,ew=call(getattr(reference,name))
    assert error==expected_error and aw==ew
    if error is None:np.testing.assert_allclose(actual,expected,equal_nan=True)

@pytest.mark.parametrize('name',('frangi','hessian'))
@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('black',(True,False))
def test_empty_ignored_parameters(name,dtype,black):
    a=np.asfortranarray(image(dtype));compare(name,a,sigmas=[],alpha=object(),beta=0,gamma=object(),mode=object(),cval=object(),black_ridges=black)

@pytest.mark.parametrize('name',('frangi','hessian'))
def test_empty_permuted_storage(name):
    a=image(np.float32,3).transpose(2,0,1)
    compare(name,a,sigmas=[],alpha=0,beta=0,gamma=0)
