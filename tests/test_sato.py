import inspect
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import sato as reference
from skimagexx._compat import FallbackWarning
from tools.sato_parity import assert_parity

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

@pytest.fixture(autouse=True)
def native():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)

def image(dtype,rank=2,shape=None):
    rng=np.random.default_rng(7933);shape=shape or ((17,19) if rank==2 else (7,9,11))
    if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
    raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)

def compare(a,**options):
    options.setdefault('sigmas',[1])
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');value=f(a,**options)
        return value,[(type(w.message),str(w.message)) for w in caught]
    actual,aw=capture(xx.filters.sato);expected,ew=capture(reference)
    assert aw==ew
    assert type(actual) is type(expected) and actual.dtype==expected.dtype
    assert actual.shape==expected.shape and actual.strides==expected.strides and actual.dtype.metadata==expected.dtype.metadata
    assert actual.flags.writeable and not np.shares_memory(actual,a)
    assert_parity(actual,expected,a,options)
    return actual,expected

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('mode',('constant','reflect','mirror','nearest','wrap','grid-constant','grid-mirror','grid-wrap'))
@pytest.mark.parametrize('black',(True,False))
def test_types_modes(dtype,rank,mode,black):compare(image(dtype,rank),mode=mode,cval=-.3,black_ridges=black)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('case',('strided','fortran','broadcast','endian','unaligned','metadata','readonly','zero','negative_zero','constant','impulse'))
def test_layouts(dtype,rank,case):
    a=image(dtype,rank)
    if case=='strided':a=a[::-1,::2]
    elif case=='fortran':a=np.asfortranarray(a)
    elif case=='broadcast':a=np.broadcast_to(a[(slice(0,1),)*rank],a.shape)
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype,rank)
    elif case=='metadata':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='readonly':a.flags.writeable=False
    elif case=='zero':a=np.zeros_like(a)
    elif case=='negative_zero':a=np.full_like(a,-0.)
    elif case=='constant':a=np.ones_like(a)
    elif case=='impulse':a=np.zeros_like(a);a[(3,)*rank]=1
    compare(a)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('scales',([0],[0,.7,1.3],[np.float32(.8),np.float64(1.3)],[np.int64(1),np.float32(.7)],np.array([.8,1.3],dtype='f4'),[np.float16(.8),np.int16(1)],[np.bool_(False),np.bool_(True)]))
def test_scale_promotion(dtype,rank,scales):compare(image(dtype,rank),sigmas=scales)

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('rank',(2,3))
def test_default_scales(dtype,rank):compare(image(dtype,rank),sigmas=range(1,10,2))

@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('kind',('quadratic','tilted-ridge','axis-ridge'))
def test_degenerate_ridges(rank,kind):
    shape=(19,)*rank;coords=np.indices(shape,dtype=float)
    if kind=='quadratic':a=np.sum(coords,axis=0)**2
    elif kind=='tilted-ridge':a=np.exp(-((np.sum(coords,axis=0)-9*rank)/3)**2)
    else:a=np.exp(-((coords[0]-9)/3)**2)
    compare(a,sigmas=[1],mode='reflect')

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('black',(True,False))
def test_empty_scales(dtype,black):
    a=np.asfortranarray(image(dtype));compare(a,sigmas=[],black_ridges=black,mode=object(),cval=object())

@pytest.mark.parametrize('black',(True,False))
@pytest.mark.parametrize('empty',(True,False))
def test_memmap(tmp_path,black,empty):
    a=np.memmap(tmp_path/'input.bin',mode='w+',shape=(17,19),dtype='float32');a[...]=image(np.float32)
    compare(a,sigmas=[] if empty else [1],black_ridges=black)

@pytest.mark.parametrize('shape',((0,7),(1,7),(7,1),(1,1),(1,3,5),(3,1,5)))
def test_small_axes(shape):
    if np.prod(shape):compare(np.ones(shape,dtype='float32'))
    else:
        with pytest.raises(ValueError,match='cannot be an empty array'):xx.filters.sato(np.ones(shape))

def test_signature():assert inspect.signature(xx.filters.sato)==inspect.signature(reference)

def test_nested_native_alias():
    from skimagexx.filters.ridges import sato
    assert sato is xx.filters.sato

@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
def test_workers_no_upstream(dtype,rank,monkeypatch):
    a=image(dtype,rank,(513,519) if rank==2 else (65,67,69));first,_=compare(a,sigmas=[.8])
    def forbidden(*args,**kw):raise AssertionError('upstream image operation used')
    import skimage.filters.ridges as ridges
    import scipy.ndimage as ndi
    monkeypatch.setattr(ridges,'hessian_matrix',forbidden);monkeypatch.setattr(ridges,'hessian_matrix_eigvals',forbidden)
    monkeypatch.setattr(ndi,'gaussian_filter',forbidden);monkeypatch.setattr(np.linalg,'eigvalsh',forbidden)
    monkeypatch.setattr(np,'maximum',forbidden);monkeypatch.setattr(np,'prod',forbidden);monkeypatch.setattr(np,'sqrt',forbidden)
    xx.set_num_threads(4);actual=xx.filters.sato(a,sigmas=[.8]);np.testing.assert_array_equal(actual,first)

@pytest.mark.parametrize('case',('nan','inf','extreme','complex','bad_sigma','negative_sigma','overflow_sigma','bad_mode','black_protocol','generator','underflow'))
def test_fallback_once(case):
    xx.set_fallback_enabled(True)
    def call(f):
        a=image(np.float64);kw={'sigmas':[1]}
        if case=='nan':a[3,4]=np.nan
        elif case=='inf':a[3,4]=np.inf
        elif case=='extreme':a[3,4]=1e307
        elif case=='complex':a=a.astype(complex)
        elif case=='bad_sigma':kw['sigmas']=['invalid']
        elif case=='negative_sigma':kw['sigmas']=[-1]
        elif case=='overflow_sigma':kw['sigmas']=[np.int8(20)]
        elif case=='bad_mode':kw['mode']='invalid'
        elif case=='black_protocol':kw['black_ridges']=np.array([True])
        elif case=='generator':kw['sigmas']=(s for s in [1,.7])
        with np.errstate(under='raise' if case=='underflow' else 'ignore'),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=f(a,**kw);error=None
            except Exception as e:value=None;error=(type(e),str(e))
        if f is xx.filters.sato:assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return value,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,error,aw=call(xx.filters.sato);expected,expected_error,ew=call(reference)
    assert error==expected_error and aw==ew
    if error is None:np.testing.assert_allclose(actual,expected,equal_nan=True)

def test_conditioning_oracle_rejects_response_errors():
    a=image(np.float64,3);expected=reference(a,sigmas=[1])
    with pytest.raises(AssertionError):assert_parity(expected*1.001,expected,a,{'sigmas':[1]})
    a=np.ones((7,9,11));expected=reference(a,sigmas=[1])
    with pytest.raises(AssertionError):assert_parity(expected+1e-9,expected,a,{'sigmas':[1]})

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_analytic_rank_one_matrix(dtype):
    # All-one 3x3 matrices have exact eigenvalues 3,0,0 and tubeness zero.
    # This analytic check is independent of LAPACK's small residual eigenvalues.
    fields=[np.ones((2,3,4),dtype=dtype) for _ in range(6)]
    actual=xx._core.sato_accumulate(fields,None,np.dtype(dtype),1.,dtype is np.float32)
    np.testing.assert_array_equal(actual,np.zeros((2,3,4),dtype=dtype))
