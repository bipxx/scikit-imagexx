import inspect,warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import meijering as reference

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
@pytest.fixture(autouse=True)
def native():
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(True)

def image(dtype,rank=2):
    shape={1:(19,),2:(13,17),3:(5,7,9),4:(3,4,5,6),5:(2,3,4,5,6)}[rank]
    rng=np.random.default_rng(8621)
    if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
    raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)

def compare(a,**kw):
    kw.setdefault('sigmas',[1])
    def capture(f):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');value=f(a,**kw)
        return value,[(type(w.message),str(w.message)) for w in caught]
    actual,aw=capture(xx.filters.meijering);expected,ew=capture(reference)
    assert aw==ew
    assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    tol=3e-6 if actual.dtype==np.float32 else 3e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol)
    zeros=(actual==0)&(expected==0);np.testing.assert_array_equal(np.signbit(actual[zeros]),np.signbit(expected[zeros]))
    return actual,expected

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(1,2,3,4,5))
@pytest.mark.parametrize('alpha',(None,0.,-.5,.3,1.))
def test_types_ranks_alpha(dtype,rank,alpha):compare(image(dtype,rank),alpha=alpha)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('rank',(2,3))
@pytest.mark.parametrize('mode',('constant','reflect','mirror','nearest','wrap'))
@pytest.mark.parametrize('black',(True,False))
def test_modes(dtype,rank,mode,black):compare(image(dtype,rank),mode=mode,cval=-.3,black_ridges=black)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('case',('fortran','strided','metadata','endian','unaligned','broadcast','zero','constant','impulse','zero_sigma','multi_scale','typed'))
def test_layouts(dtype,case):
    a=image(dtype);kw={}
    if case=='fortran':a=np.asfortranarray(a)
    elif case=='strided':a=a[::-1,::2]
    elif case=='metadata':a=a.view(np.dtype(a.dtype,metadata={'tag':3}))
    elif case=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    elif case=='unaligned':a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=image(dtype)
    elif case=='broadcast':a=np.broadcast_to(a[:1,:1],a.shape)
    elif case=='zero':a=np.zeros_like(a)
    elif case=='constant':a=np.ones_like(a)
    elif case=='impulse':a=np.zeros_like(a);a[6,8]=1
    elif case=='zero_sigma':kw['sigmas']=[0]
    elif case=='multi_scale':kw['sigmas']=[0,.7,1.3]
    elif case=='typed':kw.update(sigmas=[np.float32(.8),np.float64(1.3)],alpha=np.float32(.3))
    compare(a,**kw)

@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('black',(True,False))
def test_empty_scales(dtype,black):compare(np.asfortranarray(image(dtype)),sigmas=[],black_ridges=black,mode=object(),cval=object())

@pytest.mark.parametrize('black',(True,False))
@pytest.mark.parametrize('empty',(True,False))
def test_memmap(tmp_path,black,empty):
    a=np.memmap(tmp_path/'a.bin',mode='w+',dtype='float32',shape=(13,17));a[...]=image(np.float32)
    compare(a,sigmas=[] if empty else [1],black_ridges=black)

def test_signature():assert inspect.signature(xx.filters.meijering)==inspect.signature(reference)

@pytest.mark.parametrize('rank',(1,2,3,4,5))
@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.uint64))
def test_default_scales(dtype,rank):compare(image(dtype,rank),sigmas=range(1,10,2))

@pytest.mark.parametrize('mode',('grid-constant','grid-mirror','grid-wrap'))
@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_grid_modes(dtype,mode):compare(image(dtype,3),mode=mode)

@pytest.mark.parametrize('shape',((1,),(1,1),(1,7),(1,3,5),(2,1,3,4),(1,2,3,4,5)))
def test_singleton_axes(shape):compare(np.ones(shape,dtype='float64'))

def test_nested_native_alias():
    from skimagexx.filters.ridges import meijering
    assert meijering is xx.filters.meijering

@pytest.mark.parametrize('rank',(2,3))
def test_workers(rank,monkeypatch):
    a=np.random.default_rng(8621).normal(size=(513,519) if rank==2 else (65,67,69))
    first,_=compare(a)
    def forbidden(*args,**kw):raise AssertionError('upstream operation used')
    import skimage.filters.ridges as ridges
    monkeypatch.setattr(ridges,'hessian_matrix',forbidden);monkeypatch.setattr(np,'tensordot',forbidden);monkeypatch.setattr(np,'take_along_axis',forbidden)
    xx.set_num_threads(4);actual=xx.filters.meijering(a,sigmas=[1]);np.testing.assert_array_equal(actual,first)

@pytest.mark.parametrize('rank',(1,2,3,4,5))
@pytest.mark.parametrize('kind',('quadratic','axis_ridge','tilted_ridge','negative_zero'))
def test_degenerate(rank,kind):
    coords=np.indices((5,)*rank,dtype=float)
    if kind=='quadratic':a=np.sum(coords,axis=0)**2
    elif kind=='axis_ridge':a=np.exp(-((coords[0]-2)/2)**2)
    elif kind=='tilted_ridge':a=np.exp(-((np.sum(coords,axis=0)-2*rank)/2)**2)
    else:a=np.full((5,)*rank,-0.)
    compare(a)

@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('alpha',(0.,.3,1.,-.5))
def test_direct_response_tie_selection(dtype,alpha):
    from scipy.linalg import circulant
    n=3;shape=(2,3,4);count=np.prod(shape)
    # Diagonal matrices avoid eigensolver approximation and expose tie order.
    diagonal=np.array([[3,3,3,-1],[-3,2,0,-2],[-3,-5,-3,-3]],dtype=dtype)
    diagonal=np.sort(diagonal,axis=0)[::-1]
    values=np.tile(diagonal,(1,count//4)).reshape(n,*shape)
    fields=[]
    for r in range(n):
        for c in range(r,n):fields.append(values[r] if r==c else np.zeros(shape,dtype=dtype))
    projected=np.tensordot(circulant([1,alpha,alpha]).astype(dtype),values,1)
    chosen=np.take_along_axis(projected,abs(projected).argmax(0)[None],0).squeeze(0)
    expected=np.maximum(chosen,0);maximum=expected.max()
    if maximum>0:expected/=maximum
    actual=xx._core.meijering_accumulate(fields,None,np.dtype(dtype),alpha)
    tol=3e-6 if dtype is np.float32 else 3e-13
    np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol)

@pytest.mark.parametrize('case',('nan','inf','extreme','complex','negative_sigma','bad_alpha','bad_mode','generator','underflow'))
def test_fallback_once(case):
    from skimagexx._compat import FallbackWarning
    xx.set_fallback_enabled(True)
    def call(f):
        a=image(np.float64);kw={'sigmas':[1]}
        if case=='nan':a[3,4]=np.nan
        elif case=='inf':a[3,4]=np.inf
        elif case=='extreme':a[3,4]=1e307
        elif case=='complex':a=a.astype(complex)
        elif case=='negative_sigma':kw['sigmas']=[-1]
        elif case=='bad_alpha':kw['alpha']='invalid'
        elif case=='bad_mode':kw['mode']='invalid'
        elif case=='generator':kw['sigmas']=(s for s in [1,.7])
        with np.errstate(under='raise' if case=='underflow' else 'ignore'),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=f(a,**kw);error=None
            except Exception as e:value=None;error=(type(e),str(e))
        if f is xx.filters.meijering:assert sum(isinstance(w.message,FallbackWarning) for w in caught)==1
        return value,error,[(type(w.message),str(w.message)) for w in caught if not isinstance(w.message,FallbackWarning)]
    actual,error,aw=call(xx.filters.meijering);expected,expected_error,ew=call(reference)
    assert error==expected_error and aw==ew
    if error is None:np.testing.assert_allclose(actual,expected,equal_nan=True)
