"""Symmetric image-field eigenvalues against both public feature APIs."""
import inspect
import warnings
import numpy as np
import pytest
import skimage.feature as reference
import skimagexx as xx

APIS=('hessian_matrix_eigvals','structure_tensor_eigenvalues')


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def fields(n,dtype,shape=None,pattern='random',scale=1):
    if shape is None:shape={1:(23,),2:(11,13),3:(5,7,9),4:(2,3,4,5),5:(2,2,2,3,3),6:(2,)*6}[n]
    rng=np.random.default_rng(916+n)
    matrix=rng.normal(size=(*shape,n,n));matrix=(matrix+np.swapaxes(matrix,-1,-2))/2
    if pattern=='zero':matrix[:]=0
    elif pattern=='identity':matrix[:]=np.eye(n)
    elif pattern=='diagonal':matrix[:]=np.diag(np.arange(n)-n//2)
    elif pattern=='rank-one':
        v=rng.normal(size=(*shape,n));matrix=v[...,None]*v[...,None,:]
    elif pattern=='near-repeat':matrix=np.eye(n)+matrix*1e-10
    matrix=(matrix*scale).astype(dtype)
    return [matrix[...,i,j].copy() for i in range(n) for j in range(i,n)]


def compare(api,elements):
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(elements)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught]
    expected,ew=outcome(getattr(reference,api));actual,aw=outcome(getattr(xx.feature,api))
    assert aw==ew
    if isinstance(expected,Exception):
        assert (type(actual),str(actual))==(type(expected),str(expected));return
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    assert actual.strides==expected.strides
    assert actual.flags.c_contiguous==expected.flags.c_contiguous
    assert actual.flags.f_contiguous==expected.flags.f_contiguous
    assert actual.flags.owndata==expected.flags.owndata
    amplitude=max((float(np.max(np.abs(a))) if np.size(a) else 0 for a in elements[:3] if isinstance(a,(np.ndarray,np.generic))),default=1.)
    tolerance=2e-6 if expected.dtype.type==np.float32 else 2e-13
    if len(elements)==3:
        np.testing.assert_array_equal(actual,expected)
    else:
        np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance*amplitude)
    return actual


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[1,2,3,4,5,6])
@pytest.mark.parametrize('pattern',['random','zero','identity','diagonal','rank-one','near-repeat'])
def test_matrices(api,dtype,n,pattern):
    compare(api,fields(n,dtype,pattern=pattern))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('layout',['reversed','fortran','readonly','unaligned','endian','tagged','tagged-endian','broadcast','stacked'])
def test_layouts(api,dtype,n,layout):
    elements=fields(n,dtype)
    if layout=='reversed':elements=[np.flip(a) for a in elements]
    elif layout=='fortran':elements=[np.asfortranarray(a) for a in elements]
    elif layout=='readonly':
        for a in elements:a.flags.writeable=False
    elif layout=='unaligned':
        prepared=[]
        for a in elements:
            raw=np.empty(a.nbytes+1,dtype=np.uint8);p=np.ndarray(a.shape,dtype=dtype,buffer=raw,offset=1);p[:]=a;prepared.append(p)
        elements=prepared
    elif layout=='endian':elements=[a.astype(a.dtype.newbyteorder('>')) for a in elements]
    elif layout=='tagged':elements=[a.view(np.dtype(dtype,metadata={'field':'symmetric'})) for a in elements]
    elif layout=='tagged-endian':elements=[a.astype(np.dtype(dtype,metadata={'field':'symmetric'}).newbyteorder('>')) for a in elements]
    elif layout=='broadcast':elements=[np.broadcast_to(a[(slice(0,1),)*a.ndim],a.shape) for a in elements]
    elif layout=='stacked':elements=np.stack(elements)
    compare(api,elements)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('scale',[1e-15,1e15])
def test_scales(api,dtype,n,scale):
    compare(api,fields(n,dtype,scale=scale))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('scalar',[False,True])
@pytest.mark.parametrize('scale',[0.,1.,1e-25,1e10])
def test_scalar_promotion(api,dtype,scalar,scale):
    values=[dtype(1.2*scale),dtype(-.4*scale),dtype(.7*scale)]
    if not scalar:values=[np.array(v) for v in values]
    compare(api,values)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('shape',[(0,3),(2,0,3),(1,1),(1,1,1),(1,)*8])
def test_empty_and_singletons(api,shape):
    n=len(shape);compare(api,fields(n,np.float64,shape=shape))


@pytest.mark.parametrize('api',APIS)
def test_broadcast_components(api):
    a,b,c=fields(2,np.float32)
    compare(api,[a,b[:1,:],c[:,:1]])


@pytest.mark.parametrize('api',APIS)
def test_ignored_extra_component(api):
    class Unused:
        def __array__(self,*args,**kwargs):raise AssertionError('ignored component inspected')
    compare(api,[*fields(2,np.float64),Unused()])


@pytest.mark.parametrize('api',APIS)
def test_signature(api):
    assert inspect.signature(getattr(xx.feature,api))==inspect.signature(getattr(reference,api))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('kind',['integer','half','complex','nonfinite','mixed','opaque'])
def test_explicit_unsupported(api,kind):
    elements=fields(2,np.float32)
    if kind=='integer':elements=[a.astype(np.int16) for a in elements]
    elif kind=='half':elements=[a.astype(np.float16) for a in elements]
    elif kind=='complex':elements=[a.astype(np.complex64) for a in elements]
    elif kind=='nonfinite':elements[0][0,0]=np.nan
    elif kind=='mixed':elements[1]=elements[1].astype(np.float64)
    elif kind=='opaque':elements=iter(elements)
    with pytest.raises(NotImplementedError):getattr(xx.feature,api)(elements)
    if kind=='opaque':return
    xx.set_fallback_enabled(True)
    def outcome(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:value=function(elements)
            except Exception as error:value=error
        return value,[(w.category,str(w.message)) for w in caught if w.category is not xx.FallbackWarning]
    expected,ew=outcome(getattr(reference,api));actual,aw=outcome(getattr(xx.feature,api));assert aw==ew
    if isinstance(expected,Exception):assert (type(actual),str(actual))==(type(expected),str(expected))
    else:np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('n',[2,3])
def test_parallel_and_no_original_solver(api,n,monkeypatch):
    elements=fields(n,np.float64,shape=(257,263) if n==2 else (31,33,35))
    expected=getattr(reference,api)(elements)
    def forbidden(*args,**kwargs):raise AssertionError('original eigenvalue body executed')
    monkeypatch.setattr(reference,api,forbidden);monkeypatch.setattr(np.linalg,'eigvalsh',forbidden)
    xx.set_num_threads(1);serial=getattr(xx.feature,api)(elements)
    xx.set_num_threads(4);parallel=getattr(xx.feature,api)(elements)
    np.testing.assert_array_equal(serial,parallel)
    np.testing.assert_allclose(serial,expected,rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('length',[1,4,17])
def test_signed_zero_eigenvalues(api,dtype,length):
    elements=[np.zeros((1,1,length),dtype=dtype) for _ in range(6)]
    elements[0][:]=-0.
    expected=getattr(reference,api)(elements)
    actual=compare(api,elements)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('api',APIS)
def test_longdouble_falls_back(api):
    elements=[np.ones((1,1,1),dtype=np.longdouble) for _ in range(6)]
    with pytest.raises(NotImplementedError):getattr(xx.feature,api)(elements)
    xx.set_fallback_enabled(True)
    with pytest.raises(TypeError):getattr(xx.feature,api)(elements)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('diagonal',[0,3,5])
def test_signed_zero_in_inactive_simd_lane(api,dtype,diagonal):
    elements=[np.zeros((1,1,17),dtype=dtype) for _ in range(6)]
    elements[diagonal][0,0,0]=-0.
    for i,a in enumerate(elements):a[0,0,1:]=np.arange(1,17)*(i+1)*.01
    expected=getattr(reference,api)(elements)
    actual=compare(api,elements)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[7,8,15,20,21,22,25,30,31,62])
@pytest.mark.parametrize('length',[1,17])
def test_high_dimension_zero_order(api,dtype,n,length):
    try:np.empty((0,)*(n+2))
    except ValueError:pytest.skip('reference intermediate exceeds this NumPy runtime rank limit')
    shape=(1,)*(n-1)+(length,)
    elements=[np.zeros(shape,dtype=dtype) for _ in range(n*(n+1)//2)]
    position=0
    for r in range(n):
        for c in range(r,n):
            if r==c and (length>1 or r==0):
                elements[position].reshape(-1)[r%length]=-0.
            position+=1
    expected=getattr(reference,api)(elements)
    actual=compare(api,elements)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[7,15,22,30,62])
def test_high_dimension_random(api,dtype,n):
    try:np.empty((0,)*(n+2))
    except ValueError:pytest.skip('reference intermediate exceeds this NumPy runtime rank limit')
    compare(api,fields(n,dtype,shape=(1,)*n))


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('n',[31,63])
def test_reference_rank_limit(api,n):
    try:
        field=np.zeros((1,)*n)
        np.empty((0,)*(n+2))
    except ValueError:
        try:field=np.zeros((1,)*n)
        except ValueError:pytest.skip('input itself exceeds this NumPy rank limit')
    else:pytest.skip('reference supports this intermediate rank')
    elements=[field]*(n*(n+1)//2)
    with pytest.raises(ValueError) as expected:getattr(reference,api)(elements)
    with pytest.raises(NotImplementedError):getattr(xx.feature,api)(elements)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning),pytest.raises(ValueError) as actual:getattr(xx.feature,api)(elements)
    assert str(actual.value)==str(expected.value)
