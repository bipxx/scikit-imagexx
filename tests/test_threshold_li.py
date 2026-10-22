"""Li's public contract and ordered floating reductions against pinned upstream."""
import inspect
import warnings
import numpy as np
import pytest
from skimage.filters import threshold_li as reference
import skimagexx as xx

@pytest.fixture(autouse=True,params=[1,4])
def strict(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)

def outcome(function,image,options):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value=function(image,**options)
            result=('value',type(value),np.asarray(value).dtype,value)
        except Exception as error:result=('error',type(error),str(error))
    return result,[(type(w.message),str(w.message)) for w in notices]

def compare(image,**options):
    expected,ew=outcome(reference,image,options)
    actual,aw=outcome(xx.filters.threshold_li,image,options)
    assert actual[:3]==expected[:3]
    assert aw==ew
    if expected[0]=='value':
        dtype=np.asarray(expected[3]).dtype
        tol=(2e-6,2e-6) if dtype==np.float32 else (2e-13,2e-13)
        np.testing.assert_allclose(actual[3],expected[3],rtol=tol[0],atol=tol[1])
        assert np.signbit(actual[3])==np.signbit(expected[3]) or np.isnan(expected[3])

def sample(dtype,shape):
    rng=np.random.default_rng(351)
    values=rng.integers(0,256,shape,dtype=np.uint8)
    return values if dtype==np.uint8 else (values.astype(dtype)*dtype(.03125)-dtype(3))

def test_signature():
    assert inspect.signature(xx.filters.threshold_li)==inspect.signature(reference)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('size',[1,7,8,9,31,32,33,127,128,129,262149])
@pytest.mark.parametrize('change',['none','first','last','nan'])
def test_uniform_scan_and_late_mismatch(dtype,size,change):
    a=np.full(size,7,dtype=dtype)
    if change=='first':a[0]=9
    if change=='last':a[-1]=9
    if change=='nan' and dtype!=np.uint8:a[-1]=np.nan
    compare(a)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(),(0,),(2,0,3),(1,),(7,),(8,),(127,),(128,),(129,),(13,17),(3,7,11),(262149,)])
def test_shapes_and_reduction_boundaries(dtype,shape):compare(sample(dtype,shape))

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('layout',['c','f','reversed','planar','sliced','unaligned','broadcast','readonly','endian','metadata'])
def test_storage_and_no_mutation(dtype,layout):
    a=sample(dtype,(19,23,3))
    if layout=='f':a=np.asfortranarray(a)
    if layout=='reversed':a=a[::-1,::-1,::-1]
    if layout=='planar':a=np.ascontiguousarray(np.moveaxis(a,-1,0))
    if layout=='sliced':a=a[:,::2,::-1]
    if layout=='unaligned':
        storage=np.empty(a.nbytes+1,np.uint8);b=np.ndarray(a.shape,dtype=dtype,buffer=storage,offset=1);b[:]=a;a=b
    if layout=='broadcast':a=np.broadcast_to(a[0:1,0:1],a.shape)
    if layout=='readonly':a.flags.writeable=False
    if layout=='endian':a=a.astype(a.dtype.newbyteorder('S'))
    if layout=='metadata':a=a.view(np.dtype(dtype,metadata={'li':'input'}))
    before=a.copy();compare(a);np.testing.assert_array_equal(a,before)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('tolerance',[None,0,.001,.5,2,999,np.inf,np.nan,np.float16(.1),np.float32(.1),np.float64(.1),True,False,np.bool_(True)])
def test_tolerance(dtype,tolerance):compare(sample(dtype,(19,23)),tolerance=tolerance)

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('guess',[-100,0,.001,.5,1,2,3,7,200,255,999,np.inf,-np.inf,np.nan,np.float16(2.1),np.float32(2.1),np.bool_(True)])
def test_numeric_guess(dtype,guess):compare(sample(dtype,(19,23)),initial_guess=guess)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('values',[[],[np.nan],[np.nan,np.nan],[np.inf],[-np.inf],[np.inf,-np.inf],
    [np.nan,3,3],[3,np.inf],[3,-np.inf],[np.nan,np.inf,-np.inf],[1,2,np.inf,np.nan],[-0.,0.],[-0.,-0.],[-3,-2,0]])
@pytest.mark.parametrize('tolerance',[None,.5,np.inf])
def test_nonfinite_and_constant_order(dtype,values,tolerance):compare(np.array(values,dtype=dtype),tolerance=tolerance)

@pytest.mark.parametrize('values',[[],[False],[True],[True,False],[False,True]])
def test_boolean_contract(values):compare(np.array(values,dtype=bool))

@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_ignored_options_on_constants(dtype):
    compare(np.full((2,3),7,dtype=dtype),tolerance=object(),initial_guess=object(),iter_callback=object())

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_errstate_raise_on_empty_foreground(dtype):
    with np.errstate(invalid='raise'):compare(np.array([3,np.inf],dtype=dtype),tolerance=.5)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_guess_rounding_to_maximum(dtype):
    compare(np.array([0,.5,1],dtype=dtype),initial_guess=np.float64(1)-np.float64(1e-9))

@pytest.mark.parametrize('size',[16,128,8192,16384])
def test_numpy_reduction_buffer(size):
    previous=np.getbufsize();np.setbufsize(size)
    try:compare(sample(np.float32,(262149,)))
    finally:np.setbufsize(previous)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('length',[9,129,8191,8192,8193,262149,1048583])
@pytest.mark.parametrize('seed',[5,17,951])
def test_continuous_float_reductions(dtype,length,seed):
    rng=np.random.default_rng(seed)
    a=(rng.standard_normal(length)*7-3).astype(dtype)
    a[::31]=np.nan;a[::67]=np.inf
    # Some upstream float32/default-tolerance inputs enter a two-cycle. Check
    # their exact bounded native iteration trace rather than hanging the suite.
    class Cycle(Exception):pass
    steps=[]
    def observe(value):
        steps.append(value)
        if len(steps)==128:raise Cycle()
    try:reference(a,iter_callback=observe)
    except Cycle:
        from skimagexx import _core
        prepared=_core.li_prepare(a)
        clean=a[np.isfinite(a)];clean-=clean.min()
        spacing=np.min(np.diff(np.unique(clean)))
        assert dtype(prepared.spacing())==spacing
        tolerance=spacing/2
        native=[]
        def capture(value,updated):
            native.append(dtype(value))
            if len(native)==len(steps):raise Cycle()
        block=np.getbufsize() if np.lib.NumpyVersion(np.__version__)<'2.3.0' else 0
        with pytest.raises(Cycle):prepared.run(float(tolerance),False,0,capture,reduction_block=block)
        np.testing.assert_array_equal(native,steps)
        assert steps[-1]==steps[-3] and steps[-2]==steps[-4]
        assert abs(steps[-1]-steps[-2])>tolerance
        return
    compare(a)

@pytest.mark.parametrize('options',[{'initial_guess':np.mean},{'iter_callback':lambda x:None}])
def test_explicit_callback_fallback(options):
    a=sample(np.float32,(13,17))
    with pytest.raises(NotImplementedError,match='outside native coverage'):xx.filters.threshold_li(a,**options)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):actual=xx.filters.threshold_li(a,**options)
    np.testing.assert_allclose(actual,reference(a,**options))
