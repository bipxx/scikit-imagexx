"""Exact numerical parity with the original morphology implementations."""
import warnings
import numpy as np
import pytest
from skimage import morphology as reference
import skimagexx as xx

NAMES=xx.morphology.__all__
DTYPES=['bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64','float32','float64']
pytestmark=pytest.mark.filterwarnings('ignore:.*is deprecated since version 0.26:FutureWarning')


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def sample(dtype,shape):
    rng=np.random.default_rng(157)
    a=rng.integers(-19,20,shape)
    if dtype=='bool': return a>0
    if dtype.startswith('float'): return (a*.125).astype(dtype)
    return a.astype(dtype)


def compare(name,a,**kwargs):
    expected=getattr(reference,name)(a,**kwargs)
    actual=getattr(xx.morphology,name)(a,**kwargs)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('shape',[(1,),(37,),(9,43),(4,5,19)])
def test_defaults(name,dtype,shape):
    compare(name,sample(dtype,shape))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['bool','uint8','int16','float32','float64'])
@pytest.mark.parametrize('kind',['even','asymmetric','rect3','rect31','rect101','sequence'])
def test_footprints_and_modes(name,dtype,kind):
    a=sample(dtype,(13,41))
    fp={'even':np.array([[1,0],[1,1]]),'asymmetric':np.array([[1,1,0],[0,0,1],[1,0,0]]),
        'rect3':np.ones((3,3)),'rect31':np.ones((31,31)),
        'rect101':np.ones((101,101)),
        'sequence':((np.ones((3,1)),2),(np.ones((1,4)),1))}[kind]
    modes=('ignore','max','min') if name.startswith('binary_') else xx.morphology._MODES
    for mode in modes:
        compare(name,a,footprint=fp,mode=mode,**({} if name.startswith('binary_') else {'cval':-2.5}))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('layout',['inplace','strided','overlap','unaligned'])
def test_out_identity(name,layout):
    original=sample('float32',(19,43))
    a=original.copy()
    if layout=='inplace': out=a
    elif layout=='strided': out=np.empty((19,86),dtype=a.dtype)[:,::2]
    elif layout=='overlap':
        backing=np.empty((19*43+1,),dtype=a.dtype)
        a=backing[:-1].reshape(original.shape); a[:]=original
        out=backing[1:].reshape(original.shape)
    else: out=np.ndarray(a.shape,dtype=a.dtype,buffer=bytearray(a.nbytes+1),offset=1)
    expected=getattr(reference,name)(original)
    result=getattr(xx.morphology,name)(a,out=out)
    assert result is out
    np.testing.assert_array_equal(result,expected)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',['int64','uint64'])
@pytest.mark.parametrize('fp',[np.ones((1,1)),np.ones((3,3)),np.ones((2,4)),np.array([[0,1,0]]),np.ones((31,31))])
def test_64bit_rounding(name,dtype,fp):
    rng=np.random.default_rng(3)
    a=rng.integers(0,64,(7,47),dtype=np.uint64)+np.uint64(2**53)
    a=a.astype(dtype)
    a[2,12]=np.iinfo(dtype).max
    compare(name,a,footprint=fp)


@pytest.mark.parametrize('name',NAMES[:6])
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('cval',[-129.,256.,2.**32,2.**63,2.**64])
@pytest.mark.parametrize('fp',[np.array([[1,0,1]]),np.ones((1,3)),np.ones((2,4))],ids=['sparse','separable','even-rectangle'])
def test_constant_casts(name,dtype,cval,fp):
    compare(name,sample(dtype,(3,41)),footprint=fp,mode='constant',cval=cval)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('shape',[(0,),(0,3),(3,0)])
def test_empty(name,shape):
    compare(name,np.empty(shape,dtype=np.float32))


@pytest.mark.parametrize('name',NAMES[6:])
@pytest.mark.parametrize('count',[0,-1,1,3])
def test_binary_iterations_and_warnings(name,count):
    a=sample('bool',(9,43))
    fp=((np.ones((3,3)),count),)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        expected=getattr(reference,name)(a,fp)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        actual=getattr(xx.morphology,name)(a,fp)
    assert [(w.category,str(w.message)) for w in actual_warnings]==[(w.category,str(w.message)) for w in expected_warnings]
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('name',NAMES[:6])
def test_nan_explicit_fallback(name):
    a=np.array([[np.nan,1.,2.]])
    with pytest.raises(NotImplementedError): getattr(xx.morphology,name)(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning): actual=getattr(xx.morphology,name)(a)
    np.testing.assert_array_equal(actual,getattr(reference,name)(a))


@pytest.mark.parametrize('name',NAMES)
def test_parallel_large(name):
    compare(name,sample('float32',(519,521)))


@pytest.mark.parametrize('name',NAMES)
def test_input_layouts(name):
    a=sample('float64',(11,17,19))
    for view in (a.T,a[::-1,:,::2],np.broadcast_to(a[0:1,0:1,:],a.shape)):
        compare(name,view,footprint=np.ones((3,2,1)))


@pytest.mark.parametrize('name',NAMES[6:])
@pytest.mark.parametrize('dtype',DTYPES)
def test_binary_output_dtypes(name,dtype):
    a=sample('float64',(9,43)); a[3,9]=np.nan
    out=np.empty(a.shape,dtype=dtype)
    expected=getattr(reference,name)(a,out=np.empty_like(out))
    assert getattr(xx.morphology,name)(a,out=out) is out
    np.testing.assert_array_equal(out,expected)


@pytest.mark.parametrize('name',NAMES[:6])
@pytest.mark.parametrize('value',[np.inf,-np.inf,np.finfo('float32').max])
def test_float_extremes(name,value):
    a=sample('float32',(3,43)); a[1,18]=value; a[1,19]=-value
    with np.errstate(invalid='ignore',over='ignore'):
        compare(name,a)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('options',[{'mode':'unknown'},{'footprint':[[1,1],[1,1]]},
                                  {'footprint':()},{'footprint':np.ones(3)}])
def test_invalid_arguments(name,options):
    a=sample('uint8',(3,7))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        with pytest.raises(Exception) as expected: getattr(reference,name)(a,**options)
        with pytest.raises(type(expected.value)) as actual: getattr(xx.morphology,name)(a,**options)
    assert str(actual.value)==str(expected.value)
