"""Per-function strict parity for native raw-value image error metrics."""
import importlib
import importlib.util
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage import metrics as reference
import skimagexx as xx

if os.environ.get('SKIMAGEXX_METRICS_CANDIDATE'):
    spec=importlib.util.spec_from_file_location('skimagexx._metrics_candidate',os.environ['SKIMAGEXX_METRICS_CANDIDATE'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native._core=importlib.import_module(os.environ['SKIMAGEXX_METRICS_BACKEND'])
else:native=importlib.import_module('skimagexx._metrics')

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
FUNCTIONS=[('mean_squared_error',{}),('normalized_root_mse',{'normalization':'euclidean'}),
           ('normalized_root_mse',{'normalization':'min-max'}),('normalized_root_mse',{'normalization':'mean'}),
           ('peak_signal_noise_ratio',{'data_range':2})]


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param);native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(1);xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def sample(dtype,shape=(17,19),seed=8849):
    rng=np.random.default_rng(seed)
    if dtype is np.bool_:return rng.integers(0,2,shape,dtype=np.uint8).view(np.bool_)
    if np.dtype(dtype).kind in 'iu':return rng.bit_generator.random_raw(int(np.prod(shape))).astype(dtype).reshape(shape)
    return rng.uniform(-.7,.9,size=shape).astype(dtype)


def observe(fn,*args,**kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=fn(*args,**kwargs),None
        except Exception as caught:result,error=None,caught
    return result,error,[(type(item.message),str(item.message)) for item in notices]


def check(name,a,b,**options):
    before=(a.tobytes(),b.tobytes())
    expected,error,notices=observe(getattr(reference,name),a,b,**options)
    actual,actual_error,actual_notices=observe(getattr(native,name),a,b,**options)
    assert actual_notices==notices
    assert (a.tobytes(),b.tobytes())==before
    if error is not None:
        assert type(actual_error) is type(error)
        assert str(actual_error)==str(error)
        return
    if actual_error is not None:raise actual_error
    assert type(actual) is type(expected)
    tolerance=3e-6 if name=='normalized_root_mse' and options.get('normalization')=='mean' and a.dtype.char in 'ef' and b.dtype.char in 'ef' else 3e-13
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=0,equal_nan=True)
    if actual==0 and expected==0:assert np.signbit(actual)==np.signbit(expected)


@pytest.mark.parametrize('first',TYPES)
@pytest.mark.parametrize('second',TYPES)
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_dtype_pairs(first,second,name,options):
    check(name,sample(first),sample(second,seed=2371),**options)


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('layout',['fortran','reverse','step','broadcast','readonly','endian','unaligned','tagged','mixed-fortran'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_layouts(dtype,layout,name,options):
    a,b=sample(dtype),sample(dtype,seed=2371)
    if layout=='fortran':a,b=np.asfortranarray(a),np.asfortranarray(b)
    elif layout=='mixed-fortran':a=np.asfortranarray(a)
    elif layout=='reverse':a,b=a[::-1,::-1],b[::-1,::-1]
    elif layout=='step':a,b=a[::2,::2],b[::2,::2]
    elif layout=='broadcast':a,b=np.broadcast_to(a[:1],a.shape),np.broadcast_to(b[:1],b.shape)
    elif layout=='readonly':a.flags.writeable=False;b.flags.writeable=False
    elif layout=='endian':a,b=a.astype(a.dtype.newbyteorder('>')),b.astype(b.dtype.newbyteorder('>'))
    elif layout=='tagged':a=a.view(np.dtype(a.dtype,metadata={'unit':'raw'}))
    else:
        original=a;a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=original
    check(name,a,b,**options)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64,np.uint64,np.int64])
@pytest.mark.parametrize('shape',[(),(0,),(0,7),(1,),(7,),(8,),(127,),(128,),(129,),(1031,),(3,5,7),(2,3,4,5)])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_shapes_and_empty(dtype,shape,name,options):
    check(name,sample(dtype,shape),sample(dtype,shape,2371),**options)


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_parallel_sized(dtype,name,options):
    check(name,sample(dtype,(257,1025)),sample(dtype,(257,1025),2371),**options)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('pattern',[[0.],[-0.],[1.],[-1.],[np.nan],[np.inf],[-np.inf],
    [np.inf,-np.inf,1.],[np.nan,-1.,0.,1.],[0.,-0.,0.,-0.]])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_special_values(dtype,pattern,name,options):
    a=np.resize(np.array(pattern,dtype),37);b=np.zeros_like(a)
    check(name,a,b,**options)


@pytest.mark.parametrize('dtype',TYPES)
def test_psnr_default_range(dtype):
    a=sample(dtype);b=sample(dtype,seed=2371)
    check('peak_signal_noise_ratio',a,b)
    check('peak_signal_noise_ratio',a,b.astype(np.float64))


@pytest.mark.parametrize('normalization',['Euclidean','MIN-MAX','MeAn','unknown',None,3,b'mean'])
def test_normalization_validation(normalization):
    check('normalized_root_mse',sample(np.float32),sample(np.float32,seed=2371),normalization=normalization)


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_complex_validation(dtype,name,options):
    a=np.ones((3,5),dtype);check(name,a,a,**options)


@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_shape_validation(name,options):
    check(name,np.ones((3,5)),np.ones((5,3)),**options)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_overflow_policies(dtype,policy,name,options):
    maximum=np.finfo(dtype).max
    with np.errstate(over=policy,invalid=policy):
        check(name,np.array([maximum,-maximum,maximum],dtype),np.array([-maximum,maximum,maximum],dtype),**options)


def test_noncanonical_bool():
    a=np.resize(np.array([0,1,2,128,255],np.uint8),1031).view(np.bool_)
    for name,options in FUNCTIONS:check(name,a,np.zeros_like(a),**options)


@pytest.mark.parametrize('dtype,bits,word',[(np.float16,np.uint16,0x7c01),
    (np.float32,np.uint32,0x7f800001),(np.float64,np.uint64,0x7ff0000000000001)])
@pytest.mark.parametrize('other',[np.float16,np.float32,np.float64,np.uint8])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_signaling_nan(dtype,bits,word,other,name,options):
    a=np.resize(np.array([word],bits).view(dtype),37)
    check(name,a,np.zeros(37,other),**options)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_underflow_policies(dtype,policy,name,options):
    a=np.array([np.finfo(dtype).tiny,0.],dtype)
    with np.errstate(under=policy):check(name,a,np.zeros_like(a),**options)


@pytest.mark.parametrize('normalization',[None,'invalid'])
def test_conversion_before_parameter_error(normalization):
    a=np.array([0x7f800001],np.uint32).view(np.float32)
    check('normalized_root_mse',a,np.zeros(1,np.float64),normalization=normalization)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('size',[1,3,8,37,128,129,8193])
@pytest.mark.parametrize('zero',[0.,-0.])
def test_zero_reference_nonzero_test(dtype,size,zero):
    a=np.full(size,zero,dtype);b=np.ones_like(a)
    for name,options in FUNCTIONS:check(name,a,b,**options)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['c','f','mixed-f','reverse','broadcast'])
def test_mean_cancellation(dtype,layout):
    a=np.resize(np.array([1e10,1.,-1e10,1.,-1.,3.,-3.,2.],dtype),(17,19));b=np.zeros_like(a)
    if layout=='f':a,b=np.asfortranarray(a),np.asfortranarray(b)
    elif layout=='mixed-f':a=np.asfortranarray(a)
    elif layout=='reverse':a,b=a[::-1,::-1],b[::-1,::-1]
    elif layout=='broadcast':a=np.broadcast_to(a[:1],a.shape)
    check('normalized_root_mse',a,b,normalization='mean')


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_numpy_scalars(dtype,name,options):
    check(name,dtype(.3),dtype(.7),**options)


def test_import_preserves_floating_policy():
    spec=importlib.util.spec_from_file_location('skimagexx._metrics_policy_probe',native.__file__)
    module=importlib.util.module_from_spec(spec)
    with np.errstate(all='raise'):
        spec.loader.exec_module(module)
        assert all(value=='raise' for value in np.geterr().values())


def test_no_python_image_algorithms(monkeypatch):
    a,b=sample(np.float32),sample(np.float32,seed=2371)
    expected=reference.mean_squared_error(a,b)
    def forbidden(*args,**kwargs):raise AssertionError('native metric called a Python image algorithm')
    for name in ['mean','sum','subtract','square','multiply']:
        monkeypatch.setattr(np,name,forbidden)
    assert native.mean_squared_error(a,b)==expected
