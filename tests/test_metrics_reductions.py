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
from skimagexx.coverage import native_functions

if os.environ.get('SKIMAGEXX_METRICS_CANDIDATE'):
    spec=importlib.util.spec_from_file_location('skimagexx._metrics_candidate',os.environ['SKIMAGEXX_METRICS_CANDIDATE'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native._core=importlib.import_module(os.environ['SKIMAGEXX_METRICS_BACKEND'])
else:native=importlib.import_module('skimagexx._metrics')

TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
FUNCTIONS=[('mean_squared_error',{}),('normalized_root_mse',{'normalization':'euclidean'}),
           ('normalized_root_mse',{'normalization':'min-max'}),('normalized_root_mse',{'normalization':'mean'}),
           ('peak_signal_noise_ratio',{'data_range':2})]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',['empty-mean-resize','energy-warning-mutation'])
def test_warning_hook_preserves_materialized_stage_boundaries(dtype,case):
    def run(backend):
        if case=='empty-mean-resize':
            first=np.empty(0,dtype);second=np.empty(0,dtype);normalization='mean'
        else:
            first=np.full(2,np.finfo(dtype).max,dtype);second=first.copy();normalization='euclidean'
        notices=[]
        def hook(message,category,*args,**kwargs):
            notices.append((category,str(message)))
            if len(notices)==1:
                if case=='empty-mean-resize':first.resize(1,refcheck=False)
                else:first[:]=np.inf;second[:]=0
        with warnings.catch_warnings(),np.errstate(all='warn'):
            warnings.simplefilter('always');warnings.showwarning=hook
            try:result=backend.normalized_root_mse(first,second,normalization=normalization);error=None
            except Exception as caught:result=None;error=(type(caught),str(caught))
        return result,error,notices
    expected=run(reference);actual=run(native)
    assert actual[1:]==expected[1:]
    if expected[1] is None:
        assert type(actual[0]) is type(expected[0])
        np.testing.assert_array_equal(actual[0],expected[0])


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_normalization_protocol_preserves_conversion_before_lower(dtype):
    def run(backend):
        first=np.ones(3,dtype);second=np.full(3,2,dtype);calls=[]
        class Normalization:
            def lower(self):
                calls.append('lower');first[:]=0;second[:]=1
                return 'euclidean'
        with np.errstate(divide='ignore',invalid='ignore'):
            result=backend.normalized_root_mse(first,second,normalization=Normalization())
        return result,calls,first,second
    expected=run(reference);actual=run(native)
    assert actual[1]==expected[1]==['lower']
    for x,y in zip(actual[::2],expected[::2]):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(actual[3],expected[3])


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    if 'metrics.mean_squared_error' in native_functions() and not os.environ.get('SKIMAGEXX_METRICS_CANDIDATE'):
        assert native._core is xx._core
        for name,_ in FUNCTIONS:
            assert getattr(native,name) is getattr(xx.metrics,name)
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
    if name=='peak_signal_noise_ratio':check(name,a,np.zeros(37,other))


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


@pytest.mark.parametrize('layout',['c','reverse','endian','unaligned'])
def test_every_finite_half_encoding(layout):
    words=np.arange(65536,dtype=np.uint16)
    a=words[(words&0x7c00)!=0x7c00].view(np.float16)
    if layout=='reverse':a=a[::-1]
    elif layout=='endian':a=a.astype('>f2')
    elif layout=='unaligned':
        original=a;a=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);a[...]=original
    for name,options in FUNCTIONS:check(name,a,np.zeros_like(a),**options)


@pytest.mark.parametrize('dtype',TYPES)
def test_singleton_extreme_strides(dtype):
    minimum=np.iinfo(np.intp).min;maximum=np.iinfo(np.intp).max
    a=np.ndarray((1,1),dtype,buffer=bytearray(np.dtype(dtype).itemsize),strides=(minimum,maximum))
    a[...]=1;b=np.zeros_like(a)
    for name,options in FUNCTIONS:check(name,a,b,**options)
    check('peak_signal_noise_ratio',a,b)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('layout',['c','fortran','reverse','endian'])
def test_psnr_nan_range_validation(dtype,layout):
    a=np.resize(np.array([np.nan,2.,-3.,np.inf],dtype),(17,19))
    if layout=='fortran':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='endian':a=a.astype(a.dtype.newbyteorder('>'))
    check('peak_signal_noise_ratio',a,np.zeros_like(a))


def test_no_python_image_algorithms(monkeypatch):
    a,b=sample(np.float32),sample(np.float32,seed=2371)
    expected=reference.mean_squared_error(a,b)
    def forbidden(*args,**kwargs):raise AssertionError('native metric called a Python image algorithm')
    for name in ['mean','sum','subtract','square','multiply']:
        monkeypatch.setattr(np,name,forbidden)
    assert native.mean_squared_error(a,b)==expected

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('value',['max','tiny'])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_scalar_power_status(dtype,value,policy,name,options):
    a=np.asarray(getattr(np.finfo(dtype),value),dtype=dtype)
    with np.errstate(all=policy):check(name,a,np.asarray(0,dtype=dtype),**options)


@pytest.mark.parametrize('block',[32,8192])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_runtime_reduction_buffer(block,name,options):
    original=np.getbufsize()
    a=np.tile(np.array([1e8,1,-1e8,3],np.float32),16385)[::-1]
    try:
        np.setbufsize(block)
        check(name,a,np.zeros_like(a),**options)
    finally:np.setbufsize(original)


def _stateful_check(factory,name,options,policy,action,capfd):
    def exercise(module):
        a,b=factory();events=[];acted=False
        def callback(*event):
            nonlocal acted
            events.append(event)
            if acted:return
            acted=True
            if action=='mutate':a.fill(2);b.fill(1)
            elif action=='policy':np.seterr(all='ignore')
            elif action=='resize':a.resize((1,),refcheck=False)
            elif action=='handler':np.seterrcall(lambda *args:events.append(('replacement',)+args))
            elif action=='reentrant':
                with np.errstate(all='ignore'):
                    nested=module.mean_squared_error(np.asarray([1.,2.]),np.asarray([0.,0.]))
                assert nested==2.5
        class Sink:
            def write(self,text):callback(text)
        original=np.geterrcall()
        try:
            np.seterrcall(Sink() if policy=='log' else callback)
            with np.errstate(all=policy):
                result,error,notices=observe(getattr(module,name),a,b,**options)
                after=np.geterr().copy()
            captured=capfd.readouterr()
        finally:np.seterrcall(original)
        return result,error,notices,events,a.tobytes(),b.tobytes(),after,captured
    expected=exercise(reference);actual=exercise(native)
    assert actual[2:]==expected[2:]
    if expected[1] is not None:
        assert type(actual[1]) is type(expected[1])
        assert str(actual[1])==str(expected[1])
    else:
        if actual[1] is not None:raise actual[1]
        assert type(actual[0]) is type(expected[0])
        np.testing.assert_allclose(actual[0],expected[0],rtol=3e-6,atol=0,equal_nan=True)
        if actual[0]==expected[0]==0:assert np.signbit(actual[0])==np.signbit(expected[0])


@pytest.mark.parametrize('policy',['call','log','print'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
@pytest.mark.parametrize('scenario',['cast-first','cast-second','denominator','denominator-mean','denominator-reduce','subtract','square','scalar-over','scalar-under'])
@pytest.mark.parametrize('action',['none','mutate','policy','handler','reentrant','resize'])
def test_stateful_boundaries(policy,name,options,scenario,action,capfd):
    def factory():
        if scenario=='cast-first':
            return np.asarray([0x7f800001,0x3f000000],np.uint32).view(np.float32),np.asarray([.3,.6],np.float64)
        if scenario=='cast-second':
            return np.asarray([.3,.6],np.float64),np.asarray([0x7f800001,0x3f000000],np.uint32).view(np.float32)
        if scenario=='denominator-mean':return np.full(2,np.finfo(np.float32).max,np.float32),np.zeros(2,np.float32)
        if scenario=='denominator-reduce':return np.asarray([1e154,1e154],np.float64),np.zeros(2,np.float64)
        if scenario=='denominator':
            return np.asarray([np.finfo(np.float32).max,-np.finfo(np.float32).max],np.float32),np.zeros(2,np.float32)
        if scenario=='subtract':return np.asarray([np.finfo(np.float32).max,1e-30,np.inf],np.float32),np.asarray([-np.finfo(np.float32).max,0,np.inf],np.float32)
        if scenario=='square':return np.asarray([1e30,1e-30],np.float32),np.zeros(2,np.float32)
        return np.asarray(np.finfo(np.float64).max if scenario=='scalar-over' else np.finfo(np.float64).tiny),np.asarray(0.)
    _stateful_check(factory,name,options,policy,action,capfd)


@pytest.mark.parametrize('first',TYPES)
@pytest.mark.parametrize('second',TYPES)
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_stateful_dtype_pairs(first,second,name,options,capfd):
    _stateful_check(lambda:(sample(first),sample(second,seed=2371)),name,options,'call','none',capfd)


@pytest.mark.parametrize('layout',['fortran','mixed-fortran','reverse','broadcast','permuted3d','endian','unaligned'])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_stateful_layouts(layout,dtype,name,options,capfd):
    def factory():
        a,b=sample(dtype,shape=(17,19,23)),sample(dtype,shape=(17,19,23),seed=2371)
        if layout=='fortran':a,b=np.asfortranarray(a),np.asfortranarray(b)
        elif layout=='mixed-fortran':a=np.asfortranarray(a)
        elif layout=='reverse':a,b=a[::-1,::-1,::-1],b[::-1,::-1,::-1]
        elif layout=='broadcast':a,b=np.broadcast_to(a[:1],a.shape),np.broadcast_to(b[:1],b.shape)
        elif layout=='permuted3d':a,b=a.transpose(1,2,0),b.transpose(1,2,0)
        elif layout=='endian':a,b=a.astype(a.dtype.newbyteorder()),b.astype(b.dtype.newbyteorder())
        elif layout=='unaligned':
            aa=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);aa[:]=a;a=aa
            bb=np.ndarray(b.shape,b.dtype,buffer=bytearray(b.nbytes+1),offset=1);bb[:]=b;b=bb
        return a,b
    _stateful_check(factory,name,options,'call','none',capfd)

@pytest.mark.parametrize('policy',['call','log'])
@pytest.mark.parametrize('action',['none','mutate','policy'])
@pytest.mark.parametrize('name,options',FUNCTIONS)
def test_stateful_parallel_status(policy,action,name,options,capfd):
    def factory():
        a=np.full(257*1025,.7,np.float32);b=np.full(a.shape,.3,np.float32)
        a[0]=np.finfo(np.float32).max;b[0]=-np.finfo(np.float32).max
        a[len(a)//2]=1e-30;b[len(a)//2]=0
        a[-1]=b[-1]=np.inf
        return a,b
    _stateful_check(factory,name,options,policy,action,capfd)


@pytest.mark.parametrize('policy',['call','log','print'])
@pytest.mark.parametrize('dtype',TYPES)
def test_stateful_psnr_default_range(policy,dtype,capfd):
    _stateful_check(lambda:(sample(dtype),sample(dtype,seed=2371)),
                    'peak_signal_noise_ratio',{},policy,'none',capfd)


@pytest.mark.parametrize('policy',['call','log'])
@pytest.mark.parametrize('normalization',['euclidean','mean','min-max'])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_stateful_signaling_nan_range(policy,normalization,dtype,capfd):
    def factory():
        unsigned={np.float16:np.uint16,np.float32:np.uint32,np.float64:np.uint64}[dtype]
        word={np.float16:0x7c01,np.float32:0x7f800001,np.float64:0x7ff0000000000001}[dtype]
        a=np.ones(37,dtype);a.view(unsigned)[0]=word
        return a,np.zeros_like(a)
    _stateful_check(factory,'normalized_root_mse',{'normalization':normalization},policy,'mutate',capfd)
