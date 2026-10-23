"""Strict private rank parity; no forwarding or public registration adapters."""
import importlib
import importlib.util
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference
import skimagexx as xx

_spec=importlib.util.spec_from_file_location('rank_contract_provider',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
contracts=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(contracts)
if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    _spec=importlib.util.spec_from_file_location('skimagexx._rank_candidate',os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(native)
    native._core=importlib.import_module(os.environ['SKIMAGEXX_RANK_BACKEND'])
else:native=importlib.import_module('skimagexx._rank')


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads();other=native._core.get_num_threads()
    xx.set_num_threads(request.param);native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(other);xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('case',contracts.CASES,ids=[case[0] for case in contracts.CASES])
def test_frozen_native_contract(case):
    key,name,spec=case
    expected=contracts.observe(getattr(reference,name),spec)
    actual=contracts.observe(getattr(native,name),spec)
    assert actual==expected,key


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('masked',[False,True])
def test_parallel_large_definition(name,dtype,ndim,masked):
    shape=(513,519) if ndim==2 else (17,129,131)
    rng=np.random.default_rng(8947);a=rng.integers(0,256,shape,dtype=dtype)
    fp=np.ones((7,9) if ndim==2 else (3,5,7),np.uint8);fp.flat[::4]=0
    options={'mask':rng.integers(0,2,shape,np.uint8)} if masked else {}
    actual=getattr(native,name)(a,fp,**options);expected=getattr(reference,name)(a,fp,**options)
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_upstream_rank_calls(name,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('upstream rank forwarding')
    monkeypatch.setattr(reference,name,forbidden)
    a=np.arange(35,dtype=np.uint8).reshape(5,7)
    result=getattr(native,name)(a,np.ones((3,3),np.uint8))
    assert result.shape==a.shape


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('kind',['mask-too-small','out-too-small','zero-depth'])
def test_undefined_upstream_buffers_rejected_safely(name,kind):
    a=np.arange(35,dtype=np.uint8).reshape(5,7);options={}
    if kind=='mask-too-small':options['mask']=np.ones((1,1),np.uint8)
    elif kind=='out-too-small':options['out']=np.zeros((1,1),np.uint8)
    else:options['out']=np.zeros((5,7,0),np.uint8)
    # Do not invoke the bounds-disabled original with these allocations.
    with pytest.raises(ValueError):getattr(native,name)(a,np.ones((3,3),np.uint8),**options)

@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('uniform',[False,True])
def test_simd_long_runs_and_output_raw_bytes(name,dtype,uniform):
    rng=np.random.default_rng(29)
    a=np.full((17,71),7,dtype) if uniform else rng.integers(0,1024 if dtype is np.uint16 else 256,(17,71),dtype=dtype)
    fp=np.ones((3,65),np.uint8)
    with warnings.catch_warnings(record=True) as actual_notices:
        actual=getattr(native,name)(a,fp,out=np.zeros(a.shape,bool))
    with warnings.catch_warnings(record=True) as expected_notices:
        expected=getattr(reference,name)(a,fp,out=np.zeros(a.shape,bool))
    assert [(type(w.message),str(w.message)) for w in actual_notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('kind',['quiet','signal','quiet-max','quiet-inf','signal-max'])
@pytest.mark.parametrize('policy',['warn','raise','ignore'])
def test_nonfinite_conversion_status(name,dtype,kind,policy):
    a=np.full((3,5),.5,dtype)
    unsigned={np.float16:np.uint16,np.float32:np.uint32,np.float64:np.uint64}[dtype]
    signal={np.float16:0x7c01,np.float32:0x7f800001,np.float64:0x7ff0000000000001}[dtype]
    if kind in ('signal','signal-max'):a.view(unsigned).flat[0]=signal
    else:a.flat[0]=np.nan
    if kind in ('quiet-max','signal-max'):a.flat[-1]=np.finfo(dtype).max
    elif kind=='quiet-inf':a.flat[-1]=np.inf
    def observe(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            with np.errstate(all=policy):
                try:result,error=getattr(module,name)(a,np.ones((3,3),np.uint8)),None
                except Exception as caught:result,error=None,(type(caught),str(caught))
        return None if result is None else result.tobytes(),error,[(type(w.message),str(w.message)) for w in notices]
    assert observe(native)==observe(reference)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',[2,3])
def test_noncanonical_boolean_conversion(name,ndim):
    shape=(16,16) if ndim==2 else (1,16,16)
    a=np.arange(256,dtype=np.uint8).reshape(shape).view(np.bool_)
    def observe(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:result,error=getattr(module,name)(a,np.ones((3,)*ndim,np.uint8)).tobytes(),None
            except Exception as caught:result,error=None,(type(caught),str(caught))
        return result,error,[(type(w.message),str(w.message)) for w in notices]
    assert observe(native)==observe(reference)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64,np.complex64,np.complex128])
def test_signaling_footprint_normalization(name,dtype):
    a=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,5),dtype)
    base=np.float32 if dtype is np.complex64 else np.float64 if dtype is np.complex128 else dtype
    unsigned={np.float16:np.uint16,np.float32:np.uint32,np.float64:np.uint64}[base]
    bits={np.float16:0x7c01,np.float32:0x7f800001,np.float64:0x7ff0000000000001}[base]
    fp.view(unsigned).flat[0]=bits
    def observe(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');out=getattr(module,name)(a,fp)
        return out.tobytes(),[(type(w.message),str(w.message)) for w in notices]
    assert observe(native)==observe(reference)

@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('kind',['u16-safe-alias','mask-alias','buffer-before-shift'])
def test_aliases_and_conversion_precedence(name,kind):
    def observe(module):
        a=np.arange(35,dtype=np.uint16 if kind=='u16-safe-alias' else np.uint8).reshape(5,7)
        fp=np.ones((1,1) if kind=='u16-safe-alias' else (3,3),np.uint8);options={}
        if kind=='u16-safe-alias':options['out']=a.view()
        elif kind=='mask-alias':
            mask=np.ones(a.shape,np.uint8);mask.flat[::4]=0;options.update(mask=mask,out=mask)
        else:a.flags.writeable=False;options['shift_x']=np.asarray([0])
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:out,error=getattr(module,name)(a,fp,**options),None
            except Exception as caught:out,error=None,(type(caught),str(caught))
        return None if out is None else out.tobytes(),error,a.tobytes(),[(type(w.message),str(w.message)) for w in notices]
    assert observe(native)==observe(reference)

@pytest.mark.parametrize('policy',['call','log','print'])
@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('stage',['conversion','footprint'])
def test_stateful_float_provider_explicit_pending(policy,name,stage):
    a=np.ones((3,5),np.float32 if stage=='conversion' else np.uint8)
    fp=np.ones((3,3),np.float32 if stage=='footprint' else np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with np.errstate(all=policy):
            with pytest.raises(NotImplementedError,match='Stateful floating-point'):
                getattr(native,name)(a,fp)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_every_half_conversion_encoding_in_range(name):
    from skimage.util import img_as_ubyte
    words=np.arange(0x3c01,dtype=np.uint16)
    a=np.concatenate((words,words|0x8000)).view(np.float16).reshape(2,-1)
    actual=native._core.rank_convert(a)[0]
    np.testing.assert_array_equal(actual,img_as_ubyte(a))
    with warnings.catch_warnings(record=True) as actual_notices:
        warnings.simplefilter('always');actual=getattr(native,name)(a,np.ones((1,1),np.uint8))
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always');expected=getattr(reference,name)(a,np.ones((1,1),np.uint8))
    assert [(type(w.message),str(w.message)) for w in actual_notices]==[(type(w.message),str(w.message)) for w in expected_notices]
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',contracts.NAMES)
def test_precision_warning_shape_mutation_order(name):
    def observe(module):
        a=np.full((5,7),.5,np.float32);events=[]
        def hook(message,category,*args,**kwargs):
            events.append((category,str(message)))
            if len(events)==1:a.shape=(1,5,7)
        with warnings.catch_warnings():
            warnings.simplefilter('always');warnings.showwarning=hook
            try:result,error=getattr(module,name)(a,np.ones((3,3),np.uint8)),None
            except Exception as caught:result,error=None,(type(caught),str(caught))
        return result,error,events,a.shape
    assert observe(native)==observe(reference)



def test_parallel_half_conversion():
    from skimage.util import img_as_ubyte
    words=np.arange(0x3c01,dtype=np.uint16)
    a=np.tile(np.concatenate((words,words|0x8000)).view(np.float16).reshape(2,-1),(10,1))
    actual=native._core.rank_convert(a)
    np.testing.assert_array_equal(actual[0],img_as_ubyte(a))
    assert not any(bool(value) for value in actual[1:])

@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('kind',['timedelta','datetime','longdouble','clongdouble','complex64-nan','complex128-nan'])
def test_extended_footprint_comparisons(name,kind):
    dtype={'timedelta':'timedelta64[D]','datetime':'datetime64[D]','longdouble':np.longdouble,'clongdouble':np.clongdouble,'complex64-nan':np.complex64,'complex128-nan':np.complex128}[kind]
    a=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,5),dtype)
    if kind.endswith('nan'):fp.flat[0]=np.nan
    def observe(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:result,error=getattr(module,name)(a,fp).tobytes(),None
            except Exception as caught:result,error=None,(type(caught),str(caught))
        return result,error,[(type(w.message),str(w.message)) for w in notices]
    assert observe(native)==observe(reference)
