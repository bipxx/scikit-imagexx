"""Exact speculative leaf equivalence to the frozen pre-optimization core.

The private adapter uses unchanged Python wrappers; no operation is timed.
Production proof must separately validate the actual complete module.
"""
import importlib.util
import os
from pathlib import Path
import struct
import numpy as np
import pytest
import skimagexx as xx

spec=importlib.util.spec_from_file_location('metrics_leaf_helpers',Path(__file__).with_name('test_metrics_reductions.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
baseline=xx._core
if os.environ.get('SKIMAGEXX_METRICS_PROVEN_BASELINE'):
    # An explicitly supplied same-build-mode frozen core distinguishes
    # edit-induced changes from Debug/Release NaN operand-selection variance.
    spec=importlib.util.spec_from_file_location('metrics_leaf_baseline._metrics_load_core',os.environ['SKIMAGEXX_METRICS_PROVEN_BASELINE'])
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)

@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=helpers.native._core.get_num_threads()
    helpers.native._core.set_num_threads(request.param)
    yield
    helpers.native._core.set_num_threads(previous)

def bits(value):
    if isinstance(value,tuple):return tuple(bits(item) for item in value)
    if isinstance(value,np.generic):return type(value),value.dtype.str,value.tobytes()
    return struct.pack('=d',value) if isinstance(value,float) else value

@pytest.mark.parametrize('dtype',[np.bool_,np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','step','reverse'])
@pytest.mark.parametrize('pattern',['finite','zero-ties','nonfinite'])
def test_exact_leaf_tree_buffer_equivalence(dtype,layout,pattern):
    # Baseline module is the previously sealed complete product. Its metrics
    # sources precede this optimization; private and baseline worker states
    # are independent, so the baseline is held at one worker explicitly.
    if baseline is not helpers.native._core:baseline.set_num_threads(1)
    for size in (1,3,4,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,257,8193):
        rng=np.random.default_rng(319+size)
        if dtype is np.bool_:
            a=rng.integers(0,255,size*3,dtype=np.uint8).view(np.bool_)
            b=rng.integers(0,255,size*3,dtype=np.uint8).view(np.bool_)
        else:
            a=np.ldexp(rng.uniform(-1,1,size*3),rng.integers(-20,20,size*3)).astype(dtype)
            b=np.ldexp(rng.uniform(-1,1,size*3),rng.integers(-20,20,size*3)).astype(dtype)
            if pattern=='zero-ties':a[::3]=dtype(-0.);a[1::5]=dtype(0.)
            if pattern=='nonfinite':
                a[::11]=np.inf;a[1::13]=-np.inf;a[2::17]=np.nan
                words=a.view(np.uint32 if dtype is np.float32 else np.uint64)
                words[3::19]=0x7f800001 if dtype is np.float32 else 0x7ff0000000000001
        if layout=='step':a,b=a[::3],b[::3]
        elif layout=='reverse':a,b=a[size-1::-1],b[size-1::-1]
        else:a,b=a[:size],b[:size]
        before=a.tobytes(),b.tobytes()
        for mode in range(4):
            for capture in (False,True):
                # After integration, the ordinary installed module may be
                # the candidate itself. Its flagged scalar replay then serves
                # as the oracle instead of comparing the fast path to itself.
                oracle_capture=True if baseline is helpers.native._core else capture
                expected=baseline.metrics_reduce(a,b,dtype is np.float32,mode,oracle_capture,False)
                actual=helpers.native._core.metrics_reduce(a,b,dtype is np.float32,mode,capture,False)
                assert actual.keys()==expected.keys()
                assert {k:bits(v) for k,v in actual.items()}=={k:bits(v) for k,v in expected.items()}
        assert before==(a.tobytes(),b.tobytes())

@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.bool_])
def test_measured_layouts_reference_parity(dtype):
    rng=np.random.default_rng(482)
    first=rng.uniform(-.8,.9,(256,771)).astype(dtype)[:,::3]
    second=rng.uniform(-.8,.9,first.shape).astype(dtype)
    for name,options in helpers.FUNCTIONS:helpers.check(name,first,second,**options)
