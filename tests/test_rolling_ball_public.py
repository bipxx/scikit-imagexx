"""Strict admitted75/profile parity; explicit private component or public entry."""
import builtins
import ctypes
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import pytest
from rolling_ball_contract import recipes
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def target():
    import numpy as np
    path=os.environ.get('ROLLING_BALL_CORE_EXTENSION')
    if path:
        name='rolling_ball_component';package=types.ModuleType(name);package.__path__=[str(ROOT/'python/skimagexx')];sys.modules[name]=package
        spec=importlib.util.spec_from_file_location(name+'._core',path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);sys.modules[spec.name]=core
        module=importlib.import_module(name+'._rolling_ball')
    else:
        module=importlib.import_module('skimagexx.restoration');core=importlib.import_module('skimagexx._core')
    assert inspect.isbuiltin(core.rolling_ball_apply)and inspect.isbuiltin(core.ball_kernel_apply)
    assert str(inspect.signature(module.rolling_ball))=='(image, *, radius=100, kernel=None, nansafe=False, num_threads=<DEPRECATED>, workers=None)'
    core.set_num_threads(int(os.environ.get('ROLLING_BALL_TEST_WORKERS','1')))
    return np,module


@pytest.mark.parametrize('spec',recipes(),ids=lambda s:s['id'])
def test_saved_original(target,spec,monkeypatch):
    from rolling_ball_observer import observe
    from util_views_fenv import FEnv,backend
    np,module=target;profile='minimum'if np.__version__.split('.')[0]=='1'else'modern'
    saved=json.loads((ROOT/'tests/fixtures/rolling_ball_original_v1.json').read_bytes())['profiles'][profile]['rows']
    expected=next(r for r in saved if r['id']==spec['id'])
    libc,helper=backend()
    def state():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def numerical(value):return {k:getattr(value,k)for k in('control','status','mxcsr')}
    outer=state();incoming=state()
    # Trusted same-profile input envelope only; live pointers/reserved words.
    for key in('control','status','mxcsr'):setattr(incoming,key,expected['target_immediate']['before'][key])
    actual_state={};original_import=builtins.__import__
    def refused(name,*args,**kwargs):
        if name=='skimage'or name.startswith('skimage.'):raise AssertionError('Original request forbidden: '+name)
        return original_import(name,*args,**kwargs)
    def call(*args,**kwargs):
        restore(incoming);actual_state['before']=numerical(state())
        try:return module.rolling_ball(*args,**kwargs)
        finally:actual_state['after']=numerical(state())
    try:
        with monkeypatch.context()as scope:
            scope.setattr(builtins,'__import__',refused)
            actual=observe(call,spec,np)
        assert actual=={k:expected[k]for k in actual}
        assert actual_state=={when:{key:expected['target_immediate'][when][key]for key in('control','status','mxcsr')}for when in('before','after')}
    finally:
        restore(outer);assert bytes(state())==bytes(outer)


def test_owned_kernel_refusal_has_no_provider_algorithm_fallback(target,monkeypatch):
    np,module=target
    implementation=importlib.import_module(module.rolling_ball.__module__)
    called=[]
    def refuse(*args):
        called.append(True);return False
    with monkeypatch.context()as scope:
        scope.setattr(implementation._core,'ball_kernel_apply',refuse)
        with pytest.raises(NotImplementedError,match='owned kernel refused'):
            implementation.rolling_ball(np.ones((2,3),dtype=np.float32),radius=1)
    assert called==[True]


def test_private_output_overlap_is_refused_without_mutation(target):
    np,module=target;core=importlib.import_module(module.rolling_ball.__module__)._core
    a=np.array([1.,2.,3.]);before=a.tobytes()
    with pytest.raises(ValueError,match='overlaps'):
        core.rolling_ball_apply(a,np.array([0.]),[1],a,False,1)
    assert a.tobytes()==before


def test_private_non_numeric_dtype_is_refused_before_scan(target):
    np,module=target;core=importlib.import_module(module.rolling_ball.__module__)._core
    a=np.array([object()],dtype=object)
    with pytest.raises(TypeError,match='native float32/float64'):
        core.rolling_ball_apply(a,np.array([0.]),[1],np.zeros(1),False,1)


def test_user_workers_do_not_mutate_process_global_limit(target):
    np,module=target;core=importlib.import_module(module.rolling_ball.__module__)._core
    before=core.get_num_threads()
    actual=module.rolling_ball(np.array([1.,2.,3.]),radius=0,workers=2)
    assert core.get_num_threads()==before and np.array_equal(actual,[1.,2.,3.])


def test_degenerate_stride_magnitude_has_no_signed_overflow(target):
    np,module=target;core=importlib.import_module(module.rolling_ball.__module__)._core
    owner=np.array([1.]);a=np.ndarray((1,),dtype=np.float64,buffer=owner,
                                    strides=(int(np.iinfo(np.intp).min),))
    out=np.empty(1)
    core.rolling_ball_apply(a,np.array([0.]),[1],out,False,1)
    assert out[0]==1. and owner[0]==1.


def test_one_global_snapshot_spans_kernel_and_traversal(target,monkeypatch):
    np,module=target;implementation=importlib.import_module(module.rolling_ball.__module__);core=implementation._core
    before=core.get_num_threads();kernel=core.ball_kernel_apply;apply=core.rolling_ball_apply;seen=[]
    def change_between_stages(*args):
        accepted=kernel(*args);core.set_num_threads(4 if before!=4 else 1);return accepted
    def capture(*args):
        seen.append(args[-1]);return apply(*args)
    try:
        with monkeypatch.context()as scope:
            scope.setattr(core,'ball_kernel_apply',change_between_stages);scope.setattr(core,'rolling_ball_apply',capture)
            actual=implementation.rolling_ball(np.array([1.,2.,3.]),radius=0)
        assert seen==[before]and np.array_equal(actual,[1.,2.,3.])
    finally:core.set_num_threads(before)
