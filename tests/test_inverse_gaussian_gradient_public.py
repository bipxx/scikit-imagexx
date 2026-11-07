"""Strict admitted79/profile parity; explicit private component or public entry."""
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
from inverse_gaussian_gradient_contract import recipes
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def target():
    import numpy as np
    path=os.environ.get('IGG_CORE_EXTENSION')
    if path:
        name='igg_component';package=types.ModuleType(name);package.__path__=[str(ROOT/'python/skimagexx')];sys.modules[name]=package
        spec=importlib.util.spec_from_file_location(name+'._core',path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);sys.modules[spec.name]=core
        module=importlib.import_module(name+'._inverse_gaussian_gradient')
    else:
        module=importlib.import_module('skimagexx.segmentation');core=importlib.import_module('skimagexx._core')
    assert inspect.isbuiltin(core.inverse_gaussian_magnitude)and inspect.isbuiltin(core.inverse_gaussian_finish)
    assert str(inspect.signature(module.inverse_gaussian_gradient))=='(image, alpha=100.0, sigma=5.0)'
    assert core.get_num_threads()==1
    core.set_num_threads(int(os.environ.get('IGG_TEST_WORKERS','1')))
    # Global donor reports hardware capability, independently of this leaf's
    # compile-time scalar mode (proved by flags/disassembly in its receipt).
    assert isinstance(core.has_avx2(),bool)
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD')=='1':assert not core.has_avx2()
    return np,module


@pytest.mark.parametrize('spec',recipes(),ids=lambda s:s['id'])
def test_saved_original(target,spec,monkeypatch):
    from inverse_gaussian_gradient_observer import observe
    from util_views_fenv import FEnv,backend
    np,module=target;profile='minimum'if np.__version__.split('.')[0]=='1'else'modern'
    saved=json.loads((ROOT/'tests/fixtures/inverse_gaussian_gradient_original_v1.json').read_bytes())['profiles'][profile]['rows']
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
        try:return module.inverse_gaussian_gradient(*args,**kwargs)
        finally:actual_state['after']=numerical(state())
    try:
        with monkeypatch.context()as scope:
            scope.setattr(builtins,'__import__',refused)
            actual=observe(call,spec,np)
        assert actual=={k:expected[k]for k in actual}
        assert actual_state=={when:{key:expected['target_immediate'][when][key]for key in('control','status','mxcsr')}for when in('before','after')}
    finally:
        restore(outer);assert bytes(state())==bytes(outer)


def test_finish_rejects_incompatible_stage_span_before_read(target):
    np,module=target;core=importlib.import_module(module.inverse_gaussian_gradient.__module__)._core
    a=np.ones(1,dtype=np.float32)
    with pytest.raises(ValueError,match='stage shape'):
        core.inverse_gaussian_finish(a,a,[100],1,1)
    with pytest.raises(ValueError,match='output broadcast extent'):
        core.inverse_gaussian_finish(a,a,[100],0,1)
    assert np.array_equal(a,[1.])


def test_one_global_snapshot_spans_public_stages(target,monkeypatch):
    np,module=target;implementation=importlib.import_module(module.inverse_gaussian_gradient.__module__);core=implementation._core
    before=core.get_num_threads();magnitude=core.inverse_gaussian_magnitude;finish=core.inverse_gaussian_finish;seen=[]
    def change(*args):
        value=magnitude(*args);core.set_num_threads(4 if before!=4 else 1);return value
    def capture(*args):seen.append(args[-1]);return finish(*args)
    try:
        with monkeypatch.context()as scope:
            scope.setattr(core,'inverse_gaussian_magnitude',change);scope.setattr(core,'inverse_gaussian_finish',capture)
            actual=implementation.inverse_gaussian_gradient(np.ones((2,3),dtype=np.float32),sigma=0)
        assert seen==[before]*4 and actual.shape==(2,3)
    finally:core.set_num_threads(before)


def test_numpy_sqrt_is_only_a_scalar_warning_probe(target,monkeypatch):
    np,module=target;original=np.sqrt;seen=[]
    def metadata_only(value,*args,**kwargs):
        assert np.asarray(value).size<=1,'Image sqrt provider forbidden'
        seen.append(True);return original(value,*args,**kwargs)
    with monkeypatch.context()as scope:
        scope.setattr(np,'sqrt',metadata_only)
        with pytest.warns(RuntimeWarning,match='invalid value encountered in sqrt'):
            actual=module.inverse_gaussian_gradient(np.ones((2,3),dtype=np.float32),alpha=-100.,sigma=0)
    assert seen==[True]and actual.shape==(2,3)


def test_coefficient_callback_cannot_resnapshot_worker_budget(target,monkeypatch):
    np,module=target;implementation=importlib.import_module(module.inverse_gaussian_gradient.__module__);core=implementation._core
    before=core.get_num_threads();coefficient=implementation._coefficients;magnitude=core.inverse_gaussian_magnitude;seen=[]
    def change(*args):
        value=coefficient(*args);core.set_num_threads(4 if before!=4 else 1);return value
    def capture(*args):seen.append(args[-1]);return magnitude(*args)
    try:
        with monkeypatch.context()as scope:
            scope.setattr(implementation,'_coefficients',change);scope.setattr(core,'inverse_gaussian_magnitude',capture)
            actual=implementation.inverse_gaussian_gradient(np.ones((2,3),dtype=np.float32),sigma=1)
        assert seen==[before]and actual.shape==(2,3)
    finally:core.set_num_threads(before)
