"""Real installed TV pipeline, with an explicitly selected private proof mode.

The330 frozen observations deliberately set all FP modes to warn. Those prove
provider orchestration, not compiled pixel usage. A separately labeled default
underflow-ignore projection and finite stresses prove actual builtin calls.
"""
import ast
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import skimagexx as xx

ROOT=Path(__file__).parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


BACKEND=os.environ.get('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE')


def installed_admission():
    """Fail before fixtures for an incomplete registered public product."""
    from skimagexx.coverage import native_functions
    path='restoration.denoise_tv_chambolle'
    if path not in native_functions():
        pytest.skip('TV public pipeline pending registration; no private backend substituted',allow_module_level=True)
    from skimagexx import _core, _tv_chambolle
    from skimage.restoration import denoise_tv_chambolle as reference
    restoration=importlib.import_module('skimagexx.restoration')
    function=restoration.denoise_tv_chambolle
    assert restoration is xx.restoration
    assert function is _tv_chambolle.denoise_tv_chambolle
    assert function is not reference and inspect.isfunction(function)
    assert function.__module__==_tv_chambolle.__name__=='skimagexx._tv_chambolle'
    assert inspect.getmodule(function) is _tv_chambolle
    assert inspect.signature(function)==inspect.signature(reference)
    assert _tv_chambolle._core is _core
    package=Path(xx.__file__).resolve().parent
    assert Path(_tv_chambolle.__file__).resolve().parent==package
    assert Path(_core.__file__).resolve().parent==package
    assert xx.get_num_threads is _core.get_num_threads
    assert xx.set_num_threads is _core.set_num_threads
    assert _core.get_num_threads()==1
    _tv_chambolle._require_stages()
    _tv_chambolle._require_finite()
    return _tv_chambolle


wrapper=(load('skimagexx._tv_chambolle_private_source',ROOT/'python/skimagexx/_tv_chambolle.py')
         if BACKEND else installed_admission())
contracts=load('tv_private_frozen_reference',ROOT/'tests/test_tv_chambolle_contracts.py')
MATRIX=json.loads(contracts.MATRIX.read_text())


@pytest.fixture(scope='module')
def core():
    module=load('_tv_stage_core',Path(BACKEND).resolve(strict=True)) if BACKEND else wrapper._core
    assert module.tv_chambolle_interface=='tv-chambolle-stages-v1'
    assert type(module.tv_chambolle_capabilities) is int and module.tv_chambolle_capabilities==63
    for name in wrapper._ENTRIES:
        entry=getattr(module,name);assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==module.__name__
    assert module.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':
        assert getattr(module,'gil_assertions' if BACKEND else '_consumer_gil_assertions') is True
    return module


@pytest.fixture(params=(1,4))
def active(request,core,monkeypatch):
    before=core.get_num_threads()
    if request.param==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==before==1
        pytest.skip('Portable build explicitly rejects4; no serial worker4 parity is claimed')
    core.set_num_threads(request.param)
    if BACKEND:monkeypatch.setattr(wrapper,'_core',core)
    else:
        assert wrapper._core is core and xx.get_num_threads()==request.param
        assert xx.restoration.denoise_tv_chambolle is wrapper.denoise_tv_chambolle
    fallback=xx.set_fallback_enabled(False)
    try:yield wrapper
    finally:
        core.set_num_threads(before);xx.set_fallback_enabled(fallback)


@pytest.fixture
def calls(active,monkeypatch,request):
    counts=Counter();dispatch=active._stage_call
    def counted(name,*arguments):
        entry=getattr(active._core,name)
        assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==active._core.__name__
        counts[name]+=1
        return dispatch(name,*arguments)
    monkeypatch.setattr(active,'_stage_call',counted)
    yield counts
    request.node.user_properties.append(('actual_compiled_stage_calls',json.dumps(dict(counts),sort_keys=True)))
    request.node.user_properties.append(('actual_private_workers',str(active._core.get_num_threads())))
    request.node.user_properties.append(('actual_default',str('stage_compatibility' not in request.fixturenames)))


@pytest.fixture
def stage_compatibility(active,monkeypatch,request):
    """Explicit forced legacy coverage, never default finite-loop evidence."""
    monkeypatch.setattr(active,'_try_finite_loop',lambda *_:None)
    request.node.user_properties.append(('forced_legacy_stage_path','True'))


@pytest.fixture
def provider_loop_watch(active,monkeypatch,request):
    observed=dict(attempts=0,prewrite_rejections=0,executed=0);dispatch=active._finite_call
    def counted(*arguments):
        observed['attempts']+=1
        arrays=arguments[1:5];before=[v.tobytes() for v in arrays]
        try:result=dispatch(*arguments)
        except NotImplementedError:
            assert before==[v.tobytes() for v in arrays]
            observed['prewrite_rejections']+=1;raise
        observed['executed']+=1;return result
    monkeypatch.setattr(active,'_finite_call',counted)
    yield observed
    request.node.user_properties.append(('natural_provider_finite_status',json.dumps(observed,sort_keys=True)))


@contextmanager
def no_forward():
    original=contracts.reference;helper=contracts.original_module._denoise_tv_chambolle_nd
    forbidden={original.__code__,helper.__code__};previous=sys.getprofile()
    def guard(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:raise AssertionError('upstream TV algorithm/helper forwarding')
        if previous is not None:previous(frame,event,arg)
    try:sys.setprofile(guard);yield
    finally:sys.setprofile(previous)


def default_policy_observe(function,spec):
    # Literal original observer derivation, changing ONLY the explicit policy
    # configuration. Frozen330 observer/matrix and assertions are not edited.
    text=inspect.getsource(contracts.observe)
    old="np.seterr(all=spec.get('policy','warn'))"
    new="np.seterr(divide='warn',over='warn',under='ignore',invalid='warn')"
    assert text.count(old)==1
    original=ast.parse(text);changed=ast.parse(text.replace(old,new))
    restored=ast.parse(ast.unparse(changed).replace("np.seterr(divide='warn', over='warn', under='ignore', invalid='warn')",old))
    assert ast.dump(original)==ast.dump(restored)
    namespace=dict(contracts.__dict__);exec(compile(changed,'<tv-default-policy-observer>','exec'),namespace)
    return namespace['observe'](function,spec)


def test_signature_frozen_source_and_explicit_pending_scope():
    assert str(inspect.signature(wrapper.denoise_tv_chambolle))==MATRIX['signature']
    assert len(contracts.CASES)==330 and contracts.digest(contracts.CASES)==MATRIX['case_contract_sha256']
    source_sha=hashlib.sha256((ROOT/'tests/test_tv_chambolle_contracts.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    migration_path=ROOT/'coverage/evidence/tv-chambolle-public-assertion-migration.json'
    if not BACKEND:assert migration_path.is_file(),'public assertion transition needs its authenticated source record'
    if migration_path.is_file():
        migration=json.loads(migration_path.read_bytes())
        assert migration['schema']=='tv-chambolle-public-assertion-migration-v1'
        assert migration['source_file']=='tests/test_tv_chambolle_contracts.py'
        assert migration['original_source_sha256']==MATRIX['test_source_canonical_lf_sha256']
        assert migration['reference_matrix_sha256']==hashlib.sha256(contracts.MATRIX.read_bytes()).hexdigest()
        assert source_sha==migration['public_source_sha256']
    else:assert source_sha==MATRIX['test_source_canonical_lf_sha256']
    assert not any('skimage.restoration' in str(getattr(value,'__module__','')) for name,value in vars(wrapper).items() if callable(value))
    # Materialized stages alone do not create a public registered denoiser.
    assert MATRIX['native_checks']==0


@pytest.mark.parametrize('spec',contracts.CASES,ids=[p['id'] for p in contracts.CASES])
def test_all330_frozen_warn_provider_orchestration_not_native_usage(spec,active,calls,provider_loop_watch):
    expected=MATRIX['observations'][MATRIX['profiles'][contracts.profile()]['snapshots'][spec['id']]]
    with no_forward():actual=contracts.observe(active.denoise_tv_chambolle,spec)
    assert contracts.result_nan_encoding_contract(actual)==contracts.result_nan_encoding_contract(expected),spec['id']
    assert not calls,'frozen observer sets underflow=warn: do not falsely count it as compiled pixel parity'
    assert provider_loop_watch['executed']==0,'frozen330 naturally retain their provider path, not a forced compatibility override'
    expected_attempt=(spec.get('policy')=='ignore' and spec.get('pattern') in ('nonfinite','large')
                      and hasattr(active._core,'tv_chambolle_finite_interface'))
    assert provider_loop_watch['attempts']==provider_loop_watch['prewrite_rejections']==int(expected_attempt)


FINITE=tuple(p for p in contracts.CASES if p['id'].startswith(('base/','extremes/','layout/','channels/','channels4d/')) and p.get('layout')!='tagged')


@pytest.mark.parametrize('spec',FINITE,ids=[p['id'] for p in FINITE])
def test_finite_default_policy_projection_executes_real_stages(spec,active,calls,stage_compatibility):
    expected=default_policy_observe(contracts.reference,spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected,spec['id']
    assert calls['tv_chambolle_gradient_into']>=1 and calls['tv_chambolle_norm_into']>=1 and calls['tv_chambolle_dual_into']>=1
    if spec['iterations']>1:assert calls['tv_chambolle_divergence_into']>=1


def stress_cases():
    result=[]
    for shape in ((1,),(2,),(15,),(16,),(17,),(31,),(32,),(33,),(3,17),(5,7,9),(2,3,5,7),(257,259),(513,519)):
        for dtype in ('f4','f8'):
            for count in (1,2,8):
                result.append(dict(id=f'stress/{shape}/{dtype}/i{count}',dtype=dtype,shape=list(shape),iterations=count,weight=0.5,eps=0.))
    for dtype in ('f4','f8'):
        for pattern in ('constant','ramp'):
            result.append(dict(id=f'convergence/{dtype}/{pattern}',dtype=dtype,shape=[129,131],iterations=200,pattern=pattern,weight=0.5,eps=0.0002))
    assert len(result)==82
    return tuple(result)


STRESS=stress_cases()


@pytest.mark.parametrize('spec',STRESS,ids=[p['id'] for p in STRESS])
def test_finite_vector_tails_reduction_tree_workthreshold_and_stopping(spec,active,calls,stage_compatibility):
    expected=default_policy_observe(contracts.reference,spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected,spec['id']
    assert calls['tv_chambolle_gradient_into']>=1 and calls['tv_chambolle_norm_into']>=1 and calls['tv_chambolle_dual_into']>=1


@pytest.mark.parametrize('fault',('missing','version','boolcap','floatcap','entry','borrowed'))
def test_stage_guard_precedes_every_image_metadata_access(monkeypatch,fault):
    from types import SimpleNamespace
    import math
    core=SimpleNamespace(__name__='math',tv_chambolle_interface='tv-chambolle-stages-v1',tv_chambolle_capabilities=63)
    for name in wrapper._ENTRIES:setattr(core,name,math.sin)
    if fault=='missing':del core.tv_chambolle_capabilities
    elif fault=='version':core.tv_chambolle_interface='wrong'
    elif fault=='boolcap':core.tv_chambolle_capabilities=True
    elif fault=='floatcap':core.tv_chambolle_capabilities=63.
    elif fault=='entry':setattr(core,wrapper._ENTRIES[0],lambda *_:None)
    class Image:
        @property
        def dtype(self):raise AssertionError('image metadata reached before stage rejection')
    monkeypatch.setattr(wrapper,'_core',core)
    with pytest.raises(RuntimeError,match='compiled TV|owned compiled'):wrapper.denoise_tv_chambolle(Image())


# Literal70candidate fixtures from immutable benchmark6304; no timers imported.
BENCHMARK_FIXTURES=({'id': 'tiny/u1/one', 'shape': [3, 5], 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'tiny'}, {'id': 'tiny/u1/two', 'shape': [3, 5], 'dtype': 'u1', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 2, 'cohort': 'tiny'}, {'id': 'tiny/u1/fixed8', 'shape': [3, 5], 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'tiny'}, {'id': 'tiny/u1/converge', 'shape': [3, 5], 'dtype': 'u1', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0002, 'iterations': 200, 'cohort': 'tiny'}, {'id': 'tiny/f4/one', 'shape': [3, 5], 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'tiny'}, {'id': 'tiny/f4/two', 'shape': [3, 5], 'dtype': 'f4', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'tiny'}, {'id': 'tiny/f4/fixed8', 'shape': [3, 5], 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'tiny'}, {'id': 'tiny/f4/converge', 'shape': [3, 5], 'dtype': 'f4', 'layout': 'readonly', 'pattern': 'constant', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'tiny'}, {'id': 'tiny/f8/one', 'shape': [3, 5], 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'tiny'}, {'id': 'tiny/f8/two', 'shape': [3, 5], 'dtype': 'f8', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'tiny'}, {'id': 'tiny/f8/fixed8', 'shape': [3, 5], 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'tiny'}, {'id': 'tiny/f8/converge', 'shape': [3, 5], 'dtype': 'f8', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'tiny'}, {'id': 'small/u1/one', 'shape': [33, 35], 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'small'}, {'id': 'small/u1/two', 'shape': [33, 35], 'dtype': 'u1', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 2, 'cohort': 'small'}, {'id': 'small/u1/fixed8', 'shape': [33, 35], 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'small'}, {'id': 'small/u1/converge', 'shape': [33, 35], 'dtype': 'u1', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0002, 'iterations': 200, 'cohort': 'small'}, {'id': 'small/f4/one', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'small'}, {'id': 'small/f4/two', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'small'}, {'id': 'small/f4/fixed8', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'small'}, {'id': 'small/f4/converge', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'readonly', 'pattern': 'constant', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'small'}, {'id': 'small/f8/one', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'small'}, {'id': 'small/f8/two', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'small'}, {'id': 'small/f8/fixed8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'small'}, {'id': 'small/f8/converge', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'small'}, {'id': 'raster/u1/one', 'shape': [257, 259], 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'raster'}, {'id': 'raster/u1/two', 'shape': [257, 259], 'dtype': 'u1', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 2, 'cohort': 'raster'}, {'id': 'raster/u1/fixed8', 'shape': [257, 259], 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'raster'}, {'id': 'raster/u1/converge', 'shape': [257, 259], 'dtype': 'u1', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0002, 'iterations': 200, 'cohort': 'raster'}, {'id': 'raster/f4/one', 'shape': [257, 259], 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'raster'}, {'id': 'raster/f4/two', 'shape': [257, 259], 'dtype': 'f4', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'raster'}, {'id': 'raster/f4/fixed8', 'shape': [257, 259], 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'raster'}, {'id': 'raster/f4/converge', 'shape': [257, 259], 'dtype': 'f4', 'layout': 'readonly', 'pattern': 'constant', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'raster'}, {'id': 'raster/f8/one', 'shape': [257, 259], 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'raster'}, {'id': 'raster/f8/two', 'shape': [257, 259], 'dtype': 'f8', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'raster'}, {'id': 'raster/f8/fixed8', 'shape': [257, 259], 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'raster'}, {'id': 'raster/f8/converge', 'shape': [257, 259], 'dtype': 'f8', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'raster'}, {'id': 'volume/u1/one', 'shape': [9, 33, 35], 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'volume'}, {'id': 'volume/u1/two', 'shape': [9, 33, 35], 'dtype': 'u1', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 2, 'cohort': 'volume'}, {'id': 'volume/u1/fixed8', 'shape': [9, 33, 35], 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'volume'}, {'id': 'volume/u1/converge', 'shape': [9, 33, 35], 'dtype': 'u1', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0002, 'iterations': 200, 'cohort': 'volume'}, {'id': 'volume/f4/one', 'shape': [9, 33, 35], 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'volume'}, {'id': 'volume/f4/two', 'shape': [9, 33, 35], 'dtype': 'f4', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'volume'}, {'id': 'volume/f4/fixed8', 'shape': [9, 33, 35], 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'volume'}, {'id': 'volume/f4/converge', 'shape': [9, 33, 35], 'dtype': 'f4', 'layout': 'readonly', 'pattern': 'constant', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'volume'}, {'id': 'volume/f8/one', 'shape': [9, 33, 35], 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 1, 'cohort': 'volume'}, {'id': 'volume/f8/two', 'shape': [9, 33, 35], 'dtype': 'f8', 'layout': 'F', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 2, 'cohort': 'volume'}, {'id': 'volume/f8/fixed8', 'shape': [9, 33, 35], 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'volume'}, {'id': 'volume/f8/converge', 'shape': [9, 33, 35], 'dtype': 'f8', 'layout': 'readonly', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0002, 'iterations': 200, 'cohort': 'volume'}, {'id': 'channel/0/u1/i1', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/0/u1/i8', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'channel/0/f4/i1', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/0/f4/i8', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'channel/0/f8/i1', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/0/f8/i8', 'shape': [3, 51, 53], 'axis': 0, 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'channel/-1/u1/i1', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'u1', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/-1/u1/i8', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'u1', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'channel/-1/f4/i1', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'f4', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/-1/f4/i8', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'f4', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'channel/-1/f8/i1', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'f8', 'layout': 'C', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 1, 'cohort': 'channels'}, {'id': 'channel/-1/f8/i8', 'shape': [51, 53, 3], 'axis': -1, 'dtype': 'f8', 'layout': 'step', 'pattern': 'ramp', 'weight': 0.1, 'eps': 0.0, 'iterations': 8, 'cohort': 'channels'}, {'id': 'layout/reverse/f4', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'reverse', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/reverse/f8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'reverse', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/broadcast/f4', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'broadcast', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/broadcast/f8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'broadcast', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/endian/f4', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'endian', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/endian/f8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'endian', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/unaligned/f4', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'unaligned', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/unaligned/f8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'unaligned', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/tagged/f4', 'shape': [33, 35], 'dtype': 'f4', 'layout': 'tagged', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'}, {'id': 'layout/tagged/f8', 'shape': [33, 35], 'dtype': 'f8', 'layout': 'tagged', 'pattern': 'ramp', 'weight': 0.5, 'eps': 0.0, 'iterations': 8, 'cohort': 'layout'})
BENCHMARK_FIXTURE_SHA256='c16fa4e2af58a5d385cef6c6b6c2c28d9f3b9e502f0d6094be4420cda17b8e79'


@pytest.mark.parametrize('spec',BENCHMARK_FIXTURES,ids=[p['id'] for p in BENCHMARK_FIXTURES])
def test_all70_frozen_benchmark_finite_candidates_with_real_stage_calls(spec,active,calls,stage_compatibility):
    assert contracts.digest(list(BENCHMARK_FIXTURES))==BENCHMARK_FIXTURE_SHA256
    expected=default_policy_observe(contracts.reference,spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected,spec['id']
    assert calls['tv_chambolle_norm_into'] and calls['tv_chambolle_dual_into']
    if spec.get('layout')=='tagged':assert calls['tv_chambolle_gradient_into']==0
    else:assert calls['tv_chambolle_gradient_into']>0
    if spec['iterations']>1:assert calls['tv_chambolle_divergence_into']>0


def test_actual_private_worker_policy(core):
    assert core.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==1
    else:
        try:core.set_num_threads(4);assert core.get_num_threads()==4
        finally:core.set_num_threads(1)


@pytest.fixture
def finite_calls(active,monkeypatch,request):
    if not hasattr(active._core,'tv_chambolle_finite_interface'):
        pytest.skip('PRIVATE finite-loop binding pending: stage63 alone is not a whole-loop implementation')
    function=active._require_finite();assert inspect.isbuiltin(function)
    records=[];rejections=[]
    def traced(entry,*arguments):
        assert entry is function and inspect.isbuiltin(entry) and entry.__name__=='tv_chambolle_finite_run' and entry.__module__==active._core.__name__
        image,p,g,d,weight,eps,maximum,profile,buffer=arguments
        before=[(v.dtype.str,v.shape,v.strides,v.tobytes()) for v in (image,p,g,d)]
        try:result=entry(*arguments,True)
        except NotImplementedError as error:
            assert before==[(v.dtype.str,v.shape,v.strides,v.tobytes()) for v in (image,p,g,d)]
            rejections.append(str(error));raise
        out,iterations,alias,trace=result
        assert type(iterations) is int and 1<=iterations<=maximum
        assert type(alias) is bool and alias==(iterations==1)
        assert len(trace)==iterations and all(len(row)==7 for row in trace)
        assert (out is image)==alias
        if not alias:assert out.flags.owndata and out.flags.c_contiguous and not np.shares_memory(out,image)
        records.append(dict(iterations=iterations,input_alias=alias,trace=[list(row) for row in trace],trace_instrumentation=True,profile=profile,buffer_elements=buffer,dtype=image.dtype.str,shape=list(image.shape)))
        return result
    monkeypatch.setattr(active,'_finite_call',traced)
    yield records,rejections
    request.node.user_properties.append(('actual_default','True'))
    request.node.user_properties.append(('actual_private_workers',str(active._core.get_num_threads())))
    request.node.user_properties.append(('actual_compiled_finite_loops',json.dumps(records,sort_keys=True)))
    request.node.user_properties.append(('prewrite_finite_rejections',json.dumps(rejections)))


def expected_body_counts(spec):
    """Inert original-only comparison provider counts actual loop bodies."""
    passed,image,owner,options,events=contracts.fixture(spec)
    maximum=options.get('max_num_iter',200);seen=[]
    class Count:
        def __gt__(self,index):
            keep=index<maximum
            if keep:seen.append(index)
            return keep
    options['max_num_iter']=Count()
    with np.errstate(divide='warn',over='warn',under='ignore',invalid='warn'):
        contracts.reference(passed,**options)
    assert not events
    # Original channels execute complete spatial loops in channel order. A
    # repeated zero comparison marks the next channel's first update body.
    counts=[]
    for index in seen:
        if index==0:counts.append(0)
        counts[-1]+=1
    return counts


DEFAULT_FINITE=tuple(FINITE)+tuple(BENCHMARK_FIXTURES)+tuple(STRESS)


@pytest.mark.parametrize('spec',DEFAULT_FINITE,ids=[p['id'] for p in DEFAULT_FINITE])
def test_actual_default_finite_loop_once_per_spatial_channel(spec,active,calls,finite_calls):
    expected=default_policy_observe(contracts.reference,spec)
    bodies=expected_body_counts(spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected,spec['id']
    loops,rejections=finite_calls
    assert not rejections
    if spec.get('layout')=='tagged':
        assert not loops and calls['tv_chambolle_norm_into'] and calls['tv_chambolle_dual_into']
        assert calls['tv_chambolle_gradient_into']==0
    else:
        assert len(loops)==len(bodies) and [row['iterations'] for row in loops]==bodies
        assert not calls,'one real finite builtin replaces all Python-dispatched stage calls'
        assert [row['input_alias'] for row in loops]==[n==1 for n in bodies]


def scalar_bits(value,dtype):
    return int(np.asarray(value,dtype=dtype).view(np.uint32 if np.dtype(dtype).itemsize==4 else np.uint64))


def trace_definition(image,weight,eps,maximum):
    """Materialized NumPy definition, including original scalar expression tree."""
    ndim=image.ndim;p=np.zeros((ndim,)+image.shape,dtype=image.dtype);g=np.zeros_like(p);d=np.zeros_like(image);rows=[]
    for index in range(maximum):
        if index:
            d=-p.sum(0)
            for axis in range(ndim):
                target=[slice(None)]*ndim;previous=[slice(None)]*(ndim+1)
                target[axis]=slice(1,None);previous[0]=axis;previous[axis+1]=slice(0,-1)
                d[tuple(target)]+=p[tuple(previous)]
            out=image+d
        else:out=image
        square=(d**2).sum()
        for axis in range(ndim):
            target=[slice(None)]*(ndim+1);target[0]=axis;target[axis+1]=slice(0,-1)
            g[tuple(target)]=np.diff(out,axis=axis)
        norm=np.sqrt((g**2).sum(axis=0))[np.newaxis,...];norm_sum=norm.sum()
        energy=square;energy+=weight*norm_sum
        tau=1./(2.*ndim);norm*=tau/weight;norm+=1.;p-=tau*g;p/=norm
        energy/=float(image.size)
        energy_dtype=np.float32 if type(energy) is np.float32 else np.float64
        if index:
            difference=np.abs(previous_energy-energy);threshold=eps*initial;stopped=bool(difference<threshold)
        else:initial=energy;difference=energy_dtype(0.);threshold=energy_dtype(0.);stopped=False
        rows.append([scalar_bits(square,image.dtype),scalar_bits(norm_sum,image.dtype),scalar_bits(energy,energy_dtype),scalar_bits(difference,energy_dtype),scalar_bits(threshold,energy_dtype),energy_dtype is np.float32,stopped])
        if stopped:break
        previous_energy=energy
    return out,rows


@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('shape',((2,),(2,3),(2,3,5),(2,2,3,5),(1,4097),(65,129)))
@pytest.mark.parametrize('buffer',(1024,8192,16384))
def test_actual_finite_raw_energy_stop_trace_and_sum_buffer(dtype,shape,buffer,active,calls,finite_calls):
    old=np.getbufsize();np.setbufsize(buffer)
    try:
        spec=dict(id='trace',dtype=dtype,shape=list(shape),iterations=8,weight=0.5,eps=0.)
        image=contracts.fixture(spec)[1]
        with np.errstate(divide='warn',over='warn',under='ignore',invalid='warn'):
            expected,trace=trace_definition(image,0.5,0.,8)
            with no_forward():actual=active.denoise_tv_chambolle(image,weight=0.5,eps=0.,max_num_iter=8)
        loops,rejections=finite_calls;assert not rejections and len(loops)==1 and not calls
        assert actual.tobytes()==expected.tobytes() and actual.dtype==expected.dtype
        assert loops[0]['trace']==trace and loops[0]['buffer_elements']==buffer
    finally:np.setbufsize(old)


@pytest.mark.parametrize('fault',('missing','version','boolcap','floatcap','profile','entry','borrowed'))
def test_finite_interface_is_distinct_and_fail_closed(monkeypatch,fault):
    from types import SimpleNamespace
    import math
    core=SimpleNamespace(__name__='math',tv_chambolle_finite_interface='tv-chambolle-finite-v1',tv_chambolle_finite_capabilities=3,tv_chambolle_finite_profiles=('1.24.4','2.5.4'),tv_chambolle_finite_run=math.sin)
    if fault=='missing':del core.tv_chambolle_finite_capabilities
    elif fault=='version':core.tv_chambolle_finite_interface='tv-chambolle-stages-v1'
    elif fault=='boolcap':core.tv_chambolle_finite_capabilities=True
    elif fault=='floatcap':core.tv_chambolle_finite_capabilities=3.
    elif fault=='profile':core.tv_chambolle_finite_profiles=('2.5.4',)
    elif fault=='entry':core.tv_chambolle_finite_run=lambda *_:None
    monkeypatch.setattr(wrapper,'_core',core)
    with pytest.raises(RuntimeError,match='stage63 is not loop admission'):wrapper._require_finite()


@pytest.mark.parametrize('provider',wrapper._ALLOCATIONS)
def test_monkeypatched_allocation_provider_cannot_enter_finite_loop(monkeypatch,provider):
    original=getattr(np,provider)
    def hook(*args,**kwargs):return original(*args,**kwargs)
    monkeypatch.setattr(np,provider,hook)
    assert wrapper._providers_pristine() is False


@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('kind',('high','nonfinite'))
def test_native_input_rejection_is_prewrite_then_exact_owned_provider(kind,dtype,active,calls,finite_calls):
    spec=dict(id='guard',dtype=dtype,shape=[3,4],iterations=2,weight=0.5,eps=0.)
    original_fixture=contracts.fixture
    def values(spec):
        result=original_fixture(spec)
        if kind=='high':result[1].fill(2.**21)
        else:result[1].flat[0]=np.inf
        return result
    with __import__('unittest.mock',fromlist=['patch']).patch.object(contracts,'fixture',values):
        expected=default_policy_observe(contracts.reference,spec)
        with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert contracts.result_nan_encoding_contract(actual)==contracts.result_nan_encoding_contract(expected)
    loops,rejections=finite_calls;assert not loops and len(rejections)==1


@pytest.mark.parametrize('field',('p','g','d'))
def test_direct_finite_positive_zero_guard_rejects_without_workspace_writes(field,active,finite_calls):
    image=np.arange(12,dtype=np.float32).reshape(3,4)/8
    arrays=dict(p=np.zeros((2,3,4),np.float32),g=np.zeros((2,3,4),np.float32),d=np.zeros((3,4),np.float32))
    arrays[field].flat[0]=-0.0
    before={k:v.tobytes() for k,v in arrays.items()}
    function=active._require_finite()
    with pytest.raises(NotImplementedError):
        active._finite_call(function,image,arrays['p'],arrays['g'],arrays['d'],0.5,0.,2,np.__version__,np.getbufsize())
    assert before=={k:v.tobytes() for k,v in arrays.items()}
    assert not finite_calls[0] and len(finite_calls[1])==1


@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('maximum',(1,2,8))
def test_actual_default_without_trace_matches_trace_and_owned_return(dtype,maximum,active,calls,monkeypatch,request):
    if not hasattr(active._core,'tv_chambolle_finite_interface'):pytest.skip('Private finite binding is pending')
    entry=active._require_finite();count=[];dispatch=active._finite_call
    def counted(function,*args):
        result=dispatch(function,*args)
        assert result[3]==[]  # Production default disables trace allocation/stores.
        count.append((result[1],result[2]));return result
    monkeypatch.setattr(active,'_finite_call',counted)
    spec=dict(id='no-trace',dtype=dtype,shape=[17,19],iterations=maximum,weight=0.5,eps=0.)
    expected=default_policy_observe(contracts.reference,spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected and count==[(maximum,maximum==1)] and not calls
    image=contracts.fixture(spec)[1];p=np.zeros((2,17,19),dtype);g=np.zeros_like(p);d=np.zeros_like(image)
    result,iterations,alias,trace=entry(image,p,g,d,0.5,0.,maximum,np.__version__,np.getbufsize(),True)
    assert iterations==maximum and alias==(maximum==1) and len(trace)==maximum
    assert contracts.array(result)==actual['result']['array']
    request.node.user_properties.append(('actual_default_without_trace','True'))
    request.node.user_properties.append(('actual_compiled_finite_call_count','1'))


DEFAULT_PROTOCOL=tuple(p for p in contracts.CASES if p['id'].startswith(('parameter/','intake/','allocation/','allocation-zeroiter/')))


@pytest.mark.parametrize('spec',DEFAULT_PROTOCOL,ids=[p['id'] for p in DEFAULT_PROTOCOL])
def test_default_policy_custom_protocols_preserve_order_without_loop(spec,active,calls,finite_calls):
    expected=default_policy_observe(contracts.reference,spec)
    with no_forward():actual=default_policy_observe(active.denoise_tv_chambolle,spec)
    assert actual==expected,spec['id']
    assert not finite_calls[0] and not finite_calls[1]


def test_pristine_provider_latch_survives_a_self_restoring_allocation_hook(active,calls,finite_calls,monkeypatch):
    original=np.zeros
    spec=dict(id='restoring',dtype='f8',shape=[3,4],iterations=2,weight=0.5,eps=0.)
    def run(function):
        events=[]
        def hook(*args,**kwargs):
            events.append('zeros');monkeypatch.setattr(np,'zeros',original);return original(*args,**kwargs)
        monkeypatch.setattr(np,'zeros',hook)
        return default_policy_observe(function,spec),events
    expected,expected_events=run(contracts.reference)
    with no_forward():actual,actual_events=run(active.denoise_tv_chambolle)
    assert actual==expected and actual_events==expected_events==['zeros']
    assert not finite_calls[0] and not finite_calls[1]


@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('axis',(None,-1))
@pytest.mark.parametrize('mode',('raise','record'))
def test_observed_empty_provider_edge_preserves_original_call_order(dtype,axis,mode,active,calls,provider_loop_watch,monkeypatch):
    original=np.empty;spec=dict(id='empty-provider',dtype=dtype,shape=[3,4,2],iterations=2,weight=0.5,eps=0.,axis=axis)
    class EmptyProviderError(RuntimeError):pass
    def run(function):
        events=[]
        def hook(*args,**kwargs):
            events.append('np.empty')
            if mode=='raise':raise EmptyProviderError('unexpected-python-empty-provider')
            return original(*args,**kwargs)
        monkeypatch.setattr(np,'empty',hook)
        return default_policy_observe(function,spec),events
    expected,expected_events=run(contracts.reference)
    with no_forward():actual,actual_events=run(active.denoise_tv_chambolle)
    assert actual==expected and actual_events==expected_events==[]
    assert provider_loop_watch['attempts']==0 and calls['tv_chambolle_norm_into']==0
    assert calls['tv_chambolle_gradient_into'] and calls['tv_chambolle_dual_into']
