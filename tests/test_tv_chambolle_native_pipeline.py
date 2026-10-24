"""Private real-stage pipeline proof; never public namespace/SDK admission.

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


contracts=load('tv_private_frozen_reference',ROOT/'tests/test_tv_chambolle_contracts.py')
MATRIX=json.loads(contracts.MATRIX.read_text())
wrapper=load('skimagexx._tv_chambolle_private_source',ROOT/'python/skimagexx/_tv_chambolle.py')
BACKEND=os.environ.get('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE')


@pytest.fixture(scope='module')
def core():
    if not BACKEND:pytest.skip('PRIVATE TV STAGE PIPELINE PENDING: explicit immutable real compiled bridge required; no alternate/backend substituted')
    path=Path(BACKEND).resolve(strict=True)
    module=load('_tv_stage_core',path)
    assert module.tv_chambolle_interface=='tv-chambolle-stages-v1'
    assert type(module.tv_chambolle_capabilities) is int and module.tv_chambolle_capabilities==63
    for name in wrapper._ENTRIES:
        entry=getattr(module,name);assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==module.__name__
    assert module.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert module.gil_assertions is True
    return module


@pytest.fixture(params=(1,4))
def active(request,core,monkeypatch):
    before=core.get_num_threads();core.set_num_threads(request.param)
    monkeypatch.setattr(wrapper,'_core',core)
    try:yield wrapper
    finally:core.set_num_threads(before)


@pytest.fixture
def calls(active,monkeypatch):
    counts=Counter();dispatch=active._stage_call
    def counted(name,*arguments):
        entry=getattr(active._core,name)
        assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==active._core.__name__
        counts[name]+=1
        return dispatch(name,*arguments)
    monkeypatch.setattr(active,'_stage_call',counted)
    return counts


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
    assert hashlib.sha256((ROOT/'tests/test_tv_chambolle_contracts.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==MATRIX['test_source_canonical_lf_sha256']
    assert not any('skimage.restoration' in str(getattr(value,'__module__','')) for name,value in vars(wrapper).items() if callable(value))
    # Materialized stages alone do not create a public registered denoiser.
    assert MATRIX['native_checks']==0


@pytest.mark.parametrize('spec',contracts.CASES,ids=[p['id'] for p in contracts.CASES])
def test_all330_frozen_warn_provider_orchestration_not_native_usage(spec,active,calls):
    expected=MATRIX['observations'][MATRIX['profiles'][contracts.profile()]['snapshots'][spec['id']]]
    with no_forward():actual=contracts.observe(active.denoise_tv_chambolle,spec)
    assert contracts.result_nan_encoding_contract(actual)==contracts.result_nan_encoding_contract(expected),spec['id']
    assert not calls,'frozen observer sets underflow=warn: do not falsely count it as compiled pixel parity'


FINITE=tuple(p for p in contracts.CASES if p['id'].startswith(('base/','extremes/','layout/','channels/','channels4d/')) and p.get('layout')!='tagged')


@pytest.mark.parametrize('spec',FINITE,ids=[p['id'] for p in FINITE])
def test_finite_default_policy_projection_executes_real_stages(spec,active,calls):
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
def test_finite_vector_tails_reduction_tree_workthreshold_and_stopping(spec,active,calls):
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
