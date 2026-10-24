"""Separate default-policy Li reference supplement; existing312 remains immutable.

Forty IDs:38 potential numeric solver cases and2 warn-policy admission controls.
A native audit is findings, not implementation or public-acceptance evidence.
"""
import collections
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

ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'coverage/contracts/li-default-policy.json'
BASE_TEST=ROOT/'tests/test_iterative_threshold_contracts.py'
sp=importlib.util.spec_from_file_location('li_default_frozen_base',BASE_TEST)
BASE=importlib.util.module_from_spec(sp);sp.loader.exec_module(BASE)
DEFAULT_POLICY=dict(divide='warn',over='warn',under='ignore',invalid='warn')
BASE_TEST_SHA='63e7449e2458c3725bf8ea2b95ad84db94bd1a89bad03225b6a365750b577dd9'
BASE_MATRIX_SHA='34ea03c49663324c8169069884391fe3bacc311476a78dbe8999e6dbfbe3e95e'


def cases():
    rows=[]
    for s in BASE.cases():
        if s['function']!='threshold_li':continue
        ordinary=(s['id'].split('/')[-1].startswith('default-') and s['dtype'] in ('u1','f4','f8'))
        storage=s.get('layout')=='unaligned' and s['dtype'] in ('f4','f8')
        boundary=s.get('kind') in ('mixed-nan','finite-inf','full-u8','3D')
        if ordinary or storage or boundary:rows.append(dict(id='default/'+s['id'].split('/')[-1],base=s))
    assert len(rows)==24
    for dtype in ('u1','f4','f8'):
        for option in ('tolerance','initial'):
            rows.append(dict(id='numeric-option/'+dtype+'-'+option,base=dict(function='threshold_li',dtype=dtype,**{option:'explicit' if option=='tolerance' else 'scalar'})))
    for dtype in ('f4','f8'):
        for n in (127,128,8192,8193):rows.append(dict(id='reduction/'+dtype+'-'+str(n),dtype=dtype,count=n))
    for dtype in ('f4','f8'):rows.append(dict(id='warn-admission/'+dtype,base=dict(function='threshold_li',dtype=dtype),policy='warn'))
    assert len(rows)==40 and len({r['id'] for r in rows})==40
    return rows


def fixture(spec,events):
    if 'base' in spec:return BASE.fixture(spec['base'],events)
    n=spec['count'];assert n in (127,128,8192,8193)
    # Exact multiples of1/16; bounded positive histogram range and finite logs.
    image=((np.arange(n,dtype=np.int64)*37)%257).astype(spec['dtype'])/16
    return image,{}


def observe(spec,function=None):
    events=[];image,options=fixture(spec,events);before=BASE.value(image);options_before=BASE.value(options)
    old=np.geterr();policy=dict(DEFAULT_POLICY)
    if spec.get('policy')=='warn':policy['under']='warn'
    result=None;error=None
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(**policy):
            warnings.simplefilter('always');entered=np.geterr()
            try:result=(function or BASE.ORIGINAL['threshold_li'])(image,**options)
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
            policy_after_call=np.geterr()
        rows=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    finally:assert np.geterr()==old
    return dict(input_before=before,input_after=BASE.value(image),options_before=options_before,options_after=BASE.value(options),result=BASE.value(result),error=error,warnings=rows,events=events,policy_entered=entered,policy_after_call=policy_after_call,input_result_alias=bool(isinstance(result,np.ndarray) and isinstance(image,np.ndarray) and np.shares_memory(result,image)),NumPy_buffer_size=np.getbufsize())


def audit(spec,xx):
    """Use actual admitted core, never a prepared-object facade or upstream solver."""
    core=xx._core
    assert inspect.isbuiltin(core.li_prepare) and core.li_prepare.__module__==core.__name__
    assert core.li_prepare.__name__=='li_prepare'
    function=xx.filters.threshold_li;assert function.__module__=='skimagexx.filters'
    from skimagexx.coverage import native_functions
    assert 'filters.threshold_li' in native_functions()
    old_fallback=xx.set_fallback_enabled(False);assert xx.set_fallback_enabled(False) is False
    previous=sys.getprofile();counts=collections.Counter();attempts=[]
    original_codes={BASE.ORIGINAL['threshold_li'].__code__,BASE.SOURCE.histogram.__code__}
    def guard(frame,event,arg):
        if event=='call' and frame.f_code in original_codes:
            attempts.append(frame.f_code.co_name);raise AssertionError('upstream numerical forwarding forbidden')
        if event.startswith('c_'):
            owner=getattr(arg,'__module__',None);obj=getattr(arg,'__self__',None)
            if owner==core.__name__ or obj is not None and type(obj).__module__==core.__name__:
                counts[event+':'+getattr(arg,'__qualname__',getattr(arg,'__name__',type(arg).__name__))]+=1
    sys.setprofile(guard)
    try:row=observe(spec,function)
    finally:sys.setprofile(previous);xx.set_fallback_enabled(old_fallback)
    solver=sum(n for k,n in counts.items() if k.startswith('c_return:') and k.endswith('.run'))
    expected=json.loads(MATRIX.read_bytes())['profiles'][BASE.profile()]['observations'][spec['id']]
    unsupported=row['error'] is not None and row['error']['type']=='NotImplementedError'
    return dict(id=spec['id'],actual=row,expected=expected,public_parity=row==expected,CPP=dict(counts),solver_completed=solver,forward_attempts=attempts,classification='forwarding-rejected' if attempts else 'explicit-unsupported' if unsupported else 'mismatched' if row!=expected else 'matched/CPP-solver' if solver else 'matched/preparation-only')


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_original_default_policy_fixed_contract(spec):
    data=json.loads(MATRIX.read_bytes());actual=observe(spec)
    assert actual==data['profiles'][BASE.profile()]['observations'][spec['id']]
    assert actual['input_before']==actual['input_after']


def test_scope_source_signature_and_policy_are_frozen():
    data=json.loads(MATRIX.read_bytes())
    assert data['cases']==cases() and data['new_native_paths']==0
    assert data['numeric_candidates']==38 and data['warn_admission_controls']==2
    assert data['default_policy']==DEFAULT_POLICY and data['existing_API_status']=='native-partial'
    assert hashlib.sha256(BASE_TEST.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==BASE_TEST_SHA
    assert hashlib.sha256(BASE.MATRIX.read_bytes()).hexdigest()==BASE_MATRIX_SHA
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==data['test_canonical_lf_sha256']
    assert str(inspect.signature(BASE.ORIGINAL['threshold_li']))=='(image, *, tolerance=None, initial_guess=None, iter_callback=None)'
    assert data['source_safety']['image_workspace_bound_bytes']==8388608


@pytest.mark.parametrize('policy',('warn','ignore'))
def test_observer_restores_external_error_policy(policy):
    old=np.geterr()
    try:
        np.seterr(all=policy);before=np.geterr();observe(cases()[0]);assert np.geterr()==before
    finally:np.seterr(**old)
