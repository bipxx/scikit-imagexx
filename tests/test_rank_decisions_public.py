"""R7 own-wrapper prototype; genuine public admission remains independently gated."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import rank as reference
from skimagexx.coverage import native_functions


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ROOT=Path(__file__).parents[1]
contracts=load('rank_decisions_wrapper_contracts',Path(__file__).with_name('test_rank_decisions_contracts.py'))
MATRIX=json.loads(contracts.MATRIX.read_text())
PUBLIC=json.loads((ROOT/'coverage/contracts/rank-decisions-public.json').read_text())
PROTOTYPE=os.environ.get('SKIMAGEXX_RANK_DECISIONS_PROTOTYPE_CORE')
source_wrapper=load('skimagexx._rank_decisions_source_prototype',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source_wrapper._core=importlib.import_module(PROTOTYPE)
NAMES=contracts.NAMES
PATHS={prefix+name for prefix in ('filters.rank.','filters.rank.generic.') for name in NAMES}


@pytest.fixture
def target():
    if PROTOTYPE:return source_wrapper
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Genuine R7 public namespace/core/package gate pending; no alternate backend substituted')
    assert PATHS<=registered,'partial canonical/generic decisions registration is not public acceptance'
    return importlib.import_module('skimagexx._rank')


@pytest.fixture(params=(1,4))
def active(request,target):
    policy=xx.set_fallback_enabled(False);core=target._core;previous=core.get_num_threads()
    core.set_num_threads(request.param)
    try:yield target
    finally:core.set_num_threads(previous);xx.set_fallback_enabled(policy)


def block_original(monkeypatch,module):
    def forbidden(*args,**options):raise AssertionError('upstream local decision/fallback forwarding')
    monkeypatch.setattr(module,'fallback',forbidden)
    generic=importlib.import_module('skimage.filters.rank.generic')
    for provider in (reference,generic):
        for name in NAMES:monkeypatch.setattr(provider,name,forbidden)
    for name in ('_otsu','_otsu_3D','_majority','_majority_3D'):monkeypatch.setattr(generic.generic_cy,name,forbidden)


@pytest.fixture
def no_forwarding(monkeypatch,active):block_original(monkeypatch,active)


def test_source_signatures_old46_nodes_exports_and_compiled_prototype():
    before=ast.parse(PUBLIC['old_module_source']);after=ast.parse((ROOT/'python/skimagexx/_rank.py').read_text())
    assert len(before.body)==PUBLIC['old_top_level_node_count']==66
    assert len([n for n in before.body if isinstance(n,ast.FunctionDef)])==PUBLIC['old_function_count']==46
    prefix=ast.Module(body=after.body[:len(before.body)],type_ignores=[])
    assert ast.dump(prefix,include_attributes=False)==ast.dump(before,include_attributes=False)
    assert hashlib.sha256(PUBLIC['old_module_source'].encode()).hexdigest()==PUBLIC['old_source_canonical_lf_sha256']
    assert len(source_wrapper._GENERIC_FUNCTIONS)==17 and len(source_wrapper.__all__)==29
    assert source_wrapper.__all__==list(source_wrapper._GENERIC_FUNCTIONS+source_wrapper._PERCENTILE_FUNCTIONS+source_wrapper._BILATERAL_FUNCTIONS)
    assert not set(NAMES)&set(source_wrapper.__all__)
    for name in NAMES:assert inspect.signature(getattr(source_wrapper,name))==inspect.signature(getattr(reference,name))
    if PROTOTYPE:
        core=source_wrapper._core
        assert hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() in set(PUBLIC['private_module_sha256'].values())
        assert core.rank_decisions_interface=='rank-decisions-v1' and core.rank_decisions_capabilities==0x60000000
        assert core.get_num_threads()==1
        if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True


def test_actual_public_namespace_or_visible_pending():
    if PROTOTYPE:pytest.skip('DIRECT PRIVATE-CORE WRAPPER PROTOTYPE: actual public/sole-core/SDK admission pending')
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Root canonical2/generic2 decisions wiring and genuine complete product remain pending')
    assert PATHS<=registered
    rank=importlib.import_module('skimagexx.filters.rank');generic=importlib.import_module('skimagexx.filters.rank.generic');wrapper=importlib.import_module('skimagexx._rank')
    assert wrapper._core is xx._core and xx.set_num_threads is xx._core.set_num_threads and xx.get_num_threads is xx._core.get_num_threads
    assert xx.get_num_threads()==1
    for name in NAMES:assert getattr(rank,name) is getattr(generic,name) is getattr(wrapper,name)
    assert set(NAMES)<=set(rank.__all__) and 'otsu' in generic.__all__ and 'majority' not in generic.__all__


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_all502_frozen_source_observations(case,active,no_forwarding):
    key,name,parameters=case
    assert contracts.observe(getattr(active,name),name,parameters)==MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]],key


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('failure',contracts.PROVIDERS)
def test_all32_provider_traces(name,ndim,failure,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    assert contracts.provider_observe(name,ndim,failure)==MATRIX['provider_traces'][contracts.profile()][f'{name}/{ndim}/{failure}']


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','empty','masked','shifted','tied','full-bin'))
@pytest.mark.parametrize('output',('same','u1','f4','f8'))
def test_all256_independent_neighborhood_oracles(name,ndim,dtype,kind,output,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_independent_neighborhood_decision_replay(name,ndim,dtype,kind,output)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('action',('image-fill','mask-fill','footprint-fill','image-dtype','out-dtype'))
@pytest.mark.parametrize('field',('shift_x','shift_y'))
def test_defined_shift_mutations(name,ndim,action,field,active,monkeypatch):
    expected=shift_mutation_observe(getattr(reference,name),ndim,action,field)
    assert expected['error'] is None
    block_original(monkeypatch,active)
    assert shift_mutation_observe(getattr(active,name),ndim,action,field)==expected


SHIFT_PROTOCOLS=('index-only','int-only','float-only','both','int-subclass','float-subclass','trunc-only','int','float','nan','inf','none','numeric-string','np-i64','np-f64','array0','array1','np-bool')


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('token',SHIFT_PROTOCOLS)
def test_source_specific_signed_char_providers(name,token,active,monkeypatch):
    expected=shift_protocol_observe(getattr(reference,name),token)
    block_original(monkeypatch,active)
    assert shift_protocol_observe(getattr(active,name),token)==expected


def test_literal_observers_authenticate_same_runtime_ast():
    nodes={n.name:n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name,literal in PUBLIC['literal_observers'].items():
        assert ast.dump(nodes[name],include_attributes=False)==ast.dump(ast.parse(literal).body[0],include_attributes=False)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('unsafe',('output-short','zero-depth','mask-short','feedback','escaping-bin'))
@pytest.mark.parametrize('fallback_enabled',(False,True))
def test_explicit_unsupported_scopes_never_forward_or_write(name,unsafe,fallback_enabled,active,no_forwarding):
    # Native certificates only: NEVER send invalid Cython buffers upstream.
    image=(np.arange(12)%4).astype(np.uint16).reshape(3,4);fp=np.ones((3,3),np.uint8)
    out=np.full(image.shape,99,np.uint8);mask=None;options={}
    if unsafe=='output-short':out=np.full((1,4),99,np.uint8)
    elif unsafe=='zero-depth':out=np.zeros(image.shape+(0,),np.uint8)
    elif unsafe=='mask-short':mask=np.ones((1,4),np.uint8)
    elif unsafe=='feedback':out=image.view()
    elif unsafe=='escaping-bin':
        class Hook:
            def __int__(self):image.fill(2000);return 0
        options['shift_x']=Hook()
    before=out.copy();policy=xx.set_fallback_enabled(fallback_enabled)
    try:
        with pytest.raises(NotImplementedError,match='Outside qualified rank-decisions-v1 coverage'):
            getattr(active,name)(image,fp,mask=mask,out=out,**options)
    finally:xx.set_fallback_enabled(policy)
    assert out.tobytes()==before.tobytes()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('exception',(ValueError,OverflowError))
@pytest.mark.parametrize('message',('rank decisions image exceeded histogram bounds','rank decisions moment products outside defined coverage','overlapping buffers are outside rank-decisions-v1 parity'))
def test_python_callback_guard_text_keeps_exception_and_writes(name,exception,message,active,monkeypatch):
    def observe(function):
        image=np.arange(12,dtype=np.uint8).reshape(3,4);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.uint8);events=[]
        class Hook:
            def __int__(self):events.append('__int__');out.fill(77);raise exception(message)
        try:function(image,fp,out=out,shift_x=Hook());error=None
        except Exception as caught:error=(type(caught).__name__,str(caught))
        return error,events,out.tobytes()
    expected=observe(getattr(reference,name));block_original(monkeypatch,active)
    assert observe(getattr(active,name))==expected


def test_missing_or_python_backend_rejected_before_image_metadata(monkeypatch):
    # Tooling metadata guards only; these fake objects are NOT native evidence.
    class Trap:
        def __array__(self,*a,**k):raise AssertionError('image metadata touched before backend gate')
    def forwarded(*a,**k):raise AssertionError('Python forwarding entry invoked')
    for fake in (SimpleNamespace(),SimpleNamespace(__name__='fake',rank_decisions_interface='rank-decisions-v1',rank_decisions_capabilities=0x60000000,rank_decisions_run=forwarded)):
        monkeypatch.setattr(source_wrapper,'_core',fake)
        for name in NAMES:
            with pytest.raises(NotImplementedError,match='compiled backend required'):getattr(source_wrapper,name)(Trap(),Trap())


def shift_mutation_observe(function,ndim,action,field):
    shape=(3,4) if ndim==2 else (2,3,4)
    image=(np.arange(np.prod(shape))%4).astype(np.uint16).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8);mask=np.ones(shape,np.uint8);out=np.full(shape,99,np.float64);events=[]
    class Shift:
        def __int__(self):
            events.append(field+'.__int__')
            if action=='image-fill':image.fill(0)
            elif action=='mask-fill':mask.fill(0)
            elif action=='footprint-fill':fp.fill(0)
            elif action=='image-dtype':image.dtype=np.dtype('i1')
            elif action=='out-dtype':out.dtype=np.dtype('i1')
            return 0
    options=dict(out=out,mask=mask);options[field]=Shift()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,**options);error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],result=None if result is None else contracts.array(result),after={key:contracts.array(value) for key,value in dict(image=image,fp=fp,mask=mask,out=out).items()})


def shift_protocol_observe(function,token):
    events=[]
    class IndexOnly:
        def __index__(self):events.append('__index__');return 0
    class IntOnly:
        def __int__(self):events.append('__int__');return 0
    class FloatOnly:
        def __float__(self):events.append('__float__');return 0.
    class Both:
        def __int__(self):events.append('__int__');return 0
        def __index__(self):events.append('__index__');raise ValueError('unexpected-index')
    class Integer(int):
        def __int__(self):events.append('subclass.__int__');return 1
        def __index__(self):events.append('subclass.__index__');return 1
    class Floating(float):
        def __int__(self):events.append('subclass.__int__');return 0
    class TruncOnly:
        def __trunc__(self):events.append('__trunc__');return 0
    classes={'index-only':IndexOnly,'int-only':IntOnly,'float-only':FloatOnly,'both':Both,'int-subclass':lambda:Integer(0),'float-subclass':lambda:Floating(.5),'trunc-only':TruncOnly}
    value=classes[token]() if token in classes else {'int':0,'float':.5,'nan':float('nan'),'inf':float('inf'),'none':None,'numeric-string':'0','np-i64':np.int64(0),'np-f64':np.float64(.5),'array0':np.array(0),'array1':np.array([0]),'np-bool':np.bool_(False)}[token]
    image=np.arange(6,dtype=np.uint8).reshape(2,3);fp=np.zeros((3,3),np.uint8)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,shift_x=value);error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],result=None if result is None else contracts.array(result))
