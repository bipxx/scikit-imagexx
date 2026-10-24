"""Bilateral source/private-core prototype and future genuine public admission.

PROTOTYPE_CORE selects only a real frozen private extension for the source
wrapper. It never substitutes public namespaces, registry or whole SDK proof.
"""
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
contracts=load('rank_bilateral_wrapper_contracts',Path(__file__).with_name('test_rank_bilateral_contracts.py'))
MATRIX=json.loads(contracts.MATRIX.read_text())
PROTOTYPE=os.environ.get('SKIMAGEXX_RANK_BILATERAL_PROTOTYPE_CORE')
source_wrapper=load('skimagexx._rank_bilateral_source_prototype',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source_wrapper._core=importlib.import_module(PROTOTYPE)
NAMES=contracts.NAMES
PATHS={prefix+name for prefix in ('filters.rank.','filters.rank.bilateral.') for name in NAMES}


@pytest.fixture
def target():
    if PROTOTYPE:return source_wrapper
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Genuine whole bilateral namespace/core/package gate pending; no alternate backend substituted')
    assert PATHS<=registered,'partial canonical/nested bilateral registration is not public acceptance'
    return importlib.import_module('skimagexx._rank')


@pytest.fixture(params=(1,4))
def active(request,target):
    policy=xx.set_fallback_enabled(False);core=target._core;previous=core.get_num_threads()
    core.set_num_threads(request.param)
    try:yield target
    finally:core.set_num_threads(previous);xx.set_fallback_enabled(policy)


def block_original(monkeypatch):
    def forbidden(*args,**options):raise AssertionError('upstream bilateral algorithm/fallback forwarding')
    monkeypatch.setattr(source_wrapper,'fallback',forbidden)
    for provider in (reference,contracts.bilateral_module):
        for name in NAMES:monkeypatch.setattr(provider,name,forbidden)
    for name in ('_mean','_pop','_sum'):monkeypatch.setattr(contracts.bilateral_module.bilateral_cy,name,forbidden)


@pytest.fixture
def no_forwarding(monkeypatch):block_original(monkeypatch)


def test_source_signature_and_pending_shared_handoff():
    assert source_wrapper._BILATERAL_FUNCTIONS==NAMES
    assert len(source_wrapper._GENERIC_FUNCTIONS)==17 and not set(NAMES)&set(source_wrapper._GENERIC_FUNCTIONS)
    assert source_wrapper.__all__==list(source_wrapper._GENERIC_FUNCTIONS+source_wrapper._PERCENTILE_FUNCTIONS+NAMES+source_wrapper._DECISIONS_FUNCTIONS)
    assert len(source_wrapper.__all__)==31 and set(NAMES)<=set(source_wrapper.__all__)
    for name in NAMES:
        assert inspect.signature(getattr(source_wrapper,name))==inspect.signature(getattr(reference,name))
        assert getattr(reference,name) is getattr(contracts.bilateral_module,name)


def test_actual_namespace_or_visible_pending():
    if PROTOTYPE:pytest.skip('DIRECT PRIVATE-CORE WRAPPER PROTOTYPE: public namespace/registry/sole-core/SDK admission pending')
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Root canonical3/nested3 wiring and genuine complete product remain pending')
    assert PATHS<=registered
    rank=importlib.import_module('skimagexx.filters.rank');nested=importlib.import_module('skimagexx.filters.rank.bilateral');generic=importlib.import_module('skimagexx.filters.rank.generic');wrapper=importlib.import_module('skimagexx._rank')
    assert rank.bilateral is nested and list(nested.__all__)==list(NAMES)
    assert wrapper._core is xx._core and xx.set_num_threads is xx._core.set_num_threads and xx.get_num_threads is xx._core.get_num_threads
    assert xx.get_num_threads()==1
    for name in NAMES:
        assert getattr(rank,name) is getattr(nested,name) is getattr(wrapper,name)
        assert name in rank.__all__ and name not in generic.__dict__ and name not in generic.__all__ and not hasattr(generic,name)
        assert 'filters.rank.generic.'+name not in registered
        assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_frozen_values_errors_warnings_mutation_ownership(case,active,no_forwarding):
    key,name,parameters=case
    assert contracts.observe(getattr(active,name),name,parameters)==MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]],key


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('kind',('u1','f4'))
@pytest.mark.parametrize('scenario',contracts.SCENARIOS)
def test_radius_provider_warning_and_mutation_traces(name,kind,scenario,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    assert contracts.provider_observe(name,kind,scenario)==MATRIX['provider_events'][contracts.profile()][name+'/'+kind+'/'+scenario]


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('radius',((0,0),(1,1),(1,5),(5,1),(-1,5),(5,-1),(300,300)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_independent_integer_neighborhood(name,dtype,kind,radius,output,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_independent_integer_neighborhood_oracle(name,dtype,kind,radius,output)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('output',('u1','u2','f4','f8'))
def test_full_bin_capacity_and_unsigned_wrap(name,dtype,output,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_full_capacity_counts_and_integer_wrap(name,dtype,output)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('edge',('upper-max','lower-min','upper-min-safe','lower-max-safe'))
def test_signed_size_predicate_edges(name,dtype,edge,active,no_forwarding,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_signed_size_safe_predicate_edges_against_python_integers(name,dtype,edge)


MODES=('missing-interface','wrong-interface','missing-cap','wrong-cap','float-cap','bool-cap','negative-cap','extra-bit','missing-bit','missing-entry','non-callable','python-forwarder','wrong-builtin','wrong-owned-builtin')
@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('mode',MODES)
def test_harness_only_capability_guard_precedes_metadata(name,mode,monkeypatch):
    # Guard stubs are intentionally not native parity/compiled evidence.
    core=source_wrapper._core;entry=getattr(core,'rank_bilateral_run',None)
    fields=dict(__name__=getattr(core,'__name__','harness_only'),rank_bilateral_interface='rank-bilateral-v1',rank_bilateral_capabilities=0x1c000000,rank_bilateral_run=entry)
    if mode=='missing-interface':fields.pop('rank_bilateral_interface')
    elif mode=='wrong-interface':fields['rank_bilateral_interface']='rank-bilateral-v0'
    elif mode=='missing-cap':fields.pop('rank_bilateral_capabilities')
    elif mode=='wrong-cap':fields['rank_bilateral_capabilities']=8191
    elif mode=='float-cap':fields['rank_bilateral_capabilities']=float(0x1c000000)
    elif mode=='bool-cap':fields['rank_bilateral_capabilities']=True
    elif mode=='negative-cap':fields['rank_bilateral_capabilities']=-1
    elif mode=='extra-bit':fields['rank_bilateral_capabilities']|=1<<29
    elif mode=='missing-bit':fields['rank_bilateral_capabilities']&=~(1<<(26+NAMES.index(name)))
    elif mode=='missing-entry':fields.pop('rank_bilateral_run')
    elif mode=='non-callable':fields['rank_bilateral_run']=7
    elif mode=='python-forwarder':fields['rank_bilateral_run']=lambda *args:entry(*args)
    elif mode=='wrong-builtin':fields['rank_bilateral_run']=len
    else:fields['rank_bilateral_run']=getattr(core,'rank_run',len)
    monkeypatch.setattr(source_wrapper,'_core',SimpleNamespace(**fields))
    class Bomb:
        def __array__(self,*args,**options):raise AssertionError('array work before compiled capability admission')
    with pytest.raises(NotImplementedError,match='rank-bilateral-v1 compiled backend required'):getattr(source_wrapper,name)(Bomb(),None)


@pytest.mark.parametrize('kind',('zero-depth','short-output','short-mask','footprint-alias','mixed-type-image','narrow-u16-alias','radius-overflow-domain'))
def test_native_unsafe_scope_explicit_without_original_probe(kind,active,no_forwarding):
    image=np.arange(15,dtype=np.uint8).reshape(3,5);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.uint8);options=dict(out=out)
    if kind=='zero-depth':out=np.empty(image.shape+(0,),np.uint8);options['out']=out
    elif kind=='short-output':out=np.full((2,5),99,np.uint8);options['out']=out
    elif kind=='short-mask':options['mask']=np.ones((1,1),np.uint8)
    elif kind=='footprint-alias':
        # Prepared footprints are fresh, so an ordinary caller overlap is
        # normalized away. Test this native certificate through direct entry.
        core=active._core;prepared=image.copy();output=prepared.reshape(3,5,1);before=prepared.copy()
        with pytest.raises(ValueError,match='footprint/output alias'):core.rank_bilateral_run(image,prepared,None,output,[0,0],256,26,10,10)
        assert prepared.tobytes()==before.tobytes();return
    elif kind=='mixed-type-image':out=image.view(bool);options['out']=out;image=np.arange(15,dtype=np.uint16).reshape(3,5);out=np.ndarray(image.shape,np.uint8,buffer=image.data);options['out']=out
    elif kind=='narrow-u16-alias':image=np.arange(15,dtype=np.uint16).reshape(3,5);out=image.view();options['out']=out
    else:options['s0']=-os.sys.maxsize
    before=image.tobytes();previous=out.tobytes()
    with pytest.raises(NotImplementedError,match='Outside qualified rank-bilateral-v1 coverage'):getattr(active,'mean_bilateral')(image,fp,**options)
    assert image.tobytes()==before and out.tobytes()==previous


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('message',('rank image exceeded histogram bounds','rank bilateral radius arithmetic outside defined coverage'))
def test_user_index_exception_payload_is_not_reclassified(name,message,active):
    class Provider:
        def __index__(self):raise ValueError(message)
    image=np.arange(12,dtype=np.uint8).reshape(3,4);fp=np.ones((3,3),np.uint8)
    for function in (getattr(reference,name),getattr(active,name)):
        out=np.full(image.shape,99,np.uint8)
        with pytest.raises(ValueError,match=message):function(image,fp,out=out,s0=Provider())
        assert out.tobytes()==np.full(image.shape,99,np.uint8).tobytes()


def test_compiled_entry_and_old_interfaces_unchanged(active):
    core=active._core
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_statistics_capabilities,core.rank_vector_capabilities,core.rank_percentile_capabilities)==(8191,480,0xe000,0x10000,0x3fe0000)
    for operation in (26,27,28):assert active._require_bilateral(operation) is core.rank_bilateral_run

STATES=('zero-fp-high-all','zero-fp-high-tail','zero-mask-high-all','zero-mask-high-tail','zero-fp-enable-mask-high','full-fp-disable-mask-high','full-fp-enable-mask-valid','full-fp-rewrite-valid','full-fp-empty-selected')
MUTATION_CASES=tuple((name,state,field,output,radius) for name in NAMES for state in STATES for field in ('s0','s1') for output in ('u1','f8') for radius in ((10,10),(0,0),(-3,7)))

def mutation_observe(function,name,state,field,output,radius):
    assert state in STATES
    image=(np.arange(12,dtype=np.uint16)%4).reshape(3,4)
    fp=np.zeros((3,3),np.uint8) if state.startswith('zero-fp') else np.ones((3,3),np.uint8)
    mask=np.zeros(image.shape,np.uint8) if state.startswith('zero-mask') or state in ('zero-fp-enable-mask-high','full-fp-enable-mask-valid') else np.ones(image.shape,np.uint8)
    out=np.full(image.shape,99,np.dtype(output));events=[]
    assert not np.shares_memory(image,out) and not np.shares_memory(mask,out)
    # This certificate applies BEFORE any bounds-disabled original call. A
    # high source is sampled only as the valid center value, not histo[index].
    if 'high' in state:assert state.startswith(('zero-fp','zero-mask')) or state=='full-fp-disable-mask-high'
    pointers={key:value.ctypes.data for key,value in dict(image=image,mask=mask,out=out,fp=fp).items()}
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            events.append(self.label+'.__index__')
            if self.label==field:
                if 'high-tail' in state:image[-1,-1]=2000
                elif 'high' in state:image.fill(2000)
                elif state=='full-fp-rewrite-valid':image.fill(2)
                if 'enable-mask' in state:mask.fill(1)
                if 'disable-mask' in state:mask.fill(0)
            return 0 if state=='full-fp-empty-selected' else self.value
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result=function(image,fp,mask=mask,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]));error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    tracked=dict(image=image,mask=mask,out=out,fp=fp)
    return dict(error=error,events=events,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],after={key:contracts.array(value) for key,value in tracked.items()},pointer_stable={key:value.ctypes.data==pointers[key] for key,value in tracked.items()},result=None if result is None else dict(contracts.array(result),is_out=result is out,shares_out=bool(np.shares_memory(result,out))))

@pytest.mark.parametrize('case',MUTATION_CASES,ids=['/'.join(map(str,c)) for c in MUTATION_CASES])
def test_corrected_zero_population_mutation_observations(case,active,monkeypatch):
    name,state,field,output,radius=case
    expected=mutation_observe(getattr(reference,name),*case)
    assert expected['error'] is None and expected['events']==['s0.__index__','s1.__index__']
    if 'high' in state:assert np.count_nonzero(expected['after']['out']['values'])==0
    block_original(monkeypatch)
    assert mutation_observe(getattr(active,name),*case)==expected


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('field',('s0','s1'))
@pytest.mark.parametrize('radius',((10,10),(0,0)))
def test_nonzero_histogram_escape_is_native_only_prewrite(name,field,radius,active,no_forwarding):
    # NEVER original: nonzero source histogram access2000 exceeds captured4.
    image=(np.arange(12,dtype=np.uint16)%4).reshape(3,4);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.uint8)
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            if self.label==field:image.fill(2000)
            return self.value
    with pytest.raises(NotImplementedError,match='rank image exceeded histogram bounds'):
        getattr(active,name)(image,fp,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]))
    assert np.all(out==99) and np.all(image==2000)


@pytest.mark.parametrize('name',NAMES)
def test_corrected_zero_population_does_not_forward(name,active,monkeypatch):
    expected=mutation_observe(getattr(reference,name),name,'zero-fp-high-all','s0','f8',(10,10))
    block_original(monkeypatch)
    assert mutation_observe(getattr(active,name),name,'zero-fp-high-all','s0','f8',(10,10))==expected


def test_corrected_core_interface_and_debug(active):
    assert active._core.rank_bilateral_interface=='rank-bilateral-v1' and active._core.rank_bilateral_capabilities==0x1c000000
    assert (active._core.rank_histogram_capabilities,active._core.rank_alias_capabilities,active._core.rank_percentile_capabilities)==(8191,480,0x3fe0000)
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert active._core.gil_assertions is True


def test_mutation_observer_is_frozen_source_derivation():
    source=ast.parse((ROOT/'tests/test_rank_bilateral_mutation_supplement.py').read_text())
    original=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='observe')
    actual=ast.parse(inspect.getsource(mutation_observe)).body[0];actual.name='observe'
    class Restore(ast.NodeTransformer):
        def visit_Attribute(self,node):
            node=self.generic_visit(node)
            if node.attr=='array' and isinstance(node.value,ast.Name) and node.value.id=='contracts':
                node.value=ast.Attribute(value=ast.Name(id='pipeline',ctx=ast.Load()),attr='contracts',ctx=ast.Load())
            return node
    assert ast.dump(Restore().visit(actual))==ast.dump(original)
    assert len(MUTATION_CASES)==324
