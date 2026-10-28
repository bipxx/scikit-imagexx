"""Masked high-center bilateral parity: direct prototype or genuine public only.

The original safety certificate protects every high source value with a zero
mask entry for all histogram reads. The center remains a comparison operand.
Existing708/96/680 fixtures and wrapper code are never replaced.
"""
import ast
import hashlib
import importlib
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
from skimage.filters import rank as reference
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
NAMES=('mean_bilateral','pop_bilateral','sum_bilateral')
PATHS={prefix+name for prefix in ('filters.rank.','filters.rank.bilateral.') for name in NAMES}
PROTOTYPE=os.environ.get('SKIMAGEXX_RANK_BILATERAL_MASKED_CENTER_PROTOTYPE_CORE')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


contracts=load('bilateral_masked_public_original_contracts',ROOT/'tests/test_rank_bilateral_contracts.py')
source_wrapper=load('skimagexx._rank_bilateral_masked_source_prototype',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source_wrapper._core=importlib.import_module(PROTOTYPE)


@pytest.fixture
def target():
    if PROTOTYPE:return source_wrapper
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Genuine public bilateral namespace/core/package gate pending; no private substitution')
    assert PATHS<=registered,'partial bilateral registration is not public admission'
    return importlib.import_module('skimagexx._rank')


@pytest.fixture(params=(1,4))
def active(request,target):
    core=target._core;previous=core.get_num_threads();policy=xx.set_fallback_enabled(False)
    core.set_num_threads(request.param)
    try:yield target
    finally:core.set_num_threads(previous);xx.set_fallback_enabled(policy)


def block_original(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('upstream bilateral forwarding')
    for module in (reference,contracts.bilateral_module):
        for name in NAMES:monkeypatch.setattr(module,name,forbidden)
    for name in ('_mean','_pop','_sum'):monkeypatch.setattr(contracts.bilateral_module.bilateral_cy,name,forbidden)
    monkeypatch.setattr(source_wrapper,'fallback',forbidden)


def test_genuine_public_or_visible_pending():
    if PROTOTYPE:pytest.skip('DIRECT PRIVATE-CORE SOURCE WRAPPER PROTOTYPE: no public/sole-core/package/SDK admission')
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Root genuine whole production public acceptance remains pending')
    assert PATHS<=registered
    rank=importlib.import_module('skimagexx.filters.rank');nested=importlib.import_module('skimagexx.filters.rank.bilateral');generic=importlib.import_module('skimagexx.filters.rank.generic');wrapper=importlib.import_module('skimagexx._rank')
    assert wrapper._core is xx._core and xx.get_num_threads is xx._core.get_num_threads and xx.set_num_threads is xx._core.set_num_threads
    assert xx.get_num_threads()==1 and rank.bilateral is nested
    for name in NAMES:
        assert getattr(rank,name) is getattr(nested,name) is getattr(wrapper,name)
        assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))
        assert name not in generic.__dict__ and 'filters.rank.generic.'+name not in registered

def observe_with_pointers(function,*case):
    captured={}
    def checked(image,footprint,**options):
        arrays=dict(image=image,footprint=footprint,mask=options['mask'],out=options['out'])
        pointers={name:value.ctypes.data for name,value in arrays.items()}
        try:
            result=function(image,footprint,**options)
            captured['return_is_out']=result is options['out']
            captured['return_shares_out']=bool(np.shares_memory(result,options['out']))
            return result
        finally:
            captured['pointer_stable']={name:value.ctypes.data==pointers[name] for name,value in arrays.items()}
    result=source_observe(checked,*case)
    assert captured and all(captured['pointer_stable'].values())
    return dict(result,pointer_stable=captured['pointer_stable'],return_is_out=captured.get('return_is_out'),return_shares_out=captured.get('return_shares_out'))

RADII=((10,10),(10,65536),(1,2001),(1,65536),(-1999,10),(-65534,10),(0,0),(sys.maxsize,10),(3-sys.maxsize,10),(10,-sys.maxsize-1),(10,sys.maxsize-3))
CASES=tuple((name,high,field,radius,dtype) for name in NAMES for high in (2000,65535) for field in ('s0','s1') for radius in RADII for dtype in ('u1','u2','f8'))

def source_observe(function,name,high,field,radius,dtype):
    image=(np.arange(9,dtype=np.uint16)%4).reshape(3,3);fp=np.ones((3,3),np.uint8)
    mask=np.ones(image.shape,np.uint8);mask[1,1]=0;out=np.full(image.shape,99,np.dtype(dtype));events=[]
    assert image.max()==3 and mask[1,1]==0 and high<=65535
    assert not np.shares_memory(image,out) and not np.shares_memory(mask,out)
    # This pre-call source safety certificate uses CAPTURED bins4, not guint16
    # capacity. It includes defined edges that a65536-bin guard would reject.
    assert all(-sys.maxsize-1<=i-radius[0]<=sys.maxsize and -sys.maxsize-1<=i+radius[1]<=sys.maxsize for i in range(4))
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            events.append(self.label+'.__index__')
            if self.label==field:image[1,1]=high
            return self.value
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,mask=mask,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]));error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(error=error,events=events,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],out=contracts.array(out),image=contracts.array(image),mask=contracts.array(mask),footprint=contracts.array(fp),result=None if result is None else contracts.array(result))

def source_mask_feedback_observe(function,offset):
    image=(np.arange(25,dtype=np.uint16)%4).reshape(5,5);fp=np.ones((3,3),np.uint8)
    owner=np.ones(26,np.uint8);owner[::4]=0;mask=owner[:25].reshape(5,5);out=owner[offset:offset+25].reshape(5,5)
    # Every possible future histogram read is a valid bin, regardless of mask
    # feedback. These already-defined aliases remain controls, not new scope.
    assert image.max()==3
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');result=function(image,fp,mask=mask,out=out)
    return dict(owner=contracts.array(owner),mask=contracts.array(mask),out=contracts.array(out),result=contracts.array(result),warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])


@pytest.mark.parametrize('case',CASES,ids=['/'.join(map(str,c)) for c in CASES])
def test_exact_masked_center_observations_and_pointers(case,active,monkeypatch):
    name,high,field,radius,dtype=case
    expected=observe_with_pointers(getattr(reference,name),*case)
    assert expected['error'] is None and expected['events']==['s0.__index__','s1.__index__']
    assert expected['return_is_out'] is False and expected['return_shares_out'] is True
    block_original(monkeypatch)
    assert observe_with_pointers(getattr(active,name),*case)==expected


@pytest.mark.parametrize('name',NAMES)
def test_wide_radius_masked_center_has_real_population(name,active):
    expected={'mean_bilateral':1,'pop_bilateral':8,'sum_bilateral':12}[name]
    actual=observe_with_pointers(getattr(active,name),name,65535,'s0',(10,65536),'f8')
    assert actual['error'] is None and actual['out']['values'][1][1]==expected


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('offset',(0,1))
def test_safe_narrow_mask_feedback_control(name,offset,active,monkeypatch):
    expected=source_mask_feedback_observe(getattr(reference,name),offset)
    block_original(monkeypatch)
    assert source_mask_feedback_observe(getattr(active,name),offset)==expected


@pytest.mark.parametrize('operation',(26,27,28))
def test_uncertified_reenable_feedback_is_native_only_prewrite(operation,active,monkeypatch):
    # Potential histogram escape: NEVER call original bounds-disabled Cython.
    block_original(monkeypatch)
    image=(np.arange(25,dtype=np.uint16)%4).reshape(5,5);image[2,2]=2000
    fp=np.ones((3,3),np.uint8);owner=np.ones(26,np.uint8);owner[12]=0
    mask=owner[:25].reshape(5,5);out=owner[1:].reshape(5,5,1);before=owner.copy()
    with pytest.raises(ValueError,match='rank image exceeded histogram bounds'):
        active._core.rank_bilateral_run(image,fp,mask,out,[0,0],4,operation,10,65536)
    assert owner.tobytes()==before.tobytes()


@pytest.mark.parametrize('name',NAMES)
def test_masked_center_never_forwards_original(name,active,monkeypatch):
    expected=observe_with_pointers(getattr(reference,name),name,65535,'s0',(10,65536),'f8')
    block_original(monkeypatch)
    assert observe_with_pointers(getattr(active,name),name,65535,'s0',(10,65536),'f8')==expected


def test_frozen_source_observer_derivation_and_scope():
    data=json.loads((ROOT/'coverage/contracts/rank-bilateral-masked-center-public.json').read_text())
    assert data['native_public_admission']==0 and data['defined_reference_cases']==len(CASES)==396
    class Restore(ast.NodeTransformer):
        def visit_Attribute(self,node):
            node=self.generic_visit(node)
            if node.attr=='array' and isinstance(node.value,ast.Name) and node.value.id=='contracts':
                node.value=ast.Attribute(value=ast.Name(id='pipeline',ctx=ast.Load()),attr='contracts',ctx=ast.Load())
            return node
    for function,name in ((source_observe,'observe'),(source_mask_feedback_observe,'mask_feedback_observe')):
        node=ast.parse(inspect.getsource(function)).body[0];node.name=name
        frozen=data['source_observer_text'][name]
        assert hashlib.sha256(frozen.encode()).hexdigest()==data['source_observer_text_sha256'][name]
        assert ast.dump(Restore().visit(node))==ast.dump(ast.parse(frozen).body[0])
    for filename,digest in data['unchanged_source_pins'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest


def test_source_pin_migration_preserves_frozen_observers_and_assertions():
    import copy
    decisions=load('masked_decisions_migration',ROOT/'tests/test_rank_decisions_contracts.py')
    decisions.test_source_facts_safety_and_strict_pending(json.loads(decisions.MATRIX.read_text()))
    decisions_migration=json.loads((ROOT/'coverage/evidence/rank-decisions-fixture-assertion-migration.json').read_text())
    migration=json.loads((ROOT/'coverage/evidence/rank-bilateral-masked-source-pin-migration.json').read_text())
    frozen_text=migration['original_contract_utf8']
    assert hashlib.sha256(frozen_text.encode()).hexdigest()=='484e8ccb287722f12606095316262293734878a7e7b3411034d16ea802e148c6'
    original=json.loads(frozen_text)
    current=json.loads((ROOT/'coverage/contracts/rank-bilateral-masked-center-public.json').read_text())
    paths={'python/skimagexx/_rank.py','tests/test_rank_bilateral_public.py','tests/test_rank_bilateral_contracts.py'}
    assert set(migration['sources'])==paths
    expected=copy.deepcopy(original)
    allowed_tests={
        'tests/test_rank_bilateral_public.py':{'test_source_signature_and_pending_shared_handoff'},
        'tests/test_rank_bilateral_contracts.py':{'test_integrity_safe_scope_native0','test_strict_pending_never_counts_forwarding'},
    }
    for filename,record in migration['sources'].items():
        old_text=record['original_source_utf8']
        assert hashlib.sha256(old_text.encode()).hexdigest()==original['unchanged_source_pins'][filename]
        new_text=(ROOT/filename).read_bytes().replace(b'\r\n',b'\n').decode()
        if filename=='tests/test_rank_bilateral_contracts.py':
            from tools.rank_bilateral_fixture_source import project_qualified_fixture
            new_text,actual_fixture_digest=project_qualified_fixture(ROOT)
            assert actual_fixture_digest==hashlib.sha256((ROOT/filename).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        current_digest=hashlib.sha256(new_text.encode()).hexdigest()
        assert current_digest==record['current_canonical_lf_sha256']
        expected['unchanged_source_pins'][filename]=current_digest
        old=ast.parse(old_text);new=ast.parse(new_text)
        if filename=='python/skimagexx/_rank.py':
            # The complete live c6/ae2/export chain was authenticated above.
            # Replay the original bilateral-only projection on its frozen d6 stage.
            new=ast.parse(decisions_migration['rank_stages']['bilateral_provider']['utf8'])
            helpers=[node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=='_bilateral_ubyte']
            assert len(helpers)==1
            new.body.remove(helpers[0])
            assert ast.dump(new.body[-1])==ast.dump(ast.parse('__all__+=list(_BILATERAL_FUNCTIONS)').body[0])
            new.body.pop()
            assert len(old.body)==len(new.body)
            for before,after in zip(old.body,new.body):
                if isinstance(after,ast.FunctionDef) and after.name=='_bilateral_apply':
                    calls=[node for node in ast.walk(after) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_bilateral_ubyte']
                    assert len(calls)==2 and {node.args[0].id for node in calls}=={'image','mask'}
                    for call in calls:call.func.id='_ubyte'
                elif isinstance(after,ast.Assign) and len(after.targets)==1 and isinstance(after.targets[0],ast.Name) and after.targets[0].id=='_BILATERAL_SCOPE_DOC':
                    assert isinstance(before,ast.Assign) and before.targets[0].id=='_BILATERAL_SCOPE_DOC'
                    after.value=copy.deepcopy(before.value)
        else:
            assert len(old.body)==len(new.body)
            changed=set()
            for before,after in zip(old.body,new.body):
                if isinstance(after,ast.FunctionDef) and after.name in allowed_tests[filename]:
                    assert isinstance(before,ast.FunctionDef) and before.name==after.name
                    after.body=copy.deepcopy(before.body)
                    changed.add(after.name)
            assert changed==allowed_tests[filename]
        # Both source versions use this runtime: no cross-version AST hashes,
        # ignored fields or numerical fixture rewrites authenticate migration.
        assert ast.dump(new)==ast.dump(old)
    # The six source pins are metadata. All396 cases, literal frozen observer
    # text, safety certificates, interface and other contract values stay exact.
    assert current==expected
