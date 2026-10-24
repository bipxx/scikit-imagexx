"""R7 nonnative conversion scopes; private source/core is not public admission."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import pytest
import skimagexx as xx
from skimagexx.coverage import native_functions
import math
import warnings
import numpy as np


def conversion_cases():
    result=[]
    for name in ('otsu','majority'):
        for ndim in (2,3):
            for field in ('image','mask'):
                for maximum in (0,1,255,256,257,1023,65535):
                    for layout in ('C','step'):
                        result.append(dict(id=f'{name}/{ndim}d/{field}/max{maximum}/{layout}',name=name,ndim=ndim,field=field,maximum=maximum,layout=layout,dtype='>u2',mode='plain'))
                modes=('precision-low-to-high','precision-high-to-low','downcast-to-high','allocation-to-native','precision-error','downcast-error','allocation-error') if field=='image' else ('downcast-to-high','allocation-to-native','downcast-error','allocation-error')
                for mode in modes:
                    result.append(dict(id=f'{name}/{ndim}d/{field}/hook/{mode}',name=name,ndim=ndim,field=field,maximum=256 if mode in ('precision-high-to-low','allocation-to-native','allocation-error') else 255,layout='C',dtype='>u2',mode=mode))
                for dtype in ('>u2','>i2','>u4','>i4','>u8','>i8','u2','>f4'):
                    for maximum in (1,255,256):
                        result.append(dict(id=f'{name}/{ndim}d/{field}/type/{dtype}/{maximum}',name=name,ndim=ndim,field=field,maximum=maximum,layout='C',dtype=dtype,mode='plain'))
    assert len(result)==348 and len({v['id'] for v in result})==348
    return tuple(result)


def conversion_trace(events):
    result=[];shift_seen=False
    for event in events:
        if isinstance(event,list) and event and event[0]=='shift-int':shift_seen=True
        if shift_seen and isinstance(event,list) and len(event)==3 and event[0]=='empty' and event[1]=='int64' and tuple(event[2])==(9,):continue
        result.append(event)
    return result


def conversion_observe(function,case):
    shape=(3,)*case['ndim'];dtype=np.dtype(case['dtype']);maximum=case['maximum'];mode=case['mode'];field=case['field'];events=[]
    value=.5 if dtype.kind=='f' else maximum
    raw=(np.arange(math.prod(shape)).reshape(shape)%7).astype(dtype);raw[raw>value]=value;raw.flat[-1]=value
    if case['layout']=='step':
        owner=np.zeros(shape[:-1]+(shape[-1]*2,),dtype);owner[...,::2]=raw;raw=owner[...,::2]
    image=raw if field=='image' else (np.arange(math.prod(shape)).reshape(shape)%8).astype(np.uint8);mask=None if field=='image' else raw
    class Footprint(np.ndarray):
        def __gt__(self,other):events.append(['footprint-positive']);return super().__gt__(other)
    footprint=np.ones(shape,np.uint8).view(Footprint);out=np.full(shape,99,np.float32)
    class Shift:
        def __init__(self,label):self.label=label
        def __int__(self):events.append(['shift-int',self.label]);return 0
    original_empty=np.empty;original_floor=np.floor_divide;changed=False
    def empty(*args,**options):
        nonlocal changed
        result=original_empty(*args,**options);events.append(['empty',str(result.dtype),tuple(result.shape)])
        if mode in ('allocation-to-native','allocation-error') and not changed:
            changed=True
            if mode=='allocation-error':events.append(['allocation-error']);raise RuntimeError('conversion allocator hook')
            raw.byteswap(inplace=True);raw.dtype=np.dtype('u2');events.append(['allocation-native'])
        return result
    def floor(*args,**options):
        events.append(['floor_divide',str(options.get('dtype')),int(args[1]),options.get('casting'),tuple(args[0].shape)])
        return original_floor(*args,**options)
    def array(a):return dict(dtype=str(a.dtype),shape=list(a.shape),strides=list(a.strides),bytes=a.tobytes().hex(),values=a.tolist())
    before=dict(image=array(image),mask=None if mask is None else array(mask),out=array(out));pointers=(image.ctypes.data,None if mask is None else mask.ctypes.data,out.ctypes.data)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');original_show=warnings.showwarning
        if mode=='precision-error':warnings.filterwarnings('error',message='Possible precision loss.*')
        if mode=='downcast-error':warnings.filterwarnings('error',message='Downcasting.*')
        def show(message,category,filename,lineno,file=None,line=None):
            events.append(['warning',category.__name__,str(message)])
            if str(message).startswith('Possible precision loss'):
                if mode=='precision-low-to-high':raw.fill(256);events.append(['precision-mutated256'])
                elif mode=='precision-high-to-low':raw.fill(255);events.append(['precision-mutated255'])
            elif str(message).startswith('Downcasting') and mode=='downcast-to-high':raw.fill(256);events.append(['downcast-mutated256'])
            return original_show(message,category,filename,lineno,file,line)
        warnings.showwarning=show;np.empty=empty;np.floor_divide=floor
        try:
            result=function(image,footprint,out=out,mask=mask,shift_x=Shift('x'),shift_y=Shift('y'),shift_z=Shift('z'));error=None
            observed=dict(array=array(result),is_out=result is out,shares_out=bool(np.shares_memory(result,out)),base_is_out=result.base is out)
        except Exception as e:observed=None;error=dict(type=type(e).__name__,message=str(e))
        finally:np.empty=original_empty;np.floor_divide=original_floor;warnings.showwarning=original_show
    return dict(id=case['id'],result=observed,error=error,warnings=[dict(type=type(n.message).__name__,message=str(n.message)) for n in notices],before=before,after=dict(image=array(image),mask=None if mask is None else array(mask),out=array(out)),pointer_stable=[a==b for a,b in zip(pointers,(image.ctypes.data,None if mask is None else mask.ctypes.data,out.ctypes.data))],events=conversion_trace(events),original_raw_events=events)


def native_control_cases():
    result=[]
    for name in ('otsu','majority'):
        for ndim in (2,3):
            for field in ('image','mask'):
                for dtype in ('i2','u4','i4','u8','i8'):
                    for maximum in (1,255,256):
                        result.append(dict(id=f'{name}/{ndim}d/{field}/native/{dtype}/{maximum}',name=name,ndim=ndim,field=field,maximum=maximum,layout='C',dtype=dtype,mode='plain'))
    assert len(result)==120
    return tuple(result)

ROOT=Path(__file__).parents[1]
DATA=json.loads((ROOT/'coverage/contracts/rank-decisions-endian-public.json').read_text())
PROTOTYPE=os.environ.get(DATA['prototype_environment'])
NAMES=('otsu','majority')
PATHS={prefix+name for prefix in ('filters.rank.','filters.rank.generic.') for name in NAMES}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

source_wrapper=load('skimagexx._rank_decisions_endian_source',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source_wrapper._core=importlib.import_module(PROTOTYPE)

@pytest.fixture
def target():
    if PROTOTYPE:return source_wrapper
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Genuine corrected R7 public core/package gate pending; no private substitution')
    assert PATHS<=registered
    return importlib.import_module('skimagexx._rank')

@pytest.fixture(params=(1,4))
def active(request,target):
    core=target._core;previous=core.get_num_threads();policy=xx.set_fallback_enabled(False);core.set_num_threads(request.param)
    try:yield target
    finally:core.set_num_threads(previous);xx.set_fallback_enabled(policy)

@pytest.fixture
def no_forward(monkeypatch,active):
    def forbidden(*args,**kw):raise AssertionError('upstream rank algorithm/conversion forwarding')
    monkeypatch.setattr(active,'fallback',forbidden)
    generic=importlib.import_module('skimage.filters.rank.generic');rank=importlib.import_module('skimage.filters.rank')
    for provider in (generic,rank):
        for name in NAMES:monkeypatch.setattr(provider,name,forbidden)
    for name in ('_otsu','_otsu_3D','_majority','_majority_3D'):monkeypatch.setattr(generic.generic_cy,name,forbidden)
    monkeypatch.setattr(generic,'img_as_ubyte',forbidden)
    monkeypatch.setattr(importlib.import_module('skimage.util.dtype'),'img_as_ubyte',forbidden)


def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'

@pytest.mark.parametrize('case',conversion_cases()+native_control_cases(),ids=lambda c:c['id'])
def test_all_original348_and120_native_conversion_controls(case,active,no_forward):
    observation=conversion_observe(getattr(active,case['name']),case)
    observation.pop('original_raw_events')
    expected=DATA['observations'][DATA['snapshots'][profile()][case['id']]]
    assert json.loads(json.dumps(observation))==expected,case['id']


def test_projection_preserves_pre_shift_and_nonidentified_allocations():
    pre=['empty','int64',(9,)];conversion=['empty','uint8',(3,3)];shift=['shift-int','x'];scratch=['empty','int64',(9,)];other=['empty','int64',(27,)];notice=['warning','must stay']
    assert conversion_trace([pre,conversion,shift,scratch,other,notice])==[pre,conversion,shift,other,notice]
    assert conversion_trace([scratch])==[scratch]


def test_exact_two_callsite_scope_and_shared_provider_body():
    old=ast.parse(DATA['prerequisite_source']);new=ast.parse((ROOT/'python/skimagexx/_rank.py').read_text())
    assert len(new.body)>=len(old.body)
    for a,b in zip(old.body,new.body):
        if isinstance(a,ast.FunctionDef) and a.name=='_decisions_apply':
            calls=[n for n in ast.walk(b) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_bilateral_ubyte'];assert len(calls)==2
            for n in calls:
                assert len(n.keywords)==1 and n.keywords[0].arg=='_integer_provider' and ast.literal_eval(n.keywords[0].value) is True
                n.func.id='_ubyte';n.keywords=[]
        if isinstance(a,ast.FunctionDef) and a.name=='_bilateral_ubyte':
            assert len(b.args.kwonlyargs)==1 and b.args.kwonlyargs[0].arg=='_integer_provider' and ast.literal_eval(b.args.kw_defaults[0]) is False
            b.args.kwonlyargs=[];b.args.kw_defaults=[]
            conditions=[n for n in b.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and isinstance(n.test.values[-1],ast.BoolOp) and isinstance(n.test.values[-1].op,ast.Or)]
            assert len(conditions)==1 and ast.unparse(conditions[0].test.values[-1])=='not dtype.isnative or _integer_provider'
            conditions[0].test.values[-1]=conditions[0].test.values[-1].values[0]
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    source=(ROOT/'python/skimagexx/_rank.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_bilateral_ubyte')
    body='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    body=body.replace('def _bilateral_ubyte(image,*,_integer_provider=False):','def _bilateral_ubyte(image):').replace("and (not dtype.isnative or _integer_provider):","and not dtype.isnative:")
    assert hashlib.sha256(body.encode()).hexdigest()==DATA['qualified_scale_helper_canonical_lf_sha256']
    assert source_wrapper._GENERIC_FUNCTIONS==tuple(ast.literal_eval(next(n.value for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GENERIC_FUNCTIONS' for t in n.targets))))
    if PROTOTYPE:assert len(source_wrapper.__all__)==29 and len(source_wrapper._GENERIC_FUNCTIONS)==17


def test_literal_observer_source_and_all468_ids():
    expected=ast.parse(DATA['observer_source']);actual=ast.parse(Path(__file__).read_text())
    methods={n.name:ast.dump(n,include_attributes=False) for n in actual.body if isinstance(n,ast.FunctionDef)}
    for n in expected.body:
        if isinstance(n,ast.FunctionDef):assert methods[n.name]==ast.dump(n,include_attributes=False)
    assert list(conversion_cases()+native_control_cases())==DATA['cases'] and len(DATA['cases'])==468
    for values in DATA['snapshots'].values():assert set(values)=={c['id'] for c in conversion_cases()+native_control_cases()}


def test_compiled_5286_core_or_explicit_public_pending(target):
    core=target._core
    assert core.rank_decisions_interface=='rank-decisions-v1' and core.rank_decisions_capabilities==0x60000000
    assert inspect.isbuiltin(core.rank_decisions_run) and core.rank_decisions_run.__module__==core.__name__
    assert core.get_num_threads()==1
    if PROTOTYPE:
        assert hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() in set(DATA['compiled_module_sha256'].values())
        if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
        pytest.skip('Immutable5286 private core/source wrapper only; genuine190 public/SDK acceptance pending')
    assert core is xx._core and xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads
    rank=importlib.import_module('skimagexx.filters.rank');generic=importlib.import_module('skimagexx.filters.rank.generic')
    for name in NAMES:assert getattr(rank,name) is getattr(generic,name) is getattr(target,name)

# Consume the unchanged fixture factory directly. This does not invoke or
# bypass its strict public/capsule CLI admission and never enters timing.
sys.path.insert(0,str(ROOT/'benchmarks'))
benchmark=load('decisions_endian_unchanged694',ROOT/'benchmarks/rank_decisions.py')
FIXED=tuple(benchmark.corpus())
FIXTURE=json.loads((ROOT/'coverage/contracts/rank-decisions-benchmark.json').read_text())
assert len(FIXED)==694 and sum(bool(c.frozen_id) for c in FIXED)==502
assert sum(not c.frozen_id and not c.timed for c in FIXED)==16

@pytest.mark.parametrize('index',range(694))
def test_unchanged694_fixed_cases_including502_source_and16_metadata(index,active,no_forward):
    case=FIXED[index]
    assert benchmark.inventory(case)==FIXTURE['case_inventory'][index]
    actual=benchmark.native_observe(benchmark.observe_case,getattr(active,case.name),case)
    assert benchmark.digest(actual)==FIXTURE['reference_profiles'][profile()]['observations'][str(index)],index
