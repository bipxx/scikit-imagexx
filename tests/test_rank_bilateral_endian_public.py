"""R6 conversion-provider parity; explicit private prototype is not public proof."""
import ast
import hashlib
import importlib
import importlib.util
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
DATA=json.loads((ROOT/'coverage/contracts/rank-bilateral-endian-public.json').read_text())
NAMES=('mean_bilateral','pop_bilateral','sum_bilateral')
PATHS={prefix+name for prefix in ('filters.rank.','filters.rank.bilateral.') for name in NAMES}
PROTOTYPE=os.environ.get(DATA['prototype_environment'])

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

source=load('skimagexx._rank_endian_source_prototype',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source._core=importlib.import_module(PROTOTYPE)

@pytest.fixture
def target():
    if PROTOTYPE:return source
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Genuine bilateral endian public package pending; no alternate backend substitution')
    assert PATHS<=registered
    return importlib.import_module('skimagexx._rank')

@pytest.fixture(params=(1,4))
def active(request,target):
    previous=target._core.get_num_threads();policy=xx.set_fallback_enabled(False);target._core.set_num_threads(request.param)
    try:yield target
    finally:target._core.set_num_threads(previous);xx.set_fallback_enabled(policy)

@pytest.fixture
def no_forward(monkeypatch,active):
    def forbidden(*a,**k):raise AssertionError('upstream bilateral image algorithm/converter forwarding')
    monkeypatch.setattr(active,'fallback',forbidden)
    bilateral=importlib.import_module('skimage.filters.rank.bilateral')
    for provider in (reference,bilateral):
        for name in NAMES:monkeypatch.setattr(provider,name,forbidden)
    for name in ('_mean','_pop','_sum'):monkeypatch.setattr(bilateral.bilateral_cy,name,forbidden)
    monkeypatch.setattr(importlib.import_module('skimage.util.dtype'),'img_as_ubyte',forbidden)
    monkeypatch.setattr(importlib.import_module('skimage.filters.rank.generic'),'img_as_ubyte',forbidden)


def profile():return 'windows' if os.name=='nt' else 'minimum' if np.lib.NumpyVersion(np.__version__)<'2.0.0' else 'modern'

def trace(events):
    # Only identified post-radius original Cython scratch allocations are
    # outside the conversion-provider trace. Earlier allocations stay exact.
    result=[];radius_seen=False
    for item in events:
        if item=='radius' or isinstance(item,list) and item and item[0] in ('s0-index','s1-index'):radius_seen=True
        if radius_seen and isinstance(item,list) and len(item)>=2 and item[0] in ('empty','allocation') and item[1]=='int64':continue
        result.append(item)
    return result


def test_projection_keeps_all_pre_radius_and_conversion_events():
    earlier=['empty','int64',[9]];conversion=['empty','uint8',[3,3]];radius=['s0-index'];scratch=['empty','int64',[9]];signed=['empty','int16',[3,3]];notice=['warning','must stay']
    assert trace([earlier,conversion,radius,scratch,signed,notice])==[earlier,conversion,radius,signed,notice]


def primary_observe(function,key):
    name,field,token,layout=key.split('/');maximum=int(token[3:]);events=[]
    raw=np.array([[0,1,2],[3,4,5],[6,7,maximum]],dtype='>u2');raw[raw>maximum]=maximum
    if layout=='step':owner=np.zeros((3,6),dtype='>u2');owner[:,::2]=raw;raw=owner[:,::2]
    image=raw if field=='image' else np.arange(9,dtype=np.uint8).reshape(3,3);mask=None if field=='image' else raw
    class Footprint(np.ndarray):
        def __gt__(self,value):events.append(['footprint-positive']);return super().__gt__(value)
    footprint=np.ones((3,3),np.uint8).view(Footprint);output=np.full((3,3),99,np.float32)
    class Radius:
        def __init__(self,label):self.label=label
        def __index__(self):events.append([self.label+'-index']);return 10
    old_floor=np.floor_divide;old_empty=np.empty
    def floor(*args,**kw):events.append(['floor_divide',str(kw.get('dtype')),kw.get('casting'),int(args[1]),tuple(args[0].shape)]);return old_floor(*args,**kw)
    def empty(*args,**kw):
        result=old_empty(*args,**kw);events.append(['empty',str(result.dtype),tuple(result.shape)]);return result
    before=image.tobytes();mask_before=None if mask is None else mask.tobytes()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');old_show=warnings.showwarning
        def show(message,category,filename,lineno,file=None,line=None):events.append(['warning',category.__name__,str(message)]);return old_show(message,category,filename,lineno,file,line)
        warnings.showwarning=show;np.floor_divide=floor;np.empty=empty
        try:
            result=function(image,footprint,out=output,mask=mask,s0=Radius('s0'),s1=Radius('s1'))
            observed=dict(error=None,result_dtype=str(result.dtype),result_bytes=result.tobytes().hex(),shares_output=bool(np.shares_memory(result,output)))
        except Exception as e:observed=dict(error=dict(type=type(e).__name__,message=str(e)))
        finally:np.floor_divide=old_floor;np.empty=old_empty;warnings.showwarning=old_show
    return dict(id=key,maximum=maximum,field=field,layout=layout,events=trace(events),warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],image_unchanged=image.tobytes()==before,mask_unchanged=mask is None or mask.tobytes()==mask_before,output_unchanged=bool(np.all(output==99)),output_bytes=output.tobytes().hex(),**observed)


def hook_observe(function,key):
    name,mode=key.split('/');events=[];maximum=256 if mode in ('high','precision-high-to-low','allocation-to-native') else 255
    image=np.full((3,3),maximum,dtype='>u2');out=np.full((3,3),99,np.float32);fp=np.ones((3,3),np.uint8);changed=False
    class Radius:
        def __index__(self):events.append('radius');return 10
    old_empty=np.empty
    def empty(*a,**kw):
        nonlocal changed
        b=old_empty(*a,**kw);events.append(['allocation',str(b.dtype)])
        if mode=='allocation-to-native' and not changed:
            changed=True;image.byteswap(inplace=True);image.dtype=np.dtype('u2');events.append('allocation-mutated-native')
        return b
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');old_show=warnings.showwarning
        if mode=='precision-error':warnings.filterwarnings('error',message='Possible precision loss.*')
        if mode=='downcast-error':warnings.filterwarnings('error',message='Downcasting.*')
        def show(msg,cat,filename,lineno,file=None,line=None):
            events.append(['warning',str(msg)])
            if str(msg).startswith('Possible precision loss'):
                if mode=='precision-low-to-high':image.fill(256);events.append('precision-mutated256')
                elif mode=='precision-high-to-low':image.fill(255);events.append('precision-mutated255')
            elif str(msg).startswith('Downcasting') and mode=='downcast-to-high':image.fill(256);events.append('downcast-mutated256')
            return old_show(msg,cat,filename,lineno,file,line)
        warnings.showwarning=show;np.empty=empty
        try:result=function(image,fp,out=out,s0=Radius(),s1=Radius());error=None;result_bytes=result.tobytes().hex()
        except Exception as e:error=dict(type=type(e).__name__,message=str(e));result_bytes=None
        finally:warnings.showwarning=old_show;np.empty=old_empty
    return dict(id=key,error=error,events=trace(events),warnings=[str(n.message) for n in notices],image_dtype=str(image.dtype),image_values=image.tolist(),output_bytes=out.tobytes().hex(),output_unchanged=bool(np.all(out==99)),result_bytes=result_bytes)


def type_observe(function,key):
    _,dtype,maximum=key.split('/');maximum=int(maximum);value=.5 if dtype=='>f4' else maximum
    image=np.full((3,3),value,dtype=dtype);out=np.full((3,3),99,np.float32)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:function(image,np.ones((3,3),np.uint8),out=out);error=None
        except Exception as e:error=dict(type=type(e).__name__,message=str(e))
    return dict(id=key,error=error,warnings=[str(n.message) for n in notices],output_bytes=out.tobytes().hex(),output_unchanged=bool(np.all(out==99)))


def normalize(value):return json.loads(json.dumps(value))
CASES=tuple(row['id'] for row in DATA['observations']['windows'])

@pytest.mark.parametrize('key',CASES)
def test_all132_source_defined_endian_boundaries_hooks_and_types(key,active,no_forward):
    expected=next(r for r in DATA['observations'][profile()] if r['id']==key)
    if key.startswith('type/'):actual=type_observe(active.mean_bilateral,key)
    elif '/max' in key:actual=primary_observe(getattr(active,key.split('/')[0]),key)
    else:actual=hook_observe(getattr(active,key.split('/')[0]),key)
    assert normalize(actual)==expected,key


def test_nonbilateral_ast_and_two_callsite_scope():
    before=ast.parse(DATA['nonbilateral_source']);after=ast.parse((ROOT/'python/skimagexx/_rank.py').read_text())
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name=='_bilateral_apply':
            calls=[n for n in ast.walk(b) if isinstance(n,ast.Name) and n.id=='_bilateral_ubyte'];assert len(calls)==2
            for n in calls:n.id='_ubyte'
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert len(after.body)==len(before.body)+1 and after.body[-1].name=='_bilateral_ubyte'
    # Outside the one new integer branch, the new helper duplicates old intake
    # and native/provider stages exactly, without an extra asarray call.
    old=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='_ubyte')
    new=ast.parse((ROOT/'python/skimagexx/_rank.py').read_text()).body[-1];new.name='_ubyte'
    extras=[n for n in new.body if isinstance(n,ast.If) and 'not dtype.isnative' in ast.unparse(n.test)];assert len(extras)==1
    new.body.remove(extras[0]);assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)


def test_actual_package_or_explicit_private_pending(target):
    core=target._core
    assert core.rank_bilateral_interface=='rank-bilateral-v1' and core.rank_bilateral_capabilities==0x1c000000
    assert core.get_num_threads()==1
    if PROTOTYPE:
        assert hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() in set(DATA['compiled_module_sha256'].values())
        if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
        pytest.skip('Source wrapper with immutable b7 private core only; genuine186-successor public package pending')
    assert core is xx._core
    assert xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads
    rank=importlib.import_module('skimagexx.filters.rank');nested=importlib.import_module('skimagexx.filters.rank.bilateral')
    for name in NAMES:assert getattr(rank,name) is getattr(nested,name) is getattr(target,name)

# Frozen benchmark fixtures are consumed directly, without running a timing CLI
# or substituting its strict registry/capsule admission. This is private parity.
sys.path.insert(0,str(ROOT/'benchmarks'))
benchmark=load('rank_bilateral_endian_fixed_corpus',ROOT/'benchmarks/rank_bilateral.py')
FIXED=tuple(benchmark.corpus())
FROZEN=json.loads((ROOT/'coverage/contracts/rank-bilateral-benchmark.json').read_text())
assert len(FIXED)==len(FROZEN['case_inventory'])==1008

@pytest.mark.parametrize('index',range(1008))
def test_unchanged1008_corpus_including708_original_contracts(index,active,no_forward):
    case=FIXED[index]
    assert benchmark.inventory(case)==FROZEN['case_inventory'][index]
    actual=benchmark.native_observe(benchmark.observe_case,getattr(active,case.name),case)
    assert benchmark.digest(actual)==FROZEN['reference_profiles'][profile()]['observations'][str(index)],index
