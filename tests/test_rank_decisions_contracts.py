"""Source-qualified original local rank decisions; forwarding is never native proof.

The Cython core has unchecked buffers. Cases reuse the previously qualified
neighborhood fixture, excluding feedback overlaps, and bound all allocations.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
import scipy
import skimage
from skimage.filters import rank as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/rank-decisions.json'
NAMES=('otsu','majority')
spec=importlib.util.spec_from_file_location('rank_decisions_qualified_base',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


def cases():
    rows=[]
    for name in NAMES:
        for key,old,parameters in base.CASES:
            if old!='minimum' or parameters.get('out') in ('view','overlap'):continue
            rows.append((name+'/'+key.split('/',1)[1],name,dict(parameters)))
        for ndim in (2,3):
            for pattern in ('flat-zero','flat-seven','two-tied','three-tied','two-modes','zero-window','single-bin','wide-u16','full-u16','disjoint-owner'):
                rows.append((f'{name}/{ndim}d/decision={pattern}',name,dict(ndim=ndim,decision=pattern)))
            for output in ('u1','u2','f4','f8','bool'):
                rows.append((f'{name}/{ndim}d/ties/out={output}',name,dict(ndim=ndim,decision='three-tied',decision_out=output)))
        rows.append((name+'/missing-footprint',name,dict(omitted=True)))
        rows.append((name+'/positional-out',name,dict(positional=True)))
    assert len({r[0] for r in rows})==len(rows)
    return rows


CASES=cases()


def make_case(parameters):
    if parameters.get('omitted'):
        image=np.arange(6,dtype=np.uint8).reshape(2,3);return (image,),{},dict(image=image)
    if parameters.get('positional'):
        image=np.arange(6,dtype=np.uint8).reshape(2,3);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.uint8)
        return (image,fp,out),{},dict(image=image,footprint=fp,out=out)
    if 'decision' not in parameters:return base.make_case(parameters)
    ndim=parameters['ndim'];pattern=parameters['decision'];shape=(5,7) if ndim==2 else (3,4,5)
    values={'flat-zero':[0],'flat-seven':[7],'two-tied':[3,9],'three-tied':[0,4,8],'two-modes':[0,1,1,8,8,9],'single-bin':[255],'wide-u16':[0,1023,1024,4095],'full-u16':[0,1,32767,65534,65535]}.get(pattern,[0,1,4,7,8,15])
    dtype=np.uint16 if pattern in ('wide-u16','full-u16') else np.uint8
    image=np.resize(np.array(values,dtype),math.prod(shape)).reshape(shape);fp=np.ones((3,)*ndim,np.uint8);options={};owners={}
    if pattern=='zero-window':options['mask']=np.zeros(shape,np.uint8)
    if pattern=='disjoint-owner':
        owner=np.full(image.size*2+16,99,np.uint8);owner[:image.size]=image.ravel();image=owner[:image.size].reshape(shape);options['out']=owner[image.size+16:].reshape(shape);owners['shared_owner']=owner
    if parameters.get('decision_out'):options['out']=np.full(shape,99,np.dtype(parameters['decision_out']))
    return (image,fp),options,dict(image=image,footprint=fp,**options,**owners)


def array(value):
    if isinstance(value,np.ndarray):
        data=repr(value.tolist()).encode() if value.dtype.hasobject else value.tobytes(order='C')
        result=dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),nbytes=value.nbytes,values_sha256=hashlib.sha256(data).hexdigest())
        if value.size<=64 and value.dtype.kind in 'buif' and np.isfinite(value).all():result['values']=value.tolist()
        return result
    if isinstance(value,(list,tuple)):return dict(type=type(value).__name__,items=[array(v) for v in value])
    return dict(type=type(value).__name__,value=repr(value.item() if isinstance(value,np.generic) else value))


def safety(args,options,tracked):
    assert all(v.nbytes<=8*1024*1024 for v in tracked.values() if isinstance(v,np.ndarray))
    # For these tiny neighborhoods, all integer histogram moment intermediates
    # are well within signed64 even at value65535. No negative histogram counts
    # from feedback buffers are admitted. Converted float aliases are copied
    # before the kernel; same-object native input/out is rejected upstream.
    assert all(v.size<=65536 for v in tracked.values() if isinstance(v,np.ndarray))
    if len(args)>=2 and isinstance(args[1],np.ndarray):assert args[1].size<=1024


def observe(function,name,parameters):
    args,options,tracked=make_case(parameters);safety(args,options,tracked)
    inputs=[v for v in tracked.values() if isinstance(v,np.ndarray)];before={k:array(v) for k,v in tracked.items()};pointers={k:v.ctypes.data for k,v in tracked.items() if isinstance(v,np.ndarray)}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value=function(*args,**options);result=array(value)
            if isinstance(value,np.ndarray):result.update(identity=[value is a for a in inputs],alias=[bool(np.shares_memory(value,a)) for a in inputs])
            error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={k:array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==ptr for k,ptr in pointers.items()})


class FootprintEvent(np.ndarray):
    def __gt__(self,other):
        self.events.append('footprint>0')
        if self.fail:raise ValueError('footprint conversion')
        return super().__gt__(other)


class OutputEvent(np.ndarray):
    def __array_finalize__(self,parent):self.events=getattr(parent,'events',[])
    def reshape(self,*args,**kwargs):
        self.events.append('out.reshape')
        return super().reshape(*args,**kwargs)


class ShiftEvent:
    def __init__(self,name,events,fail=False):self.name=name;self.events=events;self.fail=fail
    def __index__(self):
        self.events.append(self.name)
        if self.fail:raise ValueError(self.name+' conversion')
        return 0


PROVIDERS=('success','footprint','shift_x','shift_y','shift_z','bad-out','bad-image','image-array-only')


class ImageArrayOnly:
    def __init__(self,image,events):self.image=image;self.events=events
    def __array__(self,dtype=None,copy=None):
        self.events.append('image.__array__')
        return np.asarray(self.image,dtype=dtype)


def provider_observe(name,ndim,failure):
    events=[];shape=(3,4) if ndim==2 else (2,3,4);image=np.arange(math.prod(shape),dtype=np.uint8).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8).view(FootprintEvent);fp.events=events;fp.fail=failure=='footprint'
    out=np.full(shape,99,np.int32 if failure=='bad-out' else np.uint8).view(OutputEvent);out.events=events
    if failure=='bad-image':image=image.astype(bool)
    elif failure=='image-array-only':image=ImageArrayOnly(image,events)
    # Cython's signed-char conversion need not invoke __index__. Freeze that
    # fact rather than assuming all index providers are accepted as integers.
    options=dict(out=out,shift_x=0,shift_y=0,shift_z=0)
    if failure in ('shift_x','shift_y','shift_z'):options[failure]=ShiftEvent(failure,events,True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:value=getattr(reference,name)(image,fp,**options);error=None
        except Exception as caught:value=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,output=array(out),result=None if value is None else array(value),warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


def test_source_facts_safety_and_strict_pending(frozen):
    assert frozen['pinned_skimage']==skimage.__version__=='0.26.0'
    assert frozen['native_checks']==0 and frozen['canonical_cases']==len(CASES)
    assert frozen['public_paths']==4 and frozen['algorithms']==2
    assert frozen['fixture_source_sha256']==text_sha(Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
    assert frozen['case_contract_sha256']==digest(CASES)
    assert all(p.get('out') not in ('view','overlap') for _,_,p in CASES)
    assert all(digest(value)==key for key,value in frozen['observations'].items())
    for snapshots in frozen['snapshots'].values():assert set(snapshots)=={key for key,_,_ in CASES}
    assert reference.otsu is not importlib.import_module('skimage.filters').threshold_otsu

    import ast, copy
    root=Path(__file__).parents[1]
    migration=json.loads((root/'coverage/evidence/rank-decisions-fixture-assertion-migration.json').read_text())
    old_text=migration['original_fixture_utf8']
    assert hashlib.sha256(old_text.encode()).hexdigest()==migration['original_fixture_canonical_lf_sha256']=='49909f317f19fb31ae2c41d4084ece5d97e69bf54bd45370a3068984699c53f1'
    new_text=Path(__file__).read_bytes().replace(b'\r\n',b'\n').decode()
    assert hashlib.sha256(new_text.encode()).hexdigest()==migration['current_fixture_canonical_lf_sha256']
    old=ast.parse(old_text);new=ast.parse(new_text)
    allowed={'test_source_facts_safety_and_strict_pending','test_strict_pending_never_counts_forwarding'}
    assert set(migration['allowed_fixture_test_bodies'])==allowed and len(old.body)==len(new.body)
    changed=set()
    for a,b in zip(old.body,new.body):
        if isinstance(b,ast.FunctionDef) and b.name in allowed:
            assert isinstance(a,ast.FunctionDef) and a.name==b.name
            b.body=copy.deepcopy(a.body);changed.add(b.name)
        assert ast.dump(a)==ast.dump(b)
    assert changed==allowed
    pins={
        'namespace29':'12eb630bb4be789be2693b99e543b7fa280d593ab14151a4791f856cc748acfc',
        'bilateral_provider':'c86646bd9314637a34a3628fef31d513817858277b8ff89dbc1d04ca6c0cc6e4',
        'decisions_append':'da6fd4934da9348ba4ed0e04940b21561ffc2e876498a730672cb26f6ef96b28',
        'composed':'d05a3c6cc82edce9f2ed7d9522155b159f4c639c5bed9ed6f0955562457eb6c4',
        'integer_provider':'18bd8cfa53f585262fada6a9bfdb5d421c907971cb08f809d308081b44611033',
        'base':'18bd8cfa53f585262fada6a9bfdb5d421c907971cb08f809d308081b44611033'}
    stages=migration['rank_stages'];assert set(stages)==set(pins)
    trees={}
    for name,pin in pins.items():
        assert hashlib.sha256(stages[name]['utf8'].encode()).hexdigest()==stages[name]['canonical_lf_sha256']==pin
        trees[name]=ast.parse(stages[name]['utf8'])
    assert len(trees['namespace29'].body)==66 and len(trees['bilateral_provider'].body)==67
    assert len(trees['decisions_append'].body)==71 and len(trees['composed'].body)==72
    assert ast.dump(ast.Module(body=trees['decisions_append'].body[:66],type_ignores=[]))==ast.dump(trees['namespace29'])
    assert [n.name for n in trees['decisions_append'].body[66:]]==['_require_decisions','_decisions_call','_decisions_apply','otsu','majority']
    assert ast.dump(trees['composed'])==ast.dump(ast.Module(body=trees['bilateral_provider'].body+trees['decisions_append'].body[66:],type_ignores=[]))
    assert ast.dump(trees['integer_provider'])==ast.dump(trees['base'])
    # Authenticate the sole approved provider delta on the exact composed source.
    normalized=copy.deepcopy(trees['integer_provider'])
    for a,b in zip(trees['composed'].body,normalized.body):
        if isinstance(b,ast.FunctionDef) and b.name=='_decisions_apply':
            calls=[n for n in ast.walk(b) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_bilateral_ubyte']
            assert len(calls)==2 and {n.args[0].id for n in calls}=={'image','mask'}
            for n in calls:
                assert len(n.keywords)==1 and n.keywords[0].arg=='_integer_provider' and ast.literal_eval(n.keywords[0].value) is True
                n.func.id='_ubyte';n.keywords=[]
        if isinstance(b,ast.FunctionDef) and b.name=='_bilateral_ubyte':
            assert len(b.args.kwonlyargs)==1 and b.args.kwonlyargs[0].arg=='_integer_provider' and ast.literal_eval(b.args.kw_defaults[0]) is False
            b.args.kwonlyargs=[];b.args.kw_defaults=[]
            conditions=[n for n in b.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and isinstance(n.test.values[-1],ast.BoolOp) and isinstance(n.test.values[-1].op,ast.Or)]
            assert len(conditions)==1 and ast.unparse(conditions[0].test.values[-1])=='not dtype.isnative or _integer_provider'
            conditions[0].test.values[-1]=conditions[0].test.values[-1].values[0]
        assert ast.dump(a)==ast.dump(b)
    current=ast.parse((root/'python/skimagexx/_rank.py').read_text())
    exports=ast.parse("_DECISIONS_FUNCTIONS=('otsu','majority')\n__all__+=list(_DECISIONS_FUNCTIONS)\n")
    assert ast.dump(ast.parse(migration['appended_export_utf8']))==ast.dump(exports)
    assert ast.dump(current)==ast.dump(ast.Module(body=trees['base'].body+exports.body,type_ignores=[]))
    assert migration['numerical_cases']==502 and migration['provider_traces']==32 and migration['independent_oracles']==256
    assert migration['public_admission'] is False


@pytest.mark.parametrize('name',NAMES)
def test_original_signatures_exports_identity(name,frozen):
    generic=importlib.import_module('skimage.filters.rank.generic');function=getattr(reference,name);facts=frozen['functions'][name]
    assert function is getattr(generic,name)
    assert str(inspect.signature(function))==facts['signature']
    assert (name in reference.__all__)==facts['canonical_in_all']
    assert (name in generic.__all__)==facts['generic_in_all']
    assert text_sha(inspect.getsourcefile(function))==frozen['generic_python_sha256']
    assert hashlib.sha256(inspect.getsource(function).replace('\r\n','\n').encode()).hexdigest()==facts['python_source_sha256']


@pytest.mark.parametrize('namespace',('filters.rank','filters.rank.generic'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_frozen_contract(case,namespace,frozen):
    key,name,parameters=case;function=getattr(importlib.import_module('skimage.'+namespace),name)
    assert observe(function,name,parameters)==frozen['observations'][frozen['snapshots'][profile()][key]],key


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('namespace',('filters.rank','filters.rank.generic'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_strict_pending_never_counts_forwarding(case,namespace,workers):
    key,name,parameters=case;policy=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        paths={prefix+n for prefix in ('filters.rank.','filters.rank.generic.') for n in NAMES}
        assert paths<=set(native_functions()),'genuine four-path public decisions registration is required'
        frozen=json.loads(MATRIX.read_text())
        actual=observe(getattr(importlib.import_module('skimagexx.'+namespace),name),name,parameters)
        assert actual==frozen['observations'][frozen['snapshots'][profile()][key]],key
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(policy)


def neighborhoods(image,footprint,mask,shifts):
    center=tuple(n//2+s for n,s in zip(footprint.shape,(shifts[1],shifts[0]) if image.ndim==2 else shifts))
    for position in np.ndindex(image.shape):
        counts=Counter()
        for offset in np.ndindex(footprint.shape):
            point=tuple(p+i-c for p,i,c in zip(position,offset,center))
            if footprint[offset] and all(0<=i<n for i,n in zip(point,image.shape)) and mask[point]:counts[int(image[point])]+=1
        yield position,counts


def decision(name,counts):
    pop=sum(counts.values())
    if not pop:return 0
    if name=='majority':return min(counts,key=lambda i:(-counts[i],i))
    # Reconstruct the source's expression tree: pop is float64 while moments,
    # q1 and denominators are signed integers. Occupied bins only, strict >.
    mean=sum(i*n for i,n in counts.items());q1=counts.get(0,0);mu1=0;best=0.;result=0
    for i in sorted(k for k in counts if k):
        n=counts[i];q1+=n
        if q1==pop:break
        mu1+=i*n;mu2=mean-mu1
        t=(float(pop)-q1)*mu1-mu2*q1
        score=(t*t)/(q1*(float(pop)-q1))
        if score>best:best=score;result=i
    return result


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','empty','masked','shifted','tied','full-bin'))
@pytest.mark.parametrize('output',('same','u1','f4','f8'))
def test_independent_neighborhood_decision_replay(name,ndim,dtype,kind,output):
    shape=(5,7) if ndim==2 else (3,4,5);image=(np.arange(math.prod(shape)).reshape(shape)%31).astype(dtype)
    fp=np.ones(((2,4) if ndim==2 else (2,3,4)) if kind=='even' else (3,)*ndim,np.uint8);mask=np.ones(shape,np.uint8);shifts=(0,0,0)
    if kind=='disconnected':fp.flat[1:-1]=0
    elif kind=='empty':mask.fill(0)
    elif kind=='masked':mask.flat[::3]=0
    elif kind=='shifted':shifts=(1,-1,0)
    elif kind=='tied':image=np.resize(np.array([0,4,8],dtype),image.size).reshape(shape)
    elif kind=='full-bin':image=np.resize(np.array([0,1,np.iinfo(dtype).max-1,np.iinfo(dtype).max],dtype),image.size).reshape(shape)
    target=image.dtype if output=='same' else np.dtype(output);expected=np.empty(shape,target)
    for point,counts in neighborhoods(image,fp,mask,shifts):
        value=decision(name,counts)
        expected[point]=value%256 if target==np.dtype('u1') else value
    options=dict(mask=mask,shift_x=shifts[0],shift_y=shifts[1],shift_z=shifts[2])
    if output!='same':options['out']=np.full(shape,99,target)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');actual=getattr(reference,name)(image,fp,**options)
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('failure',PROVIDERS)
def test_provider_order_and_prewrite_state(name,ndim,failure,frozen):
    assert provider_observe(name,ndim,failure)==frozen['provider_traces'][profile()][f'{name}/{ndim}/{failure}']


def test_otsu_endpoint_and_majority_tie_definitions():
    assert decision('majority',Counter({0:2,4:2,8:2}))==0
    assert decision('majority',Counter({4:2,8:2}))==4
    assert decision('otsu',Counter({1:3,9:3}))==1
    assert decision('otsu',Counter({0:3,9:3}))==0
    assert decision('otsu',Counter({7:6}))==0
    assert decision('otsu',Counter())==0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True);args=parser.parse_args()
    assert skimage.__version__=='0.26.0'
    report=dict(profile=profile(),runtime=dict(python=os.sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__),generic_python_sha256=text_sha(inspect.getsourcefile(reference.otsu)),generic_extension_sha256=hashlib.sha256(Path(importlib.import_module('skimage.filters.rank.generic').generic_cy.__file__).read_bytes()).hexdigest(),snapshots={key:observe(getattr(reference,name),name,p) for key,name,p in CASES},provider_traces={f'{name}/{ndim}/{failure}':provider_observe(name,ndim,failure) for name in NAMES for ndim in (2,3) for failure in PROVIDERS})
    args.record.parent.mkdir(parents=True,exist_ok=True);args.record.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(profile(),len(CASES),'original-only cases; native0')
