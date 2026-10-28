"""Pinned, allocation-safe original rank bilateral contracts; native0.

Public bilateral aliases are separate paths, not extra algorithms. Arithmetic
limits are certified before unchecked Cython; no unsafe extent/feedback probes.
"""
import argparse
from collections import Counter
import hashlib,importlib,importlib.util,inspect,json,math,os,sys,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import scipy,skimage,skimagexx as xx
from skimage.filters import rank as reference
from skimage.filters.rank import bilateral as bilateral_module
from skimagexx.coverage import native_functions

ROOT=Path(__file__).resolve().parents[1];MATRIX=ROOT/'coverage/contracts/rank-bilateral.json'
NAMES=('mean_bilateral','pop_bilateral','sum_bilateral')
CANONICAL=tuple('filters.rank.'+n for n in NAMES)
ALIASES=tuple('filters.rank.bilateral.'+n for n in NAMES)
LIMIT=8*1024*1024
_spec=importlib.util.spec_from_file_location('rank_bilateral_base_contract',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


class IndexValue:
 def __init__(self,label,value,events):self.label,self.value,self.events=label,value,events
 def __index__(self):self.events.append(self.label+'.__index__');return self.value
class IntOnly:
 def __init__(self,label,events):self.label,self.events=label,events
 def __int__(self):self.events.append(self.label+'.__int__');return 3
class Both(IndexValue):
 def __int__(self):self.events.append(self.label+'.__int__');return self.value+1
class FloatOnly:
 def __init__(self,label,events):self.label,self.events=label,events
 def __float__(self):self.events.append(self.label+'.__float__');return 3.

TOKENS=('default','zero','one','negative','float','np-f16','np-f32','np-f64','nan','inf','np-nan','np-i8','np-i64','np-u64','bool','np-bool','array0-int','array1-int','array0-float','numeric-string','bad-string','list','tuple','none','object','py-min','py-max','np-min','np-max','positive-overflow','negative-overflow','u64-overflow','index','int-only','both','float-only','index-bool','index-bad')


def parameter(token,label,events):
 values={'zero':0,'one':1,'negative':-3,'float':2.75,'np-f16':np.float16(2.75),'np-f32':np.float32(2.75),'np-f64':np.float64(2.75),'nan':float('nan'),'inf':float('inf'),'np-nan':np.float32(np.nan),'np-i8':np.int8(-3),'np-i64':np.int64(3),'np-u64':np.uint64(3),'bool':True,'np-bool':np.bool_(True),'array0-int':np.array(3),'array1-int':np.array([3]),'array0-float':np.array(3.),'numeric-string':'3','bad-string':'bad','list':[3],'tuple':(3,),'none':None,'object':SimpleNamespace(),'py-min':-sys.maxsize-1,'py-max':sys.maxsize,'np-min':np.int64(-sys.maxsize-1),'np-max':np.int64(sys.maxsize),'positive-overflow':sys.maxsize+1,'negative-overflow':-sys.maxsize-2,'u64-overflow':np.uint64(sys.maxsize)+np.uint64(1)}
 if token in values:return values[token]
 if token=='index':return IndexValue(label,3,events)
 if token=='int-only':return IntOnly(label,events)
 if token=='both':return Both(label,3,events)
 if token=='float-only':return FloatOnly(label,events)
 if token=='index-bool':return IndexValue(label,True,events)
 if token=='index-bad':return IndexValue(label,3.5,events)
 raise AssertionError(token)


def cases():
 rows=[]
 for name in NAMES:
  for key,old,p in base.CASES:
   if old=='minimum' and p.get('ndim')==2:rows.append((name+'/'+key.split('/',1)[1],name,dict(p)))
  for field in ('s0','s1'):
   for token in TOKENS:rows.append((name+f'/{field}={token}',name,dict(ndim=2,parameter=token,field=field)))
  for radius in ((0,0),(0,1),(1,0),(1,1),(1,2),(2,1),(-1,5),(5,-1),(-3,-3),(300,300)):
   rows.append((name+'/radii='+str(radius),name,dict(ndim=2,radii=radius)))
  for edge in ('upper-max','lower-min','upper-min-safe','lower-max-safe'):
   rows.append((name+'/safe-edge='+edge,name,dict(ndim=2,safe_edge=edge)))
  for shape in ((),(3,),(0,3),(3,0),(2,3,4),(1,1,1),(2,2,2,2)):
   rows.append((name+'/dimensions='+str(shape),name,dict(ndim=2,shape=shape)))
  rows.append((name+'/missing-footprint',name,dict(omitted=True)))
  for output in ('tagged-u1','tagged-u2','tagged-f4','tagged-f8','owned-view'):
   rows.append((name+'/output-owner='+output,name,dict(ndim=2,output_owner=output)))
  for kind in ('image-view','image-offset','mask-view','mask-offset'):
   for dtype in ('u1','u2'):
    for layout in ('C','F-image','reverse-image','step-image'):
     if kind.startswith('mask') and layout!='C':continue
     rows.append((name+f'/safe-alias/{dtype}/{kind}/{layout}',name,dict(ndim=2,alias=kind,alias_dtype=dtype,alias_layout=layout)))
 assert len(rows)==len({key for key,_,_ in rows})
 return rows
CASES=cases()


def alias_fixture(p):
 shape=(3,5);n=math.prod(shape);dtype=np.dtype(p['alias_dtype']);kind=p['alias'];layout=p['alias_layout'];tracked={}
 # Full uint16 capacity is required for image feedback: every stored uint16
 # remains an accessible bin, including wrapped signed intermediate values.
 if kind.startswith('image'):
  owner=(np.arange(2*n+2)%29).astype(dtype)
  if dtype==np.uint16:owner[n-1]=65535
  if layout=='F-image':image=owner[:n].reshape((shape[1],shape[0])).T
  elif layout=='reverse-image':image=owner[:n].reshape(shape)[::-1,::-1]
  elif layout=='step-image':image=owner[:2*n:2].reshape(shape)
  else:image=owner[:n].reshape(shape)
  if dtype==np.uint16:image.flat[-1]=65535
  offset=1 if kind=='image-offset' else 0;out=owner[offset:offset+n].reshape(shape)
  fp=np.ones((3,3),np.uint8);mask=(np.indices(shape).sum(axis=0)%3!=0).astype(np.uint8)
  tracked['owner']=owner
 else:
  image=(np.arange(n).reshape(shape)%29).astype(dtype)
  if dtype==np.uint16:image.flat[-1]=65535
  owner=np.ones(n+1,np.uint8);mask=owner[:n].reshape(shape);offset=1 if kind=='mask-offset' else 0;out=owner[offset:offset+n].reshape(shape);fp=np.ones((3,3),np.uint8);tracked['mask_owner']=owner
 options=dict(out=out,mask=mask,s0=10,s1=10);tracked.update(image=image,footprint=fp,out=out,mask=mask)
 return (image,fp),options,tracked


def make_case(name,p):
 if p.get('omitted'):
  image=np.arange(6,dtype=np.uint8).reshape(2,3);return (image,),{},dict(image=image)
 if p.get('alias'):return alias_fixture(p)
 args,options,tracked=base.make_case(p)
 if 'parameter' in p:
  # Extreme accepted integers are exercised only under the source's explicit
  # population-zero guard: no i-s0/i+s1 arithmetic is evaluated.
  events=[];fp=np.zeros((3,3),np.uint8);out=np.full(args[0].shape,99,np.uint8);args=(args[0],fp);options=dict(out=out)
  token=p['parameter']
  if token!='default':options[p['field']]=parameter(token,p['field'],events)
  tracked=dict(image=args[0],footprint=fp,out=out,parameter_events=events)
 if 'radii' in p:options.update(zip(('s0','s1'),p['radii']))
 if 'safe_edge' in p:
  table={'upper-max':(sys.maxsize,10),'lower-min':(10,-sys.maxsize-1),'upper-min-safe':(65535-sys.maxsize,10),'lower-max-safe':(10,sys.maxsize-65535)}
  options.update(zip(('s0','s1'),table[p['safe_edge']]))
 if 'output_owner' in p:
  token=p['output_owner']
  if token=='owned-view':
   owner=np.full(args[0].size+2,99,np.uint8);out=owner[1:-1].reshape(args[0].shape);tracked['out_owner']=owner
  else:out=np.full(args[0].shape,99,np.dtype(token[7:],metadata={'rank':'bilateral-output'}))
  options['out']=out;tracked['out']=out
 return args,options,tracked


def array(value):
 if isinstance(value,np.ndarray):
  data=repr(value.tolist()).encode() if value.dtype.hasobject else value.tobytes(order='C')
  result=dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),nbytes=value.nbytes,values_sha256=hashlib.sha256(data).hexdigest())
  if value.size<=64 and value.dtype.kind in 'buif' and np.isfinite(value).all():result['values']=value.tolist()
  return result
 if isinstance(value,(list,tuple)):return dict(type=type(value).__name__,items=[array(v) for v in value])
 return dict(type=type(value).__name__,value=repr(value.item() if isinstance(value,np.generic) else value))


def safety(name,args,options,tracked,p):
 assert all(v.nbytes<=LIMIT for v in tracked.values() if isinstance(v,np.ndarray))
 # All ordinary fixtures are small; C Py_ssize sums/counts cannot overflow.
 if 'parameter' in p:
  fp=tracked['footprint'];assert fp.dtype==np.uint8 and fp.shape==(3,3) and not fp.any()
 elif p.get('safe_edge'):
  for field,default in (('s0',10),('s1',10)):
   x=options.get(field,default);assert isinstance(x,int)
   assert (65535-sys.maxsize<=x<=sys.maxsize) if field=='s0' else (-sys.maxsize-1<=x<=sys.maxsize-65535)
 else:
  assert all(abs(int(options.get(field,10)))<=2**30 for field in ('s0','s1'))
 if p.get('alias') and p['alias'].startswith('image') and args[0].dtype==np.uint16:assert int(args[0].max())==65535
 out=options.get('out');image=args[0]
 if isinstance(image,np.ndarray) and image.ndim==2 and image.size:
  # Common preprocessing rejects bool before footprint/mask buffer access;
  # the inherited precedence fixture deliberately has a scalar bad mask.
  if image.dtype==bool:return
  if isinstance(out,np.ndarray) and out.ndim>=2:
   assert all(out.shape[k]>=image.shape[k] for k in range(2))
   if out.ndim>=3:assert out.shape[2]>0
  mask=options.get('mask')
  if isinstance(mask,np.ndarray):assert mask.size>=image.size


def observe(function,name,p):
 args,options,tracked=make_case(name,p);safety(name,args,options,tracked,p);inputs=[x for x in tracked.values() if isinstance(x,np.ndarray)];before={k:array(v) for k,v in tracked.items()};ptr={k:v.ctypes.data for k,v in tracked.items() if isinstance(v,np.ndarray)}
 with warnings.catch_warnings(record=True) as notices:
  warnings.simplefilter('always')
  try:
   value=function(*args,**options);result=array(value)
   if isinstance(value,np.ndarray):result.update(identity=[value is x for x in inputs],alias=[bool(np.shares_memory(value,x)) for x in inputs])
   error=None
  except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
 return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={k:array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==v for k,v in ptr.items()})


SCENARIOS=('none','s0','s1','bad-output','bad-shift','bad-footprint','mutate-image','mutate-mask','mutate-footprint','fill-out-fail-s1','image-dtype','out-dtype','warn-mutate-image','warn-mutate-mask','warn-error','notimpl-s0')

def provider_observe(name,kind,scenario):
 image=np.arange(12,dtype=np.uint8).reshape(3,4) if kind=='u1' else np.linspace(0,1,12,dtype=np.float32).reshape(3,4)
 fp=np.ones((3,3),np.uint8);mask=np.ones((3,4),np.uint8);out=np.full((3,4),99,np.int32 if scenario=='bad-output' else np.uint8);events=[]
 tracked=dict(image=image,footprint=fp,mask=mask,out=out);before={k:array(v) for k,v in tracked.items()};pointers={k:v.ctypes.data for k,v in tracked.items()}
 def action():
  if scenario=='mutate-image':image.fill(0)
  elif scenario=='mutate-mask':mask.fill(0)
  elif scenario=='mutate-footprint':fp.fill(0)
  elif scenario=='fill-out-fail-s1':out.fill(77)
  elif scenario=='image-dtype':image.dtype=np.dtype('i1')
  elif scenario=='out-dtype':out.dtype=np.dtype('i1')
 class Event:
  def __init__(self,label,value):self.label,self.value=label,value
  def __index__(self):
   events.append(self.label+'.__index__')
   if self.label=='s0':action()
   if scenario=='notimpl-s0' and self.label=='s0':raise NotImplementedError('s0-callback-payload')
   if scenario==self.label or (scenario=='fill-out-fail-s1' and self.label=='s1'):raise ValueError(self.label+' conversion')
   return self.value
  def __int__(self):events.append(self.label+'.__int__');return self.value+1
 options=dict(out=out,mask=mask,s0=Event('s0',2),s1=Event('s1',3))
 if scenario=='bad-shift':options['shift_x']='bad'
 if scenario=='bad-footprint':fp=None
 with warnings.catch_warnings(record=True) as notices:
  warnings.simplefilter('always')
  previous=warnings.showwarning
  def warning_hook(message,category,*args,**kwargs):
   events.append('warning:'+type(message).__name__+':'+str(message))
   if scenario=='warn-mutate-image':image.fill(0)
   elif scenario=='warn-mutate-mask':mask.fill(0)
   elif scenario=='warn-error':raise NotImplementedError('warning-hook-payload')
   return previous(message,category,*args,**kwargs)
  warnings.showwarning=warning_hook
  try:result=getattr(reference,name)(image,fp,**options);error=None
  except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
 result_info=None if result is None else dict(array(result),identity=[result is value for value in tracked.values()],alias=[bool(np.shares_memory(result,value)) for value in tracked.values()])
 return dict(events=events,error=error,result=result_info,before=before,after={k:array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==ptr for k,ptr in pointers.items()},warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])


def snapshot():return {key:observe(getattr(reference,name),name,p) for key,name,p in CASES}


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


def test_integrity_safe_scope_native0(frozen):
 import ast,copy
 assert frozen['pinned_skimage']==skimage.__version__=='0.26.0' and frozen['native_checks']==0 and frozen['native_pending']==len(CASES)
 assert frozen['cases']==[dict(id=k,function=n,parameters=json.loads(json.dumps(p))) for k,n,p in CASES]
 evidence=json.loads((ROOT/'coverage/evidence/rank-bilateral-fixture-assertion-migration.json').read_text())
 before=evidence['old_fixture_utf8'];current=Path(__file__).read_text()
 reuse=json.loads((ROOT/'coverage/evidence/rank-bilateral-fixture-reuse.json').read_text())
 assert text_sha(ROOT/'coverage/evidence/rank-bilateral-fixture-assertion-migration.json')==reuse['historical_migration_sha256']
 qualified=reuse['qualified_fixture_utf8']
 assert hashlib.sha256(qualified.encode()).hexdigest()==reuse['qualified_fixture_canonical_lf_sha256']==evidence['current_fixture_canonical_lf_sha256']
 signature='\ndef test_strict_pending_never_counts_forwarding(case,namespace,workers,frozen):'
 assertion="\n   assert actual==frozen['observations'][frozen['snapshots'][profile()][key]],(path,key)"
 assert current.count(signature)==current.count(assertion)==1
 inverse=current.replace(signature,signature.replace(',frozen', ''),1).replace(assertion,'\n   frozen=json.loads(MATRIX.read_text())'+assertion,1)
 restored=ast.parse(inverse);prior=ast.parse(qualified)
 assert len(restored.body)==len(prior.body)
 for actual,previous in zip(restored.body,prior.body):
  if isinstance(previous,ast.FunctionDef) and previous.name=='test_integrity_safe_scope_native0':
   assert isinstance(actual,ast.FunctionDef) and actual.name==previous.name
   actual.body=copy.deepcopy(previous.body)
 assert ast.dump(restored)==ast.dump(prior),'fixture reuse changed more than its dependency and integrity proof'
 current=qualified
 assert hashlib.sha256(before.replace('\r\n','\n').encode()).hexdigest()==frozen['fixture_source_sha256']['tests/test_rank_bilateral_contracts.py']==evidence['old_fixture_canonical_lf_sha256']
 assert hashlib.sha256(current.encode()).hexdigest()==evidence['current_fixture_canonical_lf_sha256']
 old=ast.parse(before);new=ast.parse(current)
 allowed={'test_integrity_safe_scope_native0','test_strict_pending_never_counts_forwarding'}
 assert set(evidence['allowed_test_bodies'])==allowed and len(old.body)==len(new.body)
 changed=set()
 for previous,actual in zip(old.body,new.body):
  if isinstance(previous,ast.FunctionDef) and previous.name in allowed:
   assert isinstance(actual,ast.FunctionDef) and actual.name==previous.name
   restored=copy.deepcopy(actual);restored.body=previous.body
   assert ast.dump(restored)==ast.dump(previous)
   assert ast.dump(previous)!=ast.dump(actual)
   changed.add(previous.name)
  else:assert ast.dump(previous)==ast.dump(actual)
 assert changed==allowed
 for name,h in frozen['fixture_source_sha256'].items():
  if name!='tests/test_rank_bilateral_contracts.py':assert text_sha(ROOT/name)==h
 for rows in frozen['snapshots'].values():assert set(rows)=={k for k,_,_ in CASES}
 assert all(digest(value)==key for key,value in frozen['observations'].items())


def test_actual_export_alias_source_facts(frozen):
 generic=importlib.import_module('skimage.filters.rank.generic')
 assert list(bilateral_module.__all__)==list(NAMES) and inspect.ismodule(reference.bilateral)
 assert importlib.import_module('skimage.filters.rank.bilateral') is reference.bilateral
 namespace={};exec('from skimage.filters.rank.bilateral import *',namespace)
 for name in NAMES:
  fn=getattr(reference,name);assert fn is getattr(bilateral_module,name) is namespace[name]
  assert name in reference.__all__ and not hasattr(generic,name)
  assert str(inspect.signature(fn))==frozen['functions'][name]['signature']
  assert hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()==frozen['functions'][name]['python_source_sha256']


@pytest.mark.parametrize('namespace',('canonical','public-bilateral'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_frozen_contract(case,namespace,frozen):
 key,name,p=case;module=reference if namespace=='canonical' else bilateral_module
 assert observe(getattr(module,name),name,p)==frozen['observations'][frozen['snapshots'][profile()][key]],key


def pending(name,namespace,*args,**options):return getattr(importlib.import_module('skimagexx.filters.rank'+('.bilateral' if namespace=='public-bilateral' else '')),name)(*args,**options)


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('namespace',('canonical','public-bilateral'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_strict_pending_never_counts_forwarding(case,namespace,workers,frozen):
 key,name,p=case;previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
 try:
  registered=set(native_functions())
  path=('filters.rank.' if namespace=='canonical' else 'filters.rank.bilateral.')+name
  actual=observe(lambda *a,**kw:pending(name,namespace,*a,**kw),name,p)
  if path in registered:
   assert set(CANONICAL+ALIASES)<=registered,'partial canonical/nested registration is not public parity'
   assert actual==frozen['observations'][frozen['snapshots'][profile()][key]],(path,key)
  else:
   assert actual['result'] is None and actual['error']['type']=='NotImplementedError'
   assert 'fallback is disabled' in actual['error']['message'] and not actual['warnings'] and actual['before']==actual['after']
 finally:xx.set_num_threads(threads);xx.set_fallback_enabled(previous)



@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('kind',('u1','f4'))
@pytest.mark.parametrize('scenario',SCENARIOS)
def test_index_provider_order_and_safe_mutation(name,kind,scenario,frozen):
 assert provider_observe(name,kind,scenario)==frozen['provider_events'][profile()][name+'/'+kind+'/'+scenario]


def integer_oracle(name,image,fp,mask,shift,s0,s1,out_dtype):
 result=np.empty(image.shape,out_dtype);centre=(fp.shape[0]//2+shift[1],fp.shape[1]//2+shift[0])
 for r,c in np.ndindex(image.shape):
  selected=[];g=int(image[r,c])
  for u,v in np.ndindex(fp.shape):
   rr,cc=r+u-centre[0],c+v-centre[1]
   if fp[u,v] and 0<=rr<image.shape[0] and 0<=cc<image.shape[1] and mask[rr,cc]:
    i=int(image[rr,cc])
    if g>i-s0 and g<i+s1:selected.append(i)
  count=len(selected);value=(sum(selected)//count if count else 0) if name=='mean_bilateral' else count if name=='pop_bilateral' else sum(selected)
  if result.dtype.kind=='u':value%=np.iinfo(result.dtype).max+1
  result[r,c]=value
 return result


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('radius',((0,0),(1,1),(1,5),(5,1),(-1,5),(5,-1),(300,300)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_independent_integer_neighborhood_oracle(name,dtype,kind,radius,output):
 image=(np.arange(35).reshape(5,7)%29).astype(dtype);fp=np.ones((2,4) if kind=='even' else (3,3),np.uint8);mask=np.ones(image.shape,np.uint8);shift=(0,0)
 if kind=='disconnected':fp.fill(0);fp[0,0]=fp[-1,-1]=1
 elif kind=='zero':fp.fill(0)
 elif kind=='masked':mask.flat[::3]=0
 elif kind=='shift':shift=(1,-1)
 target=np.dtype(dtype if output=='same' else output);s0,s1=radius
 expected=integer_oracle(name,image,fp,mask,shift,s0,s1,target);options=dict(mask=mask,shift_x=shift[0],shift_y=shift[1],s0=s0,s1=s1)
 if output!='same':options['out']=np.full(image.shape,99,target)
 actual=getattr(reference,name)(image,fp,**options)
 assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('output',('u1','u2','f4','f8'))
def test_full_capacity_counts_and_integer_wrap(name,dtype,output):
 image=np.full((17,19),255 if dtype=='u1' else 65535,np.dtype(dtype));fp=np.ones((17,17),np.uint8);mask=np.ones(image.shape,np.uint8);target=np.dtype(output)
 expected=integer_oracle(name,image,fp,mask,(0,0),10,10,target)
 with warnings.catch_warnings():warnings.simplefilter('ignore');actual=getattr(reference,name)(image,fp,out=np.full(image.shape,99,target))
 assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('edge',('upper-max','lower-min','upper-min-safe','lower-max-safe'))
def test_signed_size_safe_predicate_edges_against_python_integers(name,dtype,edge):
 image=(np.arange(35).reshape(5,7)%29).astype(dtype)
 if dtype=='u2':image.flat[-1]=65535
 fp=np.ones((3,3),np.uint8);mask=np.ones(image.shape,np.uint8)
 table={'upper-max':(sys.maxsize,10),'lower-min':(10,-sys.maxsize-1),'upper-min-safe':(65535-sys.maxsize,10),'lower-max-safe':(10,sys.maxsize-65535)}
 s0,s1=table[edge];expected=integer_oracle(name,image,fp,mask,(0,0),s0,s1,np.float64)
 with warnings.catch_warnings():warnings.simplefilter('ignore');actual=getattr(reference,name)(image,fp,s0=s0,s1=s1,out=np.empty(image.shape,np.float64))
 assert actual.tobytes()==expected.tobytes()


def test_strict_reversed_radius_endpoints_and_integer_mean():
 image=np.array([[5,7,8],[9,10,11],[12,13,15]],np.uint8);fp=np.ones((3,3),np.uint8)
 assert reference.pop_bilateral(image,fp,s0=2,s1=3)[1,1]==4
 assert reference.sum_bilateral(image,fp,s0=2,s1=3)[1,1]==38
 assert reference.mean_bilateral(image,fp,s0=2,s1=3,out=np.zeros(image.shape,np.float64))[1,1]==9
 image.fill(0);image[1,1]=1
 assert reference.mean_bilateral(image,fp,out=np.zeros(image.shape,np.float64))[1,1]==0


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True);args=parser.parse_args()
 args.record.write_bytes((json.dumps(dict(profile=profile(),runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__),snapshots=snapshot(),provider_events={name+'/'+kind+'/'+scenario:provider_observe(name,kind,scenario) for name in NAMES for kind in ('u1','f4') for scenario in SCENARIOS}),indent=2,allow_nan=False)+'\n').encode())
