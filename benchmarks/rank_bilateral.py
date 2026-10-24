"""Bounded R6 original contracts and gated public benchmarks; Phase1 native0.

All708 source-backed cases remain untimed. Only successful balanced fixtures
can be timed after the complete corpus passes on exact public native modules.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib,importlib,importlib.util,inspect,json,math,os,platform,random,sys,tempfile,traceback,uuid,warnings,zipfile
from pathlib import Path
import numpy as np
import scipy,skimage,skimagexx as xx
from skimagexx.coverage import native_functions
try:
 from .run import measure
 from .resize import binary_hashes,python_hashes,source_revision,cpu_metadata,save,summarize
 from .provenance import source_fingerprint
except ImportError:
 from run import measure
 from resize import binary_hashes,python_hashes,source_revision,cpu_metadata,save,summarize
 from provenance import source_fingerprint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import ci_measurement_artifacts as artifacts
from tools.public_qualification import SDK_MEMBERS
MANIFEST=ROOT/'coverage/contracts/rank-bilateral-benchmark.json'
SOURCE_CONTRACT=ROOT/'coverage/contracts/rank-bilateral.json'
NAMES=('mean_bilateral','pop_bilateral','sum_bilateral')
SCALAR=NAMES;VECTOR=()
CANONICAL=tuple('filters.rank.'+n for n in NAMES);ALIASES=tuple('filters.rank.bilateral.'+n for n in NAMES);APIS=CANONICAL+ALIASES
VERSION='rank-bilateral-benchmark-v1';SEED=9599;LIMIT=8*1024*1024
_spec=importlib.util.spec_from_file_location('rank_bilateral_benchmark_qualified_contract',ROOT/'tests/test_rank_bilateral_contracts.py')
qualified=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(qualified)

@dataclass(frozen=True)
class Case:
 name:str
 label:str
 fixture:dict
 timed:bool=True
 frozen_id:str|None=None

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def runtime_profile():return qualified.profile()
def resolve(name,original=False,alias=False):
 return getattr(importlib.import_module(('skimage' if original else 'skimagexx')+'.filters.rank'+('.bilateral' if alias else '')),name)

REGIMES={'symmetric':(10,10),'asymmetric':(2,9),'zero':(0,0),'signed-safe':(-1,5),'lower-only':(5,0),'upper-only':(0,5)}

def profiles():
 patterns=('uniform','ramp','low-entropy','random');fps=('dense3','dense7','sparse','disconnected','even','shifted');rows=[]
 def add(label,shape,dtype,pattern,bins,regime='symmetric',layout='C',fp=None,out=None):
  i=len(rows);rows.append(dict(id=label,shape=list(shape),dtype=dtype,pattern=pattern,bins=bins,regime=regime,footprint=fp or fps[i%6],layout=layout,mask='sparse' if i%2 else 'none',out=out or ('default','same','f4','f8','bool')[i%5]))
 for si,shape in enumerate(((17,19),(257,259))):
  for di,dtype in enumerate(('u1','u2')):
   for pi,pattern in enumerate(patterns):
    for qi,regime in enumerate(('symmetric','asymmetric','zero')):
     add(f'balanced/s{si}/d{di}/p{pi}/q{qi}',shape,dtype,pattern,256 if di==0 else (4,256,1024)[(si+pi+qi)%3],regime,'step' if (si+pi+qi)%2 else 'C')
 for di,dtype in enumerate(('u1','u2')):
  for regime in REGIMES:add(f'radius/d{di}/{regime}',(17,19),dtype,'random',256 if di==0 else 1024,regime,fp='dense3')
 for di,dtype in enumerate(('u1','u2')):
  for layout in ('C','F','reverse','step','broadcast','readonly','unaligned','tagged','endian'):
   add(f'layout/d{di}/{layout}',(17,19),dtype,'random',256 if di==0 else 1024,layout=layout,fp='sparse')
 for bins in (4,256,1024,65536):
  for pattern in patterns:add(f'bins/b{bins}/{pattern}',(3,5),'u2',pattern,bins,fp='dense3')
 for di,dtype in enumerate(('u1','u2')):
  for shape in ((1,1),(2,3),(3,5)):add(f'tiny/d{di}/s{shape}',shape,dtype,'low-entropy',256 if di==0 else 256,fp='dense3')
 assert len(rows)==100 and len({r['id'] for r in rows})==100
 return rows

def corpus():
 for name in NAMES:
  for fixture in profiles():yield Case(name,fixture['id'],fixture,fixture['layout'] not in ('readonly','endian'))
 for key,name,parameters in qualified.CASES:yield Case(name,'contract/'+key,parameters,False,key)

def raster_fixture(case):
 f=case.fixture;shape=tuple(f['shape']);count=math.prod(shape);bins=f['bins'];dtype=np.dtype(f['dtype'])
 assert len(shape)==2 and all(type(n) is int and n>0 for n in shape) and 1<=bins<=65536
 assert dtype in (np.dtype('u1'),np.dtype('u2')) and count*8<=LIMIT
 assert count*dtype.itemsize*(2 if f['layout']=='step' else 1)<=LIMIT
 if dtype==np.dtype('u1'):assert bins<=256
 assert f['regime'] in REGIMES and f['out'] in ('default','same','f4','f8','bool')
 i=np.arange(count,dtype=np.uint64)
 if f['pattern']=='uniform':v=np.full(count,bins-1,np.uint64)
 elif f['pattern']=='ramp':v=i*(bins-1)//max(count-1,1)
 elif f['pattern']=='low-entropy':v=np.array([0,bins//2,bins-1],np.uint64)[i%3]
 else:
  assert f['pattern']=='random'
  x=i+np.uint64(SEED)+np.uint64(0x9e3779b97f4a7c15);x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);v=(x^(x>>np.uint64(31)))%np.uint64(bins)
 image=v.astype(dtype).reshape(shape);image.flat[-1]=bins-1;tracked={};layout=f['layout']
 if layout=='step':
  owner=np.full((shape[0],shape[1]*2),17,dtype);owner[:,::2]=image;image=owner[:,::2];tracked['image_owner']=owner
 elif layout=='F':image=np.asfortranarray(image)
 elif layout=='reverse':image=image[::-1,::-1]
 elif layout=='broadcast':image=np.broadcast_to(image[:1],shape)
 elif layout=='readonly':image.flags.writeable=False
 elif layout=='unaligned':
  owner=bytearray(image.nbytes+1);a=np.ndarray(shape,dtype,buffer=owner,offset=1);a[...]=image;image=a
 elif layout=='tagged':image=image.view(np.dtype(dtype,metadata={'benchmark':'bilateral'}))
 elif layout=='endian':image=image.astype(dtype.newbyteorder('>'))
 else:assert layout=='C'
 kind=f['footprint'];assert kind in ('dense3','dense7','sparse','disconnected','even','shifted')
 fp=np.ones((2,4) if kind=='even' else (7 if kind in ('dense7','sparse','disconnected') else 3,)*2,np.uint8)
 if kind=='sparse':fp.flat[::3]=0;fp.flat[0]=1
 elif kind=='disconnected':fp.fill(0);fp.flat[:2]=1;fp.flat[-2:]=1
 s0,s1=REGIMES[f['regime']];options=dict(s0=s0,s1=s1)
 if kind=='shifted':options.update(shift_x=1,shift_y=-1)
 if f['mask']=='sparse':options['mask']=((i*13+SEED)%7!=0).astype(np.uint8).reshape(shape)
 else:assert f['mask']=='none'
 if f['out']!='default':options['out']=np.full(shape,99,dtype if f['out']=='same' else np.dtype(f['out']))
 tracked.update(image=image,footprint=fp,**{k:v for k,v in options.items() if k in ('mask','out')})
 qualified.safety(case.name,(image,fp),options,tracked,{})
 return (image,fp),options,tracked

def make_case(case):
 if case.frozen_id:
  args,options,tracked=qualified.make_case(case.name,case.fixture);qualified.safety(case.name,args,options,tracked,case.fixture);return args,options,tracked
 return raster_fixture(case)
def describe(value,inputs=()):
 result=qualified.array(value)
 if isinstance(value,np.ndarray):result.update(identity=[value is a for a in inputs],alias=[bool(np.shares_memory(value,a)) for a in inputs])
 return result
def inventory(case):return json.loads(json.dumps(dict(api='filters.rank.'+case.name,case=case.label,fixture=case.fixture,timed_candidate=case.timed,source_contract_id=case.frozen_id,api_kind='scalar',timing_class='defined-nonalias-raster' if case.timed else 'frozen-source-contract' if case.frozen_id else 'source-metadata-layout',work_kind='frozen-contract' if case.frozen_id else 'tiny-raster' if math.prod(case.fixture['shape'])<200 else 'small-raster' if math.prod(case.fixture['shape'])<1000 else 'raster'),allow_nan=False))
def parameter_spec(value):
 if type(value) in (qualified.IndexValue,qualified.IntOnly,qualified.Both,qualified.FloatOnly):
  return dict(type=type(value).__name__,provider_state={k:qualified.array(v) for k,v in vars(value).items()})
 return describe(value)

def spec(case):
 args,options,tracked=make_case(case)
 return dict(**inventory(case),positional_count=len(args),options={k:parameter_spec(v) for k,v in options.items()},input_contract={k:qualified.array(v) for k,v in tracked.items()})
def observe_case(function,case):
 if case.frozen_id:return qualified.observe(function,case.name,case.fixture)
 args,options,tracked=make_case(case);inputs=[a for a in tracked.values() if isinstance(a,np.ndarray)];before={k:qualified.array(v) for k,v in tracked.items()};ptr={k:a.ctypes.data for k,a in tracked.items() if isinstance(a,np.ndarray)}
 with warnings.catch_warnings(record=True) as notices:
  warnings.simplefilter('always')
  try:result,error=describe(function(*args,**options),inputs),None
  except Exception as e:result,error=None,dict(type=type(e).__name__,message=str(e))
 return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={k:qualified.array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==p for k,p in ptr.items()})

def native_observe(observer,*args):
 # Check-only guard, never entered by the timing callbacks. Match exact pinned
 # Python codes and Cython entry identities; do not classify providers merely
 # by module name. Scoped Cython attribute guards cover dynamic entry lookup
 # even when an extension was compiled without Python profiling events.
 module=importlib.import_module('skimage.filters.rank.bilateral');cython=module.bilateral_cy
 python_functions=tuple(getattr(module,name) for name in NAMES)
 fields={'_'+name.removesuffix('_bilateral'):None for name in NAMES}
 fields={name:getattr(cython,name) for name in fields}
 entries=list(fields.values())
 for entry in tuple(entries):entries.extend(getattr(entry,'__signatures__',{}).values())
 codes=tuple(code for entry in (*python_functions,*entries) if (code:=getattr(entry,'__code__',None)) is not None)
 code_ids={id(code) for code in codes};entry_ids={id(entry) for entry in entries}
 def forbidden(*unused,**options):raise AssertionError('upstream original rank entry called during native check')
 def profile(frame,event,arg):
  if (event=='call' and id(frame.f_code) in code_ids) or (event=='c_call' and id(arg) in entry_ids):forbidden()
 previous=sys.getprofile()
 try:
  for name,entry in fields.items():setattr(cython,name,forbidden)
  sys.setprofile(profile)
  return observer(*args)
 finally:
  sys.setprofile(previous)
  for name,entry in fields.items():setattr(cython,name,entry)
def check_aliases(workers,reference_only):
 rows=[]
 for name in NAMES:
  ref=resolve(name,True);assert ref is resolve(name,True,True)
  case=Case(name,'bilateral-identity',profiles()[0],False);expected=observe_case(ref,case)
  assert observe_case(resolve(name,True,True),case)==expected
  row=dict(canonical='filters.rank.'+name,alias='filters.rank.bilateral.'+name,timed=False,reference_identity=True,native_identity=None,algorithm_count=1,reference=expected)
  if not reference_only:
   assert resolve(name) is resolve(name,False,True);row['native_identity']=True;row['actual_by_workers']={}
   for worker in workers:
    xx.set_num_threads(worker);actual=native_observe(observe_case,resolve(name,False,True),case);row['actual_by_workers'][str(worker)]=actual
    assert actual==expected,json.dumps(dict(scope='public-alias',name=name,worker=worker,reference=expected,actual=actual),sort_keys=True)
  rows.append(row)
 return rows

array=qualified.array
def provider_observation(function,name,kind,scenario):
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
  try:result=function(image,fp,**options);error=None
  except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
 result_info=None if result is None else dict(array(result),identity=[result is value for value in tracked.values()],alias=[bool(np.shares_memory(result,value)) for value in tracked.values()])
 return dict(events=events,error=error,result=result_info,before=before,after={k:array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==ptr for k,ptr in pointers.items()},warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])

def check_providers(workers,reference_only,source):
 rows=[]
 for name in NAMES:
  for kind in ('u1','f4'):
   for scenario in qualified.SCENARIOS:
    expected=provider_observation(resolve(name,True),name,kind,scenario)
    assert expected==source['provider_events'][runtime_profile()][f'{name}/{kind}/{scenario}']
    row=dict(api='filters.rank.'+name,case=f'{kind}/{scenario}',timed=False,reference=expected)
    if not reference_only:
     row['actual_by_workers']={}
     for worker in workers:
      xx.set_num_threads(worker);actual=native_observe(provider_observation,resolve(name),name,kind,scenario);row['actual_by_workers'][str(worker)]=actual
      assert actual==expected,json.dumps(dict(scope='provider',name=name,kind=kind,scenario=scenario,worker=worker,reference=expected,actual=actual),sort_keys=True)
    rows.append(row)
 return rows

def policy_observation(function,name,policy):
 case=Case(name,'policy/nan-image',dict(ndim=2,special='nan-image'),False,'policy-only')
 previous=np.geterr();handler=np.geterrcall();events=[];saved=os.dup(2);reader,writer=os.pipe()
 def callback(message,flags):events.append([message,flags])
 class Logger:
  def write(self,message):events.append(message)
 try:
  np.seterrcall(Logger() if policy=='log' else callback);np.seterr(all=policy)
  sys.stderr.flush();os.dup2(writer,2);os.close(writer);writer=None
  try:result=observe_case(function,case);sys.stderr.flush()
  finally:os.dup2(saved,2)
  # The pinned tiny metadata conversion emits bounded diagnostic lines. Pipes
  # avoid anonymous fd-capture files on Windows-mounted Linux filesystems.
  chunks=[]
  while True:
   data=os.read(reader,4096)
   if not data:break
   chunks.append(data)
  printed=b''.join(chunks);assert len(printed)<4096,'bounded policy metadata exceeded pipe budget'
  return dict(observation=result,events=events,printed=printed.decode('utf8',errors='replace'))
 finally:
  os.dup2(saved,2);os.close(saved);os.close(reader)
  if writer is not None:os.close(writer)
  np.seterr(**previous);np.seterrcall(handler)

def check_policies(workers,reference_only):
 rows=[]
 for name in NAMES:
  for policy in ('warn','raise','ignore','call','log','print'):
   expected=policy_observation(resolve(name,True),name,policy);row=dict(api='filters.rank.'+name,policy=policy,timed=False,reference=expected)
   if not reference_only:
    row['actual_by_workers']={}
    for worker in workers:
     xx.set_num_threads(worker);actual=native_observe(policy_observation,resolve(name),name,policy);row['actual_by_workers'][str(worker)]=actual
     assert actual==expected,json.dumps(dict(scope='numpy-policy',name=name,policy=policy,worker=worker,reference=expected,actual=actual),sort_keys=True)
   rows.append(row)
 return rows

def timing_callback(function,case):
 assert case.timed and not case.frozen_id
 args,options,tracked=make_case(case);out=options.get('out');seed=out.copy() if out is not None else None
 def call():
  if out is not None:out[...] = seed
  return function(*args,**options)
 return call,tracked

def verify_record(wheel):
 import base64,csv,io
 with zipfile.ZipFile(wheel) as z:
  names=[n for n in z.namelist() if not n.endswith('/')];assert len(names)==len(set(names));record,=[n for n in names if n.endswith('.dist-info/RECORD')]
  rows=list(csv.reader(io.StringIO(z.read(record).decode())));assert {r[0] for r in rows}==set(names)
  for n,h,s in rows:
   if h:
    b=z.read(n);assert h=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode() and int(s)==len(b)
   else:assert n==record or n.endswith(('.dist-info/RECORD.jws','.dist-info/RECORD.p7s')),'unhashed wheel member'
def capsule_tree(capsule):return {p.relative_to(capsule).as_posix():artifacts.sha(p) for p in capsule.rglob('*') if p.is_file() and p.name!='rank-bilateral-artifacts.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def seal(capsule):
 path=capsule/'rank-bilateral-artifacts.json';assert not path.exists()
 runtime=artifacts.freeze(ROOT,capsule/'wheels',capsule/'rank-bilateral-runtime.json')
 with zipfile.ZipFile(capsule/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in runtime['source']['files_sha256']:z.write(ROOT/name,name)
 info=dict(schema='rank-bilateral-exact-offline-v1',runtime_manifest_sha256=artifacts.sha(capsule/'rank-bilateral-runtime.json'),corpus_manifest_sha256=artifacts.sha(MANIFEST),product_source=source_fingerprint(ROOT),files_sha256=capsule_tree(capsule))
 path.write_text(json.dumps(info,indent=2)+'\n');return verify(capsule)
def verify(capsule):
 path=capsule/'rank-bilateral-artifacts.json';info=json.loads(path.read_text());assert info['schema']=='rank-bilateral-exact-offline-v1'
 assert info['files_sha256']==capsule_tree(capsule) and info['corpus_manifest_sha256']==artifacts.sha(MANIFEST) and info['product_source']==source_fingerprint(ROOT)
 assert info['runtime_manifest_sha256']==artifacts.sha(capsule/'rank-bilateral-runtime.json')
 runtime=artifacts.verify(ROOT,capsule/'wheels',capsule/'rank-bilateral-runtime.json');assert len(runtime['runtime']['sdk_sha256'])==SDK_MEMBERS
 with zipfile.ZipFile(capsule/'source.zip') as z:
  assert set(z.namelist())==set(runtime['source']['files_sha256'])
  assert all(hashlib.sha256(z.read(n).replace(b'\r\n',b'\n')).hexdigest()==h for n,h in runtime['source']['files_sha256'].items())
 for wheel in (capsule/'wheels').glob('*.whl'):
  verify_record(wheel)
  with zipfile.ZipFile(wheel) as z:
   if 'skimagexx/__init__.py' not in z.namelist():continue
   assert {n for n in z.namelist() if n.startswith('include/skimagexx/') and not n.endswith('/')}=={p.relative_to(ROOT).as_posix() for p in (ROOT/'include/skimagexx').rglob('*') if p.is_file()}
   for n in z.namelist():
    if n.startswith('include/skimagexx/') and not n.endswith('/'):assert z.read(n).replace(b'\r\n',b'\n')==(ROOT/n).read_bytes().replace(b'\r\n',b'\n')
   assert {n.removeprefix('skimagexx/'):z.read(n).replace(b'\r\n',b'\n') for n in z.namelist() if n.startswith('skimagexx/') and n.endswith('.py')}=={p.relative_to(ROOT/'python/skimagexx').as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in (ROOT/'python/skimagexx').rglob('*.py')}
 return dict(manifest_sha256=artifacts.sha(path),**info)

def admit_native(previous):
 missing=sorted(set(APIS)-set(native_functions()))
 if missing:raise RuntimeError('public native R6 APIs are not registered: '+', '.join(missing))
 assert not {f'filters.rank.generic.{name}' for name in NAMES}&set(native_functions()),'invented generic bilateral aliases are not public source paths'
 core=xx._core;capabilities=getattr(core,'rank_bilateral_capabilities',0)
 if getattr(core,'rank_bilateral_interface',None)!='rank-bilateral-v1' or type(capabilities) is not int or capabilities<0 or capabilities&0x1c000000!=0x1c000000:raise RuntimeError('rank-bilateral-v1 capability0x1c000000 required')
 compiled_entries(core)
 assert previous==1 and xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads,'sole core/default worker1 required'
 for name in NAMES:
  current=resolve(name);ref=resolve(name,True)
  assert current is resolve(name,False,True),'canonical/bilateral public alias differs'
  assert current is not ref and not getattr(current,'__module__','').startswith('skimage.'),'upstream-forwarded R6 path'
  assert inspect.signature(current)==inspect.signature(ref),'public R6 signature differs'
  assert inspect.unwrap(current).__globals__.get('_core') is core,'R6 wrapper uses another core'

def compiled_entries(core):
 name='rank_bilateral_run';function=getattr(core,name,None)
 assert inspect.isbuiltin(function) and function.__module__==core.__name__ and function.__name__==name,'compiled R6 entry missing: '+name

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,nargs='+',default=[1]);parser.add_argument('--repeat',type=int,default=7);parser.add_argument('--target-ms',type=float,default=15.);parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true');parser.add_argument('--reference-only',action='store_true');parser.add_argument('--capsule',type=Path);parser.add_argument('--prepare-capsule',action='store_true');args=parser.parse_args(argv)
 if not args.threads or min(args.threads)<1 or max(args.threads)>1024 or len(set(args.threads))!=len(args.threads) or 1 not in args.threads or args.repeat!=7 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('distinct workers1..1024 including worker1, exactly seven paired repeats and positive target-ms required')
 if args.reference_only and (not args.check_only or args.capsule is not None):parser.error('--reference-only requires --check-only without capsule')
 if args.prepare_capsule and (not args.check_only or args.reference_only or args.capsule is None):parser.error('--prepare-capsule requires strict --check-only and capsule')
 if args.prepare_capsule and set(args.threads)!={1,4}:parser.error('capsule preparation requires mandatory workers1 and4')
 if not args.check_only and (args.output is None or args.capsule is None or os.environ.get('GITHUB_ACTIONS')!='true'):parser.error('timing requires isolated Actions host/output/capsule; local --check-only only')
 previous=xx.get_num_threads();policy=xx.set_fallback_enabled(False);fingerprint=source_fingerprint(ROOT);manifest_hash=artifacts.sha(MANIFEST)
 harness_paths=('benchmarks/rank_bilateral.py','benchmarks/run.py','benchmarks/resize.py','benchmarks/provenance.py','tools/ci_measurement_artifacts.py','tests/test_rank_bilateral_contracts.py','tests/test_rank_neighborhood_contracts.py')
 harness={n:text_sha(ROOT/n) for n in harness_paths};harness_bytes={n:artifacts.sha(ROOT/n) for n in harness_paths};extension=Path(importlib.import_module('skimage.filters.rank.bilateral').bilateral_cy.__file__)
 session=str(uuid.uuid4());cpu=dict(cpu_metadata(),logical_cpus=os.cpu_count(),cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER',''))
 report=dict(metadata=dict(version=VERSION,source_revision=source_revision(ROOT),**fingerprint,harness_sha256=harness,harness_bytes_sha256=harness_bytes,harness_hash_schema='sha256-posix-lf-v1',reference_extension=dict(path=str(extension),sha256=artifacts.sha(extension)),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,scikit_image=skimage.__version__,platform=platform.platform(),cpu_session=cpu,session_id=session,process_id=os.getpid(),backend_cpu_sessions={'scikit-image':session,'current':session},same_process_cpu_for_backends=True,avx2=xx.has_avx2(),current_binary_sha256=binary_hashes(),current_python_sha256=python_hashes(),requested_workers=args.threads,primary_workers=1,initial_default_workers=previous,fallback_enabled=False,repeat=args.repeat,check_only=args.check_only,reference_only=args.reference_only,algorithm_count=3,scalar_api_count=3,vector_api_count=0,canonical_paths=list(CANONICAL),public_alias_paths=list(ALIASES),operation_ids=dict(zip(NAMES,range(26,29))),aliases_timed=False,validation_scope='reference-only/native0' if args.reference_only else 'strict actual public R6',reference_pools=1,scope='Warmed public calls; supplied output reset/allocation/destruction included. Fixture construction excluded. Errors/metadata/aliases untimed.'),status='preflight',checks=[],alias_checks=[],provider_checks=[],policy_checks=[],results=[],untimed_cases=[],failures=[])
 save(report,args.output)
 try:
  assert skimage.__version__=='0.26.0'
  if not args.reference_only:admit_native(previous)
  manifest=json.loads(MANIFEST.read_text());assert text_sha(SOURCE_CONTRACT)==manifest['source_contract_sha256']
  for name,h in manifest['fixture_source_sha256'].items():assert text_sha(ROOT/name)==h
  for name in NAMES:
   ref=resolve(name,True);facts=manifest['reference_functions'][name];assert ref is resolve(name,True,True)
   assert str(inspect.signature(ref))==facts['signature'] and hashlib.sha256(inspect.getsource(ref).replace('\r\n','\n').encode()).hexdigest()==facts['python_source_sha256']
  frozen=manifest['reference_profiles'].get(runtime_profile())
  if not args.reference_only:
   assert frozen is not None and args.capsule is not None,'frozen original profile and authenticated capsule required'
   report['metadata']['exact_artifacts']=seal(args.capsule) if args.prepare_capsule else verify(args.capsule)
  cases=list(corpus());assert [inventory(c) for c in cases]==manifest['case_inventory'] and digest(manifest['case_inventory'])==manifest['case_contract_sha256']
  specs=[spec(c) for c in cases]
  if frozen:assert digest(specs)==frozen['case_spec_sha256']
  report['metadata'].update(profile=runtime_profile(),case_count=len(cases),case_specs=specs,case_spec_sha256=digest(specs),case_contract_sha256=manifest['case_contract_sha256'])
  report['alias_checks']=check_aliases(args.threads,args.reference_only)
  source=json.loads(SOURCE_CONTRACT.read_text());report['provider_checks']=check_providers(args.threads,args.reference_only,source);report['policy_checks']=check_policies(args.threads,args.reference_only)
  if frozen:assert digest([{k:v for k,v in row.items() if k not in ('native_identity','actual_by_workers')} for row in report['alias_checks']])==frozen['alias_sha256'] and digest([r['reference'] for r in report['provider_checks']])==frozen['provider_sha256'] and digest([r['reference'] for r in report['policy_checks']])==frozen['policy_sha256']
  report['status']='checking'
  for i,case in enumerate(cases):
   row=dict(case_index=i,**specs[i])
   try:
    expected=observe_case(resolve(case.name,True),case);row['reference']=expected
    if case.timed:
     assert all(expected['before'][k]==value for k,value in expected['after'].items() if k not in ('out','out_owner')),'timing fixture mutated source storage'
     assert all(expected['pointer_stable'].values()),'timing fixture changed storage pointers'
    if case.frozen_id:assert expected==source['observations'][source['snapshots'][runtime_profile()][case.frozen_id]],'original differs from frozen708'
    if frozen:assert digest(expected)==frozen['observations'][str(i)],'original differs from frozen benchmark profile'
    row['timing_eligible']=case.timed and expected['error'] is None
    if not args.reference_only:
     row['actual_by_workers']={}
     for worker in args.threads:
      xx.set_num_threads(worker);assert xx.get_num_threads()==worker
      actual=native_observe(observe_case,resolve(case.name),case);row['actual_by_workers'][str(worker)]=actual
      assert actual==expected,'public observation differs; complete reference and actual preserved in this row'
    row['status']='passed'
   except Exception as e:row.update(status='failed',exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc());report['failures'].append(row)
   report['checks'].append(row)
   if not row.get('timing_eligible'):report['untimed_cases'].append(dict(case_index=i,reason='source contract/error/metadata; never timed'))
   if (i+1)%32==0 or row['status']=='failed':save(report,args.output)
  if report['failures']:report['status']='failed-parity';return 1
  eligible=[r for r in report['checks'] if r['timing_eligible']];report['metadata'].update(timing_eligible_case_count=len(eligible),scalar_timing_cases=sum(r['api_kind']=='scalar' for r in eligible),vector_timing_cases=sum(r['api_kind']=='vector' for r in eligible),native_checks=0 if args.reference_only else len(cases)*len(args.threads))
  report['metadata'].update(timing_cohorts={kind:sum(r['work_kind']==kind for r in eligible) for kind in sorted({r['work_kind'] for r in eligible})},public_identity_checks=len(report['alias_checks']),provider_checks=len(report['provider_checks']),policy_checks=len(report['policy_checks']),target_ms=args.target_ms)
  def integrity():
   assert source_fingerprint(ROOT)==fingerprint and artifacts.sha(MANIFEST)==manifest_hash,'source/corpus changed'
   assert all(text_sha(ROOT/n)==v for n,v in harness.items()) and all(artifacts.sha(ROOT/n)==v for n,v in harness_bytes.items()),'harness/fixture bytes changed'
   assert binary_hashes()==report['metadata']['current_binary_sha256'] and python_hashes()==report['metadata']['current_python_sha256'],'installed package bytes changed'
   assert dict(cpu_metadata(),logical_cpus=os.cpu_count(),cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER',''))==cpu,'CPU session changed'
   if not args.reference_only:assert verify(args.capsule)==report['metadata']['exact_artifacts']
  integrity()
  if args.check_only:report['status']='complete-reference-check' if args.reference_only else 'complete-parity-check';return 0
  report['status']='measuring';rng=random.Random(81);orders=[]
  for _ in range(args.repeat):order=['scikit-image','current'];rng.shuffle(order);orders.append(order)
  report['metadata']['paired_sample_order']=orders
  for row in eligible:
   case=cases[row['case_index']]
   for worker in args.threads:
    xx.set_num_threads(worker);left,a=timing_callback(resolve(case.name,True),case);right,b=timing_callback(resolve(case.name),case)
    with warnings.catch_warnings():warnings.simplefilter('ignore');timing=measure(left,right,args.repeat,args.target_ms)
    assert {k:qualified.array(v) for k,v in a.items()}==row['reference']['after'] and {k:qualified.array(v) for k,v in b.items()}==row['reference']['after'],'timing changed workload state'
    report['results'].append(dict(case_index=row['case_index'],api=row['api'],case=row['case'],api_kind=row['api_kind'],work_kind=row['work_kind'],session_id=session,threads=worker,primary=worker==1,image_kernel_speedup_claim=False,slower_than_original=timing['speedup']<1,**timing));save(report,args.output)
  integrity();report['summary_by_algorithm']={name:summarize([r for r in report['results'] if r['api']=='filters.rank.'+name],args.threads,sum(r['api']=='filters.rank.'+name for r in eligible)) for name in NAMES}
  report['summary_by_cohort']={kind:summarize([r for r in report['results'] if r['work_kind']==kind],args.threads,sum(r['work_kind']==kind for r in eligible)) for kind in sorted({r['work_kind'] for r in eligible})};report['status']='complete-measurement';return 0
 except Exception as e:report['failures'].append(dict(phase=report['status'],exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));report['status']='failed';return 1
 finally:xx.set_num_threads(previous);xx.set_fallback_enabled(policy);save(report,args.output)

if __name__=='__main__':raise SystemExit(main())
