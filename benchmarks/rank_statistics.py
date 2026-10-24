"""Bounded R4 original contracts and gated public benchmarks; Phase1 native0.

All995 source-backed cases remain untimed. Only successful balanced fixtures
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
MANIFEST=ROOT/'coverage/contracts/rank-statistics-benchmark.json'
SOURCE_CONTRACT=ROOT/'coverage/contracts/rank-statistics.json'
NAMES=('geometric_mean','noise_filter','subtract_mean','windowed_histogram')
SCALAR=NAMES[:3];VECTOR=NAMES[3:]
CANONICAL=tuple('filters.rank.'+n for n in NAMES);ALIASES=tuple('filters.rank.generic.'+n for n in NAMES);APIS=CANONICAL+ALIASES
VERSION='rank-statistics-benchmark-v1';SEED=9599;LIMIT=8*1024*1024
_spec=importlib.util.spec_from_file_location('rank_statistics_benchmark_qualified_contract',ROOT/'tests/test_rank_statistics_contracts.py')
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
 return getattr(importlib.import_module(('skimage' if original else 'skimagexx')+'.filters.rank'+('.generic' if alias else '')),name)

def profiles(vector=False):
 patterns=('uniform','ramp','low-entropy','random');fps=('dense3','dense7','sparse','disconnected','even','shifted')
 layouts=('C','F','reverse','step','broadcast','readonly','unaligned','tagged')
 rows=[]
 def add(label,shape,dtype,pattern,bins,fp='dense3',layout='C',depth=16,out=None):
  i=len(rows);rows.append(dict(id=label,shape=list(shape),dtype=dtype,pattern=pattern,bins=bins,footprint=fp,layout=layout,mask='sparse' if i%2 else 'none',out=out or ('default','u2','f4','f8')[i%4],depth=depth if vector else None))
 if not vector:
  for s,shape in enumerate(((17,19),(257,259),(7,11,13))):
   for d,dtype in enumerate(('u1','u2')):
    for p,pattern in enumerate(patterns):add(f'balanced/s{s}/d{d}/p{p}',shape,dtype,pattern,256 if d==0 else (4,256,1024)[(s+p)%3],fps[(s+d+p)%6],layout='step' if (s+p)%2 else 'C')
  for ndim,shape in ((2,(17,19)),(3,(7,11,13))):
   for d,dtype in enumerate(('u1','u2')):
    for f,fp in enumerate(fps):add(f'footprint/{ndim}d/d{d}/f{f}',shape,dtype,'random',256 if d==0 else 1024,fp)
  for ndim,shape in ((2,(17,19)),(3,(3,5,7))):
   for b,bins in enumerate((4,256,1024,65536)):
    for p,pattern in enumerate(patterns):add(f'bins/{ndim}d/b{bins}/p{p}',shape,'u2',pattern,bins,'sparse' if p%2 else 'dense3')
 else:
  for s,shape in enumerate(((17,19),(65,67))):
   for d,dtype in enumerate(('u1','u2')):
    for p,pattern in enumerate(patterns):
     for depth in (4,64):add(f'balanced/s{s}/d{d}/p{p}/depth{depth}',shape,dtype,pattern,256 if d==0 else 1024,fps[(s+d+p)%6],depth=depth,out=('default','u2','f4','f8')[p])
  for d,dtype in enumerate(('u1','u2')):
   for f,fp in enumerate(fps):
    for depth in (4,16):add(f'footprint/2d/d{d}/f{f}/depth{depth}',(17,19),dtype,'random',256 if d==0 else 1024,fp,depth=depth)
  for bins in (4,256,1024,65536):
   for p,pattern in enumerate(patterns):add(f'bins/2d/b{bins}/p{p}',(17,19),'u2',pattern,bins,'sparse' if p%2 else 'dense3',depth=min(bins,16))
 for d,dtype in enumerate(('u1','u2')):
  for l,layout in enumerate(layouts):add(f'layout/2d/d{d}/{layout}',(17,19),dtype,'random',256 if d==0 else 1024,layout=layout,depth=16)
 if vector:
  for d,dtype in enumerate(('u1','u2')):
   for out in ('u1','u2','f4','f8'):add(f'output/2d/d{d}/{out}',(17,19),dtype,'ramp',256 if d==0 else 1024,depth=16,out=out)
 assert len(rows)==96 and len({r['id'] for r in rows})==96
 return rows

def corpus():
 for name in NAMES:
  for fixture in profiles(name in VECTOR):yield Case(name,fixture['id'],fixture)
 for key,name,parameters in qualified.CASES:yield Case(name,'contract/'+key,parameters,False,key)

def raster_fixture(case):
 f=case.fixture;shape=tuple(f['shape']);count=math.prod(shape);bins=f['bins']
 assert len(shape) in (2,3) and all(type(n) is int and n>0 for n in shape) and 1<=bins<=65536
 assert np.dtype(f['dtype']) in (np.dtype('u1'),np.dtype('u2')) and count*8<=LIMIT
 if np.dtype(f['dtype'])==np.dtype('u1'):assert bins<=256
 if case.name in VECTOR:assert len(shape)==2 and type(f['depth']) is int and 1<=f['depth']<=bins and count*f['depth']*8<=LIMIT
 assert count*np.dtype(f['dtype']).itemsize*(2 if f['layout']=='step' else 1)<=LIMIT
 i=np.arange(count,dtype=np.uint64)
 if f['pattern']=='uniform':v=np.full(count,bins-1,np.uint64)
 elif f['pattern']=='ramp':v=i*(bins-1)//max(count-1,1)
 elif f['pattern']=='low-entropy':v=np.array([0,bins//2,bins-1],np.uint64)[i%3]
 else:
  x=i+np.uint64(SEED)+np.uint64(0x9e3779b97f4a7c15);x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);v=(x^(x>>np.uint64(31)))%np.uint64(bins)
 image=v.astype(f['dtype']).reshape(shape);image.flat[-1]=bins-1;image[(0,)*(len(shape)-1)+(-1,)]=bins-1
 tracked={};layout=f['layout']
 if layout=='step':
  owner=np.full(shape[:-1]+(shape[-1]*2,),17,image.dtype);owner[...,::2]=image;image=owner[...,::2];tracked['image_owner']=owner
 elif layout=='F':image=np.asfortranarray(image)
 elif layout=='reverse':image=image[(slice(None,None,-1),)*len(shape)]
 elif layout=='broadcast':image=np.broadcast_to(image[(slice(0,1),)+(slice(None),)*(len(shape)-1)],shape)
 elif layout=='readonly':image.flags.writeable=False
 elif layout=='unaligned':
  owner=bytearray(image.nbytes+1);a=np.ndarray(shape,image.dtype,buffer=owner,offset=1);a[...]=image;image=a
 elif layout=='tagged':image=image.view(np.dtype(image.dtype,metadata={'benchmark':'statistics'}))
 ndim=len(shape);kind=f['footprint'];fp=np.ones(((2,4) if ndim==2 else (2,3,4)) if kind=='even' else (7 if kind in ('dense7','sparse','disconnected') else 3,)*ndim,np.uint8)
 if kind=='sparse':fp.flat[::3]=0;fp.flat[0]=1
 elif kind=='disconnected':fp.fill(0);fp.flat[:2]=1;fp.flat[-2:]=1
 options={}
 if kind=='shifted':options.update(shift_x=1,shift_y=-1);options.update({} if ndim==2 else {'shift_z':0})
 if f['mask']=='sparse':options['mask']=((i*13+SEED)%7!=0).astype(np.uint8).reshape(shape)
 out=f['out']
 if case.name in VECTOR:
  depth=f['depth'];options['n_bins']=depth
  if out!='default':options['out']=np.full(shape+(depth,),99,np.dtype(out))
 elif out!='default':options['out']=np.full(shape,99,np.dtype(out))
 tracked.update(image=image,footprint=fp,**{k:v for k,v in options.items() if k in ('mask','out')})
 assert all(a.nbytes<=LIMIT for a in tracked.values() if isinstance(a,np.ndarray))
 if case.name in VECTOR:
  assert ndim==2 and 1<=f['depth']<=bins and count*f['depth']*8<=LIMIT
 if case.name=='subtract_mean' and image.dtype==np.uint16:
  dtype=image.dtype if out=='default' else np.dtype(out);capacity=max(3,int(image.max()))+1
  if dtype.kind in 'ui':assert math.trunc(capacity//2-1-(capacity-1)/2)>=0 and math.trunc(capacity//2-1+(capacity-1)/2)<=np.iinfo(dtype).max
 return (image,fp),options,tracked

def make_case(case):
 if case.frozen_id:
  args,options,tracked=qualified.make_case(case.fixture);qualified.safety(case.name,args,options,tracked);return args,options,tracked
 return raster_fixture(case)
def describe(value,inputs=()):
 result=qualified.array(value)
 if isinstance(value,np.ndarray):result.update(identity=[value is a for a in inputs],alias=[bool(np.shares_memory(value,a)) for a in inputs])
 return result
def inventory(case):return json.loads(json.dumps(dict(api='filters.rank.'+case.name,case=case.label,fixture=case.fixture,timed_candidate=case.timed,source_contract_id=case.frozen_id,api_kind='vector' if case.name in VECTOR else 'scalar'),allow_nan=False))
def spec(case):
 args,options,tracked=make_case(case)
 return dict(**inventory(case),positional_count=len(args),options={k:describe(v) for k,v in options.items()},input_contract={k:qualified.array(v) for k,v in tracked.items()})
def observe_case(function,case):
 if case.frozen_id:return qualified.observe(function,case.name,case.fixture)
 args,options,tracked=make_case(case);inputs=[a for a in tracked.values() if isinstance(a,np.ndarray)];before={k:qualified.array(v) for k,v in tracked.items()};ptr={k:a.ctypes.data for k,a in tracked.items() if isinstance(a,np.ndarray)}
 with warnings.catch_warnings(record=True) as notices:
  warnings.simplefilter('always')
  try:result,error=describe(function(*args,**options),inputs),None
  except Exception as e:result,error=None,dict(type=type(e).__name__,message=str(e))
 return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={k:qualified.array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==p for k,p in ptr.items()})
def check_aliases(workers,reference_only):
 rows=[]
 for name in NAMES:
  ref=resolve(name,True);assert ref is resolve(name,True,True)
  c=Case(name,'alias-dispatch',profiles(name in VECTOR)[0],False);expected=observe_case(ref,c);assert observe_case(resolve(name,True,True),c)==expected
  row=dict(canonical='filters.rank.'+name,alias='filters.rank.generic.'+name,timed=False,reference_identity=True,native_identity=None,algorithm_count=1)
  if not reference_only:
   row['native_identity']=True
   for worker in workers:xx.set_num_threads(worker);assert observe_case(resolve(name),c)==expected and observe_case(resolve(name,alias=True),c)==expected
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
def capsule_tree(capsule):return {p.relative_to(capsule).as_posix():artifacts.sha(p) for p in capsule.rglob('*') if p.is_file() and p.name!='rank-statistics-artifacts.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def seal(capsule):
 path=capsule/'rank-statistics-artifacts.json';assert not path.exists()
 runtime=artifacts.freeze(ROOT,capsule/'wheels',capsule/'rank-statistics-runtime.json')
 with zipfile.ZipFile(capsule/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in runtime['source']['files_sha256']:z.write(ROOT/name,name)
 info=dict(schema='rank-statistics-exact-offline-v1',runtime_manifest_sha256=artifacts.sha(capsule/'rank-statistics-runtime.json'),corpus_manifest_sha256=artifacts.sha(MANIFEST),product_source=source_fingerprint(ROOT),files_sha256=capsule_tree(capsule))
 path.write_text(json.dumps(info,indent=2)+'\n');return verify(capsule)
def verify(capsule):
 path=capsule/'rank-statistics-artifacts.json';info=json.loads(path.read_text());assert info['schema']=='rank-statistics-exact-offline-v1'
 assert info['files_sha256']==capsule_tree(capsule) and info['corpus_manifest_sha256']==artifacts.sha(MANIFEST) and info['product_source']==source_fingerprint(ROOT)
 assert info['runtime_manifest_sha256']==artifacts.sha(capsule/'rank-statistics-runtime.json')
 runtime=artifacts.verify(ROOT,capsule/'wheels',capsule/'rank-statistics-runtime.json');assert len(runtime['runtime']['sdk_sha256'])==SDK_MEMBERS
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
 if missing:raise RuntimeError('public native R4 APIs are not registered: '+', '.join(missing))
 core=xx._core
 scalar=getattr(core,'rank_statistics_capabilities',0);vector=getattr(core,'rank_vector_capabilities',0)
 if getattr(core,'rank_statistics_interface',None)!='rank-statistics-v1' or type(scalar) is not int or scalar<0 or scalar&0xe000!=0xe000:raise RuntimeError('rank-statistics-v1 capability0xe000 required')
 if getattr(core,'rank_vector_interface',None)!='rank-vector-v1' or type(vector) is not int or vector<0 or vector&0x10000!=0x10000:raise RuntimeError('rank-vector-v1 capability0x10000 required')
 compiled_entries(core)
 assert previous==1 and xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads,'sole core/default worker1 required'
 for name in NAMES:
  current=resolve(name);ref=resolve(name,True)
  assert current is resolve(name,alias=True),'native generic alias differs'
  assert current is not ref and not getattr(current,'__module__','').startswith('skimage.'),'upstream-forwarded R4 path'
  assert inspect.signature(current)==inspect.signature(ref),'public R4 signature differs'
  assert inspect.unwrap(current).__globals__.get('_core') is core,'R4 wrapper uses another core'

def compiled_entries(core):
 for name in ('rank_statistics_run','rank_windowed_histogram_run','rank_noise_footprint'):
  function=getattr(core,name,None)
  assert inspect.isbuiltin(function) and function.__module__==core.__name__ and function.__name__==name,'compiled R4 entry missing: '+name

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,nargs='+',default=[1]);parser.add_argument('--repeat',type=int,default=7);parser.add_argument('--target-ms',type=float,default=15.);parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true');parser.add_argument('--reference-only',action='store_true');parser.add_argument('--capsule',type=Path);parser.add_argument('--prepare-capsule',action='store_true');args=parser.parse_args(argv)
 if not args.threads or min(args.threads)<1 or max(args.threads)>1024 or len(set(args.threads))!=len(args.threads) or 1 not in args.threads or args.repeat<1 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('positive distinct workers1..1024/repeats/target-ms required; include worker1')
 if args.reference_only and (not args.check_only or args.capsule is not None):parser.error('--reference-only requires --check-only without capsule')
 if args.prepare_capsule and (not args.check_only or args.reference_only or args.capsule is None):parser.error('--prepare-capsule requires strict --check-only and capsule')
 if not args.check_only and (args.output is None or args.capsule is None or os.environ.get('GITHUB_ACTIONS')!='true'):parser.error('timing requires isolated Actions host/output/capsule; local --check-only only')
 previous=xx.get_num_threads();policy=xx.set_fallback_enabled(False);fingerprint=source_fingerprint(ROOT);manifest_hash=artifacts.sha(MANIFEST)
 harness_paths=('benchmarks/rank_statistics.py','benchmarks/run.py','benchmarks/resize.py','benchmarks/provenance.py','tools/ci_measurement_artifacts.py','tests/test_rank_statistics_contracts.py','tests/test_rank_neighborhood_contracts.py')
 harness={n:text_sha(ROOT/n) for n in harness_paths};harness_bytes={n:artifacts.sha(ROOT/n) for n in harness_paths};extension=Path(importlib.import_module('skimage.filters.rank.generic').generic_cy.__file__)
 session=str(uuid.uuid4());cpu=dict(cpu_metadata(),logical_cpus=os.cpu_count(),cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER',''))
 report=dict(metadata=dict(version=VERSION,source_revision=source_revision(ROOT),**fingerprint,harness_sha256=harness,harness_bytes_sha256=harness_bytes,harness_hash_schema='sha256-posix-lf-v1',reference_extension=dict(path=str(extension),sha256=artifacts.sha(extension)),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,scikit_image=skimage.__version__,platform=platform.platform(),cpu_session=cpu,session_id=session,process_id=os.getpid(),backend_cpu_sessions={'scikit-image':session,'current':session},same_process_cpu_for_backends=True,avx2=xx.has_avx2(),current_binary_sha256=binary_hashes(),current_python_sha256=python_hashes(),requested_workers=args.threads,primary_workers=1,initial_default_workers=previous,fallback_enabled=False,repeat=args.repeat,check_only=args.check_only,reference_only=args.reference_only,algorithm_count=4,scalar_api_count=3,vector_api_count=1,aliases_timed=False,validation_scope='reference-only/native0' if args.reference_only else 'strict actual public R4',reference_pools=1,scope='Warmed public calls; supplied output reset/allocation/destruction included. Fixture construction excluded. Errors/metadata/aliases untimed.'),status='preflight',checks=[],alias_checks=[],results=[],untimed_cases=[],failures=[])
 save(report,args.output)
 try:
  assert skimage.__version__=='0.26.0'
  if not args.reference_only:admit_native(previous)
  manifest=json.loads(MANIFEST.read_text());assert text_sha(SOURCE_CONTRACT)==manifest['source_contract_sha256']
  for name,h in manifest['fixture_source_sha256'].items():assert text_sha(ROOT/name)==h
  for name in NAMES:
   ref=resolve(name,True);facts=manifest['reference_functions'][name];assert ref is resolve(name,True,True)
   assert str(inspect.signature(ref))==facts['signature'] and hashlib.sha256(inspect.getsource(ref).encode()).hexdigest()==facts['python_source_sha256']
  frozen=manifest['reference_profiles'].get(runtime_profile())
  if not args.reference_only:
   assert frozen is not None and args.capsule is not None,'frozen original profile and authenticated capsule required'
   report['metadata']['exact_artifacts']=seal(args.capsule) if args.prepare_capsule else verify(args.capsule)
  cases=list(corpus());assert [inventory(c) for c in cases]==manifest['case_inventory'] and digest(manifest['case_inventory'])==manifest['case_contract_sha256']
  specs=[spec(c) for c in cases]
  if frozen:assert digest(specs)==frozen['case_spec_sha256']
  report['metadata'].update(profile=runtime_profile(),case_count=len(cases),case_specs=specs,case_spec_sha256=digest(specs),case_contract_sha256=manifest['case_contract_sha256'])
  report['alias_checks']=check_aliases(args.threads,args.reference_only)
  source=json.loads(SOURCE_CONTRACT.read_text());report['status']='checking'
  for i,case in enumerate(cases):
   row=dict(case_index=i,**specs[i])
   try:
    expected=observe_case(resolve(case.name,True),case);row['reference']=expected
    if case.frozen_id:assert expected==source['observations'][source['snapshots'][runtime_profile()][case.frozen_id]],'original differs from frozen995'
    if frozen:assert digest(expected)==frozen['observations'][str(i)],'original differs from frozen benchmark profile'
    row['timing_eligible']=case.timed and expected['error'] is None
    if not args.reference_only:
     row['actual_by_workers']={}
     for worker in args.threads:
      xx.set_num_threads(worker);assert xx.get_num_threads()==worker
      actual=observe_case(resolve(case.name),case);assert actual==expected;row['actual_by_workers'][str(worker)]=actual
    row['status']='passed'
   except Exception as e:row.update(status='failed',exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc());report['failures'].append(row)
   report['checks'].append(row)
   if not row.get('timing_eligible'):report['untimed_cases'].append(dict(case_index=i,reason='source contract/error/metadata; never timed'))
   if (i+1)%32==0 or row['status']=='failed':save(report,args.output)
  if report['failures']:report['status']='failed-parity';return 1
  eligible=[r for r in report['checks'] if r['timing_eligible']];report['metadata'].update(timing_eligible_case_count=len(eligible),scalar_timing_cases=sum(r['api_kind']=='scalar' for r in eligible),vector_timing_cases=sum(r['api_kind']=='vector' for r in eligible),native_checks=0 if args.reference_only else len(cases)*len(args.threads))
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
    report['results'].append(dict(case_index=row['case_index'],api=row['api'],case=row['case'],api_kind=row['api_kind'],session_id=session,threads=worker,primary=worker==1,image_kernel_speedup_claim=False,slower_than_original=timing['speedup']<1,**timing));save(report,args.output)
  integrity();report['summary_by_algorithm']={name:summarize([r for r in report['results'] if r['api']=='filters.rank.'+name],args.threads,sum(r['api']=='filters.rank.'+name for r in eligible)) for name in NAMES};report['status']='complete-measurement';return 0
 except Exception as e:report['failures'].append(dict(phase=report['status'],exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));report['status']='failed';return 1
 finally:xx.set_num_threads(previous);xx.set_fallback_enabled(policy);save(report,args.output)

if __name__=='__main__':raise SystemExit(main())
