"""Six max-tree APIs: original-only fixed-corpus authoring, native admission pending.

No operation timer is implemented here. The original Cython boundary is certified
before every call. The captured reference helper is immutable source, not a
replacement native backend. Future public199/SDK20 qualification and isolated
one-worker measurement require a separate reviewed consumer.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import types
import warnings
import zlib

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'coverage/contracts/max-tree-benchmark.json'
MATRIX=ROOT/'coverage/contracts/max-tree.json'
MATRIX_SHA='45d5999d49f3b1a5cdb8c7ec5767de470e74b158f5fb489f288132722edc6365'
CAPTURE_SHA='188a6b7d6641463c2d4949d04878881feff252410d2c2c38f032b081f1d65788'
CAPTURE_GIT='5b92e534da06e37eb348250ebbd1692b5ecccbd2:tests/test_max_tree_contracts.py'
NAMES=('max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing')
DTYPES=('u1','u2','u4','u8','i1','i2','i4','i8','f4','f8')
ENTRIES=('_max_tree','_compute_area','_compute_extension','_direct_filter','_max_tree_local_maxima')
MEMORY_LIMIT=256*1024*1024
ACCEPTED_PRODUCT_PINS=None
BINARY_SHA={
    'windows':'9a818964d2ed98b4e1bac5a2ca453ac6e3742db52a53e21857ffb1d107d97332',
    'modern':'517d355d11602636f37dcb86ce885465f3ca89570a10412e02e5009bb7ac9654',
    'minimum':'7a8a1dc23fcdfc13767ec6a460b864b888527b1cbc4c7cab7b2488a1e8407f14'}


class SafetyError(ValueError):pass
class AdmissionPending(RuntimeError):pass


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pin(path):
    data=Path(path).read_bytes();return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def canonical_sha(data):return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()


def native_admission(*args,**kwargs):
    raise AdmissionPending('native max-tree benchmarks require a genuine qualified public199/SDK20 product; Phase1 pins are null')


def timing_cases():
    rows=[]
    def add(identifier,name,dtype,shape,**options):
        row=dict(id='timed/'+name+'/'+identifier,function=name,dtype=dtype,shape=shape,
                 layout='C',pattern='multilevel',connectivity=1,threshold=17,provided=False)
        row.update(options);rows.append(row)
    for j,name in enumerate(NAMES):
        for k,dtype in enumerate(DTYPES):
            for rank,shape in ((2,[32,33]),(3,[8,9,10])):
                add(f'coverage/{dtype}/{rank}d',name,dtype,shape,
                    layout=('C','F','reverse','step')[(j+k+rank)%4],
                    pattern=('flat','plateau','ramp','sparse','multilevel')[(j+k)%5],
                    connectivity=1 if k%2 else rank)
        for dtype in ('u1','u2','f4','f8'):
            for label,shape in (('cache',[256,256]),('memory',[1024,1024]),('volume',[64,64,64])):
                add(f'large/{dtype}/{label}',name,dtype,shape,
                    pattern='plateau' if label=='memory' else 'multilevel',
                    connectivity=len(shape) if dtype in ('u2','f8') else 1)
            add('tiny/'+dtype,name,dtype,[4,5],pattern='sparse',threshold=3)
        add('deep-tree/u2/unique65536',name,'u2',[256,256],pattern='unique',connectivity=1)
        if name!='max_tree':
            for dtype in ('u1','u2','f4'):
                for rank,shape in ((2,[128,129]),(3,[24,25,26])):
                    add(f'provided/{dtype}/{rank}d',name,dtype,shape,provided=True,
                        layout='F' if dtype=='f4' else 'C',connectivity=rank)
    assert len(rows)==252 and len({row['id'] for row in rows})==252
    return tuple(rows)


CASES=timing_cases()


def validate_case(case):
    fields={'id','function','dtype','shape','layout','pattern','connectivity','threshold','provided'}
    if set(case)!=fields:raise SafetyError('unassessed case fields')
    shape=case['shape']
    if type(shape) is not list or len(shape) not in (2,3) or any(type(s) is not int or not 4<=s<=1024 for s in shape):
        raise SafetyError('unsafe radius-one source shape')
    n=math.prod(shape)
    if n>1024*1024:raise SafetyError('unassessed original size')
    if case['function'] not in NAMES or case['dtype'] not in DTYPES:raise SafetyError('unassessed function/dtype')
    if case['layout'] not in ('C','F','reverse','step') or case['pattern'] not in ('flat','plateau','ramp','sparse','multilevel','unique'):
        raise SafetyError('unassessed storage/pattern')
    if case['pattern']=='unique' and (case['dtype']!='u2' or shape!=[256,256]):raise SafetyError('unique stress domain is pinned uint16 256x256')
    if type(case['connectivity']) is not int or not 1<=case['connectivity']<=len(shape):raise SafetyError('unsafe radius-one connectivity')
    if type(case['threshold']) is not int or not 1<=case['threshold']<=1024:raise SafetyError('unassessed finite threshold')
    if type(case['provided']) is not bool or (case['provided'] and case['function']=='max_tree'):raise SafetyError('invalid provided-tree mode')
    # Upper envelope: image/owners/copies, two trees, zpar/sort scratch,
    # diameter float64 coordinates and transient unravel/maximum vectors,
    # certificate permutation/position arrays and record byte buffers. Observations
    # are streamed, not retained as raster hex. No original array exceeds N.
    budget=240*n+1024*1024
    if budget>MEMORY_LIMIT:raise SafetyError('original workspace budget exceeded')
    return budget


def load_reference(capture_text=None):
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.environ['OMP_DYNAMIC']='FALSE'
    if pin(MATRIX)['sha256']!=MATRIX_SHA:raise SafetyError('immutable351 matrix changed')
    matrix=json.loads(MATRIX.read_text())
    if capture_text is None:
        capture_text=json.loads(MANIFEST.read_text())['reference_capture']['utf8']
    if canonical_sha(capture_text.encode())!=CAPTURE_SHA:raise SafetyError('original helper capture changed')
    # This is explicitly an immutable captured helper, NOT the current physical
    # test file. Its synthetic anchor only resolves ROOT/MATRIX; no fixture file
    # is substituted and its pending-native tests are never executed here.
    module=types.ModuleType('_max_tree_original_reference_capture')
    module.__file__=str(ROOT/'tests/_max_tree_original_capture_anchor.py')
    sys.modules[module.__name__]=module
    exec(compile(capture_text,'<authenticated max-tree reference '+CAPTURE_SHA+'>','exec'),module.__dict__)
    module.authenticate()
    for source in matrix['sources'].values():
        if hashlib.sha256(source['utf8'].encode()).hexdigest()!=source['sha256']:raise SafetyError('pinned source/license capture corrupt')
    if matrix['sources']['src/skimage/morphology/_max_tree.pyx']['sha256']!='8d7a4c96bd9c1146c58d63594b4273a3359e38c05a0d6c4995449f6e69dfea50':
        raise SafetyError('full original Cython audit source changed')
    for name in NAMES:
        if str(inspect.signature(getattr(module.original,name)))!=matrix['signatures'][name]:raise SafetyError('original signature changed')
    if pin(module.original._max_tree.__file__)['sha256']!=BINARY_SHA[module.profile()]:raise SafetyError('original Cython binary changed')
    for actual,path in ((module.original,'src/skimage/morphology/max_tree.py'),
        (sys.modules[module.original.invert.__module__],'src/skimage/util/_invert.py'),
        (sys.modules[module.original._validate_connectivity.__module__],'src/skimage/morphology/_util.py')):
        if canonical_sha(inspect.getsource(actual).encode())!=matrix['sources'][path]['sha256']:raise SafetyError('installed dependency source changed: '+path)
    return module,matrix


def fixture(case,np):
    validate_case(case);shape=tuple(case['shape']);n=math.prod(shape)
    i=np.arange(n,dtype=np.uint64).reshape(shape)
    if case['pattern']=='unique':v=i
    elif case['pattern']=='flat':v=np.full(shape,3,dtype=np.uint64)
    elif case['pattern']=='plateau':v=((i//31)%8)
    elif case['pattern']=='ramp':v=i%127
    elif case['pattern']=='sparse':v=np.where(i%97==0,31,np.where(i%17==0,7,0))
    else:v=(i*13+i//19)%64
    image=v.astype(case['dtype']);owner=image
    if case['layout']=='F':image=np.array(image,order='F');owner=image
    elif case['layout']=='reverse':image=image[tuple(slice(None,None,-1) for _ in shape)]
    elif case['layout']=='step':
        owner=np.zeros(tuple(2*s for s in shape),dtype=case['dtype'])
        image=owner[tuple(slice(None,None,2) for _ in shape)];image[...] =v
    return image,owner


def array_record(value,np,blobs=None):
    a=np.asarray(value);data=a.tobytes();sha=hashlib.sha256(data).hexdigest()
    if blobs is not None:
        path=Path(blobs)/(sha+'.bin.z')
        if not path.exists():path.write_bytes(zlib.compress(data,6))
    return {'dtype':a.dtype.str,'shape':list(a.shape),'strides':list(a.strides),
        'own':bool(a.flags.owndata),'C':bool(a.flags.c_contiguous),'F':bool(a.flags.f_contiguous),
        'writeable':bool(a.flags.writeable),'aligned':bool(a.flags.aligned),'bytes':len(data),'sha256':sha}


def certify_array(a,np,dtype=None,n=None):
    if (type(a) is not np.ndarray or not a.dtype.isnative or not a.flags.c_contiguous
        or not a.flags.aligned or not a.flags.writeable or (dtype is not None and a.dtype!=np.dtype(dtype))
        or (n is not None and a.size!=n)):raise SafetyError('unsafe original prepared buffer')


def certify_tree(image,parent,traverser,np):
    n=image.size;certify_array(parent,np,'i8',n);certify_array(traverser,np,'i8',n)
    if np.shares_memory(parent,traverser) or np.shares_memory(image,parent) or np.shares_memory(image,traverser):raise SafetyError('overlapping original tree')
    if np.any(parent<0) or np.any(parent>=n) or not np.array_equal(np.sort(traverser),np.arange(n,dtype=np.int64)):
        raise SafetyError('unsafe parent/permutation')
    pos=np.empty(n,dtype=np.int64);pos[traverser]=np.arange(n,dtype=np.int64)
    root=int(traverser[0]);pixels=np.arange(n,dtype=np.int64)
    if int(parent[root])!=root or np.any((pixels!=root)&(pos[parent]>=pos)) or np.any(image[parent]>image):
        raise SafetyError('unsafe tree root/order/value')


def certify_entry(name,args,shape,np):
    image=args[0];certify_array(image,np)
    if (image.ndim!=1 or image.dtype not in tuple(np.dtype(d) for d in DTYPES)
        or len(shape) not in (2,3) or min(shape)<4 or image.size!=math.prod(shape)
        or image.size>1024*1024 or not np.isfinite(image).all()):raise SafetyError('unsafe prepared source image')
    n=image.size
    if name=='_max_tree':
        _,mask,structure,offset,dimensions,parent,traverser=args
        for a,dtype,size in ((mask,'u1',n),(offset,'i4',len(shape)),(dimensions,'i4',len(shape)),(parent,'i8',n),(traverser,'i8',n)):
            certify_array(a,np,dtype,size)
        certify_array(structure,np,'i4')
        if not np.array_equal(dimensions,shape) or not np.all(offset==1) or structure.size>26:raise SafetyError('unsafe radius/shape')
        shaped=mask.reshape(shape)
        for axis in range(len(shape)):
            if np.moveaxis(shaped,axis,0)[[0,-1]].any():raise SafetyError('original neighbor mask must zero borders')
        center=sum(math.prod(shape[k+1:]) for k in range(len(shape)))
        for delta in map(int,structure):
            if not 0<=center+delta<n or any(abs(int(c)-1)>1 for c in np.unravel_index(center+delta,shape)):
                raise SafetyError('unsafe neighbor offset')
        if not np.array_equal(np.sort(traverser),np.arange(n,dtype=np.int64)):raise SafetyError('unsafe initial permutation')
        values=image[traverser]
        if np.any(values[1:]<values[:-1]) or np.any((values[1:]==values[:-1])&(traverser[1:]<traverser[:-1])):
            raise SafetyError('original stable sort order changed')
        for i,a in enumerate(args):
            for b in args[i+1:]:
                if np.shares_memory(a,b):raise SafetyError('overlapping builder buffers')
    elif name in ('_compute_area','_compute_extension'):
        if name=='_compute_extension':
            certify_array(args[1],np,'i4',len(shape))
            if not np.array_equal(args[1],shape):raise SafetyError('unsafe extension shape')
            certify_tree(image,args[2],args[3],np)
        else:certify_tree(image,args[1],args[2],np)
    elif name=='_direct_filter':
        _,out,parent,traverser,attribute,threshold=args;certify_tree(image,parent,traverser,np)
        certify_array(out,np,image.dtype,n);certify_array(attribute,np,'f8',n)
        if type(threshold) is not int or not 1<=threshold<=1024 or not np.isfinite(attribute).all():raise SafetyError('unsafe filter attribute/threshold')
        if any(np.shares_memory(out,a) for a in (image,parent,traverser,attribute)):raise SafetyError('unsafe filter output overlap')
    elif name=='_max_tree_local_maxima':
        _,out,parent,traverser=args;certify_tree(image,parent,traverser,np);certify_array(out,np,'u8',n)
        if not np.all(out==1) or any(np.shares_memory(out,a) for a in (image,parent,traverser)):raise SafetyError('unsafe maxima output')
    else:raise SafetyError('unknown original compiled entry')


def observe_case(case,reference,blobs=None):
    np=reference.np;image,owner=fixture(case,np);original=reference.original
    entries=[];providers=[];saved={};phase='prepare'
    def record(value):
        if isinstance(value,np.ndarray):return {'array':array_record(value,np,blobs)}
        if isinstance(value,(tuple,list)):return [record(v) for v in value]
        if isinstance(value,np.generic):return reference.scalar(value)
        if isinstance(value,type):return {'type':value.__module__+'.'+value.__name__}
        return value
    def entry(name,fn):
        def call(*args):
            certify_entry(name,args,image.shape,np)
            row={'name':name,'phase':phase,'before':[record(v) for v in args]};entries.append(row)
            result=fn(*args);row['after']=[record(v) for v in args];row['result']=record(result)
            return result
        return call
    def provider(name,fn):
        def call(*args,**kwargs):
            providers.append({'name':name,'phase':phase,'args':record(args),'kwargs':{k:record(v) for k,v in kwargs.items()}})
            return fn(*args,**kwargs)
        return call
    kwargs={'connectivity':case['connectivity']}
    if case['function'] not in ('max_tree','max_tree_local_maxima'):
        kwargs['area_threshold' if case['function'].startswith('area') else 'diameter_threshold']=case['threshold']
    before={'image':array_record(image,np,blobs),'owner':array_record(owner,np,blobs)}
    old=np.seterr(all='warn');result=None;error=None;provided={}
    try:
        for name in ENTRIES:
            fn=getattr(original._max_tree,name);saved[name]=fn;setattr(original._max_tree,name,entry(name,fn))
        for name in ('max_tree','invert','_validate_connectivity','_offsets_to_raveled_neighbors'):
            fn=getattr(original,name);saved[name]=fn;setattr(original,name,provider(name,fn))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if case['provided']:
                    work=original.invert(image) if case['function'].endswith('closing') else image
                    parent,traverser=original.max_tree(work,case['connectivity'])
                    kwargs.update(parent=parent,tree_traverser=traverser)
                    provided={key:array_record(kwargs[key],np,blobs) for key in ('parent','tree_traverser')}
                phase='call'
                result=(saved['max_tree'] if case['function']=='max_tree' else getattr(original,case['function']))(image,**kwargs)
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            warning_rows=[{'type':type(w.message).__name__,'message':str(w.message)} for w in caught]
    finally:
        for name in ENTRIES:setattr(original._max_tree,name,saved[name])
        for name in ('max_tree','invert','_validate_connectivity','_offsets_to_raveled_neighbors'):setattr(original,name,saved[name])
        np.seterr(**old)
    after={'image':array_record(image,np,blobs),'owner':array_record(owner,np,blobs)}
    provided_after={key:array_record(kwargs[key],np,blobs) for key in provided}
    def output(a):
        return {'array':array_record(a,np,blobs),'type':type(a).__name__,'is_input':a is image,
                'shares_input':bool(np.shares_memory(a,image)),
                'base':array_record(a.base,np,blobs) if isinstance(a.base,np.ndarray) else None,
                'shares_provided':{key:bool(np.shares_memory(a,kwargs[key])) for key in provided}}
    outcome=None if result is None else [output(a) for a in result] if isinstance(result,tuple) else output(result)
    row={'before':before,'after':after,'provided_before':provided,'provided_after':provided_after,
         'entries':entries,'providers':providers,'result':outcome,'error':error,'warnings':warning_rows}
    if error or warning_rows or before!=after or provided!=provided_after or not entries or outcome is None:
        raise SafetyError('source-certified finite case failed '+case['id']+': '+json.dumps(row))
    return row


SCIENTIFIC=('timing_cases','validate_case','fixture','array_record','certify_array','certify_tree','certify_entry','observe_case')


def verify_manifest(manifest):
    if (manifest.get('schema')!='max-tree-benchmark-v1' or manifest.get('native_checks')!=0
        or type(manifest.get('native_checks')) is not int or manifest.get('operation_timings')!=0
        or type(manifest.get('operation_timings')) is not int or manifest.get('accepted_product_pins','absent') is not None):
        raise SafetyError('Phase1 cannot carry native/performance admission')
    if manifest.get('cases')!=list(CASES) or manifest.get('case_spec_sha256')!=digest(CASES):raise SafetyError('fixed252 case data changed')
    if manifest.get('matrix_sha256')!=MATRIX_SHA or canonical_sha(manifest['reference_capture']['utf8'].encode())!=CAPTURE_SHA:
        raise SafetyError('immutable351/captured helper changed')
    if manifest.get('harness_canonical_lf_sha256')!=canonical_sha(Path(__file__).read_bytes()):raise SafetyError('harness source changed')
    ast_profiles=manifest.get('scientific_ast',{})
    if set(ast_profiles)!={'3.11','3.13'}:raise SafetyError('scientific AST runtime set changed')
    minor=str(sys.version_info.major)+'.'+str(sys.version_info.minor)
    if minor not in ast_profiles:raise SafetyError('unassessed Python AST schema')
    pins=ast_profiles[minor]
    if set(pins)!=set(SCIENTIFIC):raise SafetyError('scientific AST set changed')
    for name in SCIENTIFIC:
        actual=hashlib.sha256(ast.dump(ast.parse(inspect.getsource(globals()[name])),include_attributes=False).encode()).hexdigest()
        if actual!=pins[name]:raise SafetyError('scientific AST changed: '+name)
    for case in CASES:validate_case(case)


def verify_report(report,manifest):
    verify_manifest(manifest)
    if report['native_checks']!=0 or report['operation_timings']!=0 or report['accepted_product_pins'] is not None:raise SafetyError('reference receipt promoted')
    expected=manifest['profiles'][report['profile']]
    for key in ('reference351','cases','runtime'):
        if report[key]!=expected[key]:raise SafetyError('original raw observations/runtime changed: '+key)
    if len(report['cases'])!=252 or len({r['id'] for r in report['cases']})!=252:raise SafetyError('partial/duplicate252 rows')
    for row,case in zip(report['cases'],CASES):
        if row['id']!=case['id'] or row['spec']!=case or digest(row['observation'])!=row['digest']:raise SafetyError('case observation mismatch')


def check_reference(output=None,blobs=None,manifest=None,capture_text=None):
    if manifest is not None:verify_manifest(manifest)
    reference,matrix=load_reference(capture_text);profile=reference.profile();np=reference.np
    if blobs is not None:Path(blobs).mkdir(parents=True,exist_ok=True)
    oldrows=[]
    for spec in reference.CASES:
        actual=reference.observe(spec);key=matrix['profiles'][profile]['snapshots'][spec['id']]
        if actual!=matrix['observations'][key]:raise SafetyError('immutable351 original observation changed: '+spec['id'])
        raw=json.dumps(actual,sort_keys=True,separators=(',',':')).encode();sha=hashlib.sha256(raw).hexdigest()
        if blobs is not None:
            path=Path(blobs)/(sha+'.json.z')
            if not path.exists():path.write_bytes(zlib.compress(raw,6))
        oldrows.append({'id':spec['id'],'mode':spec['mode'],'observation_sha256':reference.digest(actual),'raw_json_sha256':sha})
    rows=[]
    for case in CASES:
        actual=observe_case(case,reference,blobs)
        rows.append({'id':case['id'],'spec':case,'observation':actual,'digest':digest(actual)})
        print('original-only '+profile+' '+case['id'],flush=True)
    runtime={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':reference.scipy.__version__,
        'skimage':reference.skimage.__version__,'interpreter':{'path':sys.executable,**pin(sys.executable)},
        'binary':{'path':reference.original._max_tree.__file__,**pin(reference.original._max_tree.__file__)},
        'public_source_sha256':canonical_sha(inspect.getsource(reference.original).encode()),
        'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
    report={'schema':'max-tree-benchmark-reference-v1','profile':profile,'native_checks':0,'operation_timings':0,
        'accepted_product_pins':None,'phase':'original-only-not-native-or-performance-admission',
        'reference351':oldrows,'cases':rows,'runtime':runtime}
    if output is not None:Path(output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    if manifest is not None:verify_report(report,manifest)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true');parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--record-reference',action='store_true');parser.add_argument('--capture',type=Path)
    parser.add_argument('--workers',type=int,choices=(1,4),default=1)
    parser.add_argument('--output',type=Path);parser.add_argument('--blob-dir',type=Path)
    args=parser.parse_args(argv)
    if not args.reference_only:native_admission()
    if not args.check_only:raise AdmissionPending('Phase1 is check-only; no operation timing implemented')
    if args.capture is not None and not args.record_reference:raise SafetyError('explicit capture only for original authoring')
    manifest=None if args.record_reference else json.loads(MANIFEST.read_text())
    capture=args.capture.read_text() if args.capture else None
    result=check_reference(args.output,args.blob_dir,manifest,capture)
    print(json.dumps({'profile':result['profile'],'original351':len(result['reference351']),'timing_candidates':len(result['cases']),
        'requested_future_native_workers':args.workers,'original_worker_semantics':'original serial; not a native worker test',
        'native_checks':0,'operation_timings':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
