"""Bregman fixed-corpus reference authoring; native measurement is not admitted.

Only --reference-only --check-only executes original image calls. This Phase1
module has no operation timer or native implementation selector. Qualified
whole-product consumers, three-profile preparation barrier and fresh-host
measurement are a later gate; accepted product pins stay null.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings
import zlib

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'coverage/contracts/tv-bregman-benchmark.json'
REFERENCE_MATRIX=ROOT/'coverage/contracts/tv-bregman.json'
REFERENCE_HELPERS=ROOT/'tests/test_tv_bregman_contracts.py'
MATRIX_SHA256='638a6f569348a94f36ba11ef1dbbb8f1f08f4acbc1aec1cf893e9fe968cfa85c'
HELPERS_CANONICAL_SHA256='11a951abca0c01356e746225a995eb6256caa0676ce0dff265f4e976458dc60b'
API='restoration.denoise_tv_bregman'
ACCEPTED_PRODUCT_PINS=None
MEMORY_LIMIT=64*1024*1024
REFERENCE_BINARY_SHA256={
    'windows':'0136d13d55789de97b0cdc3df00c7d72334da56293d9461c7969017194539ea1',
    'modern':'803974c8547065e2c176734956769da1749e2c55ad88f0599d2ceeb435c80643',
    'minimum':'d662cc8dd8b13ef6fb680d8ca71d61277a87fca3a90f8994507c935699dfcd90'}
SCIENTIFIC_FUNCTIONS=('timing_cases','validate_case','fixture','array_record','certify_entry',
                      'observe_timing_case','observe_nonfinite_intake')


class AdmissionPending(RuntimeError):pass
class SafetyError(ValueError):pass


def canonical_hash(data):return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(path):
    data=Path(path).read_bytes();return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def native_admission(*args,**kwargs):
    # No symbol/private-stage presence or caller-supplied path can promote a
    # Phase1 reference receipt into genuine public/SDK/native admission.
    raise AdmissionPending('Bregman native benchmarking requires a qualified whole public product; Phase1 pins are null')


def configure_reference_environment():
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['OMP_DYNAMIC']='FALSE'


def timing_cases():
    rows=[]
    def add(identifier,**options):
        row=dict(id='timed/'+identifier,dtype='f4',shape=[4,5],layout='C',pattern='texture',
                 max_num_iter=2,weight=.5,eps=1e-6,isotropic=True,channel_axis=None)
        row.update(options);rows.append(row)
    for dtype in ('u1','f4','f8'):
        for size,shape in (('tiny',[4,5]),('medium',[64,65]),('large',[512,513])):
            for iso in (False,True):
                for policy,count,eps,pattern,weight in (
                    ('i1',1,1e-6,'ramp',.5),('i2',2,1e-6,'texture',.25),
                    ('budget12',12,0.,'texture',5.),('convergent',64,1e-3,'constant',.5)):
                    add(f'{dtype}/{size}/{int(iso)}/{policy}',dtype=dtype,shape=shape,
                        isotropic=iso,max_num_iter=count,eps=eps,pattern=pattern,weight=weight)
        for layout in ('F','reverse','step'):
            add(f'{dtype}/layout/{layout}',dtype=dtype,shape=[32,35],layout=layout,
                isotropic=layout!='reverse',max_num_iter=5,weight=8.)
        for axis in (None,-1,0,1):
            for channels in (1,3):
                add(f'{dtype}/channels/{axis}/c{channels}',dtype=dtype,shape=[24,27,channels],
                    channel_axis=axis,layout='F' if axis in (0,1) else 'C',pattern='channels',max_num_iter=5)
    for dtype in ('f4','f8'):
        add(f'{dtype}/layout/unaligned-step',dtype=dtype,shape=[32,35],layout='unaligned-step',max_num_iter=5)
    add('u1/layout/byte-offset',dtype='u1',shape=[32,35],layout='offset',max_num_iter=5)
    add('u2/conversion/unaligned',dtype='u2',shape=[32,35],layout='unaligned',max_num_iter=5)
    add('u2/conversion/big-endian',dtype='>u2',shape=[32,35],max_num_iter=5)
    assert len(rows)==110 and len({row['id'] for row in rows})==110
    return tuple(rows)


TIMING_CASES=timing_cases()
UNSAFE_INTAKES=tuple({'id':f'intake-only/nonfinite/{dtype}/{kind}','dtype':dtype,'kind':kind}
                    for dtype in ('f4','f8') for kind in ('nan','inf'))


def validate_case(case):
    required={'id','dtype','shape','layout','pattern','max_num_iter','weight','eps','isotropic','channel_axis'}
    if set(case)!=required:raise SafetyError('timing case fields differ from frozen schema')
    shape=case['shape']
    if (type(shape) is not list or len(shape) not in (2,3) or any(type(s) is not int or s<1 for s in shape)
        or min(shape[:2])<4 or (len(shape)==3 and shape[2] not in (1,3))):raise SafetyError('unsafe source spatial/component shape')
    if case['dtype'] not in ('u1','u2','>u2','f4','f8'):raise SafetyError('unassessed source dtype')
    if case['layout'] not in ('C','F','reverse','step','unaligned-step','offset','unaligned'):raise SafetyError('unassessed layout')
    if case['layout']=='unaligned' and case['dtype']!='u2':raise SafetyError('unaligned contiguous float must never enter original Cython')
    if case['layout']=='offset' and case['dtype']!='u1':raise SafetyError('offset cohort is byte-typed only')
    if case['layout']=='unaligned-step' and case['dtype'] not in ('f4','f8'):raise SafetyError('unaligned float requires a non-C copying path')
    if case['pattern'] not in ('ramp','texture','constant','channels'):raise SafetyError('unassessed pattern')
    if type(case['max_num_iter']) is not int or not 1<=case['max_num_iter']<=64:raise SafetyError('iterations must be positive bounded primitive C int')
    if type(case['isotropic']) is not bool:raise SafetyError('isotropic must be a primitive bool')
    if type(case['weight']) not in (int,float) or not math.isfinite(case['weight']) or not .25<=case['weight']<=8:raise SafetyError('unsafe weight')
    if type(case['eps']) not in (int,float) or not math.isfinite(case['eps']) or not 0<=case['eps']<=1:raise SafetyError('unsafe eps')
    if case['channel_axis'] not in (None,-1,0,1) or (len(shape)==2 and case['channel_axis'] is not None):raise SafetyError('unassessed axis')
    n=math.prod(shape);padded=(shape[0]+2)*(shape[1]+2)*(shape[2] if len(shape)==3 else 1)
    # Conservative complete input/conversion/global out + four auxiliary copies
    # and channel storage. Source images/workspaces remain <=64MiB.
    memory=(n*3+padded*8)*8
    if memory>MEMORY_LIMIT:raise SafetyError('case exceeds bounded original workspace budget')
    return memory


def load_reference():
    configure_reference_environment()
    if pin(REFERENCE_MATRIX)['sha256']!=MATRIX_SHA256:raise SafetyError('frozen135 original matrix changed')
    if canonical_hash(REFERENCE_HELPERS.read_bytes())!=HELPERS_CANONICAL_SHA256:raise SafetyError('frozen scientific helper source changed')
    spec=importlib.util.spec_from_file_location('_bregman_benchmark_original_contracts',REFERENCE_HELPERS)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    matrix=json.loads(REFERENCE_MATRIX.read_text())
    if module.skimage.__version__!='0.26.0' or module.source_sha(module.reference)!=matrix['source_sha256']:
        raise SafetyError('original source/version mismatch')
    if str(inspect.signature(module.reference))!=matrix['signature'] or module.reference is not module.original_module.denoise_tv_bregman:
        raise SafetyError('original public/helper identity mismatch')
    sources=matrix['sources']
    for path,entry in sources.items():
        if hashlib.sha256(entry['utf8'].encode()).hexdigest()!=entry['sha256']:raise SafetyError('original source/license capture corrupt')
    if sources['src/skimage/restoration/_denoise_cy.pyx']['sha256']!='737fa38f080af85c41156d2dfe758c499bc22790b9a8a50c976fbbe8653aa124':
        raise SafetyError('original full Cython source changed')
    helper_module=sys.modules[module.original_module._denoise_tv_bregman.__module__]
    if pin(helper_module.__file__)['sha256']!=REFERENCE_BINARY_SHA256[module.profile()]:
        raise SafetyError('original helper binary mismatch before image calls')
    for actual,path in ((module.original_module,'src/skimage/restoration/_denoise.py'),
        (module.original_module.utils,'src/skimage/_shared/utils.py'),
        (sys.modules[module.original_module.img_as_float.__module__],'src/skimage/util/dtype.py')):
        if hashlib.sha256(inspect.getsource(actual).replace('\r\n','\n').encode()).hexdigest()!=sources[path]['sha256']:
            raise SafetyError('installed original dependency source mismatch: '+path)
    return module,matrix


def fixture(case,np):
    validate_case(case)
    shape=tuple(case['shape']);values=np.arange(math.prod(shape),dtype=np.int64).reshape(shape)
    if case['pattern']=='constant':data=np.full(shape,.25)
    elif case['pattern']=='ramp':data=(values%17)/16.
    else:data=((values*13+values//7)%33)/32.
    if case['pattern']=='channels':data=(data+np.arange(shape[-1])[None,None,:]/8.)/2.
    dtype=np.dtype(case['dtype'])
    image=(data*np.iinfo(dtype).max).astype(dtype) if dtype.kind=='u' else data.astype(dtype)
    axis=case['channel_axis']
    if len(shape)==3 and axis is not None and axis!=-1:image=np.moveaxis(image,-1,axis)
    layout=case['layout'];owner=image
    if layout=='F':image=np.array(image,order='F');owner=image
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='step':
        owner=np.zeros(tuple(2*s for s in image.shape),dtype=dtype);view=owner[tuple(slice(None,None,2) for _ in image.shape)];view[...] =image;image=view
    elif layout in ('unaligned','offset','unaligned-step'):
        full_shape=tuple(2*s for s in image.shape) if layout=='unaligned-step' else image.shape
        owner=np.zeros(math.prod(full_shape)*dtype.itemsize+1,dtype=np.uint8)
        raw=np.ndarray(full_shape,dtype,buffer=owner,offset=1)
        view=raw[tuple(slice(None,None,2) for _ in image.shape)] if layout=='unaligned-step' else raw
        view[...] =image;image=view
    options={key:case[key] for key in ('weight','max_num_iter','eps','isotropic')}
    if axis is not None:options['channel_axis']=axis
    return image,owner,options


def array_record(value,np,blob_dir=None):
    a=np.asarray(value);data=a.tobytes();sha=hashlib.sha256(data).hexdigest()
    if blob_dir is not None:
        path=Path(blob_dir)/(sha+'.bin.z')
        if not path.exists():path.write_bytes(zlib.compress(data,9))
    return {'dtype':a.dtype.str,'shape':list(a.shape),'strides':list(a.strides),'own':bool(a.flags.owndata),
            'C':bool(a.flags.c_contiguous),'F':bool(a.flags.f_contiguous),'writeable':bool(a.flags.writeable),
            'aligned':bool(a.flags.aligned),'bytes':len(data),'sha256':sha}


def certify_entry(np,image,weight,maximum,eps,isotropic,out):
    if (type(image) is not np.ndarray or image.ndim!=3 or image.dtype not in (np.dtype('f4'),np.dtype('f8'))
        or not image.dtype.isnative or not image.flags.c_contiguous or not image.flags.aligned or not image.flags.writeable):
        raise SafetyError('original Cython requires prepared native aligned writable C float32/64')
    r,c,d=image.shape
    if min(r,c)<4 or d<1 or image.size>2**22 or not np.isfinite(image).all():raise SafetyError('unsafe original image domain')
    if type(weight) not in (np.float32,np.float64) or not np.isfinite(weight) or not .25<=weight<=8:raise SafetyError('unsafe typed weight')
    if type(maximum) is not int or not 1<=maximum<=64 or type(eps) not in (int,float) or not np.isfinite(eps) or not 0<=eps<=1 or type(isotropic) is not bool:
        raise SafetyError('unsafe original scalar domain')
    if (type(out) is not np.ndarray or out.shape!=(r+2,c+2,d) or out.dtype!=image.dtype
        or not out.flags.c_contiguous or not out.flags.aligned or not out.flags.writeable or np.shares_memory(image,out)
        or not np.isfinite(out).all()):
        raise SafetyError('unsafe original padded output')
    if image.nbytes+out.nbytes*6>MEMORY_LIMIT:raise SafetyError('unsafe original allocation bound')


def observe_timing_case(case,reference,blob_dir=None):
    np=reference.np;image,owner,options=fixture(case,np)
    before={'image':array_record(image,np,blob_dir),'owner':array_record(owner,np,blob_dir)}
    entry=reference.original_module._denoise_tv_bregman;entries=[]
    def guarded(image,weight,maximum,eps,isotropic,out):
        certify_entry(np,image,weight,maximum,eps,isotropic,out)
        row={'image':array_record(image,np,blob_dir),'weight':reference.scalar(weight),
             'maximum':maximum,'eps':eps.hex() if type(eps) is float else eps,'isotropic':isotropic,
             'out_before':array_record(out,np,blob_dir)};entries.append(row)
        result=entry(image,weight,maximum,eps,isotropic,out)
        row['out_after']=array_record(out,np,blob_dir);row['returns_none']=result is None
        return result
    result=None;error=None;old=np.seterr(all='warn')
    try:
        reference.original_module._denoise_tv_bregman=guarded
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=reference.reference(image,**options)
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            warning_rows=[{'type':type(w.message).__name__,'message':str(w.message)} for w in caught]
    finally:reference.original_module._denoise_tv_bregman=entry;np.seterr(**old)
    after={'image':array_record(image,np,blob_dir),'owner':array_record(owner,np,blob_dir)}
    outcome=None if result is None else {'array':array_record(result,np,blob_dir),'type':type(result).__name__,
        'base':array_record(result.base,np,blob_dir) if isinstance(result.base,np.ndarray) else None,
        'shares_input':bool(np.shares_memory(result,image)),'is_input':result is image}
    if error is not None:raise SafetyError('finite timing candidate failed: '+case['id']+' '+str(error))
    if warning_rows or before!=after or not entries or outcome['shares_input']:
        raise SafetyError('timing observation violates source certificate')
    return {'before':before,'after':after,'result':outcome,'entries':entries,'error':error,'warnings':warning_rows}


def observe_nonfinite_intake(spec,reference):
    np=reference.np;image=np.zeros((4,5),dtype=spec['dtype']);image[1,1]=np.nan if spec['kind']=='nan' else np.inf
    before=reference.array(image);entry=reference.original_module._denoise_tv_bregman;reached=[]
    def blocked(*args):reached.append([reference.array(args[0]),reference.scalar(args[1])]);raise reference.EntryBlocked('nonfinite original entry intentionally blocked')
    error=None
    try:
        reference.original_module._denoise_tv_bregman=blocked
        try:reference.reference(image,max_num_iter=2)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
    finally:reference.original_module._denoise_tv_bregman=entry
    if not error or error['type']!='EntryBlocked' or len(reached)!=1 or reference.array(image)!=before:raise SafetyError('nonfinite intake block changed')
    return {'before':before,'after':reference.array(image),'entries':reached,'error':error,'numeric':False}


def verify_manifest_metadata(manifest):
    if (manifest.get('schema')!='tv-bregman-benchmark-corpus-v1'
        or manifest.get('accepted_product_pins','missing') is not None
        or type(manifest.get('native_checks')) is not int or manifest['native_checks']!=0
        or type(manifest.get('operation_timings')) is not int or manifest['operation_timings']!=0):
        raise SafetyError('reference corpus cannot carry native/performance admission')
    if manifest.get('timing_cases')!=list(TIMING_CASES) or manifest.get('timing_spec_sha256')!=digest(TIMING_CASES):
        raise SafetyError('frozen timing IDs/options changed')
    if manifest.get('raw_matrix_sha256')!=MATRIX_SHA256 or manifest.get('helper_source_canonical_lf_sha256')!=HELPERS_CANONICAL_SHA256:
        raise SafetyError('scientific source/matrix pin changed')
    if manifest.get('harness_source_canonical_lf_sha256')!=canonical_hash(Path(__file__).read_bytes()):
        raise SafetyError('reference harness source changed')
    if set(manifest.get('scientific_function_sha256',{}))!=set(SCIENTIFIC_FUNCTIONS):
        raise SafetyError('scientific function pin set changed')
    for name,expected in manifest['scientific_function_sha256'].items():
        if name not in globals() or hashlib.sha256(inspect.getsource(globals()[name]).replace('\r\n','\n').encode()).hexdigest()!=expected:
            raise SafetyError('scientific fixture function changed: '+name)
    for case in TIMING_CASES:validate_case(case)


def verify_report(report,manifest):
    verify_manifest_metadata(manifest)
    profile=report['profile'];expected=manifest['profiles'][profile]
    if report['native_checks']!=0 or report['operation_timings']!=0 or report['accepted_product_pins'] is not None:
        raise SafetyError('reference receipt promoted into performance admission')
    if report['reference135']!=expected['reference135'] or report['blocked_nonfinite']!=expected['blocked_nonfinite']:
        raise SafetyError('original135/intake observation changed')
    if report['timing_cases']!=expected['timing_cases']:
        raise SafetyError('timing raw-value/metadata observation changed')
    for row,case in zip(report['timing_cases'],TIMING_CASES):
        if row['id']!=case['id'] or row['spec']!=case or digest(row['observation'])!=row['digest']:
            raise SafetyError('timing observation contents/fixture source changed')
    actual={row['id']:row['digest'] for row in report['timing_cases']}
    if len(actual)!=len(report['timing_cases']) or len(actual)!=len(TIMING_CASES):
        raise SafetyError('timing observation IDs missing or duplicated')
    if actual!=expected['timing_observation_sha256']:raise SafetyError('timing fixture output/metadata/warnings changed')
    for key in ('numpy','scipy','skimage','public_source_sha256'):
        if report['runtime'][key]!=expected['runtime'][key]:raise SafetyError('reference runtime/source mismatch: '+key)


def check_reference(output=None,blob_dir=None,manifest=None):
    if manifest is not None:verify_manifest_metadata(manifest)
    reference,matrix=load_reference();np=reference.np;profile=reference.profile()
    if blob_dir is not None:Path(blob_dir).mkdir(parents=True,exist_ok=True)
    reference_rows=[]
    for spec in reference.CASES:
        actual=reference.observe(reference.reference,spec)
        key=matrix['profiles'][profile]['snapshots'][spec['id']]
        if actual!=matrix['observations'][key]:raise SafetyError('original135 differs: '+spec['id'])
        reference_rows.append({'id':spec['id'],'mode':spec['mode'],'digest':reference.digest(actual)})
    timing=[]
    for case in TIMING_CASES:
        actual=observe_timing_case(case,reference,blob_dir)
        timing.append({'id':case['id'],'spec':case,'observation':actual,'digest':digest(actual)})
    blocked=[{'id':spec['id'],'observation':observe_nonfinite_intake(spec,reference)} for spec in UNSAFE_INTAKES]
    helper_module=sys.modules[reference.original_module._denoise_tv_bregman.__module__]
    report={'schema':'tv-bregman-benchmark-reference-v1','phase':'reference-only-ineligible-for-native-or-performance',
        'native_checks':0,'operation_timings':0,'accepted_product_pins':None,'profile':profile,
        'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':reference.scipy.__version__,
                   'skimage':reference.skimage.__version__,'interpreter':{'path':sys.executable,**pin(sys.executable)},
                   'helper_binary':{'path':helper_module.__file__,**pin(helper_module.__file__)},
                   'public_source_sha256':reference.source_sha(reference.reference),
                   'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_DYNAMIC')}},
        'reference135':reference_rows,'timing_cases':timing,'blocked_nonfinite':blocked,
        'timing_spec_sha256':digest(TIMING_CASES),'raw_matrix_sha256':MATRIX_SHA256,
        'helper_source_canonical_lf_sha256':canonical_hash(REFERENCE_HELPERS.read_bytes()),
        'harness_source_canonical_lf_sha256':canonical_hash(Path(__file__).read_bytes())}
    if manifest is not None:verify_report(report,manifest)
    if output is not None:Path(output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true');parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,4),default=1)
    parser.add_argument('--output',type=Path);parser.add_argument('--blob-dir',type=Path)
    parser.add_argument('--record-reference',action='store_true',help='Author original-only observations before freezing the corpus')
    args=parser.parse_args(argv)
    if not args.reference_only:native_admission()
    if not args.check_only:raise AdmissionPending('Phase1 reference execution is check-only; no operation timing is implemented')
    manifest=None if args.record_reference else json.loads(MANIFEST.read_text())
    result=check_reference(args.output,args.blob_dir,manifest)
    print(json.dumps({'profile':result['profile'],'original135':len(result['reference135']),
        'timing_candidates':len(result['timing_cases']),'blocked_nonfinite':len(result['blocked_nonfinite']),
        'requested_future_native_workers':args.workers,'reference_global_worker_setting':'not-applicable; original helper serial',
        'native_checks':0,'operation_timings':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
