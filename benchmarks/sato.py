"""Allocation-inclusive Sato filtering versus pinned scikit-image."""
from run import measure
from provenance import source_fingerprint
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sato_parity import assert_parity
import numpy as np
import scipy,skimage
from skimage.filters import sato as reference
import skimagexx as xx


def serial(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if min(args.threads)<=0 or args.repeat<=0:parser.error('positive workers/repeats required')
    if not args.check_only and args.output is None:parser.error('--output is required for measurements')
    assert skimage.__version__=='0.26.0'
    xx.set_fallback_enabled(False);rng=np.random.default_rng(7933);cases=[]
    def make(dtype,shape):
        if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
        raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
        return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)
    for rank in (2,3):
        for dtype in (np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64):
            a=make(dtype,(128,129) if rank==2 else (17,19,21));ordinary=dict(sigmas=[1])
            cases.extend([('contiguous',a,ordinary),('strided',a[::-1,::2],ordinary),
                ('fortran-wide-sigma',np.asfortranarray(a),dict(sigmas=[np.float64(1)])),
                ('tiny',a[(slice(0,7),slice(0,9))] if rank==2 else a[:3,:4,:5],ordinary),
                ('broadcast',np.broadcast_to(a[(slice(0,1),)*rank],a.shape),ordinary),
                ('zero',np.zeros_like(a),ordinary),
                ('white-constant',a,dict(sigmas=[1],black_ridges=False,mode='constant',cval=-.3)),
                ('zero-sigma',a,dict(sigmas=[0])),('default-scales',a,{}),
                ('mirror',a,dict(sigmas=[1],mode='mirror'))])
    def sigma_types(options):return [type(s).__module__+'.'+type(s).__name__ for s in options.get('sigmas',range(1,10,2))]
    specs=[dict(case=label,dtype=a.dtype.name,shape=list(a.shape),strides=list(a.strides),options=options,sigma_types=sigma_types(options)) for label,a,options in cases]
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'scikit_image':skimage.__version__,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,
        'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':7933,'command':sys.argv,
        'parity':'source-precision budget; 3D independently checked Hessians plus geometric-mean conditioning allowance (tools/sato_parity.py)',
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256':hashlib.sha256(json.dumps(specs,sort_keys=True,default=serial).encode()).hexdigest(),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,a,options in cases:
        expected=reference(a,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.sato(a,**options)
            assert type(actual) is type(expected) is np.ndarray
            assert actual.dtype==expected.dtype and actual.shape==expected.shape and actual.strides==expected.strides
            diagnostics=assert_parity(actual,expected,a,options)
            if args.check_only:
                print(f'PARITY {a.ndim}D {a.dtype} {label} workers={workers} {diagnostics}',flush=True);continue
            row={'api':'filters.sato','rank':a.ndim,'dtype':a.dtype.name,'output_dtype':actual.dtype.name,
                'case':label,'shape':list(a.shape),'options':options,'sigma_types':sigma_types(options),'threads':workers,**diagnostics,
                **measure(lambda:reference(a,**options),lambda:xx.filters.sato(a,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2,default=serial)+'\n').encode())
            print(f'{a.ndim}D {a.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
    xx.set_num_threads(1)


if __name__=='__main__':main()
