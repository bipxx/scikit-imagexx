"""Allocation-inclusive structure tensors versus pinned scikit-image."""
from run import measure
from provenance import source_fingerprint
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.feature import structure_tensor as reference
import skimagexx as xx


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
    xx.set_fallback_enabled(False);rng=np.random.default_rng(8311);cases=[]
    def make(dtype,shape):
        if np.dtype(dtype).kind=='f':return rng.normal(size=shape).astype(dtype)
        raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
        return (raw&1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)
    for dtype in (np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64):
        a=make(dtype,(256,257));ordinary=dict(sigma=1.2)
        cases.extend([('contiguous',a,ordinary),('strided',a[::-1,::2],ordinary),
            ('fortran',np.asfortranarray(a),ordinary),('zero-sigma',a,dict(sigma=0,cval=-.3)),
            ('anisotropic',a,dict(sigma=(.7,1.3),mode=('reflect','wrap'),order='xy')),
            ('tiny',a[:7,:9],dict(sigma=.7)),('broadcast',np.broadcast_to(a[:1,:1],a.shape),ordinary),
            ('3d',make(dtype,(31,33,35)),dict(sigma=(.7,1.3,0),mode='mirror')),
            ('4d',make(dtype,(7,9,11,13)),dict(sigma=.7,mode='reflect')),
            ('5d',make(dtype,(3,5,7,9,11)),dict(sigma=.7,mode='reflect'))])
    package=Path(xx.__file__).parent
    specs=[dict(case=label,dtype=a.dtype.name,shape=list(a.shape),strides=list(a.strides),options=options) for label,a,options in cases]
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'seed':8311,'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256':hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,a,options in cases:
        expected=reference(a,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.feature.structure_tensor(a,**options)
            assert type(actual) is type(expected) is list and len(actual)==len(expected)
            error=0.
            for result,target in zip(actual,expected):
                assert type(result) is type(target) is np.ndarray
                assert result.dtype==target.dtype and result.shape==target.shape and result.strides==target.strides
                tolerance=2e-6 if result.dtype==np.float32 else 2e-13
                np.testing.assert_allclose(result,target,rtol=tolerance,atol=tolerance)
                error=max(error,float(np.max(np.abs(result-target))))
            if args.check_only:
                print(f'PARITY {a.dtype} {label} workers={workers}',flush=True);continue
            row={'api':'feature.structure_tensor','dtype':a.dtype.name,'output_dtype':actual[0].dtype.name,
                'case':label,'shape':list(a.shape),'options':options,'threads':workers,'max_abs_error':error,
                **measure(lambda:reference(a,**options),lambda:xx.feature.structure_tensor(a,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
            print(f'{a.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
    xx.set_num_threads(1)


if __name__=='__main__':main()
