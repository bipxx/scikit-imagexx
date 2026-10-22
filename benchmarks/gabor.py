"""Allocation-inclusive Gabor image filtering versus pinned scikit-image."""
from run import measure
from provenance import source_fingerprint
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import gabor as reference
import skimagexx as xx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=192)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true',help='validate all workloads without timing or writing a performance report')
    args=parser.parse_args()
    if min(args.threads+[args.size,args.repeat])<1:parser.error('positive size/workers/repeats required')
    if not args.check_only and args.output is None:parser.error('--output is required for measurements')
    assert skimage.__version__=='0.26.0'
    xx.set_fallback_enabled(False);rng=np.random.default_rng(7041);cases=[]
    for dtype in (np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64):
        shape=(args.size,args.size+1)
        if np.issubdtype(dtype,np.floating):image=rng.random(shape).astype(dtype)
        else:
            raw=rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
            image=(raw&1).astype(dtype) if dtype==np.bool_ else raw.astype(dtype)
        normal=dict(frequency=.2,theta=.37)
        cases.extend([('contiguous',image,normal),('strided',image[::-1,::2],normal),
            ('fortran',np.asfortranarray(image),normal),('tiny',image[:7,:9],normal),
            ('large-kernel',image,dict(frequency=.1,theta=.71)),
            ('constant',image,dict(frequency=.6,theta=.37,mode='constant',cval=-.3))])
    package=Path(xx.__file__).parent
    specs=[dict(case=label,dtype=image.dtype.name,shape=list(image.shape),strides=list(image.strides),options=options) for label,image,options in cases]
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'seed':7041,'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256':hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,image,options in cases:
        expected=reference(image,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.gabor(image,**options)
            assert type(actual) is type(expected) is tuple and len(actual)==2
            error=0.
            for a,e in zip(actual,expected):
                assert a.dtype==e.dtype and a.shape==e.shape and a.dtype.metadata==e.dtype.metadata
                if image.dtype.kind in 'biu':np.testing.assert_array_equal(a.view(np.uint8),e.view(np.uint8))
                else:
                    tolerance=2e-6 if e.dtype==np.float32 else 2e-13
                    np.testing.assert_allclose(a,e,rtol=tolerance,atol=tolerance)
                    error=max(error,float(np.max(np.abs(a-e))))
            if args.check_only:
                print(f'PARITY {image.dtype} {label} workers={workers}',flush=True)
                continue
            row={'api':'filters.gabor','dtype':image.dtype.name,'output_dtype':actual[0].dtype.name,
                'case':label,'shape':list(image.shape),'options':options,'threads':workers,
                'max_abs_error':error,'integer_byte_parity':'verified' if image.dtype.kind in 'biu' else 'not-applicable',
                **measure(lambda:reference(image,**options),lambda:xx.filters.gabor(image,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
            print(f'{image.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
    xx.set_num_threads(1)


if __name__=='__main__':main()
