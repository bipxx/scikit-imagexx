"""Allocation-inclusive symmetric image eigenvalues against scikit-image 0.26.0."""
from run import measure
from provenance import source_fingerprint
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage import feature as reference
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
    xx.set_fallback_enabled(False);rng=np.random.default_rng(9171);cases=[]
    shapes={1:(65536,),2:(256,257),3:(31,33,35),4:(11,13,15,17),5:(7,9,9,9,11)}
    for dtype in (np.float32,np.float64):
        for n,shape in shapes.items():
            components=[rng.normal(size=shape).astype(dtype) for _ in range(n*(n+1)//2)]
            layouts=[('contiguous',components),('strided',[a[(slice(None,None,-2),)*n] for a in components]),
                ('fortran',[np.asfortranarray(a) for a in components]),
                ('tiny',[a[(slice(0,1),)*n] for a in components]),
                ('broadcast',[np.broadcast_to(a[(slice(0,1),)*n],shape) for a in components])]
            diagonal=[]
            for r in range(n):
                for c in range(r,n):
                    diagonal.append(np.full(shape,float(n-r) if r==c else 0.,dtype=dtype))
            layouts.append(('diagonal',diagonal))
            for label,elements in layouts:cases.append((label,n,elements))
    package=Path(xx.__file__).parent
    specs=[dict(case=label,dimension=n,dtype=fields[0].dtype.name,shape=list(fields[0].shape),strides=[list(a.strides) for a in fields]) for label,n,fields in cases]
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'seed':9171,'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256':hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for api in ('hessian_matrix_eigvals','structure_tensor_eigenvalues'):
        original=getattr(reference,api);native=getattr(xx.feature,api)
        for label,n,elements in cases:
            expected=original(elements)
            for workers in args.threads:
                xx.set_num_threads(workers);actual=native(elements)
                assert type(actual) is type(expected) is np.ndarray
                assert actual.dtype==expected.dtype and actual.shape==expected.shape and actual.strides==expected.strides
                tolerance=2e-6 if actual.dtype==np.float32 else 2e-13
                if n<=2:np.testing.assert_array_equal(actual,expected)
                else:np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
                if args.check_only:
                    print(f'PARITY {api} {elements[0].dtype} n={n} {label} workers={workers}',flush=True);continue
                row={'api':'feature.'+api,'dtype':actual.dtype.name,'case':label,'dimension':n,
                    'shape':list(elements[0].shape),'threads':workers,'max_abs_error':float(np.max(np.abs(actual-expected))),
                    **measure(lambda:original(elements),lambda:native(elements),args.repeat,15)}
                report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
                print(f'{api} {actual.dtype} n={n} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
    xx.set_num_threads(1)


if __name__=='__main__':main()
