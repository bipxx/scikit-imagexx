"""Fixed rank-order workloads, including reconstruction and exact bit parity."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import rank_order as reference
import skimagexx as xx
from provenance import source_fingerprint

def check(actual,expected,image):
    assert type(actual) is type(expected) is tuple and len(actual)==2
    for a,e in zip(actual,expected):
        assert a.dtype==e.dtype and a.shape==e.shape and a.tobytes()==e.tobytes()
    np.testing.assert_array_equal(actual[1][actual[0]],image)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path)
    args=parser.parse_args()
    if args.size<3 or min(args.threads+[args.repeat])<1:parser.error('size >=3 and positive workers/repeats required')
    xx.set_fallback_enabled(False);baseline=None;snapshot=None
    if args.baseline_dir:
        sys.path.insert(0,str(args.baseline_dir.resolve()))
        import _baseline as baseline
        snapshot=json.loads((args.baseline_dir/'snapshot.json').read_bytes())
        package=Path(baseline.__file__).parent
        for group,pattern in [('binary_sha256','_core.*'),('python_sha256','*.py')]:
            assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob(pattern)}==snapshot[group]
        assert baseline.has_avx2()==xx.has_avx2()
        baseline.set_fallback_enabled(False)
    rng=np.random.default_rng(3019);cases=[]
    for dtype in (np.uint8,np.uint64,np.float16,np.float32,np.float64):
        image=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        if np.dtype(dtype).kind=='f':image=image/dtype(255)
        wide=(rng.integers(0,np.iinfo(dtype).max,image.shape,dtype=dtype,endpoint=True) if np.dtype(dtype).kind=='u'
              else rng.normal(-2,7,image.shape).astype(dtype))
        # A monotone workload with bounded representable values for every dtype.
        monotone=np.broadcast_to(np.linspace(0,127,args.size+1).astype(dtype),image.shape).copy()
        for label,a in [('quantized',image),('fortran',np.asfortranarray(image)),('strided',image[::-1,::2]),
                        ('constant',np.full_like(image,7)),('tiny',image[:3,:7].copy()),('monotone',monotone),('wide-range',wide)]:
            cases.append((label,a))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'python':sys.version,'numpy':np.__version__,
        'scipy':scipy.__version__,'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':3019,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    if snapshot:report['metadata']['baseline']=snapshot
    for label,image in cases:
        expected=reference(image)
        for workers in args.threads:
            xx.set_num_threads(workers);check(xx.filters.rank_order(image),expected,image)
            old=None
            if baseline:
                baseline.set_num_threads(workers);old=lambda:baseline.filters.rank_order(image)
                check(old(),expected,image)
            row={'api':'filters.rank_order','dtype':image.dtype.name,'shape':list(image.shape),'layout':label,
                 'threads':workers,'max_abs_error':0.,**measure(lambda:reference(image),lambda:xx.filters.rank_order(image),args.repeat,15,old)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
