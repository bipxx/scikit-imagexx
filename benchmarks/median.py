"""Fixed median footprint, layout, casting and dimensional workloads."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import median as reference
import skimagexx as xx
from provenance import source_fingerprint

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
    rng=np.random.default_rng(5227);cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        image=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        if dtype!=np.uint8:image/=dtype(255)
        for mode in ('nearest','reflect','mirror','wrap','constant'):cases.append((mode,image,np.ones((3,3),dtype=bool),mode))
        cross=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
        holey=np.zeros((7,7),dtype=bool);holey[::3,::3]=True
        volume=np.resize(image,(min(33,args.size),min(35,args.size),min(37,args.size)))
        cases.extend([('fortran',np.asfortranarray(image),np.ones((3,3),dtype=bool),'nearest'),
            ('strided',image[::-1,::2],np.ones((3,3),dtype=bool),'nearest'),
            ('dense5',image,np.ones((5,5),dtype=bool),'nearest'),
            ('sparse7',image,holey,'nearest'),('cross',image,cross,'nearest'),
            ('even',image,np.ones((2,2),dtype=bool),'nearest'),('identity',image,np.ones((1,1),dtype=bool),'nearest'),
            ('tiny',image[:3,:7].copy(),np.ones((3,3),dtype=bool),'nearest'),
            ('one-dimensional3',image.ravel(),np.ones(3,dtype=bool),'nearest'),
            ('one-dimensional31',image.ravel(),np.ones(31,dtype=bool),'nearest'),
            ('three-dimensional',volume,np.ones((3,3,3),dtype=bool),'nearest')])
        if dtype!=np.uint8:
            zeros=np.random.default_rng(5228).choice(np.array([-1.,-0.,0.,1.],dtype=dtype),image.size)
            for width in (3,31):cases.append((f'signed-zero{width}',zeros,np.ones(width,dtype=bool),'nearest'))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'python':sys.version,'numpy':np.__version__,
        'scipy':scipy.__version__,'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':5227,'signed_zero_seed':5228,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    if snapshot:report['metadata']['baseline']=snapshot
    for label,image,kernel,mode in cases:
        expected=reference(image,kernel,mode=mode)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.median(image,kernel,mode=mode)
            assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_array_equal(actual,expected)
            if actual.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
            old=None
            baseline_parity='not-requested'
            baseline_signbit_mismatches=None
            if baseline:
                baseline.set_num_threads(workers);old=lambda:baseline.filters.median(image,kernel,mode=mode)
                previous=old()
                assert type(previous) is type(expected) and previous.dtype==expected.dtype and previous.shape==expected.shape
                np.testing.assert_array_equal(previous,expected)
                baseline_parity='verified'
                baseline_signbit_mismatches=int(np.count_nonzero(np.signbit(previous)!=np.signbit(expected))) if previous.dtype.kind=='f' else 0
                if baseline_signbit_mismatches:
                    # Preserve evidence of older tie-order failures, without
                    # timing an incorrect baseline as an equivalent algorithm.
                    baseline_parity='signed-zero-mismatch-unmeasured';old=None
            row={'api':'filters.median','dtype':image.dtype.name,'shape':list(image.shape),
                 'layout':label,'kernel_shape':list(kernel.shape),'mode':mode,'threads':workers,'max_abs_error':0.,'baseline_parity':baseline_parity,'baseline_signbit_mismatches':baseline_signbit_mismatches,
                 **measure(lambda:reference(image,kernel,mode=mode),lambda:xx.filters.median(image,kernel,mode=mode),args.repeat,15,old)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
