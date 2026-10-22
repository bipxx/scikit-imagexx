"""Fixed Li threshold workloads; all timings include Python API preparation."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import skimage
from skimage.filters import threshold_li as reference
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path)
    args=parser.parse_args()
    if args.size<2 or min(args.threads+[args.repeat])<1:parser.error('size >=2 and positive workers/repeats required')
    xx.set_fallback_enabled(False)
    baseline=None;snapshot=None
    if args.baseline_dir:
        sys.path.insert(0,str(args.baseline_dir.resolve()))
        import _baseline as baseline
        snapshot=json.loads((args.baseline_dir/'snapshot.json').read_text())
        package=Path(baseline.__file__).parent
        for group,pattern in [('binary_sha256','_core.*'),('python_sha256','*.py')]:
            assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob(pattern)}==snapshot[group]
        assert baseline.has_avx2()==xx.has_avx2()
        baseline.set_fallback_enabled(False)
    rng=np.random.default_rng(817)
    cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        a=rng.integers(0,256,(args.size,args.size+1)).astype(dtype)
        for label,image in [('quantized',a),('fortran',np.asfortranarray(a)),
                            ('strided',a[::-1,::2]),('constant',np.full_like(a,73)),
                            ('tiny',a[:3,:7].copy())]:
            cases.append((label,image,{}))
        cases.append(('explicit-tolerance',a,{'tolerance':.25}))
        cases.append(('numeric-guess',a,{'initial_guess':100.}))
        if dtype!=np.uint8:
            continuous=rng.normal(-3,7,a.shape).astype(dtype)
            # Some reference float32 inputs cycle at the default tolerance.
            # Positive fixed tolerance keeps timed work bounded; parity tests
            # separately compare exact bounded traces of reference cycles.
            cases.append(('continuous',continuous,{'tolerance':.001}))
            missing=a.copy();missing.flat[::31]=np.nan;missing.flat[1::67]=np.inf
            cases.append(('nonfinite',missing,{}))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scikit_image':skimage.__version__,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,
        'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':817,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    if snapshot:report['metadata']['baseline']=snapshot
    for label,image,options in cases:
        expected=reference(image,**options)
        for threads in args.threads:
            xx.set_num_threads(threads)
            actual=xx.filters.threshold_li(image,**options)
            assert type(actual)==type(expected)
            np.testing.assert_allclose(actual,expected,rtol=2e-6 if image.dtype==np.float32 else 2e-13,atol=2e-6 if image.dtype==np.float32 else 2e-13)
            old=None
            if baseline:
                baseline.set_num_threads(threads)
                old=lambda:baseline.filters.threshold_li(image,**options)
                before=old()
                assert type(before)==type(expected)
                np.testing.assert_allclose(before,expected,rtol=2e-6 if image.dtype==np.float32 else 2e-13,atol=2e-6 if image.dtype==np.float32 else 2e-13)
            row={'api':'filters.threshold_li','dtype':image.dtype.name,'layout':label,
                 'shape':list(image.shape),'options':options,'threads':threads,
                 'max_abs_error':float(abs(actual-expected)),
                 **measure(lambda:reference(image,**options),lambda:xx.filters.threshold_li(image,**options),args.repeat,15,old)}
            report['results'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(f'{image.dtype.name} {label} t={threads}: {row["speedup"]:.3f}x original/current',flush=True)


if __name__=='__main__':main()
