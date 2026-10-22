"""Fixed threshold_local method, window, border and storage workloads."""
from run import measure
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import skimage
from skimage.filters import threshold_local as reference
import skimagexx as xx
from provenance import source_fingerprint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.size<3 or min(args.threads+[args.repeat])<1:parser.error('size >=3 and positive workers/repeats required')
    xx.set_fallback_enabled(False)
    rng=np.random.default_rng(1709);cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        image=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        if dtype!=np.uint8:image=image/dtype(255)
        for method in ('gaussian','mean','median'):
            for window in (3,15):cases.append(('contiguous',image,{'method':method,'block_size':window}))
            cases.extend([('fortran',np.asfortranarray(image),{'method':method}),
                          ('strided',image[::-1,::2],{'method':method}),
                          ('offset',image,{'method':method,'offset':.17}),
                          ('constant-border',image,{'method':method,'mode':'constant','cval':.31}),
                          ('tiny',image[:3,:7].copy(),{'method':method})])
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scikit_image':skimage.__version__,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'repeat':args.repeat,
        'target_ms':15,'cooldown_ms':20,'seed':1709,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(xx.__file__).parent.glob('_core.*')}},'results':[]}
    for label,image,options in cases:
        expected=reference(image,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.threshold_local(image,**options)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            tolerance=2e-6 if actual.dtype==np.float32 else 2e-13
            np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            row={'api':'filters.threshold_local','dtype':image.dtype.name,'shape':list(image.shape),
                 'layout':label,'options':options,'threads':workers,'max_abs_error':float(np.max(abs(actual-expected))),
                 **measure(lambda:reference(image,**options),lambda:xx.filters.threshold_local(image,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} {options} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
