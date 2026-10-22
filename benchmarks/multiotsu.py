"""Fixed multi-Otsu image/histogram workloads, allocation-inclusive."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import threshold_multiotsu as reference
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1:parser.error('positive size/workers/repeats required')
    xx.set_fallback_enabled(False)
    rng=np.random.default_rng(6521);cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        image=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        if dtype!=np.uint8:image/=dtype(255)
        for classes in (2,3):
            for layout,a in [('contiguous',image),('reversed',image[::-1,::2]),('tiny',image[:7,:9])]:
                cases.append((layout,a,{'classes':classes},dtype.__name__))
    for bins,classes in [(32,4),(256,2),(256,3),(1024,2),(1024,3)]:
        for kind in ('random','flat','gaps'):
            counts=rng.integers(1,1000,bins).astype(np.float32)
            if kind=='flat':counts[:]=1
            elif kind=='gaps':counts[::3]=0
            cases.append((f'histogram-{kind}-{bins}',None,{'classes':classes,'hist':counts},'float32'))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'seed':6521,'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,image,options,dtype in cases:
        expected=reference(image,**options)
        for workers in args.threads:
            xx.set_num_threads(workers)
            actual=xx.filters.threshold_multiotsu(image,**options)
            assert actual.dtype==expected.dtype
            np.testing.assert_array_equal(actual,expected)
            row={'api':'filters.threshold_multiotsu','dtype':dtype,'layout':label,
                'shape':list(image.shape) if image is not None else None,
                'classes':options['classes'],'threads':workers,'max_abs_error':0.,
                **measure(lambda:reference(image,**options),lambda:xx.filters.threshold_multiotsu(image,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{label} {dtype} classes={options["classes"]} workers={workers}: {row["speedup"]:.3f}x',flush=True)


if __name__=='__main__':main()
