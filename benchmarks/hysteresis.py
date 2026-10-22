"""Fixed hysteresis graph, threshold-field and storage workloads."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import apply_hysteresis_threshold as reference
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
    xx.set_fallback_enabled(False);rng=np.random.default_rng(1934);cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        a=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        scale=1 if dtype==np.uint8 else 1/255
        if dtype!=np.uint8:a*=dtype(scale)
        low,high=63*scale,191*scale
        rows=np.linspace(35*scale,90*scale,a.shape[0]).astype(dtype)[:,None]
        columns=np.linspace(160*scale,220*scale,a.shape[1]).astype(dtype)[None,:]
        snake=np.zeros_like(a);snake[::2]=dtype(127*scale);snake[1::4,-1]=dtype(127*scale);snake[3::4,0]=dtype(127*scale);snake[0,0]=dtype(255*scale)
        checker=np.zeros_like(a);checker[::2,::2]=dtype(127*scale);checker[::8,::8]=dtype(255*scale)
        for label,image,lo,hi in [('random',a,low,high),('fortran',np.asfortranarray(a),low,high),
            ('strided',a[::-1,::2],low,high),('all-weak',a,-1,high),('no-seeds',a,low,300*scale),
            ('equal',a,high,high),('threshold-fields',a,rows,columns),('snake',snake,low,high),
            ('checkerboard',checker,low,high),('tiny',a[:3,:7].copy(),low,high)]:cases.append((label,image,lo,hi))
        if dtype!=np.uint8:cases.append(('nan-background',a,np.nan,high))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'python':sys.version,'numpy':np.__version__,
        'scipy':scipy.__version__,'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':1934,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,image,low,high in cases:
        expected=reference(image,low,high)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.apply_hysteresis_threshold(image,low,high)
            assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_array_equal(actual,expected)
            row={'api':'filters.apply_hysteresis_threshold','dtype':image.dtype.name,'shape':list(image.shape),
                 'layout':label,'threshold_fields':np.ndim(low)>0,'threads':workers,'max_abs_error':0.,
                 **measure(lambda:reference(image,low,high),lambda:xx.filters.apply_hysteresis_threshold(image,low,high),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
