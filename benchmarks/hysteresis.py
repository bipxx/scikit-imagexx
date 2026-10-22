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
    rng=np.random.default_rng(1934);cases=[]
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
    if snapshot:report['metadata']['baseline']=snapshot
    for label,image,low,high in cases:
        expected=reference(image,low,high)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.apply_hysteresis_threshold(image,low,high)
            assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_array_equal(actual,expected)
            old=None
            if baseline:
                baseline.set_num_threads(workers);old=lambda:baseline.filters.apply_hysteresis_threshold(image,low,high)
                previous=old()
                assert type(previous) is type(expected) and previous.dtype==expected.dtype and previous.shape==expected.shape
                np.testing.assert_array_equal(previous,expected)
            row={'api':'filters.apply_hysteresis_threshold','dtype':image.dtype.name,'shape':list(image.shape),
                 'layout':label,'threshold_fields':np.ndim(low)>0,'threads':workers,'max_abs_error':0.,
                 **measure(lambda:reference(image,low,high),lambda:xx.filters.apply_hysteresis_threshold(image,low,high),args.repeat,15,old)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
