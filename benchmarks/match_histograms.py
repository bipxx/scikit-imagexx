"""Fixed histogram-matching workloads, with optional exact saved-package pairing."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import skimage
from skimage.exposure import match_histograms as reference
import skimagexx as xx
from provenance import source_fingerprint

def binary_hashes(package):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(package.__file__).parent.glob('_core.*')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path)
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1:parser.error('size, repeats and workers must be positive')
    xx.set_fallback_enabled(False)
    baseline=None
    if args.baseline_dir:
        sys.path.insert(0,str(args.baseline_dir.resolve()))
        import _baseline as baseline
        baseline.set_fallback_enabled(False)
        snapshot=json.loads((args.baseline_dir/'snapshot.json').read_text())
        assert binary_hashes(baseline)==snapshot['binary_sha256']
    rng=np.random.default_rng(904)
    cases=[]
    for dtype in (np.uint8,np.uint16,np.int16,np.float32,np.float64):
        if np.dtype(dtype).kind in 'ui':
            low,high=(0,256) if dtype==np.uint8 else (0,65536) if dtype==np.uint16 else (-2048,2048)
            a=rng.integers(low,high,(args.size,args.size+1,3),dtype=dtype)
            b=rng.integers(low,high,(args.size+3,args.size-1,3),dtype=dtype)
        else:
            a=rng.normal(size=(args.size,args.size+1,3)).astype(dtype)
            b=rng.normal(1,2,size=(args.size+3,args.size-1,3)).astype(dtype)
        cases.extend([('mono',a[...,0].copy(),b[...,0].copy(),None),('color',a,b,-1),
            ('planar',np.ascontiguousarray(np.moveaxis(a,-1,0)),np.ascontiguousarray(np.moveaxis(b,-1,0)),0),
            ('strided',a[:,::2,::-1],b[::-1,::2,::-1],-1)])
    for dtype in (np.float32,np.float64):
        a=rng.integers(0,256,(args.size,args.size+1,3)).astype(dtype)
        b=rng.integers(0,256,(args.size+3,args.size-1,3)).astype(dtype)
        cases.extend([('quantized-mono',a[...,0].copy(),b[...,0].copy(),None),('quantized-color',a,b,-1)])
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scikit_image':skimage.__version__,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,
        'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':904,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),'binary_sha256':binary_hashes(xx)},'results':[]}
    if baseline:report['metadata']['baseline']=snapshot
    for label,a,b,axis in cases:
        options={'channel_axis':axis}
        expected=reference(a,b,**options)
        for threads in args.threads:
            xx.set_num_threads(threads)
            actual=xx.exposure.match_histograms(a,b,**options)
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_array_equal(actual,expected)
            old=None
            if baseline:
                baseline.set_num_threads(threads)
                old=lambda:baseline.exposure.match_histograms(a,b,**options)
                np.testing.assert_array_equal(old(),expected)
            row={'api':'exposure.match_histograms','dtype':a.dtype.name,'layout':label,
                 'shape':list(a.shape),'reference_shape':list(b.shape),'channel_axis':axis,
                 'output_dtype':actual.dtype.name,'threads':threads,'max_abs_error':float(np.max(np.abs(actual-expected))),
                 **measure(lambda:reference(a,b,**options),lambda:xx.exposure.match_histograms(a,b,**options),args.repeat,15,old)}
            report['results'].append(row)
            print(f'{a.dtype.name} {label} t={threads}: {row["speedup"]:.3f}x original/current'
                  +(f'; {row["improvement"]:.3f}x baseline/current' if baseline else ''),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
