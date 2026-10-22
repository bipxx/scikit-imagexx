"""Allocation-inclusive adaptive equalization workloads versus scikit-image."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.exposure import equalize_adapthist as reference
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
    if min([args.size,args.repeat]+args.threads)<1:parser.error('positive size/workers/repeats required')
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
    rng=np.random.default_rng(6603);cases=[]
    for dtype in (np.uint8,np.uint16,np.float32,np.float64):
        data=rng.integers(0,65536,(args.size,args.size+1),dtype=np.uint16)
        image=(data>>8).astype(dtype) if dtype==np.uint8 else data.astype(dtype)
        if np.issubdtype(dtype,np.floating):image/=dtype(65535)
        volume=np.resize(image,(min(65,args.size),min(67,args.size),min(69,args.size)))
        rgb=np.resize(image,(*image.shape,3))
        cases.extend([('contiguous',image,{}),('reversed',image[::-1,::2],{}),
            ('fortran',np.asfortranarray(image),{}),('tiny',image[:7,:9],{}),
            ('large-kernel',image,{'kernel_size':127}),('unclipped',image,{'clip_limit':0.}),
            ('low-clip',image,{'kernel_size':31,'clip_limit':.0001}),
            ('bins64',image,{'nbins':64}),('volume',volume,{}),('rgb',rgb,{})])
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'seed':6603,'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    if snapshot:report['metadata']['baseline']=snapshot
    for label,image,options in cases:
        expected=reference(image,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.exposure.equalize_adapthist(image,**options)
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            if label=='rgb':
                tolerance=2e-6 if actual.dtype==np.float32 else 2e-13
                np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            else:np.testing.assert_array_equal(actual,expected)
            old=None
            if baseline:
                baseline.set_num_threads(workers)
                old=lambda:baseline.exposure.equalize_adapthist(image,**options)
                previous=old();assert previous.dtype==expected.dtype and previous.shape==expected.shape
                if label=='rgb':np.testing.assert_allclose(previous,expected,rtol=tolerance,atol=tolerance)
                else:np.testing.assert_array_equal(previous,expected)
            row={'api':'exposure.equalize_adapthist' ,'dtype':str(image.dtype),'layout':label,
                'shape':list(image.shape),'options':options,'threads':workers,
                'max_abs_error':float(np.max(np.abs(actual-expected))),
                'baseline_parity':'verified' if baseline else 'not-requested',
                **measure(lambda:reference(image,**options),lambda:xx.exposure.equalize_adapthist(image,**options),args.repeat,15,old)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)


if __name__=='__main__':main()
