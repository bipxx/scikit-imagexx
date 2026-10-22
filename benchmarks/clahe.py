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
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1:parser.error('positive size/workers/repeats required')
    xx.set_fallback_enabled(False)
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
    for label,image,options in cases:
        expected=reference(image,**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.exposure.equalize_adapthist(image,**options)
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            if label=='rgb':
                tolerance=2e-6 if actual.dtype==np.float32 else 2e-13
                np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            else:np.testing.assert_array_equal(actual,expected)
            row={'api':'exposure.equalize_adapthist','dtype':str(image.dtype),'layout':label,
                'shape':list(image.shape),'options':options,'threads':workers,
                'max_abs_error':float(np.max(np.abs(actual-expected))),
                **measure(lambda:reference(image,**options),lambda:xx.exposure.equalize_adapthist(image,**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)


if __name__=='__main__':main()
