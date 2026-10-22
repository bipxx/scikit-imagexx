"""Fixed sparse-stencil padding, storage and accumulation workloads."""
from run import measure
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import correlate_sparse as reference
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
    xx.set_fallback_enabled(False);rng=np.random.default_rng(7213);cases=[]
    for dtype in (np.uint8,np.float32,np.float64):
        image=rng.integers(0,256,(args.size,args.size+1),dtype=np.uint8).astype(dtype)
        if dtype!=np.uint8:image/=dtype(255)
        kernel=np.array([[0,.25,0],[1,-.125,0],[0,0,.5]])
        for mode in ('reflect','constant','nearest','mirror','wrap','valid'):cases.append((mode,image,kernel,mode))
        cases.extend([('fortran',np.asfortranarray(image),kernel,'reflect'),('strided',image[::-1,::2],kernel,'reflect')])
        wide=np.zeros((51,51));wide.flat[::257]=np.linspace(-.1,.2,len(wide.flat[::257]))
        cases.extend([('dense',image,rng.uniform(-.1,.1,(9,9)),'reflect'),('wide',image,wide,'reflect'),
            ('zero',image,np.zeros((3,3)),'reflect'),('identity',image,np.ones((1,1)),'reflect'),
            ('tiny',image[:3,:7].copy(),kernel,'reflect'),('one-dimensional',image.ravel(),np.array([0,.25,0,1,-.5]),'reflect'),
            ('three-dimensional',np.resize(image,(min(33,args.size),min(35,args.size),min(37,args.size))),np.ones((3,3,3))/27,'reflect')])
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'python':sys.version,'numpy':np.__version__,
        'scipy':scipy.__version__,'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,'seed':7213,'command':sys.argv,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,image,kernel,mode in cases:
        expected=reference(image,kernel,mode)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.correlate_sparse(image,kernel,mode)
            assert type(actual) is type(expected) and actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_array_equal(actual,expected)
            row={'api':'filters.correlate_sparse','dtype':image.dtype.name,'shape':list(image.shape),
                 'layout':label,'kernel_shape':list(kernel.shape),'mode':mode,'threads':workers,'max_abs_error':0.,
                 **measure(lambda:reference(image,kernel,mode),lambda:xx.filters.correlate_sparse(image,kernel,mode),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
            print(f'{image.dtype.name} {label} t={workers}: {row["speedup"]:.3f}x',flush=True)

if __name__=='__main__':main()
