"""Allocation-inclusive Gabor kernel corpus versus pinned scikit-image."""
from run import measure
from provenance import source_fingerprint
import argparse,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage.filters import gabor_kernel as reference
import skimagexx as xx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.threads+[args.repeat])<1:parser.error('positive workers and repeats required')
    assert skimage.__version__=='0.26.0'
    xx.set_fallback_enabled(False)
    cases=[]
    for dtype in (np.complex64,np.complex128):
        for frequency in (.1,.2,.5,1.):
            for theta in (0.,.37,float(np.pi/2)):
                cases.append((f'f{frequency}-theta{theta}',dict(frequency=frequency,theta=theta,dtype=dtype)))
        for label,options in [('anisotropic',dict(frequency=.2,theta=.71,sigma_x=2.,sigma_y=11.)),
                              ('phase',dict(frequency=.2,theta=.53,offset=1.3)),
                              ('tiny',dict(frequency=.2,sigma_x=.1,sigma_y=.1)),
                              ('broad',dict(frequency=.2,bandwidth=.2)),
                              ('negative-frequency',dict(frequency=-.3,theta=.23)),
                              ('large',dict(frequency=.2,sigma_x=90.,sigma_y=100.,theta=.3))]:
            cases.append((label,dict(options,dtype=dtype)))
    package=Path(xx.__file__).parent
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_image':skimage.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,'target_ms':15,'cooldown_ms':20,
        'command':sys.argv,**source_fingerprint(Path(__file__).resolve().parents[1]),
        'case_spec_sha256':hashlib.sha256(json.dumps([(label,{**options,'dtype':np.dtype(options['dtype']).name}) for label,options in cases],sort_keys=True).encode()).hexdigest(),
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}},'results':[]}
    for label,options in cases:
        expected=reference(**options)
        for workers in args.threads:
            xx.set_num_threads(workers);actual=xx.filters.gabor_kernel(**options)
            tolerance=3e-7 if options['dtype']==np.complex64 else 3e-14
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            error=float(np.max(np.abs(actual-expected)))
            scale=float(np.max(np.abs(expected)))
            np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance*scale)
            row={'api':'filters.gabor_kernel','dtype':actual.dtype.name,'case':label,
                'shape':list(actual.shape),'options':{**options,'dtype':actual.dtype.name},
                'threads':workers,'max_abs_error':error,'maximum_envelope':scale,
                **measure(lambda:reference(**options),lambda:xx.filters.gabor_kernel(**options),args.repeat,15)}
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
            print(f'{actual.dtype} {label} workers={workers}: {row["speedup"]:.3f}x',flush=True)
    xx.set_num_threads(1)


if __name__=='__main__':main()
