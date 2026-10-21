"""Perceptual color options, extra channels, and warning-count workloads."""
from run import measure
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--output',type=Path,default=Path('benchmarks/perceptual-parameters.json'))
    args=parser.parse_args()
    if min([args.size]+args.threads)<1:
        parser.error('size and workers must be positive')
    xx.set_fallback_enabled(False)
    results=[]
    for dtype in (np.float32,np.float64):
        a=np.random.default_rng(23).random((args.size,args.size,3)).astype(dtype)
        gray=np.repeat(a[...,:1],3,axis=-1)
        hsv=a.copy(); hsv[...,0]=hsv[...,0]*dtype(40)-dtype(20)
        lab=a.copy(); lab[...,0]*=dtype(100); lab[...,1:]=lab[...,1:]*dtype(40)-dtype(20)
        extra=np.concatenate((lab,a[...,:1]),axis=-1)
        lch=extra.copy(); lch[...,1]=np.abs(lch[...,1]); lch[...,2]=a[...,2]*dtype(40)-dtype(20)
        negative=np.broadcast_to(np.array([0,0,100],dtype=dtype),a.shape)
        luv=a.copy(); luv[...,0]=luv[...,0]*dtype(80)+dtype(20); luv[...,1:]=luv[...,1:]*dtype(200)-dtype(100)
        dark=np.broadcast_to(np.array([.01,.03,.04],dtype=dtype),a.shape)
        low=luv.copy(); low[...,0]=a[...,0]*dtype(7.5)+dtype(.25); low[...,1:]=a[...,1:]*dtype(2)-dtype(1)
        alt={'illuminant':'D50','observer':'10'}
        cases=[('rgb2hsv',gray,{},'grayscale'),('hsv2rgb',hsv,{},'wrapped-hue'),
               ('xyz2lab',a,alt,'D50/10'),('lab2xyz',lab,alt,'D50/10'),
               ('rgb2lab',a,alt,'D50/10'),('lab2rgb',lab,alt,'D50/10'),
               ('lab2lch',extra,{},'four-channels'),('lch2lab',lch,{},'four-channels'),
               ('lab2xyz',negative,{},'negative-Z'),('lab2rgb',negative,{},'negative-Z'),
               ('xyz2luv',a,alt,'D50/10'),('luv2xyz',luv,alt,'D50/10'),
               ('rgb2luv',dark,{},'dark-sRGB'),('luv2rgb',low,{},'low-lightness'),
               ('luv2rgb',luv,{},'Luv-midtones')]
        for name,image,options,workload in cases:
            old,current=getattr(reference,name),getattr(xx.color,name)
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter('always')
                expected=old(image,**options)
            for threads in args.threads:
                xx.set_num_threads(threads)
                with warnings.catch_warnings(record=True) as actual_warnings:
                    warnings.simplefilter('always')
                    actual=current(image,**options)
                assert [(w.category,str(w.message)) for w in actual_warnings]==[(w.category,str(w.message)) for w in expected_warnings]
                assert actual.dtype==expected.dtype
                rtol,atol=parity_tolerance('color.'+name,actual.dtype)
                np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    timing=measure(lambda: old(image,**options),lambda: current(image,**options),7,15)
                result={'api':'color.'+name,'dtype':np.dtype(dtype).name,'size':args.size,'threads':threads,
                        'workload':workload,'options':options,'channels':image.shape[-1],'backend':'native',
                        'max_abs_error':float(np.max(np.abs(actual-expected))),**timing}
                results.append(result)
                print(f'{name:10} {np.dtype(dtype).name:7} {workload:14} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+list((root/'python/skimagexx').glob('*.py'))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                         'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
                         'source_sha256':digest.hexdigest(),'avx2':xx.has_avx2(),'fallback_enabled':False,
                         'reference_thread_pools':1,'warning_policy':'warning count/text verified before timing; UserWarning ignored during timing'},
            'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Perceptual color parameter performance','',
           'Strict native dispatch versus original scikit-image. Values, dtype and warnings are '
           'checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. '
           'UserWarning is ignored during timing, while native negative-Z counting remains included. '
           'Speedup = reference/current; slower cases remain visible.','',
           '| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["workload"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
