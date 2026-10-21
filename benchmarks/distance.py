"""Raw Lab distances and composed color routes versus original scikit-image."""
from run import measure
from color_parity import check_routed_hsv
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import skimage
from skimage import color as reference
import skimagexx as xx

METRICS=('deltaE_cie76','deltaE_ciede94','deltaE_ciede2000','deltaE_cmc')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/color-distance.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1: parser.error('size, repeats and workers must be positive')
    xx.set_fallback_enabled(False)
    cases=[]
    rng=np.random.default_rng(106)
    base=rng.integers(0,256,(args.size,2*args.size,3),dtype=np.uint8)
    for dtype in (np.uint8,np.float32,np.float64):
        a=base.astype(dtype)
        if dtype!=np.uint8:
            a[...,0]*=dtype(100/255)
            a[...,1:]=a[...,1:]*dtype(200/255)-dtype(100)
        image=a[:,:args.size].copy()
        other=image[::-1,::-1].copy()
        planar=np.ascontiguousarray(np.moveaxis(image,-1,0))
        extra=np.concatenate((image,image[...,:1]),axis=-1)
        for label,first,second,axis in [('contiguous',image,other,-1),
                ('channel-first',planar,np.ascontiguousarray(np.moveaxis(other,-1,0)),0),
                ('strided',a[:,::2],other[:,::-1],-1),('broadcast',image,other[0:1,0:1],-1),
                ('extra-channels',extra,other,-1)]:
            for metric in METRICS:
                cases.append((metric,(first,second),{'channel_axis':axis},label,'native'))
        rgb=base[:,:args.size].copy()
        if dtype!=np.uint8: rgb=(rgb/255).astype(dtype)
        for source,target in [('RGB','HSV'),('HSV','XYZ'),('XYZ','HSV'),('YUV','YCbCr'),('YCbCr','YUV'),('RGB CIE','RGB')]:
            image=reference.convert_colorspace(rgb,'RGB',source)
            cases.append(('convert_colorspace',(image,source,target),{},source+'->'+target,'native'))
        cases.append(('convert_colorspace',(rgb,'RGB','RGB'),{},'RGB-identity','identity'))
    for first,second in [(np.uint16,np.float32),(np.float32,np.float64),(np.int64,np.float32),(np.bool_,np.float32)]:
        for metric in METRICS:
            cases.append((metric,(base[:,:args.size].astype(first),base[:,:args.size,::-1].astype(second)),{},'mixed-dtypes','native'))
    for dtype in (np.float32,np.float64):
        first=(rng.random((args.size,args.size,3))*100).astype(dtype)
        second=first[::-1,::-1].copy()
        for metric in METRICS[1:]:
            weights=[({'kL':2},'lightness-2'),({'kC':.5},'chroma-half')]
            if metric=='deltaE_ciede94':
                weights += [({'kL':2,'k1':.048,'k2':.014},'textile'),({'kH':2},'hue-2')]
            elif metric=='deltaE_ciede2000':weights += [({'kH':2},'hue-2')]
            for options,label in weights:
                cases.append((metric,(first,second),options,label,'native'))
            cases.append((metric,(first[0,0],second[0,0]),{},'scalar-pair','native'))
    for dtype in (np.float32,np.float64):
        image=(rng.random((args.size,args.size,3))*200-100).astype(dtype)
        cases.append(('convert_colorspace',(image,'YUV','HSV'),{},'out-of-gamut-YUV->HSV','native'))
    results=[]
    for name,call,options,workload,backend in cases:
        old,new=getattr(reference,name),getattr(xx.color,name)
        expected=old(*call,**options)
        for threads in args.threads:
            xx.set_num_threads(threads)
            actual=new(*call,**options)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            if backend=='identity': assert actual is call[0]
            rtol=atol=2e-6 if actual.dtype==np.float32 else 2e-13
            parity={'parity_contract':'base dtype tolerance','rtol':rtol,'atol':atol}
            if name=='convert_colorspace' and call[1]!='RGB' and call[2]=='HSV':
                parity.update(check_routed_hsv(call[0],call[1],actual,expected))
            else: np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
            timing=measure(lambda:old(*call,**options),lambda:new(*call,**options),args.repeat,15)
            dtype=call[0].dtype.name
            second_dtype=call[1].dtype.name if name in METRICS else None
            if second_dtype and second_dtype!=dtype: dtype+='/'+second_dtype
            results.append({'api':'color.'+name,'dtype':dtype,'second_dtype':second_dtype,'size':1 if workload=='scalar-pair' else args.size,
                'shape':list(call[0].shape),'second_shape':list(call[1].shape) if name in METRICS else None,
                'threads':threads,'workload':workload,'layout':workload,'options':options,'backend':backend,
                'max_abs_error':float(np.max(np.abs(actual-expected))),**parity,**timing})
            print(f'{name:18} {dtype:18} {workload:24} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+list((root/'python/skimagexx').glob('*.py'))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,'source_sha256':digest.hexdigest(),
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'command':sys.argv},'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Color distance and route performance','',
        'Native dispatch versus original scikit-image, with parity checked before warmed, allocation-inclusive timing. '
        'Reference pools use one thread. Raw Lab integer values are not intensity-normalized. '
        'All four Lab metrics include mixed dtypes, broadcast/strided/planar inputs, extra channels, scalar pairs '
        'and non-default weights. RGB intermediate allocations are included in composed routes. '
        'Ill-conditioned HSV routes check each stage and a propagated ratio error bound. '
        'RGB identity rows perform no image processing and are excluded from processing speedup counts.','',
        '| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["workload"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
