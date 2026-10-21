"""All standard stain matrices and custom output widths versus scikit-image."""
from run import measure
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import skimage
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/stain-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1: parser.error('size, repeat and workers must be positive')
    xx.set_fallback_enabled(False)
    families=('hed','hdx','fgx','bex','rbd','gdx','hax','bro','bpx','ahx','hpx')
    results=[]
    for dtype in ('uint8','float32','float64'):
        image=np.random.default_rng(31).integers(0,256,(args.size,args.size,3),dtype=np.uint8)
        if dtype!='uint8': image=(image/255).astype(dtype)
        for name in ('separate_stains','combine_stains'):
            matrices=[(family,getattr(reference,family+'_from_rgb' if name=='separate_stains' else 'rgb_from_'+family)) for family in families]
            matrices += [(f'custom-{k}',np.random.default_rng(k).uniform(-1,1,(3,k)).astype(np.float32)) for k in (1,7)]
            old,current=getattr(reference,name),getattr(xx.color,name)
            for workload,matrix in matrices:
                expected=old(image,matrix)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual=current(image,matrix)
                    assert actual.dtype==expected.dtype
                    rtol,atol=parity_tolerance('color.'+name,np.dtype(dtype))
                    atol*=max(1,float(np.max(np.sum(np.abs(matrix),axis=0))))
                    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
                    timing=measure(lambda:old(image,matrix),lambda:current(image,matrix),args.repeat,15)
                    results.append({'api':'color.'+name,'dtype':dtype,'output_dtype':actual.dtype.name,'size':args.size,
                                    'threads':threads,'workload':workload,'layout':workload,'channels':matrix.shape[1],
                                    'matrix':matrix.tolist(),'rtol':rtol,'atol':atol,'backend':'native',
                                    'max_abs_error':float(np.max(np.abs(actual-expected))),**timing})
                    print(f'{name:16} {dtype:7} {workload:8} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+list((root/'python/skimagexx').glob('*.py'))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                        'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
                        'source_sha256':digest.hexdigest(),'avx2':xx.has_avx2(),'fallback_enabled':False,
                        'reference_thread_pools':1,'repeat':args.repeat},'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Stain parameter performance','',
           'All 11 standard stain families and custom 3×1/3×7 float32 matrices. '
           'Values and dtype are checked against original scikit-image before timing; fallback is disabled. '
           'Input precision and matrix norm determine the error budget, including float32 logarithms '
           'promoted to float64. Seven warmed allocation-inclusive samples; reference pools use one thread. '
           'Ratios below one remain visible.','',
           '| API | Input | Output | Matrix | Channels | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["output_dtype"]} | {r["workload"]} | {r["channels"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
