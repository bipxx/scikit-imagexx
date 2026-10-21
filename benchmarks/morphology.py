"""Morphology footprints and buffer reuse versus original scikit-image."""
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
from skimage import morphology as reference
import skimagexx as xx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/morphology-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat]+args.threads)<1: parser.error('size, repeat and workers must be positive')
    xx.set_fallback_enabled(False)
    results=[]
    footprints={'rectangle-31':np.ones((31,31),dtype=bool),
                'even-asymmetric':np.array([[1,0,1,0],[0,1,0,0]],dtype=bool),
                'disk-5':reference.disk(5)}
    rng=np.random.default_rng(23)
    for dtype in ('bool','uint8','float32','float64'):
        a=rng.integers(0,256,(args.size,args.size),dtype=np.uint8)
        if dtype=='bool': a=a>127
        else: a=a.astype(dtype)
        for name in xx.morphology.__all__:
            binary=name.startswith('binary_')
            image=(a>np.median(a)).astype(dtype) if binary and dtype!='bool' else a
            old,current=getattr(reference,name),getattr(xx.morphology,name)
            for workload,fp in footprints.items():
                options={'footprint':fp}
                # Reused output buffers are timed separately from allocating defaults.
                if workload=='even-asymmetric': options['out']=np.empty(image.shape,dtype=bool if binary else image.dtype)
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter('always')
                    expected=old(image,**options).copy()
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    with warnings.catch_warnings(record=True) as actual_warnings:
                        warnings.simplefilter('always')
                        actual=current(image,**options)
                    assert [(w.category,str(w.message)) for w in actual_warnings]==[(w.category,str(w.message)) for w in expected_warnings]
                    assert actual.dtype==expected.dtype
                    np.testing.assert_array_equal(actual,expected)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',FutureWarning)
                        timing=measure(lambda:old(image,**options),lambda:current(image,**options),args.repeat,15)
                    results.append({'api':'morphology.'+name,'dtype':dtype,'size':args.size,'threads':threads,
                                    'workload':workload,'layout':workload,'out_reused':'out' in options,
                                    'backend':'native','max_abs_error':0,**timing})
                    print(f'{name:16} {dtype:7} {workload:16} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+list((root/'python/skimagexx').glob('*.py'))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                        'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
                        'source_sha256':digest.hexdigest(),'avx2':xx.has_avx2(),'fallback_enabled':False,
                        'reference_thread_pools':1,'repeat':args.repeat,
                        'warning_policy':'FutureWarning checked before timing and ignored during timed calls'},'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Morphology performance versus scikit-image','',
           'Exact values, dtype and warning parity checked before timing; fallback disabled. '
           'Reference pools limited to one thread. Warmed calls include allocations and buffer adapters; '
           'even/asymmetric cases reuse out. Speedup = original/current; below 1 is slower.','',
           '| API | Dtype | Footprint | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["workload"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
