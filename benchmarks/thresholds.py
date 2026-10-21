"""Histogram threshold selectors versus original scikit-image, strict native calls."""
from run import measure
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
from skimage import filters as reference
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/threshold-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat,*args.threads])<1:parser.error('positive size, repeat and workers required')
    xx.set_fallback_enabled(False)
    rows=[]
    names=('threshold_yen','threshold_triangle','threshold_isodata')

    def add(name,image,workload,options=None,layout='contiguous'):
        options={} if options is None else options
        native=getattr(xx.filters,name);original=getattr(reference,name)
        def call(function):return function(image,**options)
        def checked(function):
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                result=call(function)
            return result,[(type(w.message).__name__,str(w.message)) for w in notices]
        expected,reference_warnings=checked(original)
        for workers in args.threads:
            xx.set_num_threads(workers)
            actual,native_warnings=checked(native)
            assert native_warnings==reference_warnings
            assert np.asarray(actual).dtype==np.asarray(expected).dtype
            np.testing.assert_array_equal(actual,expected)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                timing=measure(lambda:call(original),lambda:call(native),args.repeat,15)
            source=image if image is not None else options['hist'][0]
            rows.append({'api':'filters.'+name,'size':args.size if image is not None else int(source.size),
                         'shape':list(source.shape),'dtype':str(source.dtype) if image is not None else str(source.dtype)+'/'+str(options['hist'][1].dtype),'threads':workers,
                         'backend':'native','workload':workload,'layout':layout,
                         'nbins':options.get('nbins',256),'return_all':options.get('return_all',False),
                         'output_dtype':str(np.asarray(actual).dtype),'warnings':reference_warnings,
                         'max_abs_error':0.,**timing})
            print(f'{name:19} {str(source.dtype):7} {workload:20} t={workers}: {timing["speedup"]:.2f}x',flush=True)

    rng=np.random.default_rng(58)
    for dtype in (np.uint8,np.float32,np.float64):
        raw=rng.integers(0,256,(args.size,args.size),dtype=np.uint8)
        image=raw if dtype==np.uint8 else (raw/255).astype(dtype)
        large=np.tile(image,(2,2))
        layouts={'contiguous':image,'reversed':image[::-1,::-1],
                 'strided':large[::2,::2],'volume':image.reshape(1,args.size,args.size)}
        for layout,a in layouts.items():
            for name in names:add(name,a,layout,layout=layout)
        for bins in (32,513):
            if dtype!=np.uint8:
                for name in names:add(name,image,'bins-'+str(bins),{'nbins':bins})
        add('threshold_isodata',image,'all-fixed-points',{'return_all':True})
        constant=np.full_like(image,13)
        # Constant floating ISODATA has no valid threshold: its IndexError and
        # warnings are covered by parity tests, rather than timed as processing.
        for name in names:
            if dtype==np.uint8 or name!='threshold_isodata':add(name,constant,'constant')

    for dtype in (np.int64,np.float32,np.float64):
        centers=np.arange(257,dtype=dtype)
        if dtype!=np.int64:centers=centers/dtype(13)-dtype(7)
        for workload in ('supplied-histogram','trimmed-histogram','large-counts'):
            counts=rng.integers(1,1000,257).astype(np.float32)
            if workload=='trimmed-histogram':counts[:31]=0;counts[-9:]=0
            if workload=='large-counts':counts*=np.float32(1e6)
            for name in ('threshold_yen','threshold_isodata'):
                add(name,None,workload,{'hist':(counts,centers)},layout='histogram-only')

    root=Path(__file__).resolve().parents[1]
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'reference':skimage.__version__,
        'numpy':np.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,**source_fingerprint(root),
        'scope':'Warmed allocation-inclusive Python API; exact thresholds, dtype and warnings; histogram metadata calls included'},'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Histogram threshold performance','',
        'Reference: original scikit-image '+skimage.__version__+'. Reference pools use one thread; native workers are explicit. '
        'Exact thresholds, output dtype and warnings are checked before timing with fallback disabled. '
        'Calls include dispatch, histogram construction, metadata and input copying. '
        'Histogram-only rows measure selectors without scanning an image. '
        'Scores below 1 indicate slower native calls; all cases and raw samples are retained.','',
        '| API | Input shape | Dtype | Workload | Workers | Original ms | C++ ms | Speedup |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape'])) or 'scalar'
        lines.append(f'| {row["api"]} | {shape} | {row["dtype"]} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
