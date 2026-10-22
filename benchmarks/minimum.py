"""Allocation-inclusive minimum thresholding against original scikit-image."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import skimage
from skimage import filters as original
import skimagexx as xx
from provenance import source_fingerprint
from run import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=513)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--output',type=Path,default=Path('benchmarks/minimum-parameters.json'))
    args=parser.parse_args()
    if min(args.size,args.repeat,*args.threads)<1:parser.error('positive dimensions, repeat and workers required')
    xx.set_fallback_enabled(False)
    rows=[]

    def add(image,workload,**options):
        def call(function):
            # Failed convergence is a real workload. Include its exception
            # construction in both timings and record it separately from values.
            try:return ('value',function(image,**options))
            except RuntimeError as error:return ('error',type(error).__name__,str(error))
        def checked(function):
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                result=call(function)
            return result,[(type(w.message).__name__,str(w.message)) for w in notices]
        expected,expected_notices=checked(original.threshold_minimum)
        for workers in args.threads:
            xx.set_num_threads(workers)
            actual,actual_notices=checked(xx.filters.threshold_minimum)
            assert actual_notices==expected_notices
            assert actual[0]==expected[0]
            if actual[0]=='value':
                assert type(actual[1]) is type(expected[1])
                assert np.asarray(actual[1]).dtype==np.asarray(expected[1]).dtype
                np.testing.assert_array_equal(actual[1],expected[1])
            else:assert actual==expected
            timing=measure(lambda:call(original.threshold_minimum),lambda:call(xx.filters.threshold_minimum),args.repeat,15)
            source=image if image is not None else options['hist'][0]
            row={'api':'filters.threshold_minimum','dtype':str(source.dtype),
                 'shape':list(source.shape),'workload':workload,'threads':workers,
                 'backend':'native','warnings':actual_notices,'outcome':actual[0],
                 'max_num_iter':options.get('max_num_iter',10000),
                 'nbins':options.get('nbins',256),'max_abs_error':0.,**timing}
            if actual[0]=='value':row['output_dtype']=str(np.asarray(actual[1]).dtype)
            else:row['exception']={'type':actual[1],'message':actual[2]}
            rows.append(row)
            print(f'{str(source.dtype):8} {workload:24} workers={workers}: {timing["speedup"]:.2f}x',flush=True)

    rng=np.random.default_rng(9371)
    for dtype in (np.uint8,np.float32,np.float64):
        image=rng.integers(0,48,(args.size,args.size+4),dtype=np.uint8)
        image[::2]+=np.uint8(200)
        if dtype!=np.uint8:image=image.astype(dtype)/dtype(255)-dtype(.3)
        add(image,'bimodal')
        add(np.asfortranarray(image),'fortran')
        add(image[::-1,::-1],'reversed')
        add(image[::2,::2],'stepped')
        add(np.broadcast_to(image[0],image.shape),'broadcast')
        add(image,'seven-bins',nbins=7)
        add(image,'513-bins',nbins=513)
        add(np.full_like(image,dtype(1)),'constant-failed')
        add(image,'iteration-limit',max_num_iter=1)
    for dtype in (np.int64,np.float32,np.float64):
        centers=np.arange(257,dtype=dtype)
        if dtype!=np.int64:centers=centers/dtype(13)-dtype(7)
        for workload in ('bimodal-histogram','random-histogram','trimmed-histogram','large-histogram'):
            counts=rng.integers(1,1000,257).astype(np.float32)
            if workload=='bimodal-histogram':counts[65:192]=0
            elif workload=='trimmed-histogram':counts[:31]=0;counts[-9:]=0
            elif workload=='large-histogram':counts*=np.float32(1e20)
            add(None,workload,hist=(counts,centers))
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'reference':skimage.__version__,
        'numpy':np.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
        'reference_thread_pools':1,'repeat':args.repeat,
        **source_fingerprint(Path(__file__).resolve().parents[1]),
        'scope':'Warmed allocation-inclusive exact public calls; unsuccessful convergence separately identified'},'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Minimum threshold performance','',
           'Original scikit-image '+skimage.__version__+'. Reference pools use one thread. '
           'Native calls run with explicit worker limits and fallback disabled. '
           'Exact output type, value, warnings and convergence exceptions are checked before timing. '
           'Every workload and raw sample is retained, including slowdowns and failed convergence.','',
           '| Dtype | Workload | Outcome | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["dtype"]} | {row["workload"]} | {row["outcome"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
