"""Allocation-inclusive minimum thresholding against original scikit-image."""
import argparse
from run import measure
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import scipy
import skimage
from skimage import filters as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=513)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--baseline-dir',type=Path,help='Exact measured initial _baseline package')
    parser.add_argument('--output',type=Path,default=Path('benchmarks/minimum-parameters.json'))
    args=parser.parse_args()
    if min(args.size,args.repeat,*args.threads)<1 or 1 not in args.threads:
        parser.error('positive dimensions/repeat/workers and a one-worker comparison required')
    root=Path(__file__).resolve().parents[1]
    fingerprint=source_fingerprint(root)
    def binary_hashes(package):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(package.__file__).parent.iterdir()) if p.suffix in ('.pyd','.so')}
    current_binaries=binary_hashes(xx)
    assert current_binaries, 'native binary required'
    xx.set_fallback_enabled(False)
    baseline=None;baseline_provenance=None
    if args.baseline_dir:
        directory=args.baseline_dir.resolve()
        baseline_provenance=json.loads((directory/'baseline-source.json').read_text(encoding='utf-8'))
        measurement=directory/baseline_provenance['measurement_report']
        assert hashlib.sha256(measurement.read_bytes()).hexdigest()==baseline_provenance['measurement_sha256']
        measured=json.loads(measurement.read_text(encoding='utf-8'))['metadata']
        assert measured['source_sha256']==baseline_provenance['source_sha256']
        sys.path.insert(0,str(directory))
        import _baseline as baseline
        assert binary_hashes(baseline)==baseline_provenance['binary_sha256']==measured['current_binary_sha256']
        assert baseline.has_avx2()==xx.has_avx2()==measured['avx2']
        baseline.set_fallback_enabled(False)
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
        def verify(result,notices):
            assert notices==expected_notices
            assert result[0]==expected[0]
            if result[0]=='value':
                assert type(result[1]) is type(expected[1])
                assert np.asarray(result[1]).dtype==np.asarray(expected[1]).dtype
                np.testing.assert_array_equal(result[1],expected[1])
            else:assert result==expected
        for workers in args.threads:
            xx.set_num_threads(workers)
            actual,actual_notices=checked(xx.filters.threshold_minimum)
            verify(actual,actual_notices)
            if baseline is not None:
                baseline.set_num_threads(workers)
                verify(*checked(baseline.filters.threshold_minimum))
            timing=measure(lambda:call(original.threshold_minimum),lambda:call(xx.filters.threshold_minimum),args.repeat,15,
                           baseline=(lambda:call(baseline.filters.threshold_minimum)) if baseline is not None else None)
            source=image if image is not None else options['hist'][0]
            descriptor=str(source.dtype) if image is not None else str(source.dtype)+'/'+str(options['hist'][1].dtype)
            row={'api':'filters.threshold_minimum','dtype':descriptor,
                 'shape':list(source.shape),'workload':workload,'threads':workers,
                 'layout':workload if workload in ('fortran','reversed','stepped','broadcast') else 'contiguous' if image is not None else 'histogram-only',
                 'backend':'native','warnings':actual_notices,'outcome':actual[0],
                 'max_num_iter':options.get('max_num_iter',10000),
                 'nbins':options.get('nbins',256),'max_abs_error':0.,**timing}
            if actual[0]=='value':
                row['output_dtype']=str(np.asarray(actual[1]).dtype)
                row['threshold']=np.asarray(actual[1]).item()
            else:row['exception']={'type':actual[1],'message':actual[2]}
            rows.append(row)
            print(f'{descriptor:16} {workload:24} workers={workers}: {timing["speedup"]:.2f}x',flush=True)

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
    assert len(rows)==39*len(args.threads)
    assert len({(r['dtype'],r['workload'],r['threads']) for r in rows})==len(rows), 'duplicate workload key'
    assert source_fingerprint(root)==fingerprint, 'source changed during timing'
    assert binary_hashes(xx)==current_binaries, 'native binary changed during timing'
    if baseline is not None:
        assert binary_hashes(baseline)==baseline_provenance['binary_sha256'], 'baseline changed during timing'
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'cpu':platform.processor(),'reference':skimage.__version__,
        'numpy':np.__version__,'scipy':scipy.__version__,'python':sys.version,
        'avx2':xx.has_avx2(),'fallback_enabled':False,
        'current_binary_sha256':current_binaries,
        'reference_thread_pools':1,'repeat':args.repeat,
        'seed':9371,'target_ms':15,'cooldown_ms':20,**fingerprint,
        'scope':'Warmed allocation-inclusive exact public calls; unsuccessful convergence separately identified'},'results':rows}
    if baseline_provenance is not None:report['metadata']['baseline']=baseline_provenance
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Minimum threshold performance','',
           'Original scikit-image '+skimage.__version__+'. Reference pools use one thread. '
           'Native calls run with explicit worker limits and fallback disabled. '
           'Exact output type, value, warnings and convergence exceptions are checked before timing. '
           'Every workload and raw sample is retained, including slowdowns and failed convergence.','',
           '| Dtype | Workload | Outcome | Workers | Original ms | C++ ms | Speedup | Preserved/current |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        paired=f'{row["improvement"]:.4f}x' if 'improvement' in row else '—'
        lines.append(f'| {row["dtype"]} | {row["workload"]} | {row["outcome"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x | {paired} |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
