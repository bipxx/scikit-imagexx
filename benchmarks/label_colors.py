"""Fixed public label-color workloads versus original scikit-image."""
from run import measure
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
import numpy as np
import scipy
import skimage
from skimage.color import label2rgb as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=513)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/label-color-parameters.json'))
    parser.add_argument('--suite',choices=('core','cardinality'),default='core')
    args=parser.parse_args()
    if min(args.size,args.repeat,*args.threads)<1 or 1 not in args.threads:parser.error('positive sizes/repeat/workers and worker1 required')
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
    def hashes(package):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(package.__file__).parent.iterdir()) if p.suffix in ('.pyd','.so')}
    binaries=hashes(xx);assert binaries
    xx.set_fallback_enabled(False);baseline=None;provenance=None
    if args.baseline_dir:
        directory=args.baseline_dir.resolve();provenance=json.loads((directory/'baseline-source.json').read_text())
        raw=(directory/provenance['measurement_report']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==provenance['measurement_sha256']
        measured=json.loads(raw)['metadata'];assert measured['source_sha256']==provenance['source_sha256']
        sys.path.insert(0,str(directory));import _baseline as baseline
        assert hashes(baseline)==provenance['binary_sha256']==measured['current_binary_sha256']
        assert baseline.has_avx2()==xx.has_avx2()==measured['avx2']
        baseline.set_fallback_enabled(False)
    rows=[]
    def add(labels,workload,**options):
        def call(function):return function(labels,**options)
        def checked(function):
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');value=call(function)
            return value,[(type(n.message).__name__,str(n.message)) for n in notices]
        expected,expected_notices=checked(original)
        image=options.get('image');rgb=image is not None and image.ndim==labels.ndim+1
        tolerant=rgb and options.get('kind','overlay')=='overlay'
        single=image is not None and image.dtype==np.float32
        def verify(value,notices):
            assert notices==expected_notices
            assert value.shape==expected.shape and value.dtype==expected.dtype and value.dtype.metadata==expected.dtype.metadata
            if tolerant:np.testing.assert_allclose(value,expected,rtol=2e-6 if single else 2e-13,atol=2e-6 if single else 2e-13)
            else:np.testing.assert_array_equal(value,expected)
            np.testing.assert_array_equal(np.signbit(value),np.signbit(expected))
        for workers in args.threads:
            xx.set_num_threads(workers);actual,notices=checked(xx.color.label2rgb);verify(actual,notices)
            if baseline is not None:
                baseline.set_num_threads(workers);verify(*checked(baseline.color.label2rgb))
            timing=measure(lambda:call(original),lambda:call(xx.color.label2rgb),args.repeat,15,
                           baseline=(lambda:call(baseline.color.label2rgb)) if baseline is not None else None)
            descriptor=str(labels.dtype) if image is None else str(labels.dtype)+'/'+str(image.dtype)
            rows.append({'api':'color.label2rgb','backend':'native','dtype':descriptor,'shape':list(labels.shape),
                'threads':workers,'workload':workload,'kind':options.get('kind','overlay'),
                'parameters':{k:v for k,v in options.items() if k!='image'},'output_dtype':str(actual.dtype),
                'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),**timing})
            print(f'{descriptor:18} {workload:24} workers={workers}: {timing["speedup"]:.2f}x',flush=True)
    rng=np.random.default_rng(4129);shape=(args.size,args.size+4)
    if args.suite=='cardinality':
        for dtype in (np.uint64,np.int64,np.float64):
            for groups in (31,512,513,4096,shape[0]*shape[1]):
                raw=rng.integers(0,groups,shape,dtype=np.uint64)
                labels=np.where(raw==0,0,raw*10000000019+2**50).astype(dtype)
                labels.flat[0]=0
                add(labels,'sparse-groups-'+str(groups))
    else:
        for dtype in (np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64):
            labels=rng.integers(0,32,shape,dtype=np.uint8).astype(dtype);labels.flat[0]=0
            add(labels,'labels-only');add(np.asfortranarray(labels),'fortran')
            add(labels[::-1,::-1],'reversed');add(labels[::2,::2],'stepped')
            add(np.broadcast_to(labels[:1],shape),'broadcast')
            add(labels,'custom-background',bg_label=2,bg_color=None);add(labels,'channel-first',channel_axis=0)
            if np.dtype(dtype).kind in 'iu' and np.dtype(dtype).itemsize>=2:
                width=np.dtype(dtype).itemsize;base=0 if width==2 else 2**28 if width==4 else 2**50
                wide=np.where(labels==0,0,labels.astype(np.uint64)*1024+base).astype(dtype)
                add(wide,'sparse-wide')
        labels=rng.integers(0,19,shape,dtype=np.uint8);labels.flat[0]=0
        for dtype in (np.uint8,np.float32,np.float64):
            for rgb in (False,True):
                image=rng.integers(0,256,(*shape,3) if rgb else shape,dtype=np.uint8)
                if dtype!=np.uint8:image=(image/255).astype(dtype)
                add(labels,'overlay-rgb' if rgb else 'overlay-gray',image=image,image_alpha=.7)
                add(labels,'average-rgb' if rgb else 'average-gray',image=image,kind='avg')
                if rgb:
                    for saturation in (.5,1):add(labels,'saturation-'+str(saturation),image=image,saturation=saturation,image_alpha=.7)
    assert len(rows)==(101 if args.suite=='core' else 15)*len(args.threads)
    assert len({(r['dtype'],r['workload'],r['threads']) for r in rows})==len(rows)
    assert source_fingerprint(root)==fingerprint and hashes(xx)==binaries,'source/build changed during timing'
    if baseline is not None:assert hashes(baseline)==provenance['binary_sha256']
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'cpu':platform.processor(),
        'reference':skimage.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'python':sys.version,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'seed':4129,'repeat':args.repeat,
        'suite':args.suite,'target_ms':15,'cooldown_ms':20,'current_binary_sha256':binaries,**fingerprint,
        'scope':'Warmed allocation-inclusive public calls; rank construction, copies, HSV, compositing and grouped averages included'},'results':rows}
    if provenance is not None:report['metadata']['baseline']=provenance
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Label-color performance','',
        'Original scikit-image '+skimage.__version__+'. Reference pools use one thread; native worker limits are explicit. '
        'Fallback is disabled. Values, dtype, metadata and warnings are checked before timing; RGB desaturation uses the documented HSV error budget. '
        'Every raw sample and slower workload is retained.','',
        '| Input | Workload | Workers | Original ms | Native ms | Original/native | Preserved/native |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        paired=f'{row["improvement"]:.4f}x' if 'improvement' in row else '—'
        lines.append(f'| {row["dtype"]} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.4f}x | {paired} |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
