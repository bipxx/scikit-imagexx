"""Fixed public boolean and uint16-to-int16 conversion workloads."""
from run import measure
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,sys,warnings
from pathlib import Path
import numpy as np
import scipy,skimage
from skimage import util as reference
import skimagexx as xx
from provenance import source_fingerprint


def hashes(package):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(package.__file__).parent.iterdir()) if p.suffix in ('.pyd','.so')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=1024)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/conversion-packets.json'))
    args=parser.parse_args()
    if min(args.size,args.repeat,*args.threads)<1 or 1 not in args.threads:parser.error('positive values and worker1 required')
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root);binary=hashes(xx)
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
    rng=np.random.default_rng(817);shape=(args.size,args.size+3);rows=[]
    def add(name,image,workload):
        def call(function):return function(image)
        def checked(function):
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');result=call(function)
            return result,[(n.category.__name__,str(n.message)) for n in notices]
        expected,expected_notices=checked(getattr(reference,name))
        def verify(result,notices):
            assert notices==expected_notices
            assert result.dtype==expected.dtype and result.dtype.metadata==expected.dtype.metadata
            assert result.shape==expected.shape and type(result)==type(expected)
            assert (result is image)==(expected is image)
            assert np.shares_memory(result,image)==np.shares_memory(expected,image)
            np.testing.assert_array_equal(result,expected)
        saved=image.copy()
        for workers in args.threads:
            xx.set_num_threads(workers);result,notices=checked(getattr(xx.util,name));verify(result,notices)
            if baseline is not None:
                baseline.set_num_threads(workers);verify(*checked(getattr(baseline.util,name)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                timing=measure(lambda:call(getattr(reference,name)),lambda:call(getattr(xx.util,name)),args.repeat,15,
                    baseline=(lambda:call(getattr(baseline.util,name))) if baseline is not None else None)
            np.testing.assert_array_equal(image,saved)
            kind='identity' if result is image else 'native'
            rows.append(dict(api='util.'+name,dtype=str(image.dtype),shape=list(image.shape),workload=workload,threads=workers,
                backend=kind,output_dtype=str(result.dtype),warnings=notices,**timing))
            print(f'{name:12} {str(image.dtype):7} {workload:14} t={workers}: {timing["speedup"]:.3f}x',flush=True)
    def layouts(a):
        return (('contiguous',a),('fortran',np.asfortranarray(a)),('reversed',a[::-1,::-1]),('stepped',a[::2,::2]))
    for dtype in (np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64):
        descriptor=np.dtype(dtype)
        if descriptor.kind=='f':image=rng.random(shape).astype(dtype)
        elif descriptor.kind=='b':image=rng.integers(0,2,shape,dtype=np.uint8).astype(dtype)
        else:image=rng.integers(0,256,shape[0]*shape[1]*descriptor.itemsize,dtype=np.uint8).view(dtype).reshape(shape)
        for layout,array in layouts(image):add('img_as_bool',array,layout)
    for maximum,family in ((65535,'full'),(32766,'narrow')):
        image=rng.integers(0,maximum+1,shape,dtype=np.uint16)
        image.flat[-1]=maximum
        for layout,array in layouts(image):add('img_as_int',array,family+'-'+layout)
    assert len(rows)==52*len(args.threads)
    assert source_fingerprint(root)==fingerprint and hashes(xx)==binary
    if baseline is not None:assert hashes(baseline)==provenance['binary_sha256']
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,python=sys.version,avx2=xx.has_avx2(),fallback_enabled=False,
        reference_thread_pools=1,seed=817,repeat=args.repeat,target_ms=15,cooldown_ms=20,current_binary_sha256=binary,
        **fingerprint,scope='Warmed allocation-inclusive public calls; validation, contiguous copies, threshold comparisons and output destruction included'),results=rows)
    if provenance is not None:report['metadata']['baseline']=provenance
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Boolean and signed conversion performance','','Exact values, dtype, warnings, identity and input immutability are checked before timing. Fallback is disabled. Reference pools use one thread; native worker limits are explicit. All raw samples and adverse cases are retained.','','| API | Input | Workload | Workers | Execution | Original ms | Current ms | Original/current | Preserved/current |','|---|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        paired=f"{r['improvement']:.4f}x" if 'improvement' in r else '—'
        lines.append(f"| {r['api']} | {r['dtype']} | {r['workload']} | {r['threads']} | {r['backend']} | {r['reference_ms']:.4f} | {r['cpp_ms']:.4f} | {r['speedup']:.4f}x | {paired} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
