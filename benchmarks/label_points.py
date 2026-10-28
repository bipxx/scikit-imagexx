"""Nine frozen coordinate workloads; whole-call allocation-inclusive comparison."""
import argparse
from datetime import datetime, timezone
import hashlib, importlib, inspect, json, math, os
from pathlib import Path
import platform, struct, sys

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/label-points-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def inputs(spec,np):
    rng=np.random.default_rng(2026);shape=tuple(spec['output_shape']);points=spec['point_count'];kind=spec['kind']
    if kind=='dense':
        order=rng.permutation(math.prod(shape));coords=np.empty((points,len(shape)),dtype=spec['coordinate_dtype'])
        divisor=math.prod(shape)
        for axis,extent in enumerate(shape):
            divisor//=extent;coords[:,axis]=(order//divisor)%extent
    else:
        count=spec.get('duplicate_pool',points)
        pool=np.column_stack([rng.integers(0,extent-1,count,dtype=np.int64) for extent in shape]).astype(spec['coordinate_dtype'])
        if pool.dtype.kind=='f':pool+=((np.arange(count)%4)/4).astype(pool.dtype)[:,None]
        coords=pool[np.arange(points)%count] if kind=='duplicates' else pool
    backings=[]
    if kind=='strided':
        storage=np.empty((points*2,len(shape)),dtype=coords.dtype);storage[::2]=coords
        storage[1::2]=-37;coords=storage[::2,::-1];backings.append(storage)
    return coords,shape,backings


def oracle_words(coords,shape):
    # Decode logical input words, round independently, visit last point first.
    formats={'int64':'q','float64':'d','float32':'f'};fmt='='+formats[coords.dtype.name]*len(shape)
    width=struct.calcsize(fmt);raw=coords.tobytes();assert len(raw)==coords.shape[0]*width
    output=bytearray(math.prod(shape)*8);seen=set()
    for point in range(coords.shape[0]-1,-1,-1):
        values=struct.unpack_from(fmt,raw,point*width);offset=0
        for value,extent in zip(values,shape):
            index=round(value);assert 0<=index<extent
            offset=offset*extent+index
        if offset not in seen:
            struct.pack_into('=Q',output,offset*8,point+1);seen.add(offset)
    return bytes(output)


def snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,shape=list(array.shape),strides=list(array.strides),
        own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),
        base_type=type(array.base).__name__,raw=array.tobytes())


def exported(record):
    return dict(**{k:v for k,v in record.items() if k!='raw'},sha256=hashlib.sha256(record['raw']).hexdigest())


def checked_outputs(actual,expected,oracle):
    a,b=snapshot(actual),snapshot(expected)
    assert a==b and a['raw']==oracle
    return dict(output_metadata={k:v for k,v in a.items() if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),
        reference_sha256=hashlib.sha256(b['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/ci-label-points.json'));args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.util._label');reference,native=skimage.util.label_points,xx.util.label_points
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();source_before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference_code,native_code=(inspect.unwrap(f).__code__ for f in (reference,native));rows=[]
    for spec in case_specs():
        coords,shape,backings=inputs(spec,np)
        assert coords.dtype.name==spec['coordinate_dtype'] and coords.shape==(spec['point_count'],len(shape))
        assert math.prod(shape)<=contract['max_output_elements']
        arrays=(coords,*backings);before=[snapshot(a) for a in arrays];oracle=oracle_words(coords,shape)
        expected=reference(coords,shape);returns=[];accepted=[];prior=sys.getprofile()
        def reject(*args,**kwargs):raise AssertionError('native label_points forwarded to original whole API')
        saved=(skimage.util.label_points,original.label_points)
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is reference_code:reject()
            if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='label_points_apply':returns.append(arg.__name__)
            if event=='return' and frame.f_code is native_code:accepted.append(frame.f_locals.get('native_result') is not None)
        skimage.util.label_points=original.label_points=reject;sys.setprofile(profile)
        try:actual=native(coords,shape)
        finally:sys.setprofile(prior);skimage.util.label_points,original.label_points=saved
        assert returns==['label_points_apply'] and accepted==[True]
        output=checked_outputs(actual,expected,oracle)
        assert not np.shares_memory(coords,actual) and not np.shares_memory(coords,expected)
        assert [snapshot(a) for a in arrays]==before
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(coords,shape),lambda:native(coords,shape),args.repeat,19)
        assert [snapshot(a) for a in arrays]==before
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.label_points',backend='native',threads=1,seed=2026,
            inputs=[exported(v) for v in before],actual_C_returns=returns,native_accepted=True,
            **output,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==9 and all(list(map(len,row['samples_ms']))==[7,7] for row in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==contract_raw
    scope=contract['scope']+' '+contract['timed_boundary']+' '+contract['frontier']
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,
        core_path=str(core),core_sha256=core_sha,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],
        threads=1,repeat=7,target_ms=19,raw_samples=126,seed=2026,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=['util.label_points'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Label points: one-thread comparison','',scope,'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
