"""Nine frozen grid-label constructors; whole-public-call one-thread comparison."""
import argparse
from datetime import datetime, timezone
import hashlib, importlib, inspect, itertools, json, math, os
from pathlib import Path
import platform, struct, sys

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/regular-seeds-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def oracle_labels(shape,grid,dtype):
    formats={'uint8':'B','int32':'i','int64':'q','float32':'f','float64':'d'}
    fmt='='+formats[dtype];width=struct.calcsize(fmt)
    raw=bytearray(math.prod(shape)*width);strides=[];stride=1
    for dimension in reversed(shape):strides.append(stride);stride*=dimension
    strides.reverse();ranges=[range(*s.indices(d)) for d,s in zip(shape,grid)]
    assert len(ranges)==len(shape) and all(r.step>0 for r in ranges)
    selected=math.prod(map(len,ranges))
    assert dtype!='float32' or selected<=2**24
    assert dtype!='float64' or selected<=2**53
    for label,coordinates in enumerate(itertools.product(*ranges),1):
        offset=sum(c*s for c,s in zip(coordinates,strides))*width
        struct.pack_into(fmt,raw,offset,label%256 if dtype=='uint8' else label)
    return dict(raw=bytes(raw),shape=tuple(shape),selected_count=selected)


def snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,shape=list(array.shape),strides=list(array.strides),
        own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),
        base_type=type(array.base).__name__,raw=array.tobytes())


def checked_outputs(actual,expected,oracle,dtype):
    a,b=snapshot(actual),snapshot(expected)
    assert actual.dtype.name==dtype and a==b
    assert a['raw']==oracle['raw'] and tuple(a['shape'])==oracle['shape']
    return dict(output_metadata={k:v for k,v in a.items() if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),
        reference_sha256=hashlib.sha256(b['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle['raw']).hexdigest(),selected_count=oracle['selected_count'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/ci-regular-seeds.json'));args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.util._regular_grid');reference,native=skimage.util.regular_seeds,xx.util.regular_seeds
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();source_before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference_code,native_code=(inspect.unwrap(f).__code__ for f in(reference,native));rows=[]
    for spec in case_specs():
        shape=tuple(spec['shape']);points=spec['n_points'];dtype=np.dtype(spec['dtype'])
        assert math.prod(shape)<=contract['max_output_elements']
        grid=original.regular_grid(shape,points);oracle=oracle_labels(shape,grid,spec['dtype'])
        expected=reference(shape,points,dtype=dtype);returns=[];accepted=[];prior=sys.getprofile()
        def reject(*args,**kwargs):raise AssertionError('native regular_seeds forwarded to original whole API')
        saved=skimage.util.regular_seeds,original.regular_seeds
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is reference_code:reject()
            if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='regular_seeds_apply':returns.append(arg.__name__)
            if event=='return' and frame.f_code is native_code:accepted.append(frame.f_locals.get('native_result') is True)
        skimage.util.regular_seeds=original.regular_seeds=reject;sys.setprofile(profile)
        try:actual=native(shape,points,dtype=dtype)
        finally:sys.setprofile(prior);skimage.util.regular_seeds,original.regular_seeds=saved
        assert returns==['regular_seeds_apply'] and accepted==[True]
        output=checked_outputs(actual,expected,oracle,spec['dtype'])
        assert not np.shares_memory(actual,expected)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(shape,points,dtype=dtype),lambda:native(shape,points,dtype=dtype),args.repeat,19)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.regular_seeds',backend='native',threads=1,actual_C_returns=returns,native_accepted=True,
            grid_slices=[[None if v is None else int(v)for v in(s.start,s.stop,s.step)]for s in grid],**output,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==9 and all(list(map(len,row['samples_ms']))==[7,7]for row in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==contract_raw
    scope=contract['scope']+' '+contract['timed_boundary']+' '+contract['provider_frontier']
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,
        core_path=str(core),core_sha256=core_sha,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=19,
        raw_samples=126,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),numpy_error_policy=policy,numpy_buffer_elements=buffer,
        command=sys.argv,selected_apis=['util.regular_seeds'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Regular seeds: one-thread comparison','',scope,'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
