"""Nine frozen byte-row workloads; whole-public-call one-thread comparison."""
import argparse
from datetime import datetime, timezone
import hashlib, importlib, inspect, json, math, os
from pathlib import Path
import platform, struct, sys

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/unique-rows-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def inputs(spec,np):
    rng=np.random.default_rng(2026);n,d=spec['shape'];pool=spec['unique_pool']
    keys=np.arange(pool,dtype=np.int64);rows=np.empty((pool,d),dtype=spec['dtype'])
    if spec['kind']=='prefix':
        rows.fill(65);rows[:,-2]=keys&255;rows[:,-1]=(keys>>8)&255
    elif rows.dtype.kind=='u':
        for axis in range(d):rows[:,axis]=(keys>>(8*axis))&255
    else:
        rows[:,0]=keys.astype(rows.dtype)+.25
        for axis in range(1,d):rows[:,axis]=(keys*(axis+3)%97-48).astype(rows.dtype)/4
    selection=np.arange(n,dtype=np.int64)%pool;rng.shuffle(selection);ar=rows[selection];backings=[]
    if spec['layout']=='row-and-column-step2':
        storage=np.full((n*2,d*2),-37,dtype=ar.dtype);storage[::2,::2]=ar;ar=storage[::2,::2];backings.append(storage)
    elif spec['layout']=='F':ar=np.asfortranarray(ar)
    elif spec['layout']=='readonly':ar.flags.writeable=False
    return ar,backings


def oracle_rows(ar):
    # Fixed-width bytes, not numeric values or either implementation's sorter.
    raw=ar.tobytes();width=ar.shape[1]*ar.dtype.itemsize
    assert width>0 and len(raw)==ar.shape[0]*width
    rows=[raw[i:i+width] for i in range(0,len(raw),width)];first={}
    for index,row in enumerate(rows):first.setdefault(row,index)
    ordered=sorted(set(rows))
    return dict(raw=b''.join(ordered),shape=(len(ordered),ar.shape[1]),first_indices=[first[row] for row in ordered])


def snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,shape=list(array.shape),strides=list(array.strides),
        own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),
        base_type=type(array.base).__name__,raw=array.tobytes())


def exported(record):
    return dict(**{k:v for k,v in record.items() if k!='raw'},sha256=hashlib.sha256(record['raw']).hexdigest())


def checked_outputs(actual,expected,oracle):
    a,b=snapshot(actual),snapshot(expected)
    assert a==b and a['raw']==oracle['raw'] and tuple(a['shape'])==oracle['shape']
    return dict(output_metadata={k:v for k,v in a.items() if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),
        reference_sha256=hashlib.sha256(b['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle['raw']).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/ci-unique-rows.json'));args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.util.unique');reference,native=skimage.util.unique_rows,xx.util.unique_rows
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();source_before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference_code,native_code=(inspect.unwrap(f).__code__ for f in (reference,native));unique_call=np.unique;rows=[]
    unique_code=getattr(inspect.unwrap(unique_call),'__code__',None)
    for spec in case_specs():
        ar,backings=inputs(spec,np)
        assert ar.dtype.name==spec['dtype'] and ar.shape==tuple(spec['shape']) and ar.shape[0]<=contract['max_rows']
        assert ar.size<=contract['max_input_elements'] and ar.dtype.itemsize*ar.shape[1]==spec['row_width_bytes']
        arrays=(ar,*backings);before=[snapshot(a) for a in arrays];oracle=oracle_rows(ar)
        assert oracle['shape'][0]==spec['unique_pool']
        expected=reference(ar);returns=[];accepted=[];indices_records=[];prior=sys.getprofile()
        def reject(*args,**kwargs):raise AssertionError('native unique_rows forwarded to original API or NumPy unique')
        saved=(skimage.util.unique_rows,original.unique_rows,np.unique)
        def profile(frame,event,arg):
            if event=='call' and frame.f_code in (reference_code,unique_code):reject()
            if event=='c_call' and arg is unique_call:reject()
            if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='unique_rows_apply':returns.append(arg.__name__)
            if event=='return' and frame.f_code is native_code:
                result=frame.f_locals.get('native_result');accepted.append(result is not None)
                if result is not None:indices_records.append(snapshot(result))
        skimage.util.unique_rows=original.unique_rows=np.unique=reject;sys.setprofile(profile)
        try:actual=native(ar)
        finally:sys.setprofile(prior);skimage.util.unique_rows,original.unique_rows,np.unique=saved
        assert returns==['unique_rows_apply'] and accepted==[True] and len(indices_records)==1
        indices=indices_records[0];index_format='q' if np.dtype(np.intp).itemsize==8 else 'i'
        assert indices['dtype']==np.dtype(np.intp).str and indices['shape']==[len(oracle['first_indices'])]
        assert indices['raw']==struct.pack('='+index_format*len(oracle['first_indices']),*oracle['first_indices'])
        output=checked_outputs(actual,expected,oracle)
        assert not np.shares_memory(ar,actual) and not np.shares_memory(ar,expected)
        assert [snapshot(a) for a in arrays]==before
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(ar),lambda:native(ar),args.repeat,19)
        assert [snapshot(a) for a in arrays]==before
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.unique_rows',backend='native',threads=1,seed=2026,inputs=[exported(v) for v in before],
            actual_C_returns=returns,native_accepted=True,native_indices=exported(indices),**output,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==9 and all(list(map(len,row['samples_ms']))==[7,7] for row in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==contract_raw
    scope=contract['scope']+' '+contract['timed_boundary']+' '+contract['provider_frontier']
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,
        core_path=str(core),core_sha256=core_sha,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],
        threads=1,repeat=7,target_ms=19,raw_samples=126,seed=2026,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=['util.unique_rows'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Unique rows: one-thread comparison','',scope,'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
