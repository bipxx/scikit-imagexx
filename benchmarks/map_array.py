"""Nine frozen map-array workloads; independent exact prechecks before timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/map-array-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def inputs(spec,np):
    rng=np.random.default_rng(2026);kind=spec['kind'];backings=[]
    if kind in ('dense','strided-output','strided-input'):
        bins=1024 if kind=='strided-input' else 256
        keys=rng.permutation(np.arange(bins,dtype=spec['input_dtype']))
        keys=np.concatenate((keys,keys[:1]))
        if kind=='strided-output':values=rng.integers(-4096,4097,len(keys)).astype(np.float32)/np.float32(8)
        else:values=rng.integers(1,1<<np.dtype(spec['output_dtype']).itemsize*8,len(keys),dtype=spec['output_dtype'])
        if kind=='strided-input':
            storage=rng.integers(0,2048,spec['backing_shape'],dtype=np.uint16);image=storage[::2,::2];backings.append(storage)
        else:image=rng.integers(0,256,spec['shape'],dtype=np.uint8)
    else:
        if kind=='signed-sparse':
            keys=np.array([-(1<<55),-7,0,17,1<<40,-7],np.int64)
            values=rng.integers(-4096,4097,len(keys)).astype(np.float64)/8
            choices=np.array([-(1<<55),-7,0,17,1<<40,99,1<<55],np.int64)
        else:
            keys=np.array([(1<<63)+3,(1<<64)-1,17,(1<<63)+(1<<40),0,(1<<63)+3],np.uint64)
            values=rng.integers(-4096,4097,len(keys)).astype(np.float64)/8+1j*(rng.integers(-4096,4097,len(keys)).astype(np.float64)/16)
            choices=np.array([0,17,(1<<63)+3,(1<<64)-1,(1<<63)+(1<<40),(1<<63)+5],np.uint64)
        image=choices[rng.integers(0,len(choices),spec['shape'],dtype=np.uint8)]
    reference_out=native_out=None;output_backings=[]
    if spec['preallocated_out']:
        for fill in (91.25,-37.5):output_backings.append(np.full(spec['output_backing_shape'],fill,np.float32))
        reference_out,native_out=(value[::2] for value in output_backings)
    return (image,keys,values),reference_out,native_out,backings,output_backings


def oracle_words(image,keys,values):
    # Independent copied-value dictionary, not an invocation of either mapper.
    formats={('i',1):'b',('i',2):'h',('i',4):'i',('i',8):'q',
             ('u',1):'B',('u',2):'H',('u',4):'I',('u',8):'Q'}
    prefix='<' if sys.byteorder=='little' else '>'
    fmt=prefix+formats[image.dtype.kind,image.dtype.itemsize]
    assert (keys.dtype.kind,keys.dtype.itemsize)==(image.dtype.kind,image.dtype.itemsize)
    width=values.dtype.itemsize;raw=values.tobytes();lookup={}
    for k,(label,) in enumerate(struct.iter_unpack(fmt,keys.tobytes())):lookup[label]=raw[k*width:(k+1)*width]
    zero=bytes(width);return b''.join(lookup.get(label,zero) for (label,) in struct.iter_unpack(fmt,image.tobytes()))


def snapshot(value):
    return dict(dtype=value.dtype.str,metadata=value.dtype.metadata,shape=list(value.shape),strides=list(value.strides),
        own=bool(value.flags.owndata),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),
        writeable=bool(value.flags.writeable),base_type=type(value.base).__name__,raw=value.tobytes())


def exported(record):
    return dict(**{k:v for k,v in record.items() if k!='raw'},sha256=hashlib.sha256(record['raw']).hexdigest())


def checked_outputs(actual,expected,oracle,reference_out,native_out):
    a,b=snapshot(actual),snapshot(expected)
    assert a==b and a['raw']==oracle
    assert (actual is native_out)==(native_out is not None)
    assert (expected is reference_out)==(reference_out is not None)
    return dict(output_metadata={k:v for k,v in a.items() if k!='raw'},
        output_sha256=hashlib.sha256(a['raw']).hexdigest(),reference_sha256=hashlib.sha256(b['raw']).hexdigest(),
        oracle_sha256=hashlib.sha256(oracle).hexdigest(),result_is_out=actual is native_out,
        reference_result_is_out=expected is reference_out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-map-array.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    original=importlib.import_module('skimage.util._map_array');primitive=importlib.import_module('skimage.util._remap')
    reference,native=skimage.util.map_array,xx.util.map_array
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    limits={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();source_before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference_code,native_code=(inspect.unwrap(f).__code__ for f in (reference,native))
    primitive_call=primitive._map_array;primitive_code=getattr(primitive_call,'__code__',None)
    rows=[]
    for spec in case_specs():
        call_args,reference_out,native_out,backings,output_backings=inputs(spec,np)
        image,keys,values=call_args
        assert image.dtype.name==spec['input_dtype'] and values.dtype.name==spec['output_dtype']
        assert image.shape==tuple(spec['shape']) and len(keys)==len(values)==spec['key_count']
        arrays=(*call_args,*backings);before=[snapshot(a) for a in arrays]
        gaps=[value[1::2].tobytes() for value in output_backings]
        oracle=oracle_words(*call_args)
        expected=reference(*call_args,out=reference_out)
        returns=[];accepted=[];prior=sys.getprofile()
        def reject(*args,**kwargs):raise AssertionError('native mapping forwarded to original Python/Cython mapper')
        saved=(skimage.util.map_array,original.map_array,primitive._map_array)
        def profile(frame,event,arg):
            if event=='call' and frame.f_code in (reference_code,primitive_code):reject()
            if event=='c_call' and arg is primitive_call:reject()
            if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='map_array_apply':returns.append(arg.__name__)
            if event=='return' and frame.f_code is native_code:accepted.append(frame.f_locals.get('native_result') is not None)
        skimage.util.map_array=original.map_array=primitive._map_array=reject
        sys.setprofile(profile)
        try:actual=native(*call_args,out=native_out)
        finally:
            sys.setprofile(prior)
            skimage.util.map_array,original.map_array,primitive._map_array=saved
        assert returns==['map_array_apply'] and accepted==[True]
        assert type(actual) is type(expected) is np.ndarray
        output=checked_outputs(actual,expected,oracle,reference_out,native_out)
        assert [snapshot(a) for a in arrays]==before and gaps==[a[1::2].tobytes() for a in output_backings]
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(*call_args,out=reference_out),lambda:native(*call_args,out=native_out),args.repeat,19)
        assert [snapshot(a) for a in arrays]==before and gaps==[a[1::2].tobytes() for a in output_backings]
        assert actual.tobytes()==expected.tobytes()==oracle
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.map_array',backend='native',threads=1,seed=2026,
            inputs=[exported(r) for r in before],actual_C_returns=returns,native_accepted=True,
            **output,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==9 and all(list(map(len,r['samples_ms']))==[7,7] for r in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==contract_raw
    scope=contract['scope']+' All nine workloads preserve input and mapping bytes and check complete output words, storage metadata and out identity against an independent copied-value last-key oracle and fresh scikit-image0.26 before timing. Genuine C admission and original Python/Cython forwarding guards complete before timers. Output hashes only are exported. This first original/current comparison is not prior-C++ gain or full-profile qualification.'
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],
        primitive_sha256=hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest(),threads=1,repeat=7,target_ms=19,
        raw_samples=126,seed=2026,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=['util.map_array'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Map array: one-thread comparison','',scope,'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
