"""Fourteen frozen B-spline workloads; validation is outside timed calls."""
import argparse
import json
import math
from pathlib import Path
CONTRACT=Path(__file__).resolve().parents[1]/'coverage/contracts/subdivide-polygon-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def integer_masks(degree):
    # Independent Pascal-row derivation; degree1 is the source's special pair.
    if degree==1:return [1,1],[1,1]
    n=degree+1;row=[math.comb(n,k)for k in range(n+1)]
    return (row[1::2],row[::2])if degree%2==0 else(row[::2],[0]+row[1::2])


def points(spec):
    values=[(i/16.,(((i*17)%31)-15)/16.+.1)for i in range(spec['base_points'])]
    if spec['closed']:values.append(values[0])
    if spec['layout']=='reverse-rows':values.reverse()
    return values


def fixture(spec,np):
    values=np.array(points(spec),dtype=np.float64);layout=spec['layout']
    if layout=='F':return np.asfortranarray(values)
    if layout=='reverse-rows':return values[::-1].copy()[::-1]
    if layout=='step-rows':
        owner=np.full((2*len(values),2),-911.,dtype=np.float64);owner[::2]=values
        return owner[::2]
    if layout=='step-columns':
        owner=np.full((len(values),4),-911.,dtype=np.float64);owner[:,::2]=values
        return owner[:,::2]
    assert layout=='C'
    return values


def subdivision_oracle(values,degree,preserve):
    # Independent scalar convolution using the source's bounded tap order.
    circular=values[0]==values[-1]
    data=values[:-1]if circular else values;n=len(data)
    even,odd=integer_masks(degree);length=len(even)
    count=n if circular else n-length+1;phase=(length-1)//2 if circular else length-1
    result=[]
    for i in range(count):
        for mask in(odd,even):
            pair=[]
            for channel in(0,1):
                total=0.
                for k,weight in enumerate(mask):
                    index=i+phase-k
                    if circular:index%=n
                    total=total+data[index][channel]*(weight/(1<<degree))
                pair.append(total)
            result.append(tuple(pair))
    if circular:result.append(result[0])
    elif preserve:result=[data[0],*result,data[-1]]
    return result


def snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,
        shape=list(array.shape),strides=list(array.strides),own=bool(array.flags.owndata),
        writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),
        f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,raw=array.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-subdivide-polygon.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import hashlib,importlib,inspect,os,platform,sys
    from datetime import datetime,timezone
    import numpy as np,scipy,skimage,skimagexx as xx
    root=Path(__file__).resolve().parents[1]
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    original=importlib.import_module('skimage.measure._polygon')
    public=importlib.import_module('skimage.measure')
    implementation=importlib.import_module('skimagexx._subdivide_polygon')
    assert skimage.__version__=='0.26.0'and xx.get_num_threads()==1
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={name:os.environ[name]for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_hash=hashlib.sha256(core.read_bytes()).hexdigest()
    source_before=source_fingerprint(root)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference,native=public.subdivide_polygon,xx.measure.subdivide_polygon
    entry=implementation._core.subdivide_polygon_apply
    assert inspect.isbuiltin(entry)and native is implementation.subdivide_polygon
    rows=[]
    for spec in case_specs():
        coordinates=fixture(spec,np);degree,preserve=spec['degree'],spec['preserve_ends']
        assert coordinates.shape==(spec['points'],2)and coordinates.dtype==np.float64
        owner=coordinates
        while isinstance(owner.base,np.ndarray):owner=owner.base
        assert owner.size<=contract['max_backing_elements']
        before=snapshot(coordinates);backing_before=snapshot(owner)
        oracle=np.array(subdivision_oracle(coordinates.tolist(),degree,preserve),dtype=np.float64)
        assert list(oracle.shape)==spec['expected_output_shape']
        expected=reference(coordinates,degree,preserve)
        calls=[];accepted=[];prior=sys.getprofile();code=native.__code__
        def reject(*a,**kw):raise AssertionError('owned subdivision forwarded to original whole API')
        def profile(frame,event,value):
            if event=='call'and frame.f_code is reference.__code__:reject()
            if event=='c_return'and value is entry:calls.append(value.__name__)
            if event=='return'and frame.f_code is code:accepted.append(frame.f_locals.get('native_result')is True)
        saved=[(module,module.approximate_polygon)for module in(public,original)]
        for module,_ in saved:module.approximate_polygon=reject
        sys.setprofile(profile)
        try:actual=native(coordinates,degree,preserve)
        finally:
            sys.setprofile(prior)
            for module,value in saved:module.approximate_polygon=value
        assert calls==['subdivide_polygon_apply']and accepted==[True]
        a,b,c=map(snapshot,(actual,expected,oracle));assert a==b==c
        assert a['own']and a['writeable']and a['c']and a['dtype_metadata']is None
        assert not np.shares_memory(actual,coordinates)and not np.shares_memory(actual,expected)
        assert before==snapshot(coordinates)and backing_before==snapshot(owner)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        checked=dict(output_metadata={k:v for k,v in a.items()if k!='raw'},
            output_sha256=hashlib.sha256(a['raw']).hexdigest(),
            reference_sha256=hashlib.sha256(b['raw']).hexdigest(),
            oracle_sha256=hashlib.sha256(c['raw']).hexdigest(),
            input_metadata={k:v for k,v in before.items()if k!='raw'},input_sha256=hashlib.sha256(before['raw']).hexdigest())
        timing=measure(lambda:reference(coordinates,degree,preserve),lambda:native(coordinates,degree,preserve),args.repeat,19)
        assert before==snapshot(coordinates)and backing_before==snapshot(owner)and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,backend='native',threads=1,actual_C_returns=calls,native_accepted=True,native_entry_module=entry.__module__,**checked,max_abs_error=0.,**timing))
        print(spec['case_id']+': '+str(timing['speedup']),flush=True)
    assert len(rows)==14 and sum(sum(map(len,r['samples_ms']))for r in rows)==196
    source_after=source_fingerprint(root)
    assert source_before==source_after and CONTRACT.read_bytes()==contract_raw and hashlib.sha256(core.read_bytes()).hexdigest()==core_hash
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,core_path=str(core),core_sha256=core_hash,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=19,raw_samples=196,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),numpy_error_policy=policy,numpy_error_handler_type=type(handler).__name__,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=contract['apis'],provider_frontier=contract['provider_frontier']+' '+contract['timed_boundary']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Subdivide polygon: one-thread comparison','',contract['provider_frontier'],'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
