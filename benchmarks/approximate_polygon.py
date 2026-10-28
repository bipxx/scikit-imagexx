"""Frozen nine-case RDP comparison; validation is outside all timed calls."""
import argparse
import json
import math
from pathlib import Path

CONTRACT=Path(__file__).resolve().parents[1]/'coverage/contracts/approximate-polygon-benchmark.json'


def case_specs():
    return json.loads(CONTRACT.read_bytes())['cases']


def points(spec):
    """Logical coordinates, independent of their subsequent storage layout."""
    n=spec['points'];kind=spec['geometry']
    if kind=='collinear':values=[(float(i),float(2*i+3))for i in range(n)]
    elif kind=='zigzag':values=[(float(i),1. if i%2 else -1.)for i in range(n)]
    elif kind=='smooth':
        values=[(i*(16.*math.pi/(n-1)),math.sin(i*(16.*math.pi/(n-1))))for i in range(n)]
    else:raise ValueError('Unknown frozen polygon geometry')
    if spec['layout']=='reverse-rows':values.reverse()
    return values


def fixture(spec,np):
    values=np.array(points(spec),dtype=np.float64)
    layout=spec['layout']
    if layout=='F':return np.asfortranarray(values)
    if layout=='reverse-rows':return values[::-1].copy()[::-1]
    if layout=='step-rows':
        owner=np.full((2*len(values),2),-911.,dtype=np.float64)
        owner[::2]=values
        return owner[::2]
    if layout=='step-columns':
        owner=np.full((len(values),4),-911.,dtype=np.float64)
        owner[:,::2]=values
        return owner[:,::2]
    assert layout=='C'
    return values


def selection_oracle(values,tolerance,trig=None):
    """Independent scalar RDP placement, with an explicit elementary provider.

    Keep source subtraction/multiply/add/sqrt order, strict projection and
    tolerance predicates, first maximum and left-first stack traversal. The
    provider returns sin(-atan2(dr,dc)),cos(-atan2(dr,dc)); timed Python can pass
    the exact NumPy scalar provider without sharing native selection logic.
    """
    if trig is None:
        def trig(dr,dc):
            angle=-math.atan2(dr,dc)
            return math.sin(angle),math.cos(angle)
    keep=[False]*len(values);keep[0]=keep[-1]=True
    stack=[(0,len(values)-1)]
    while stack:
        start,end=stack.pop();r0,c0=values[start];r1,c1=values[end]
        dr,dc=r1-r0,c1-c0;sine,cosine=trig(dr,dc)
        segment_dist=c0*sine+r0*cosine
        maximum=-math.inf;index=None;split=False
        for i in range(start+1,end):
            r,c=values[i];dr0,dc0=r-r0,c-c0;dr1,dc1=r-r1,c-c1
            p0=dr0*dr+dc0*dc;p1=(-dr1)*dr-dc1*dc
            if p0>0 and p1>0:distance=abs(r*cosine+c*sine-segment_dist)
            else:distance=min(math.sqrt(dc0*dc0+dr0*dr0),math.sqrt(dc1*dc1+dr1*dr1))
            split=split or distance>tolerance
            if distance>maximum:maximum,index=distance,i
        if split:
            assert index is not None
            keep[index]=True;stack.extend(((index,end),(start,index)))
    return [i for i,value in enumerate(keep)if value]


def snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,
        shape=list(array.shape),strides=list(array.strides),own=bool(array.flags.owndata),
        writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),
        f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,raw=array.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-approximate-polygon.json'))
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
    implementation=importlib.import_module('skimagexx._approximate_polygon')
    assert skimage.__version__=='0.26.0'and xx.get_num_threads()==1
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={name:os.environ[name]for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_hash=hashlib.sha256(core.read_bytes()).hexdigest()
    source_before=source_fingerprint(root)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference,native=public.approximate_polygon,xx.measure.approximate_polygon
    entry=implementation._core.approximate_polygon_apply
    assert inspect.isbuiltin(entry)and native is implementation.approximate_polygon
    rows=[]
    for spec in case_specs():
        coordinates=fixture(spec,np);tolerance=spec['tolerance']
        assert coordinates.shape==(spec['points'],2)and coordinates.dtype==np.float64
        owner=coordinates
        while isinstance(owner.base,np.ndarray):owner=owner.base
        assert owner.size<=contract['max_backing_elements']
        before=snapshot(coordinates);backing_before=snapshot(owner)
        indices=selection_oracle(coordinates.tolist(),tolerance,implementation._scalar_trig)
        oracle=coordinates[indices,:];expected=reference(coordinates,tolerance)
        calls=[];accepted=[];prior=sys.getprofile();code=native.__code__
        def reject(*a,**kw):raise AssertionError('owned RDP selection forwarded to original whole API')
        def profile(frame,event,value):
            if event=='call'and frame.f_code is reference.__code__:reject()
            if event=='c_return'and value is entry:calls.append(value.__name__)
            if event=='return'and frame.f_code is code:accepted.append(frame.f_locals.get('native_result')is True)
        saved=[(module,module.approximate_polygon)for module in(public,original)]
        for module,_ in saved:module.approximate_polygon=reject
        sys.setprofile(profile)
        try:actual=native(coordinates,tolerance)
        finally:
            sys.setprofile(prior)
            for module,value in saved:module.approximate_polygon=value
        assert calls==['approximate_polygon_apply']and accepted==[True]
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
        timing=measure(lambda:reference(coordinates,tolerance),lambda:native(coordinates,tolerance),args.repeat,19)
        assert before==snapshot(coordinates)and backing_before==snapshot(owner)and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,backend='native',threads=1,actual_C_returns=calls,native_accepted=True,native_entry_module=entry.__module__,**checked,max_abs_error=0.,**timing))
        print(spec['case_id']+': '+str(timing['speedup']),flush=True)
    assert len(rows)==9 and sum(sum(map(len,r['samples_ms']))for r in rows)==126
    source_after=source_fingerprint(root)
    assert source_before==source_after and CONTRACT.read_bytes()==contract_raw and hashlib.sha256(core.read_bytes()).hexdigest()==core_hash
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,core_path=str(core),core_sha256=core_hash,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=19,raw_samples=126,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),numpy_error_policy=policy,numpy_error_handler_type=type(handler).__name__,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=contract['apis'],provider_frontier=contract['provider_frontier']+' '+contract['timed_boundary']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Approximate polygon: one-thread comparison','',contract['provider_frontier'],'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
