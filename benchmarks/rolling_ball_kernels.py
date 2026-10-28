"""Frozen constructor parameters and exact native coordinate/reduction kernels.

No image or dtype grid stands in for constructor parameters. Timings require a
genuine authenticated product on a separately authorized measurement host.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/rolling-ball-kernels-benchmark.json'
ENTRIES={'restoration.ball_kernel':'ball_kernel_apply','restoration.ellipsoid_kernel':'ellipsoid_kernel_apply'}


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def output_shape(spec):
    params=spec['parameters']
    if spec['api']=='restoration.ball_kernel':
        return (2*math.ceil(params['radius'])+1,)*params['ndim']
    if spec['api']=='restoration.ellipsoid_kernel':
        return tuple(2*max(dimension//2,1)+1 for dimension in params['shape'])
    raise ValueError('Unknown frozen constructor API')


def constructor_arguments(spec):
    params=spec['parameters']
    if spec['api']=='restoration.ball_kernel':return (params['radius'],params['ndim'])
    if spec['api']=='restoration.ellipsoid_kernel':return (list(params['shape']),params['intensity'])
    raise ValueError('Unknown frozen constructor API')


def source_axis_sum(terms):
    """Pinned NumPy float64 fast-axis order for this rank<=8 cohort."""
    assert 1<=len(terms)<=8
    if len(terms)<8:
        total=-0.0
        for value in terms:total+=value
    else:
        total=((terms[0]+terms[1])+(terms[2]+terms[3]))+((terms[4]+terms[5])+(terms[6]+terms[7]))
    return 0.0+total


def coordinate_oracle(spec):
    """Independent placement/arithmetic with the pinned binary64 reduction order.

    The source materializes both square roots for ball and applies its distance
    mask only afterwards. Ellipsoid normalizes before square/reduction, then
    subtracts from one, clips, roots, multiplies and applies the scaling mask.
    No original constructor, meshgrid or native function is called here.
    """
    shape=output_shape(spec);params=spec['parameters'];ball=spec['api']=='restoration.ball_kernel'
    axes=[(dimension-1)//2 for dimension in shape]
    raw=bytearray()
    for point in itertools.product(*(range(-axis,axis+1)for axis in axes)):
        if ball:
            squares=[float(coordinate)*float(coordinate)for coordinate in point]
            total=source_axis_sum(squares)
            distance=math.sqrt(total)
            value=math.sqrt(max(params['radius']**2-total,0.))
            if distance>params['radius']:value=math.inf
        else:
            normalized=[float(coordinate)/float(axis)for coordinate,axis in zip(point,axes)]
            total=source_axis_sum([coordinate*coordinate for coordinate in normalized])
            scaling=1.-total
            value=params['intensity']*math.sqrt(max(scaling,0.))
            if scaling<0.:value=math.inf
        raw.extend(struct.pack('=d',value))
    return bytes(raw),shape


def output_snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,shape=list(array.shape),strides=list(array.strides),own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,raw=array.tobytes())


def checked_outputs(actual,expected,oracle,shape):
    import hashlib
    a,b=output_snapshot(actual),output_snapshot(expected)
    assert a==b and a['raw']==oracle and tuple(a['shape'])==tuple(shape)
    assert a['dtype'].endswith('f8')and a['dtype_metadata']is None and a['own']and a['writeable']and a['c']and a['base_type']=='NoneType'
    return dict(output_metadata={k:v for k,v in a.items()if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),reference_sha256=hashlib.sha256(b['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-rolling-ball-kernels.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import hashlib,importlib,inspect,os,platform,sys
    from datetime import datetime,timezone
    import numpy as np,scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.restoration._rolling_ball')
    public=importlib.import_module('skimage.restoration');implementation=importlib.import_module('skimagexx._rolling_ball_kernels')
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert skimage.__version__=='0.26.0'and xx.get_num_threads()==1
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={name:os.environ[name]for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    core=Path(xx._core.__file__);core_hash=hashlib.sha256(core.read_bytes()).hexdigest();source_before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    reference_codes={getattr(original,api.split('.')[-1]).__code__ for api in contract['apis']}
    for spec in case_specs():
        api=spec['api'];name=api.split('.')[-1];arguments=constructor_arguments(spec);input_before=json.dumps(arguments)
        reference=getattr(public,name);native=getattr(xx.restoration,name);entry=getattr(implementation._core,ENTRIES[api])
        assert inspect.isbuiltin(entry)and native is getattr(implementation,name)
        oracle,shape=coordinate_oracle(spec);assert list(shape)==spec['expected_output_shape']and math.prod(shape)<=contract['max_output_elements']
        expected=reference(*arguments);calls=[];accepted=[];prior=sys.getprofile();code=native.__code__
        def reject(*a,**kw):raise AssertionError('owned rolling constructor forwarded to original whole API')
        def profile(frame,event,value):
            if event=='call'and frame.f_code in reference_codes:reject()
            if event=='c_return'and value is entry:calls.append(value.__name__)
            if event=='return'and frame.f_code is code:accepted.append(frame.f_locals.get('native_result')is True)
        saved=[(module,fn,getattr(module,fn))for module in (public,original)for fn in ('ball_kernel','ellipsoid_kernel')]
        for module,fn,_ in saved:setattr(module,fn,reject)
        sys.setprofile(profile)
        try:actual=native(*arguments)
        finally:
            sys.setprofile(prior)
            for module,fn,value in saved:setattr(module,fn,value)
        assert calls==[ENTRIES[api]]and accepted==[True]
        checked=checked_outputs(actual,expected,oracle,shape)
        assert json.dumps(arguments)==input_before and not np.shares_memory(actual,expected)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(*arguments),lambda:native(*arguments),args.repeat,19)
        assert json.dumps(arguments)==input_before and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,backend='native',threads=1,actual_C_returns=calls,native_accepted=True,native_entry_module=entry.__module__,**checked,max_abs_error=0.0,**timing))
        print(spec['case_id']+': '+str(timing['speedup']),flush=True)
    assert len(rows)==12 and sum(sum(map(len,r['samples_ms']))for r in rows)==168
    assert all(list(map(len,r['samples_ms']))==[7,7]for r in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after and CONTRACT.read_bytes()==contract_raw and hashlib.sha256(core.read_bytes()).hexdigest()==core_hash
    scope=contract['provider_frontier']+' '+contract['timed_boundary']
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,core_path=str(core),core_sha256=core_hash,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),contract_sha256=hashlib.sha256(contract_raw).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=19,raw_samples=168,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),numpy_error_policy=policy,numpy_error_handler_type=type(handler).__name__,numpy_buffer_elements=buffer,command=sys.argv,selected_apis=contract['apis'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Rolling kernel constructors: one-thread comparison','',scope,'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
