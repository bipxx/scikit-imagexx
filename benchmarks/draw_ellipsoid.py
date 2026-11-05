"""Twelve source-bounded ellipsoid public workloads, exact one-worker parity."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/ellipsoid-benchmark.json'
REFERENCE=ROOT/'coverage/contracts/ellipsoid.json'


def source_safe(spec):
    """Allocation audit, not an expected-value/shape oracle or native grid."""
    assert spec['api']=='draw.ellipsoid' and type(spec['levelset']) is bool
    axes,spacing=spec['axes'],spec['spacing'];assert len(axes)==len(spacing)==3
    assert all(type(v) in (int,float) and math.isfinite(v) and 0<v<=48 for v in axes)
    assert all(type(v) is float and math.isfinite(v) and .3<=v<=2 for v in spacing)
    bounds=[]
    for axis,step in zip(axes,spacing):
        low=math.ceil(-axis-step);high=math.floor((axis+step)+1)
        if (high-low)%2==0:low-=1
        # np.arange first allocates ceil((high-low)/nominal_step) doubles.
        # If zero is missing, source subtracts a strictly negative probe value
        # from low. This increases low, so the subsequent nominal-step mgrid
        # cannot have more elements than this PRE-correction bound. All small
        # integer limits are exact doubles; step>0 has nonzero effective step.
        effective=(float(low)+step)-float(low)
        count=math.ceil((high-low)/step)
        assert low<0<high and effective>0 and 0<count<=99
        bounds.append(count)
    count=math.prod(bounds);assert 0<count<=1_000_000
    return dict(axis_upper_bounds=bounds,max_output_elements=count,
        max_probe_elements=sum(bounds),max_dense_grid_elements=3*count,
        conservative_peak_working_bytes=80*count+8*sum(bounds))


def fixture(spec):
    source_safe(spec)
    return tuple(spec['axes']),dict(spacing=tuple(spec['spacing']),levelset=spec['levelset'])


def checked_output(actual,expected,np,snapshot,bound):
    assert type(actual) is type(expected) is np.ndarray
    assert actual is not expected and not np.shares_memory(actual,expected)
    assert actual.ndim==3 and actual.size<=bound
    assert actual.dtype==expected.dtype and actual.dtype in (np.dtype(bool),np.dtype('float64'))
    assert actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable and actual.flags.aligned
    assert actual.base is None and actual.dtype.metadata is None
    assert snapshot((actual,))==snapshot((expected,))  # Includes exact raw words.
    output=snapshot((actual,))
    output['arrays'][0]['raw_sha256']=hashlib.sha256(output['arrays'][0].pop('raw')).hexdigest()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output',type=Path,default=Path('build/ci-ellipsoid.json'))
    args=parser.parse_args()
    deadline=args.deadline_utc or (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists() and not journal.exists()
    sys.path.insert(0,str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    from draw_coordinates import snapshot
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimagexx import _compat
    original=importlib.import_module('skimage.draw.draw3d');public=importlib.import_module('skimage.draw')
    owned=importlib.import_module('skimagexx._draw_ellipsoid')
    grid=importlib.import_module(type(np.mgrid).__module__)
    prefix='numpy._core' if np.__version__.startswith('2.') else 'numpy.core'
    binary=importlib.import_module(prefix+'._multiarray_umath')
    contract_raw,reference_raw=CONTRACT.read_bytes(),REFERENCE.read_bytes()
    contract,reference=json.loads(contract_raw),json.loads(reference_raw)
    assert hashlib.sha256(reference_raw).hexdigest()==contract['reference_contract_sha256']
    assert skimage.__version__=='0.26.0' and np.__version__ in reference['original_runtime_provenance']
    runtime=reference['original_runtime_provenance'][np.__version__]
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==reference['original_source_sha256']
    assert hashlib.sha256(Path(grid.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==runtime['numpy_grid_source_sha256']
    binary_sha=hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()
    assert binary_sha==runtime['numpy_binary_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    for name,pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    limits={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'} and xx.get_num_threads()==1
    xx.set_fallback_enabled(False);native,upstream=xx.draw.ellipsoid,public.ellipsoid
    assert native is owned.ellipsoid and str(inspect.signature(native))==str(inspect.signature(upstream))==reference['signature']
    entry,getter=xx._core.draw_ellipsoid,xx._core.get_num_threads
    assert inspect.isbuiltin(entry) and entry.__module__=='skimagexx._core'
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    cpu_model=platform.processor();cpu_info=Path('/proc/cpuinfo')
    if cpu_info.is_file():cpu_model=next((s.partition(':')[2].strip() for s in cpu_info.read_text().splitlines() if s.startswith('model name')),cpu_model)
    cpu_count,avx2_available=os.cpu_count(),bool(xx._core.has_avx2())
    before=source_fingerprint(ROOT);policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    for index,spec in enumerate(contract['cases']):
        bound=source_safe(spec);positional,options=fixture(spec);input_state=(positional,options.copy())
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');expected=upstream(*positional,**options)
        assert not notices and (positional,options)==input_state
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        calls,accepted,snapshots=[],[],[]
        def denied(*args,**kwargs):raise AssertionError('native ellipsoid forwarded to original provider')
        def profile(frame,event,value):
            assert not (event=='call' and frame.f_code is upstream.__code__),'cached original algorithm called by native'
            if event=='c_return' and value is entry:calls.append('draw_ellipsoid')
            if event=='c_return' and value is getter:snapshots.append('get_num_threads')
            if event=='return' and frame.f_code is owned.ellipsoid.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
        prior_profile=sys.getprofile();prior=(original.ellipsoid,public.ellipsoid,_compat.fallback)
        try:
            original.ellipsoid=public.ellipsoid=denied;_compat.fallback=denied;sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native(*positional,**options)
        finally:
            sys.setprofile(prior_profile);original.ellipsoid,public.ellipsoid,_compat.fallback=prior
        assert not notices and calls==['draw_ellipsoid'] and accepted==[True] and snapshots==['get_num_threads']
        output=checked_output(actual,expected,np,snapshot,bound['max_output_elements'])
        assert actual.dtype==np.dtype('float64' if spec['levelset'] else bool)
        assert (positional,options)==input_state and xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:upstream(*positional,**options),lambda:native(*positional,**options),args.repeat,100,
            journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert (positional,options)==input_state and xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(case_id=spec['id'],api='draw.ellipsoid',parameters=spec,bounds=bound,threads=1,backend='native',actual_C_returns=calls,native_accepted=True,global_worker_snapshots=snapshots,output=output,error=None,warnings=[],max_abs_error=0.,**timing))
    assert len(rows)==12 and sum(sum(map(len,r['samples_ms'])) for r in rows)==168
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    assert CONTRACT.read_bytes()==contract_raw and REFERENCE.read_bytes()==reference_raw
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu_model=cpu_model,logical_cpu_count=cpu_count,avx2_available=avx2_available,
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**before,core_sha256=core_sha,numpy_binary_sha256=binary_sha,
        threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,raw_samples=168,reference_pool_limits=limits,fallback_enabled=False,deadline_utc=deadline,
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),method_sha256=contract['method_sha256'],scope=contract['scope']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Ellipsoid: initial one-worker comparison','',contract['scope'],'','| Case | Original ms | C++ ms | Original/current |','|---|---:|---:|---:|']
    lines.extend(f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |" for r in rows)
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
