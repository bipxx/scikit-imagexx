"""Frozen rational Bezier workloads; allocation-inclusive public calls, one worker."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-bezier-benchmark-v1.json'
REFERENCE=ROOT/'coverage/contracts/draw-bezier-v1.json'


def source_safe(spec):
    args=spec['args'];shape=spec['shape']
    assert spec['api']=='draw.bezier_curve' and len(args)==7
    assert all(type(x) is int and -4096<=x<=4096 for x in args[:6])
    assert type(args[6]) is float and args[6] in (0.,.25,.5,1.,2.)
    assert shape is None or (type(shape) is list and len(shape)==2 and all(type(x) is int and 0<=x<=4096 for x in shape))
    # The independent source-stage trace certificate covers these exact cases;
    # this coarse finite allocation cap is not a substitute for that proof.
    span=max(args[:6])-min(args[:6])
    bound=3*(2*span+5)
    assert bound<=50000
    return {'coordinate_pairs_upper_bound':bound,'payload_bytes_upper_bound':16*bound}


def fixture(spec):
    bounds=source_safe(spec)
    return tuple(spec['args']),{'shape':None if spec['shape'] is None else tuple(spec['shape'])},bounds


def checked_output(actual,expected,np,snapshot,bound):
    assert type(actual) is type(expected) is tuple and len(actual)==len(expected)==2
    for value in actual:
        assert type(value) is np.ndarray and value.dtype==np.dtype(np.intp) and value.dtype.metadata is None
        assert value.ndim==1 and value.flags.owndata and value.base is None and value.flags.writeable and value.flags.aligned
        assert all(value is not other and not np.shares_memory(value,other) for other in expected)
        assert value.size<=bound['coordinate_pairs_upper_bound']
    assert actual[0] is not actual[1] and not np.shares_memory(*actual)
    assert snapshot(actual)==snapshot(expected)
    result=snapshot(actual)
    for a in result['arrays']:a['raw_sha256']=hashlib.sha256(a.pop('raw')).hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output',type=Path,default=Path('build/ci-bezier-curve.json'))
    args=parser.parse_args()
    deadline=args.deadline_utc or (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl')
    assert not args.output.exists() and not journal.exists()
    contract_raw,reference_raw=CONTRACT.read_bytes(),REFERENCE.read_bytes()
    contract,reference=json.loads(contract_raw),json.loads(reference_raw)
    for name,pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert hashlib.sha256(reference_raw).hexdigest()==contract['reference_contract_sha256']
    sys.path.insert(0,str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    from draw_coordinates import snapshot
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    owned=importlib.import_module('skimagexx._bezier_curve')
    public=importlib.import_module('skimage.draw')
    original=importlib.import_module('skimage.draw.draw')
    compiled=importlib.import_module('skimage.draw._draw')
    runtime=reference['original_profiles'][np.__version__]
    assert skimage.__version__=='0.26.0' and np.dtype(np.intp).itemsize==8
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==runtime['source_canonical_lf_sha256']
    binary=hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest()
    assert binary==runtime['binary_sha256']
    pools={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(pools.values())=={'1'} and xx.get_num_threads()==1
    xx.set_fallback_enabled(False)
    native,upstream=xx.draw.bezier_curve,public.bezier_curve
    assert native is owned.bezier_curve
    assert str(inspect.signature(native))==str(inspect.signature(upstream))==reference['signature']
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    before=source_fingerprint(ROOT)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    getter=xx._core.get_num_threads
    entries={getattr(xx._core,name):name for name in contract['native_entries']}
    assert all(inspect.isbuiltin(entry) for entry in entries)
    original_codes={original.bezier_curve.__code__}
    original_compiled={compiled._bezier_curve,compiled._bezier_segment,compiled._line}
    rows=[]
    for index,spec in enumerate(contract['cases']):
        positional,options,bounds=fixture(spec);input_state=json.dumps((positional,options))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');expected=upstream(*positional,**options)
        assert not notices
        calls=[];snapshots=[];accepted=[];clip_accepted=[]
        def denied(*a,**k):raise AssertionError('Native Bezier forwarded to original algorithm')
        def profile(frame,event,value):
            assert not(event=='call' and frame.f_code in original_codes)
            assert not(event=='c_call' and value in original_compiled)
            if event=='c_return' and value in entries:calls.append(entries[value])
            if event=='c_return' and value is getter:snapshots.append('get_num_threads')
            if event=='return' and frame.f_code is native.__code__:
                accepted.append(frame.f_locals.get('native_result') is not None)
                clip_accepted.append(frame.f_locals.get('clipped'))
        saved=[(public,'bezier_curve',public.bezier_curve),(original,'bezier_curve',original.bezier_curve),
            (original,'_bezier_curve',original._bezier_curve),(compiled,'_bezier_curve',compiled._bezier_curve),
            (compiled,'_bezier_segment',compiled._bezier_segment),(compiled,'_line',compiled._line),(_compat,'fallback',_compat.fallback)]
        prior=sys.getprofile()
        try:
            for obj,name,old in saved:setattr(obj,name,denied)
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native(*positional,**options)
        finally:
            sys.setprofile(prior)
            for obj,name,old in saved:setattr(obj,name,old)
        assert not notices and calls==contract['native_entries'] and snapshots==['get_num_threads'] and accepted==[True]
        assert clip_accepted==[spec['shape'] is not None]
        output=checked_output(actual,expected,np,snapshot,bounds)
        assert json.dumps((positional,options))==input_state
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        timing=measure(lambda:upstream(*positional,**options),lambda:native(*positional,**options),
            args.repeat,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert json.dumps((positional,options))==input_state
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        rows.append(dict(case_id=spec['id'],api=spec['api'],parameters=spec,bounds=bounds,threads=1,
            actual_C_returns=calls,global_worker_snapshots=snapshots,native_accepted=True,native_clipping=clip_accepted[0],
            output=output,error=None,warnings=[],max_abs_error=0.,**timing))
    assert len(rows)==12 and sum(sum(map(len,r['samples_ms'])) for r in rows)==168
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    assert CONTRACT.read_bytes()==contract_raw and REFERENCE.read_bytes()==reference_raw
    cpu=platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu=next((s.partition(':')[2].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu)
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        cpu_model=cpu,logical_cpu_count=os.cpu_count(),avx2_available=bool(xx.has_avx2()),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**before,core_sha256=core_sha,
        original_draw_binary_sha256=binary,threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,
        raw_samples=168,reference_pool_limits=pools,fallback_enabled=False,deadline_utc=deadline,
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),method_sha256=contract['method_sha256'],scope=contract['scope']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Bezier curve: initial one-worker comparison','',contract['scope'],'',
        '| Case | Original ms | C++ ms | Original/C++ |','|---|---:|---:|---:|']
    lines += [f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |" for r in rows]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
