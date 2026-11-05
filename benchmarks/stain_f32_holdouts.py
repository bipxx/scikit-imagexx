"""Four fixed legacy stain holdouts; no numerical work on import."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import warnings


def fixture(spec,np,reference):
    assert spec['operation'] in ('combine_stains','separate_stains')
    assert spec['k'] in (1,3,7) and spec['matrix'] in ('custom','rgb_from_hed')
    image=np.random.default_rng(31).integers(0,256,(512,512,3),dtype=np.uint8)
    image=(image/255).astype(np.float32)
    matrix=(np.random.default_rng(spec['k']).uniform(-1,1,(3,spec['k'])).astype(np.float32)
            if spec['matrix']=='custom' else reference.rgb_from_hed)
    assert image.size==786432 and matrix.size<=21 and 512*512*spec['k']<=1835008
    return image,matrix


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args(argv);root=a.source_root.resolve();raw=a.contract.read_bytes();contract=json.loads(raw)
    assert contract['schema']=='region-stain-same-host-v1'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(a.deadline_utc)
    journal=a.output.with_suffix('.batches.jsonl')
    assert not a.output.exists() and not journal.exists()
    sys.path[:0]=[str(root/'benchmarks'),str(root/'tools')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    from draw_coordinates import snapshot
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimage import color as reference
    from skimagexx import _compat
    from skimagexx.coverage import parity_tolerance
    owned=importlib.import_module('skimagexx.color');original=importlib.import_module('skimage.color.colorconv')
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    assert parity_tolerance('color.combine_stains',np.dtype('float32'))==(2e-6,2e-6)
    assert parity_tolerance('color.separate_stains',np.dtype('float32'))==(2e-6,2e-6)
    xx.set_fallback_enabled(False);entry=xx._core.stain_transform
    assert inspect.isbuiltin(entry)
    before=source_fingerprint(root);core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    for index,spec in enumerate(contract['holdout_cases']):
        image,matrix=fixture(spec,np,reference);input_state=snapshot((image,matrix))
        upstream,native=getattr(reference,spec['operation']),getattr(owned,spec['operation'])
        assert str(inspect.signature(native))==str(inspect.signature(upstream))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');expected=upstream(image,matrix)
        assert not notices
        calls=[];accepted=[]
        def denied(*args,**kwargs):raise AssertionError('native stain forwarded numerical algorithm')
        def profile(frame,event,value):
            if event=='call' and frame.f_globals.get('__name__')=='skimage.color.colorconv':
                raise AssertionError('original color algorithm in native validation')
            if event=='c_return' and value is entry:calls.append('stain_transform')
            if event=='return' and frame.f_code is owned._stain.__code__:
                accepted.append(frame.f_locals['moved'].dtype.type is np.float32 and type(frame.f_locals.get('result')) is np.ndarray)
        old_profile=sys.getprofile();prior=(getattr(reference,spec['operation']),getattr(original,spec['operation']),owned.fallback,_compat.fallback)
        try:
            setattr(reference,spec['operation'],denied);setattr(original,spec['operation'],denied)
            owned.fallback=_compat.fallback=denied;sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native(image,matrix)
        finally:
            sys.setprofile(old_profile);setattr(reference,spec['operation'],prior[0]);setattr(original,spec['operation'],prior[1]);owned.fallback,_compat.fallback=prior[2:]
        assert not notices and calls==['stain_transform'] and accepted==[True]
        assert type(actual) is type(expected) is np.ndarray and actual.dtype==expected.dtype
        assert actual is not expected and not np.shares_memory(actual,expected)
        assert all(actual is not v and not np.shares_memory(actual,v) for v in (image,matrix))
        assert actual.flags.owndata and actual.base is None and actual.flags.writeable and actual.flags.aligned
        assert actual.size<=1835008
        left,right=snapshot((actual,)),snapshot((expected,))
        left_raw=left['arrays'][0].pop('raw');right_raw=right['arrays'][0].pop('raw')
        assert left==right
        rtol,atol=parity_tolerance('color.'+spec['operation'],image.dtype)
        assert (rtol,atol)==(2e-6,2e-6)
        atol*=max(1.,float(np.max(np.sum(np.abs(matrix),axis=0))))
        np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
        error=float(np.max(np.abs(actual-expected)))
        assert snapshot((image,matrix))==input_state
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        timing=measure(lambda:upstream(image,matrix),lambda:native(image,matrix),7,100,
            journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=a.deadline_utc)
        assert snapshot((image,matrix))==input_state
        assert np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        rows.append(dict(case_id=spec['id'],parameters=spec,threads=1,backend='native',actual_C_returns=calls,native_accepted=True,
            dtype=image.dtype.name,output_dtype=actual.dtype.name,original_output_dtype=expected.dtype.name,output=left,
            output_sha256=hashlib.sha256(left_raw).hexdigest(),original_output_sha256=hashlib.sha256(right_raw).hexdigest(),rtol=rtol,atol=atol,max_abs_error=error,**timing))
    assert source_fingerprint(root)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and a.contract.read_bytes()==raw
    cpu_model=platform.processor();cpu_info=Path('/proc/cpuinfo')
    if cpu_info.is_file():cpu_model=next((s.partition(':')[2].strip() for s in cpu_info.read_text().splitlines() if s.startswith('model name')),cpu_model)
    metadata=dict(**before,core_sha256=core_sha,contract_sha256=hashlib.sha256(raw).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        method_sha256=hashlib.sha256(inspect.getsource(measure).encode()).hexdigest(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,
        cpu_model=cpu_model,logical_cpu_count=os.cpu_count(),avx2_available=bool(xx.has_avx2()),threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,raw_samples=56,
        reference_pool_limits=limits,fallback_enabled=False,deadline_utc=a.deadline_utc,timestamp_utc=datetime.now(timezone.utc).isoformat())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(metadata=metadata,results=rows),indent=2)+'\n')


if __name__=='__main__':main()
