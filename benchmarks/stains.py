"""All standard stain matrices and custom output widths versus scikit-image."""
from run import measure
import argparse
from datetime import datetime,timezone
from provenance import source_fingerprint
import json
from pathlib import Path
import platform
import numpy as np
import skimage
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--half-only',action='store_true',help='Only four frozen half-input cases; requires size512, workers1, repeat7 and fresh output')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/stain-parameters.json'))
    args=parser.parse_args()
    if args.half_only:
        return _half_main(args)
    if min([args.size,args.repeat]+args.threads)<1: parser.error('size, repeat and workers must be positive')
    xx.set_fallback_enabled(False)
    families=('hed','hdx','fgx','bex','rbd','gdx','hax','bro','bpx','ahx','hpx')
    results=[]
    for dtype in ('uint8','float32','float64'):
        image=np.random.default_rng(31).integers(0,256,(args.size,args.size,3),dtype=np.uint8)
        if dtype!='uint8': image=(image/255).astype(dtype)
        for name in ('separate_stains','combine_stains'):
            matrices=[(family,getattr(reference,family+'_from_rgb' if name=='separate_stains' else 'rgb_from_'+family)) for family in families]
            matrices += [(f'custom-{k}',np.random.default_rng(k).uniform(-1,1,(3,k)).astype(np.float32)) for k in (1,7)]
            old,current=getattr(reference,name),getattr(xx.color,name)
            for workload,matrix in matrices:
                expected=old(image,matrix)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual=current(image,matrix)
                    assert actual.dtype==expected.dtype
                    rtol,atol=parity_tolerance('color.'+name,np.dtype(dtype))
                    atol*=max(1,float(np.max(np.sum(np.abs(matrix),axis=0))))
                    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
                    timing=measure(lambda:old(image,matrix),lambda:current(image,matrix),args.repeat,15)
                    results.append({'api':'color.'+name,'dtype':dtype,'output_dtype':actual.dtype.name,'size':args.size,
                                    'threads':threads,'workload':workload,'layout':workload,'channels':matrix.shape[1],
                                    'matrix':matrix.tolist(),'rtol':rtol,'atol':atol,'backend':'native',
                                    'max_abs_error':float(np.max(np.abs(actual-expected))),**timing})
                    print(f'{name:16} {dtype:7} {workload:8} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root=Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    report={'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                        'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
                        **fingerprint,'avx2':xx.has_avx2(),'fallback_enabled':False,
                        'reference_thread_pools':1,'repeat':args.repeat},'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Stain parameter performance','',
           'All 11 standard stain families and custom 3×1/3×7 float32 matrices. '
           'Values and dtype are checked against original scikit-image before timing; fallback is disabled. '
           'Input precision and matrix norm determine the error budget, including float32 logarithms '
           'promoted to float64. Seven warmed allocation-inclusive samples; reference pools use one thread. '
           'Ratios below one remain visible.','',
           '| API | Input | Output | Matrix | Channels | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["output_dtype"]} | {r["workload"]} | {r["channels"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _half_main(args):
    """New half-only scope; legacy loops and sampler remain unchanged."""
    import hashlib
    import importlib
    import inspect
    import os
    import sys
    import warnings
    import scipy
    from datetime import timedelta
    root = Path(__file__).resolve().parents[1]
    assert args.size == 512 and args.threads == [1] and args.repeat == 7, 'half-only freezes size512/workers1/repeat7'
    assert args.output.resolve() != (root/'benchmarks/stain-parameters.json').resolve(), 'half-only requires a separate fresh --output'
    journal = args.output.with_suffix('.batches.jsonl')
    assert not args.output.exists() and not journal.exists()
    deadline = args.deadline_utc or (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc) < datetime.fromisoformat(deadline)
    sys.path.insert(0, str(root/'tools'))
    from montage_sustained import measure as sustained_measure
    from draw_coordinates import snapshot
    from skimagexx import _compat
    original = importlib.import_module('skimage.color.colorconv')
    utils = importlib.import_module('skimage._shared.utils')
    dtype_module = importlib.import_module('skimage.util.dtype')
    owned = importlib.import_module('skimagexx.color')
    contract_path = root/'coverage/contracts/combine-stains-half-f32-runtime-v1.json'
    reference_path = root/'coverage/contracts/combine-stains-half.json'
    contract_raw, reference_raw = contract_path.read_bytes(), reference_path.read_bytes()
    contract, saved = json.loads(contract_raw), json.loads(reference_raw)
    assert skimage.__version__ == '0.26.0' and np.__version__ in saved['original_provenance']
    assert hashlib.sha256(reference_raw).hexdigest() == contract['reference_contract_sha256']
    assert hashlib.sha256(inspect.getsource(sustained_measure).encode()).hexdigest() == contract['method_sha256']
    for module, name in ((original,'colorconv'),(utils,'utils'),(dtype_module,'dtype')):
        assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == saved['original_source_sha256'][name]
    for name, pin in contract['source_closure'].items():
        assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin
    limits = {key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values()) == {'1'} and xx.get_num_threads() == 1
    assert parity_tolerance('color.combine_stains',np.dtype('float16')) == (2e-6,2e-6)
    xx.set_fallback_enabled(False)
    native, upstream = xx.color.combine_stains, reference.combine_stains
    assert native is owned.combine_stains and xx.combine_stains is native
    assert str(inspect.signature(native)) == str(inspect.signature(upstream)) == saved['signature']
    entry = xx._core.stain_transform
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    core = Path(xx._core.__file__)
    core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        cpu_model = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    cpu_count, avx2_available = os.cpu_count(), bool(xx._core.has_avx2())
    before = source_fingerprint(root)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    base = np.random.default_rng(31).integers(0,256,(args.size,args.size,3),dtype=np.uint8)
    base = (base*(0.75/255.0)).astype(np.float16)
    rows = []
    for index, spec in enumerate(contract['cases']):
        image = base if spec['layout'] == 'C' else base[::-1,::-1,::-1]
        matrix = np.array(spec['matrix'],dtype=spec['matrix_dtype'])
        input_before = snapshot((image,matrix))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            expected = upstream(image,matrix)
        assert not notices
        calls, casts, accepted = [], [], []
        def denied(*unused_args, **unused_options):
            raise AssertionError('half combine_stains forwarded to provider')
        def profile(frame,event,value):
            if event == 'call' and frame.f_globals.get('__name__') == 'skimage.color.colorconv':
                raise AssertionError('original color algorithm in native validation')
            if event == 'c_return' and value is entry:
                calls.append('stain_transform')
            if event == 'c_call' and frame.f_code is owned._stain.__code__ and getattr(value,'__name__','') == 'astype':
                receiver = getattr(value,'__self__',None)
                if type(receiver) is np.ndarray and receiver.dtype.type is np.float16:
                    casts.append(receiver.dtype.str)
            if event == 'return' and frame.f_code is owned._stain.__code__:
                accepted.append(frame.f_locals['moved'].dtype.type is np.float32 and type(frame.f_locals.get('result')) is np.ndarray)
        previous = sys.getprofile()
        originals = reference.combine_stains, original.combine_stains, owned.fallback, _compat.fallback
        try:
            reference.combine_stains = original.combine_stains = denied
            owned.fallback = _compat.fallback = denied
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                actual = native(image,matrix)
        finally:
            sys.setprofile(previous)
            reference.combine_stains, original.combine_stains, owned.fallback, _compat.fallback = originals
        assert not notices and calls == ['stain_transform'] and casts == [np.dtype(np.float16).str] and accepted == [True]
        assert type(actual) is type(expected) is np.ndarray and actual.dtype == expected.dtype
        assert actual is not expected and not np.shares_memory(actual,expected)
        assert all(actual is not value and not np.shares_memory(actual,value) for value in (image,matrix))
        assert actual.flags.owndata and actual.base is None and actual.flags.writeable and actual.flags.aligned
        assert actual.size <= contract['max_output_elements']
        actual_output, expected_output = snapshot((actual,)), snapshot((expected,))
        actual_raw = actual_output['arrays'][0].pop('raw')
        expected_raw = expected_output['arrays'][0].pop('raw')
        assert actual_output == expected_output
        rtol, atol = parity_tolerance('color.combine_stains',image.dtype)
        assert (rtol,atol) == (2e-6,2e-6)
        atol *= max(1.,float(np.max(np.sum(np.abs(matrix),axis=0))))
        np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
        max_error = float(np.max(np.abs(actual-expected)))
        assert snapshot((image,matrix)) == input_before
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        timing = sustained_measure(lambda:upstream(image,matrix),lambda:native(image,matrix),
            args.repeat,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert snapshot((image,matrix)) == input_before
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        rows.append(dict(case_id=spec['id'],api='color.combine_stains',dtype='float16',output_dtype=actual.dtype.name,
            original_output_dtype=expected.dtype.name,size=args.size,threads=1,parameters=spec,backend='native',
            actual_C_returns=calls,half_normalizations=casts,native_accepted=True,output=actual_output,
            output_sha256=hashlib.sha256(actual_raw).hexdigest(),original_output_sha256=hashlib.sha256(expected_raw).hexdigest(),
            rtol=rtol,atol=atol,max_abs_error=max_error,**timing))
    assert len(rows) == 4 and sum(sum(map(len,row['samples_ms'])) for row in rows) == 56
    assert source_fingerprint(root) == before and hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    assert contract_path.read_bytes() == contract_raw and reference_path.read_bytes() == reference_raw
    report = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,cpu_model=cpu_model,
        logical_cpu_count=cpu_count,avx2_available=avx2_available,**before,core_sha256=core_sha,
        threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,raw_samples=56,
        deadline_utc=deadline,reference_pool_limits=limits,method_sha256=contract['method_sha256'],
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        scope=contract['scope'],selected_mode='half-only',fallback_enabled=False),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# combine_stains: half-input one-thread comparison','',contract['scope'],'',
        '| Case | Input | Original output | Current output | Original ms | C++ ms | Original/current |',
        '|---|---|---|---|---:|---:|---:|']
    lines.extend(f"| {row['case_id']} | float16 | {row['original_output_dtype']} | {row['output_dtype']} | {row['reference_ms']:.9f} | {row['cpp_ms']:.9f} | {row['speedup']:.9f}x |" for row in rows)
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
