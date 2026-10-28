"""Eight frozen default level-set generators; exact checks precede timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/level-set.json'


def case_specs():
    contract = json.loads(CONTRACT.read_bytes())
    selected = contract['future_benchmark']['cases']
    cases = {row['id']: row for row in contract['cases']}
    return [dict(case_id=name, api=cases[name]['api'], shape=cases[name]['shape'])
            for name in selected]


def checked_outputs(actual, expected, frozen, np):
    # Compare every word and all source storage metadata BEFORE exporting hashes.
    def snapshot(value):
        assert type(value) is np.ndarray
        return dict(type=type(value).__module__+'.'+type(value).__name__,
            dtype=value.dtype.str, dtype_metadata=value.dtype.metadata, shape=list(value.shape),
            strides=list(value.strides), own=bool(value.flags.owndata),
            c=bool(value.flags.c_contiguous), f=bool(value.flags.f_contiguous),
            writeable=bool(value.flags.writeable), base_type=type(value.base).__name__)
    actual_record, expected_record = snapshot(actual), snapshot(expected)
    frozen_record = {k:v for k,v in frozen.items() if k not in ('raw_sha256','raw_zlib_base64')}
    assert actual_record == expected_record == frozen_record
    assert actual.tobytes() == expected.tobytes()
    actual_sha = hashlib.sha256(actual.tobytes()).hexdigest()
    reference_sha = hashlib.sha256(expected.tobytes()).hexdigest()
    assert actual_sha == reference_sha == frozen['raw_sha256']
    return dict(output_metadata=actual_record, output_sha256=actual_sha,
                reference_sha256=reference_sha, frozen_reference_sha256=frozen['raw_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-level-set.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx

    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    limits={name:os.environ[name] for name in
            ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(args.threads)
    source_before=source_fingerprint(ROOT)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    contract_bytes=CONTRACT.read_bytes();contract=json.loads(contract_bytes)
    frozen={row['id']:row for row in contract['original_observations'][np.__version__]}
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    rows=[]
    for spec in case_specs():
        shape=tuple(spec['shape']);name=spec['api'].split('.')[-1]
        reference,native=getattr(skimage.segmentation,name),getattr(xx.segmentation,name)
        reference_code,native_code=(inspect.unwrap(call).__code__ for call in (reference,native))
        source_path=Path(inspect.getsourcefile(reference))
        assert hashlib.sha256(source_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['source']['canonical_lf_sha256']
        expected=reference(shape)
        returns=[];accepted=[];prior=sys.getprofile()
        entry='level_set_checkerboard' if name=='checkerboard_level_set' else 'level_set_disk'
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is reference_code:
                raise AssertionError('native generator forwarded to the original whole API')
            if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)==entry:
                returns.append(arg.__name__)
            if event=='return' and frame.f_code is native_code:
                accepted.append(frame.f_locals.get('native_result') is not None)
        sys.setprofile(profile)
        try:actual=native(shape)
        finally:sys.setprofile(prior)
        assert returns==[entry] and accepted==[True]
        output=checked_outputs(actual,expected,frozen[spec['case_id']]['result'],np)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(shape),lambda:native(shape),args.repeat,19)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,backend='native',threads=1,default_arguments=True,
            actual_C_returns=returns,native_accepted=True,max_abs_error=0.0,**output,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==8 and all([len(s) for s in row['samples_ms']]==[7,7] for row in rows)
    source_after=source_fingerprint(ROOT);assert source_after==source_before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==contract_bytes
    scope=('Eight frozen default calls: checkerboard_level_set and disk_level_set on shapes '
           '256x256,1024x1024,64x64x64,128x64x32. These generators accept shape tuples, '
           'not image buffers of different dtypes; outputs are int8. Full public default '
           'evaluation, normalization, native admission, allocation, owned scalar/SIMD geometry '
           'and output destruction are timed. Checker uses square_size=5; disk uses default '
           'integer center and radius=min(shape)*3/8, preserving sqrt/subtract/strict>0. '
           'Fresh reference results are verified against the original-first frozen metadata/SHA; '
           'full output bytes and metadata then match before timing, with hashes only exported. '
           'Unsupported protocols/floating centers/exceptional providers are outside this cohort. '
           'All raw samples and losses remain included. This original/current comparison is '
           'neither a prior-C++ gain nor full supported-profile qualification.')
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        original_source_sha256=contract['source']['canonical_lf_sha256'],
        numpy_error_policy=policy,numpy_buffer_elements=buffer,threads=1,repeat=7,target_ms=19,
        raw_samples=112,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),
        command=sys.argv,selected_apis=contract['apis'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Level-set generators: one-thread comparison','',scope,'',
           '| Case | Original ms | Current ms | Original/current | Max error |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
