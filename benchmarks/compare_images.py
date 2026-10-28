"""Fixed eighteen-case public image comparison; exact checks are untimed."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys


def case_specs():
    return [dict(case_id=f'side{s}:{d}:{m}',shape=[s,s],dtype=d,method=m,n_tiles=[8,8])
            for s in (256,1024) for d in ('uint8','float32','float64')
            for m in ('diff','blend','checkerboard')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-compare-images.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx

    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    limits={name:os.environ[name]for name in
            ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(args.threads)
    root=Path(__file__).resolve().parents[1];source_before=source_fingerprint(root)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    reference,native=skimage.util.compare_images,xx.util.compare_images
    original_code,native_code=(inspect.unwrap(call).__code__ for call in (reference,native))
    def snapshot(value):
        assert type(value)is np.ndarray
        return dict(dtype=value.dtype.str,metadata=value.dtype.metadata,shape=list(value.shape),
                    strides=list(value.strides),own=bool(value.flags.owndata),
                    writeable=bool(value.flags.writeable),base_type=type(value.base).__name__,
                    raw=value.tobytes())
    rows=[]
    for spec in case_specs():
        images=[]
        for seed in (2026,2027):
            gray=np.random.default_rng(seed).integers(0,256,spec['shape'],dtype=np.uint8)
            images.append(gray if spec['dtype']=='uint8'else(gray/255).astype(spec['dtype']))
        assert all(image.dtype.name==spec['dtype'] and image.flags.c_contiguous for image in images)
        input_shas=[hashlib.sha256(image.tobytes()).hexdigest()for image in images]
        options=dict(method=spec['method'],n_tiles=tuple(spec['n_tiles']))
        captured={original_code:[],native_code:[]};returns=[];accepted=[]
        prior=sys.getprofile()
        def profile(frame,event,arg):
            if event=='return'and frame.f_code in captured:
                captured[frame.f_code].append({name:snapshot(frame.f_locals[name])for name in ('img1','img2')})
                if frame.f_code is native_code:
                    accepted.append(frame.f_locals.get('native_comparison')is not None)
            if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='compare_images':
                returns.append(arg.__name__)
        sys.setprofile(profile)
        try:
            expected=reference(*images,**options);actual=native(*images,**options)
        finally:sys.setprofile(prior)
        assert returns==['compare_images'] and accepted==[True]
        assert len(captured[original_code])==len(captured[native_code])==1
        assert captured[original_code]==captured[native_code]
        expected_record,actual_record=snapshot(expected),snapshot(actual)
        assert actual_record==expected_record and actual.tobytes()==expected.tobytes()
        assert [hashlib.sha256(image.tobytes()).hexdigest()for image in images]==input_shas
        converted={}
        for name,record in captured[native_code][0].items():
            payload=record.pop('raw');converted[name]=dict(**record,sha256=hashlib.sha256(payload).hexdigest())
        timing=measure(lambda:reference(*images,**options),lambda:native(*images,**options),args.repeat,19)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        assert [hashlib.sha256(image.tobytes()).hexdigest()for image in images]==input_shas
        raw_actual=actual_record.pop('raw');raw_expected=expected_record.pop('raw')
        rows.append(dict(**spec,api='util.compare_images',backend='native',threads=1,
            input_dtype_str=[image.dtype.str for image in images],input_sha256=input_shas,
            converted_inputs=converted,output_metadata=actual_record,
            output_sha256=hashlib.sha256(raw_actual).hexdigest(),
            reference_sha256=hashlib.sha256(raw_expected).hexdigest(),actual_C_returns=returns,native_accepted=True,
            max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==18 and all([len(s)for s in row['samples_ms']]==[7,7]for row in rows)
    source_after=source_fingerprint(root);assert source_after==source_before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    scope=('Eighteen positive C-order 2D cases: sides256/1024, uint8/float32/float64, '
           'diff/blend/checkerboard with default8x8 tiles. Independent seed2026/2027 uint8 images; '
           'float normalization outside timing. Public decorator, conversion, method/tile validation, '
           'storage preparation, owned arithmetic, allocation and output destruction are timed. '
           'Untimed exact converted-input storage/words and public metadata/words precede seven paired rounds. '
           'Literal img_as_float remains an explicit conversion provider. Exceptional arithmetic and '
           'unassessed scalar promotion are outside this cohort. This first original/current comparison '
           'is not a prior-C++ gain or full supported-profile qualification.')
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,threads=1,repeat=7,raw_samples=252,
        reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,
        selected_apis=['util.compare_images'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Compare images: one-thread comparison','',scope,'',
           '| Case | Original ms | Current ms | Original/current | Max error |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
