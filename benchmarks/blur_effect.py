"""Fixed twenty-four-case public blur comparison; exact checks are untimed."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys


def case_specs():
    rows=[]
    shapes=((4096,),(65536,),(256,256),(1024,1024),(32,32,32),(64,64,64))
    for shape,h_size in [*((s,11)for s in shapes),((256,256),1),((1024,1024),21)]:
        for dtype in ('uint8','float32','float64'):
            rows.append(dict(case_id='x'.join(map(str,shape))+f':{dtype}:h{h_size}',
                             shape=list(shape),dtype=dtype,h_size=h_size,
                             axes=len(shape),pixels=math.prod(shape)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-blur-effect.json'))
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
    reference,native=skimage.measure.blur_effect,xx.measure.blur_effect
    append_lines={}
    for call in(reference,native):
        lines,first=inspect.getsourcelines(call)
        found=[first+i for i,line in enumerate(lines)if 'B.append('in line]
        assert len(found)==1
        append_lines[call.__code__]=found[0]
    stages=('filt_im','im_sharp','im_blur','T','M1','M2')
    # Preserve scalar types and full array storage metadata, not just values.
    def snapshot(value):
        assert isinstance(value,(np.ndarray,np.generic))
        array=np.asarray(value)
        return dict(type=type(value).__name__,dtype=array.dtype.str,shape=list(array.shape),
                    strides=list(array.strides),own=bool(array.flags.owndata),
                    writeable=bool(array.flags.writeable),base_type=type(array.base).__name__,
                    raw=array.tobytes())
    rows=[];total_uniform=0;total_sparse=0
    for spec in case_specs():
        gray=np.random.default_rng(2026).integers(0,256,spec['shape'],dtype=np.uint8)
        image=gray if spec['dtype']=='uint8'else(gray/255).astype(spec['dtype'])
        assert image.dtype.name==spec['dtype'] and image.flags.c_contiguous
        options=dict(h_size=spec['h_size'])
        input_sha=hashlib.sha256(image.tobytes()).hexdigest()
        captured={reference.__code__:[],native.__code__:[]};returns=[];accepted=[]
        prior_trace,prior_profile=sys.gettrace(),sys.getprofile()
        def trace(frame,event,arg):
            if frame.f_code in append_lines:
                if event=='line'and frame.f_lineno==append_lines[frame.f_code]:
                    local=frame.f_locals
                    captured[frame.f_code].append(dict(axis=local['ax'],**{
                        name:snapshot(local[name])for name in stages}))
                    if frame.f_code is native.__code__:accepted.append(local['owned'])
                return trace
            return None
        def profile(frame,event,arg):
            if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__:
                name=getattr(arg,'__name__',None)
                if name in('uniform_filter1d','correlate_sparse'):returns.append(name)
        sys.settrace(trace);sys.setprofile(profile)
        try:
            expected=reference(image,**options);actual=native(image,**options)
        finally:
            sys.settrace(prior_trace);sys.setprofile(prior_profile)
        assert accepted==[True]*spec['axes']
        assert returns==['uniform_filter1d','correlate_sparse','correlate_sparse']*spec['axes']
        expected_stages,actual_stages=captured[reference.__code__],captured[native.__code__]
        assert len(actual_stages)==spec['axes'] and actual_stages==expected_stages
        expected_record,actual_record=snapshot(expected),snapshot(actual)
        assert actual_record==expected_record
        assert actual.tobytes()==expected.tobytes()
        assert hashlib.sha256(image.tobytes()).hexdigest()==input_sha
        stage_records=[]
        for axis in actual_stages:
            record=dict(axis=axis['axis'])
            for name in stages:
                payload=axis[name].pop('raw')
                record[name]=dict(**axis[name],sha256=hashlib.sha256(payload).hexdigest())
            stage_records.append(record)
        timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,19)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        assert hashlib.sha256(image.tobytes()).hexdigest()==input_sha
        total_uniform+=spec['axes'];total_sparse+=2*spec['axes']
        raw_actual=actual_record.pop('raw');raw_expected=expected_record.pop('raw')
        rows.append(dict(**spec,api='measure.blur_effect',backend='native',threads=1,
            input_dtype_str=image.dtype.str,input_sha256=input_sha,intermediates=stage_records,
            output_metadata=actual_record,output_raw_hex=raw_actual.hex(),reference_raw_hex=raw_expected.hex(),
            actual_C_returns=returns,native_accepted=True,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==24 and total_uniform==48 and total_sparse==96
    assert all([len(s)for s in row['samples_ms']]==[7,7]for row in rows)
    source_after=source_fingerprint(root);assert source_after==source_before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    scope=('Twenty-four positive rank1/2/3 cases, uint8/float32/float64, default h_size11 plus '
           'h_size1 and21 controls. Seed2026 uint8 input is normalized by255 for float storage outside timing. '
           'Public conversion, uniform filtering, Sobel sampling, allocations, clamps, cropped sums, ratio '
           'and final default np.max are timed. Exact typed six-intermediate snapshots and public scalar '
           'metadata/words precede seven paired rounds; 48 accepted uniform and96 sparse stages are checked. '
           'h_size1 retains the ordered rolling recurrence. Literal NumPy clamps, cropped reductions and '
           'final reducer remain providers, including the installed NumPy EPSILON promotion rules. '
           'This first original/current comparison is not a prior-C++ gain or full-API qualification.')
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,threads=1,repeat=7,raw_samples=336,
        accepted_uniform_stages=total_uniform,accepted_sparse_stages=total_sparse,
        reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,
        selected_apis=['measure.blur_effect'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Blur effect: one-thread comparison','',scope,'',
           '| Case | Pixels | Original ms | Current ms | Original/current | Max error |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['pixels']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
