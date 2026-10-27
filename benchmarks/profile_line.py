"""Fixed eighteen-case public profile sampling comparison; checks are untimed."""
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
    for side in (256,1024):
        src=(.25,1.5);dst=(side-1.25,side-1.75)
        length=math.ceil(math.hypot(dst[0]-src[0],dst[1]-src[1])+1)
        for dtype in ('uint8','float32','float64'):
            for width in (1,9,33):
                rows.append(dict(case_id=f'side{side}:{dtype}:width{width}:length{length}',
                    shape=[side,side],dtype=dtype,src=src,dst=dst,linewidth=width,
                    profile_length=length,sampled_values=length*width))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-profile-line.json'))
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
    reference,native=skimage.measure.profile_line,xx.measure.profile_line
    flip_code=inspect.unwrap(np.flip).__code__
    sample_code=native.__globals__['_sample_plane'].__code__
    rows=[]
    for spec in case_specs():
        gray=np.random.default_rng(2026).integers(0,256,spec['shape'],dtype=np.uint8)
        image=gray if spec['dtype']=='uint8'else(gray/255).astype(spec['dtype'])
        assert image.dtype.name==spec['dtype']
        length=int(np.ceil(np.hypot(*(np.asarray(spec['dst'])-spec['src']))+1))
        assert length==spec['profile_length']
        call_args=(image,spec['src'],spec['dst'])
        options=dict(linewidth=spec['linewidth'])
        input_sha=hashlib.sha256(image.tobytes()).hexdigest()
        captured=[];outputs=[];returns=[];accepted=[];prior=sys.getprofile()
        def profile(frame,event,arg):
            if event=='return'and frame.f_code is flip_code:
                # Snapshot immediately; never hold the live pre-reducer array.
                captured.append(dict(shape=list(arg.shape),dtype=arg.dtype.str,
                    strides=list(arg.strides),own=bool(arg.flags.owndata),
                    writeable=bool(arg.flags.writeable),base_type=type(arg.base).__name__,
                    raw=arg.tobytes()))
            if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='profile_sample':
                returns.append(arg.__name__)
            if event=='return'and frame.f_code is sample_code:
                accepted.append(frame.f_locals.get('native_pixels')is not None)
        sys.setprofile(profile)
        try:
            for call in(reference,native):outputs.append(call(*call_args,**options))
        finally:sys.setprofile(prior)
        expected,actual=outputs
        assert returns==['profile_sample'] and accepted==[True] and len(captured)==2
        assert captured[0]==captured[1] and captured[0]['shape']==[length,spec['linewidth']]
        assert type(actual)is type(expected)is np.ndarray
        assert (actual.shape,actual.dtype,actual.strides,actual.flags.owndata,actual.flags.writeable,type(actual.base))==(
            expected.shape,expected.dtype,expected.strides,expected.flags.owndata,expected.flags.writeable,type(expected.base))
        assert actual.tobytes()==expected.tobytes()
        assert hashlib.sha256(image.tobytes()).hexdigest()==input_sha
        pixel_sha=hashlib.sha256(captured[0].pop('raw')).hexdigest()
        timing=measure(lambda:reference(*call_args,**options),lambda:native(*call_args,**options),args.repeat,19)
        assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        assert hashlib.sha256(image.tobytes()).hexdigest()==input_sha
        rows.append(dict(**spec,backend='native',threads=1,input_dtype_str=image.dtype.str,
            input_sha256=input_sha,pre_reducer_metadata=captured[0],sample_sha256=pixel_sha,
            output_dtype_str=actual.dtype.str,output_shape=list(actual.shape),output_strides=list(actual.strides),
            output_raw_hex=actual.tobytes().hex(),reference_raw_hex=expected.tobytes().hex(),
            actual_C_returns=returns,native_accepted=True,max_abs_error=0.0,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==18 and all([len(s)for s in row['samples_ms']]==[7,7]for row in rows)
    source_after=source_fingerprint(root);assert source_after==source_before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    scope=('Eighteen positive 2D cases: sides256/1024, uint8/float32/float64, linewidth1/9/33. '
           'Seed2026 uint8 images are normalized by255 before float storage, outside timing. '
           'Length is ceil(hypot(dst-src)+1); workload counts sampled values, not image pixels. '
           'Public coordinates, storage preparation, typed sampling, flip and literal mean are timed. '
           'Untimed exact sampled-byte and public metadata/word checks precede seven paired rounds. '
           'Only the assessed finite default order1/reflect routes are measured; coordinate construction '
           'and width reduction retain literal NumPy providers. This is a first original/current comparison, '
           'not a prior-C++ gain or full-API qualification.')
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,threads=1,repeat=7,raw_samples=252,
        reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,
        selected_apis=['measure.profile_line'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Profile line: one-thread comparison','',scope,'',
           '| Case | Sampled values | Original ms | Current ms | Original/current | Max error |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['sampled_values']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
