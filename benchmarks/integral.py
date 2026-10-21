"""Ordered integral-image parity and allocation-inclusive original-library timings."""
from run import measure
import argparse
from datetime import datetime,timezone
import json
import hashlib
import sys
from pathlib import Path
import platform
import warnings
import numpy as np
import scipy
import skimage
from skimage import transform as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--baseline-dir',type=Path,help='Directory with preserved _baseline package and baseline-source.json')
    parser.add_argument('--output',type=Path,default=Path('benchmarks/integral-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat,*args.threads])<1:parser.error('size, repeats and workers must be positive')
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
    xx.set_fallback_enabled(False);rows=[];rng=np.random.default_rng(544);replacement=xx.transform
    baseline=None;baseline_provenance=None
    if args.baseline_dir:
        sys.path.insert(0,str(args.baseline_dir.resolve()))
        import _baseline as baseline
        baseline.set_fallback_enabled(False)
        baseline_provenance=json.loads((args.baseline_dir/'baseline-source.json').read_text(encoding='utf-8'))
        binaries=[*Path(baseline.__file__).parent.glob('*.pyd'),*Path(baseline.__file__).parent.glob('*.so')]
        assert binaries,'preserved baseline binary is missing'
        for file in binaries:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==baseline_provenance['binary_sha256'][file.name]
        assert baseline.has_avx2()==xx.has_avx2(),'baseline/current ISA settings differ'
    baseline_transform=None if baseline is None else baseline.transform
    real=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

    def sample(dtype):
        dtype=np.dtype(dtype);shape=(args.size,args.size)
        if dtype.kind=='b':return rng.random(shape)>.5
        if dtype.kind in 'iu':
            limits=np.iinfo(dtype)
            return rng.integers(int(limits.min),int(limits.max)+1,shape,dtype=dtype)
        return rng.normal(size=shape).astype(dtype)

    def add(image,workload,dtype=None):
        def reference():return original.integral_image(image,dtype=dtype)
        def native():return replacement.integral_image(image,dtype=dtype)
        def previous():return baseline_transform.integral_image(image,dtype=dtype)
        with warnings.catch_warnings(record=True) as expected_notices:
            warnings.simplefilter('always');expected=reference()
        expected_warnings=[(type(w.message).__name__,str(w.message)) for w in expected_notices]
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native()
            assert [(type(w.message).__name__,str(w.message)) for w in notices]==expected_warnings
            assert type(actual) is type(expected)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.strides==expected.strides
            assert actual.dtype.metadata==expected.dtype.metadata
            np.testing.assert_array_equal(actual,expected,strict=True)
            if expected.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
            if baseline is not None:
                baseline.set_num_threads(workers)
                with warnings.catch_warnings(record=True) as baseline_notices:
                    warnings.simplefilter('always');old=previous()
                assert [(type(w.message).__name__,str(w.message)) for w in baseline_notices]==expected_warnings
                assert old.dtype==expected.dtype and old.dtype.metadata==expected.dtype.metadata and old.strides==expected.strides
                np.testing.assert_array_equal(old,expected,strict=True)
                if old.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(old),np.signbit(expected))
            maximum_error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
            timing=measure(reference,native,args.repeat,15,baseline=previous if baseline is not None else None)
            rows.append({'api':'transform.integral_image','size':args.size,'shape':list(image.shape),
                         'dtype':str(image.dtype),'output_dtype':str(actual.dtype),'threads':workers,
                         'backend':'identity' if actual is image else 'native','layout':workload,'workload':workload,
                         'input_dtype_metadata':None if image.dtype.metadata is None else dict(image.dtype.metadata),
                         'output_dtype_metadata':None if actual.dtype.metadata is None else dict(actual.dtype.metadata),
                         'parameter_repr':{'dtype':repr(dtype)},'warnings':expected_warnings,
                         'max_abs_error':maximum_error,**timing})
            print(f'integral_image {str(image.dtype):7} {workload:22} t={workers}: {timing["speedup"]:.2f}x',flush=True)

    for dtype in real:
        image=sample(dtype)
        for label,view in [('contiguous',image),('fortran',np.asfortranarray(image)),
                           ('reversed',image[::-1,::-1]),('stepped',np.tile(image,(2,2))[::2,::2]),
                           ('line',image.reshape(-1)),('single-row',image[:1,:]),
                           ('single-column',image[:,:1]),('volume',np.stack((image,image[::-1])))]:add(view,label)
        add(image,'float64',np.float64)
        add(image.astype(np.dtype(dtype,metadata={'units':'raw','source':'benchmark'})),'dtype-metadata')
        if np.dtype(dtype).kind in 'biu':
            add(image,'int64',np.int64);add(image,'uint64',np.uint64)
        if dtype==np.float32:add(image,'float32',np.float32)
    for dtype in ('>i2','>u2','>i4','>u4','>i8','>u8','>f2','>f4','>f8'):
        add(sample(np.dtype(dtype).newbyteorder('=')).astype(dtype),'endian')
    assert len(rows)==148*len(args.threads)
    assert fingerprint==source_fingerprint(root),'source changed during measurement'
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
              'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
              'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'repeat':args.repeat,**fingerprint,
              'scope':'Allocation-inclusive public API; exact ordered sums, dtype descriptors, output strides and warnings checked before timing'}
    if baseline_provenance is not None:metadata['baseline']=baseline_provenance
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metadata':metadata,'results':rows},indent=2)+'\n',encoding='utf-8')
    lines=['# Integral-image performance','','Original scikit-image '+skimage.__version__+
           '. Reference thread pools use one worker; native worker limits are explicit. Raw casting, copies, validation, allocation and destruction are included. Every case requires exact numerical parity, matching output dtype/metadata, strides and warnings. All samples and slower results are retained in JSON.','',
           '| Input | Shape | Workload | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape']))
        lines.append(f'| {row["dtype"]} | {shape} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
