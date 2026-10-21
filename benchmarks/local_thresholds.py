"""Niblack/Sauvola parity and allocation-inclusive reference performance."""
from run import measure
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
from skimage import filters as original
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/local-threshold-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat,*args.threads])<1:parser.error('size, repeats and workers must be positive')
    xx.set_fallback_enabled(False)
    names=('threshold_niblack','threshold_sauvola');rows=[];rng=np.random.default_rng(93)

    def add(name,image,workload,options=None,layout='contiguous'):
        options={} if options is None else options
        reference=getattr(original,name);native=getattr(xx.filters,name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');expected=reference(image,**options)
        reference_notices=[(type(w.message).__name__,str(w.message)) for w in caught]
        tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');actual=native(image,**options)
            assert reference_notices==[(type(w.message).__name__,str(w.message)) for w in caught]
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,15)
            window=options.get('window_size',15)
            if np.ndim(window)==0:window=[window]*image.ndim
            rows.append({'api':'filters.'+name,'size':args.size,'shape':list(image.shape),
                'dtype':str(image.dtype),'output_dtype':str(actual.dtype),'threads':workers,'backend':'native',
                'layout':layout,'workload':workload,'window':list(map(int,window)),
                'parameter_repr':{key:repr(value) for key,value in options.items()},
                'rtol':tolerance,'atol':tolerance,'warnings':reference_notices,
                'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),**timing})
            print(f'{name:19} {str(image.dtype):7} {workload:20} t={workers}: {timing["speedup"]:.2f}x',flush=True)

    for dtype in (np.uint8,np.uint16,np.int16,np.float32,np.float64):
        if np.issubdtype(dtype,np.integer):
            info=np.iinfo(dtype);a=rng.integers(int(info.min),int(info.max)+1,(args.size,args.size),dtype=dtype)
        else:a=rng.normal(size=(args.size,args.size)).astype(dtype)
        for window in (15,3,31,127,(3,31)):
            for name in names:add(name,a,'window-'+str(window),{'window_size':window})
        layouts={'reversed':a[::-1,::-1],'strided':np.tile(a,(2,2))[::2,::2],
                 'fortran':np.asfortranarray(a),'volume':a.reshape(1,*a.shape)}
        for layout,image in layouts.items():
            options={'window_size':(1,15,15)} if layout=='volume' else {}
            for name in names:add(name,image,layout,options,layout)
        if dtype in (np.float32,np.float64):
            for name in names:add(name,a,'int64-window',{'window_size':np.int64(15)})
            for k in (0.,-.2,np.float32(.17),np.float64(.17)):
                for name in names:add(name,a,'k-'+repr(k),{'k':k})
            for r in (.5,128.,np.float64(.5)):add('threshold_sauvola',a,'r-'+repr(r),{'r':r})
    for dtype in (np.bool_,np.int8,np.uint32,np.int32,np.uint64,np.int64):
        if dtype==np.bool_:a=rng.random((args.size,args.size))>.5
        else:
            info=np.iinfo(dtype);a=rng.integers(int(info.min),int(info.max)+1,(args.size,args.size),dtype=dtype)
        for name in names:add(name,a,'additional-width')
    root=Path(__file__).resolve().parents[1]
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
        'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'repeat':args.repeat,
        **source_fingerprint(root),'scope':'Warmed allocation-inclusive Python API; raw numeric input range; exact shape/dtype and bounded numerical parity before every timing'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metadata':metadata,'results':rows},indent=2)+'\n',encoding='utf-8')
    lines=['# Local threshold performance','',
        'Reference: original scikit-image '+skimage.__version__+'. Reference thread pools use one worker; native workers are explicit. '
        'Each case checks dtype, shape, warnings and numerical parity with fallback disabled. '
        'Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. '
        'Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. '
        'All raw samples and slower cases are retained in the JSON.','',
        '| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape']))
        lines.append(f'| {row["api"]} | {shape} | {row["dtype"]} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
