"""Laplace/Roberts layout parity and allocation-inclusive reference timings."""
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
    parser.add_argument('--output',type=Path,default=Path('benchmarks/edge-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat,*args.threads])<1:parser.error('size, repeats and workers must be positive')
    xx.set_fallback_enabled(False)
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
    rows=[];rng=np.random.default_rng(417)
    names=('laplace','roberts','roberts_pos_diag','roberts_neg_diag')
    real=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

    def sample(dtype):
        dtype=np.dtype(dtype);shape=(args.size,args.size)
        if dtype.kind=='b':return rng.random(shape)>.5
        if dtype.kind in 'iu':
            info=np.iinfo(dtype);return rng.integers(int(info.min),int(info.max)+1,shape,dtype=dtype)
        return rng.normal(size=shape).astype(dtype)

    def add(name,image,workload,options=None,layout='contiguous'):
        options={} if options is None else options
        reference,native=getattr(original,name),getattr(xx.filters,name)
        with warnings.catch_warnings(record=True) as expected_notices:
            warnings.simplefilter('always');expected=reference(image,**options)
        notices_expected=[(type(w.message).__name__,str(w.message)) for w in expected_notices]
        tolerance=2e-6 if expected.dtype==np.float32 else 2e-13
        for workers in args.threads:
            xx.set_num_threads(workers)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native(image,**options)
            assert notices_expected==[(type(w.message).__name__,str(w.message)) for w in notices]
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,15)
            rows.append({'api':'filters.'+name,'size':args.size,'shape':list(image.shape),
                         'dtype':str(image.dtype),'output_dtype':str(actual.dtype),'threads':workers,
                         'backend':'native','layout':layout,'workload':workload,
                         'parameter_repr':{key:(f'array(shape={value.shape},dtype={value.dtype})' if isinstance(value,np.ndarray) else repr(value)) for key,value in options.items()},
                         'rtol':tolerance,'atol':tolerance,'warnings':notices_expected,
                         'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),**timing})
            print(f'{name:18} {str(image.dtype):7} {workload:22} t={workers}: {timing["speedup"]:.2f}x',flush=True)

    for dtype in real:
        image=sample(dtype)
        for layout,view in {'contiguous':image,'reversed':image[::-1,::-1],
                            'strided':np.tile(image,(2,2))[::2,::2]}.items():
            for name in names:add(name,view,layout,layout=layout)
        if dtype in (np.float32,np.float64):
            storage=np.empty(image.nbytes+1,dtype=np.uint8)
            unaligned=np.ndarray(image.shape,dtype=dtype,buffer=storage,offset=1);unaligned[:]=image
            for layout,view in {'fortran':np.asfortranarray(image),'unaligned':unaligned,
                                'single-row':image[:1,:],'single-column':image[:,:1]}.items():
                for name in names:add(name,view,layout,layout=layout)
        if dtype in (np.uint8,np.float32,np.float64):
            dimensions={'volume':np.stack((image,image[::-1]),axis=0),
                        'singleton-tail':image.reshape(*image.shape,1),
                        'singleton-head':image.reshape(1,*image.shape),'line':image.reshape(-1)}
            for layout,view in dimensions.items():add('laplace',view,layout,{'ksize':5},layout)
            mask=np.ones(image.shape,dtype=bool)
            mask[args.size//4:args.size//2,args.size//3:2*args.size//3]=False
            masks={'mask-bool':mask,'mask-float-reverse':mask.astype(np.float32)[::-1,::-1],
                   'mask-broadcast':np.ones((args.size,),dtype=bool)}
            for label,value in masks.items():
                for name in names:add(name,image,label,{'mask':value})
    for dtype in ('>i2','>u2','>i4','>u4','>i8','>u8','>f2','>f4','>f8'):
        image=sample(np.dtype(dtype).newbyteorder('=')).astype(dtype)
        for name in names:add(name,image,'endian')
    assert len(rows)==260*len(args.threads)
    assert fingerprint==source_fingerprint(root),'source changed during measurement'
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
              'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
              'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,
              'repeat':args.repeat,**fingerprint,
              'scope':'Warmed allocation-inclusive API; dtype/shape/warnings and bounded numerical parity before every measurement'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metadata':metadata,'results':rows},indent=2)+'\n',encoding='utf-8')
    lines=['# Small edge-filter performance','','Original scikit-image '+skimage.__version__+
           '. Reference thread pools use one worker; native workers are explicit. Timings include validation, '
           'dtype/endian conversion, input copies, reflected boundaries, mask erosion/broadcast handling, arithmetic and output allocation. '
           'All raw samples and slower cases are retained in JSON.','',
           '| API | Input | Shape | Workload | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape']))
        lines.append(f'| {row["api"]} | {row["dtype"]} | {shape} | {row["workload"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
