"""Allocation-inclusive integral-window queries against original scikit-image."""
from run import measure
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
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
    parser.add_argument('--output',type=Path,default=Path('benchmarks/integrate-parameters.json'))
    args=parser.parse_args()
    if min([args.size,args.repeat,*args.threads])<1:parser.error('sizes, repeats and workers must be positive')
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
    xx.set_fallback_enabled(False);rows=[];rng=np.random.default_rng(882)
    real=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)

    def add(table,label,count=257):
        a=np.column_stack([rng.integers(0,n,count) for n in table.shape])
        b=np.column_stack([rng.integers(0,n,count) for n in table.shape])
        first,last=np.minimum(a,b),np.maximum(a,b)
        # Include boundary suppression and upstream's negative-index contract.
        if count>1:
            first[0]=0;last[0]=np.array(table.shape)-1
            first[1]-=np.array(table.shape);last[1]-=np.array(table.shape)
        def reference():return original.integrate(table,first,last)
        def native():return xx.transform.integrate(table,first,last)
        expected=reference()
        for workers in args.threads:
            xx.set_num_threads(workers);actual=native()
            assert type(actual) is type(expected)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.strides==expected.strides
            assert actual.dtype.metadata==expected.dtype.metadata
            np.testing.assert_array_equal(actual,expected,strict=True)
            np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
            timing=measure(reference,native,args.repeat,15)
            rows.append({'api':'transform.integrate','size':args.size,'shape':list(table.shape),
                         'dtype':str(table.dtype),'output_dtype':str(actual.dtype),'threads':workers,
                         'backend':'native','layout':label,'workload':label,'windows':count,
                         'input_dtype_metadata':None if table.dtype.metadata is None else dict(table.dtype.metadata),
                         'output_dtype_metadata':None,'max_abs_error':0.,**timing})
            print(f'integrate {str(table.dtype):7} {label:15} windows={count} t={workers}: {timing["speedup"]:.2f}x',flush=True)

    for dtype in real:
        descriptor=np.dtype(dtype);shape=(args.size,args.size)
        if descriptor.kind=='b':table=rng.random(shape)>.5
        elif descriptor.kind in 'iu':
            limits=np.iinfo(dtype);table=rng.integers(int(limits.min),int(limits.max)+1,shape,dtype=dtype)
        else:table=rng.normal(size=shape).astype(dtype)
        storage=np.empty(table.nbytes+1,dtype=np.uint8)
        unaligned=np.ndarray(shape,dtype=dtype,buffer=storage,offset=1);unaligned[...]=table
        readonly=table.view();readonly.flags.writeable=False
        for label,view in [('contiguous',table),('fortran',np.asfortranarray(table)),
                           ('reversed',table[::-1,::-1]),('stepped',np.tile(table,(2,2))[::2,::2]),
                           ('endian',table.astype(descriptor.newbyteorder('>'))),
                           ('dtype-metadata',table.view(np.dtype(dtype,metadata={'units':'raw'}))),
                           ('readonly',readonly),('unaligned',unaligned),
                           ('zero-stride',np.broadcast_to(table[:1],table.shape)),
                           ('line',table.reshape(-1)),('volume',np.stack((table,table[::-1])))]:add(view,label)
        if dtype in (np.uint8,np.float32,np.float64):
            for count in (1,17,4097,65539):add(table,f'batch-{count}',count)
    assert len(rows)==144*len(args.threads)
    assert fingerprint==source_fingerprint(root),'source changed during measurement'
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
              'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
              'avx2':xx.has_avx2(),'fallback_enabled':False,'reference_thread_pools':1,'repeat':args.repeat,**fingerprint,
              'scope':'Public raw integral-table corner queries; no image copying or preprocessing; exact corner order, float64 conversion and signed zero checked before timing'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metadata':metadata,'results':rows},indent=2)+'\n',encoding='utf-8')
    lines=['# Integral-window performance','','Original scikit-image '+skimage.__version__+
           '. Reference pools use one thread; native global worker limits are explicit. Inputs are raw integral tables. Coordinate validation, normalization, allocation and destruction are included; table construction is excluded. All raw samples and slower cases are retained.','',
           '| Input | Shape | Layout | Windows | Workers | Original ms | C++ ms | Speedup |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        shape='×'.join(map(str,row['shape']))
        lines.append(f'| {row["dtype"]} | {shape} | {row["layout"]} | {row["windows"]} | {row["threads"]} | {row["reference_ms"]:.4f} | {row["cpp_ms"]:.4f} | {row["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
