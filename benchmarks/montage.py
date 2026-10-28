"""Frozen real-batch Montage comparison; prepared native fill/copy credit only."""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,importlib,inspect,json,math,os,platform,struct,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/montage-benchmark.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def fixture(spec,np,index):
    rng=np.random.default_rng(2026+index)
    shape=tuple(spec['shape']);dtype=np.dtype(spec['dtype']);layout=spec['layout']
    if layout=='strided-reverse':
        owner=rng.integers(0,256,(shape[0],shape[1],shape[2]*2),dtype=np.uint16).astype(dtype)
        image=owner[:,::-1,::2]
    elif layout=='channels-first':
        image=rng.integers(0,256,(shape[0],shape[2],shape[3],shape[1]),dtype=np.uint16).astype(dtype)
        image=np.moveaxis(image,-1,1)
    else:
        image=rng.integers(0,256,shape,dtype=np.uint16).astype(dtype)
        if layout=='F':image=np.asfortranarray(image)
    assert image.shape==shape
    options=dict(fill=spec['fill'],grid_shape=tuple(spec['grid']),padding_width=spec['padding'],channel_axis=spec['channel_axis'])
    return image,options


def snapshot(value):
    return dict(dtype=value.dtype.str,metadata=value.dtype.metadata,shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__,raw=value.tobytes())


def oracle_tiles(logical_bytes,shape,spec):
    images,height,width=shape[:3];channels=1 if len(shape)==3 else shape[3]
    fmt={'uint8':'B','uint16':'H','float32':'f','float64':'d'}[spec['dtype']];itemsize=struct.calcsize('='+fmt)
    fill=spec['fill']
    if fill=='mean':
        assert spec['dtype']=='float64' and channels==1
        values=struct.iter_unpack('=d',logical_bytes)
        fill=float(Fraction(sum(int(v[0])for v in values),images*height*width))
    values=fill if isinstance(fill,list)else[fill]
    assert len(values)==channels
    pattern=b''.join(struct.pack('='+fmt,value)for value in values)
    rows,columns=spec['grid'];padding=spec['padding'];out_h=(height+padding)*rows+padding;out_w=(width+padding)*columns+padding
    result=bytearray(pattern*(out_h*out_w));span=width*channels*itemsize
    for image in range(images):
        tile_r,tile_c=divmod(image,columns)
        for row in range(height):
            source=(image*height+row)*span
            target=((padding+(height+padding)*tile_r+row)*out_w+padding+(width+padding)*tile_c)*channels*itemsize
            result[target:target+span]=logical_bytes[source:source+span]
    return bytes(result),[out_h,out_w]+([channels]if spec['channel_axis']is not None else[])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/ci-montage.json'));args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np,scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.util._montage');implementation=importlib.import_module('skimagexx._montage')
    reference,native=skimage.util.montage,xx.util.montage
    raw_contract=CONTRACT.read_bytes();contract=json.loads(raw_contract)
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={key:os.environ[key]for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    entry=implementation._core.montage_apply;assert inspect.isbuiltin(entry)
    source_before=source_fingerprint(ROOT);core=Path(xx._core.__file__);core_hash=hashlib.sha256(core.read_bytes()).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    reference_code=inspect.unwrap(reference).__code__;native_code=inspect.unwrap(native).__code__
    for index,spec in enumerate(case_specs()):
        image,options=fixture(spec,np,index);before=snapshot(image)
        logical=image if spec['channel_axis']is None else np.moveaxis(image,spec['channel_axis'],-1)
        oracle,shape=oracle_tiles(logical.tobytes(),logical.shape,spec);assert math.prod(shape)<=contract['max_output_elements']
        expected=reference(image,**options);calls=[];accepted=[];prior=sys.getprofile()
        def reject(*a,**kw):raise AssertionError('owned montage forwarded to original whole API')
        def profile(frame,event,value):
            if event=='call'and frame.f_code is reference_code:reject()
            if event=='c_return'and value is entry:calls.append(value.__name__)
            if event=='return'and frame.f_code is native_code:accepted.append(frame.f_locals.get('native_result')is True)
        saved=skimage.util.montage,original.montage;skimage.util.montage=original.montage=reject;sys.setprofile(profile)
        try:actual=native(image,**options)
        finally:sys.setprofile(prior);skimage.util.montage,original.montage=saved
        assert calls==['montage_apply']and accepted==[True]
        a,b=snapshot(actual),snapshot(expected);assert a==b and a['raw']==oracle and a['shape']==shape
        assert not np.shares_memory(actual,image)and not np.shares_memory(expected,image)
        assert snapshot(image)==before and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,19)
        assert snapshot(image)==before and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.montage',backend='native',threads=1,actual_C_returns=calls,native_accepted=True,native_entry_module=entry.__module__,output_metadata={k:v for k,v in a.items()if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle).hexdigest(),max_abs_error=0.0,**timing))
        print(spec['case_id']+': '+str(timing['speedup']),flush=True)
    assert len(rows)==9 and all(list(map(len,row['samples_ms']))==[7,7]for row in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after and CONTRACT.read_bytes()==raw_contract
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_hash
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,core_path=str(core),core_sha256=core_hash,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),contract_sha256=hashlib.sha256(raw_contract).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=19,raw_samples=126,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,selected_apis=['util.montage'],provider_frontier=contract['provider_frontier'],scope=contract['scope'],timed_boundary=contract['timed_boundary']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Montage: one-thread comparison','',contract['provider_frontier'],'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
