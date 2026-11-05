"""Prospective public in-place draw.set_color, fresh image allocation per call."""
import argparse,hashlib,importlib,inspect,json,os,platform,sys,warnings
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-set-color-benchmark.json'

def source_safe(spec):
    assert spec['kind']in('ordinary','duplicates','clipped','color-alias')and spec['order']in('C','F')
    assert spec.get('dtype','float64')in('float32','float64')
    h,w,c=spec['shape'];n=spec['count']
    assert all(type(x)is int for x in(h,w,c,n))and 1<=h<=384 and 1<=w<=384 and c in(1,3)and 0<=n<=h*w*4
    assert h*w*c<=442368 and n*c<=786432
    return dict(image_elements=h*w*c,index_pairs=n,arithmetic_elements=n*c,max_gather_elements=n*c)

def fixture(spec,np):
    source_safe(spec);h,w,c=spec['shape'];n=spec['count']
    base=np.full((h,w)if c==1 else(h,w,c),.5,dtype=spec.get('dtype','float64'),order=spec['order'])
    q=np.arange(n,dtype=np.intp);rr=(q//w)%h;cc=q%w
    if spec['kind']=='duplicates':rr=rr//2;cc=cc//2
    if spec['kind']=='clipped':rr[::3]=-1
    alpha=np.full(n,.5);color=np.array([.25]*c)
    def call(function):
        # Fresh image is required by in-place API; same copy/setup for both arms.
        image=base.copy(order='K');prepared_color=image[0,0]if spec['kind']=='color-alias'else color
        result=function(image,(rr,cc),prepared_color,alpha=alpha)
        return image
    return call,(base,rr,cc,color,alpha)

def main():
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,choices=[1],default=1);p.add_argument('--repeat',type=int,choices=[7],default=7);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-utc');a=p.parse_args()
    deadline=a.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    assert not a.output.exists()and not a.output.with_suffix('.batches.jsonl').exists()
    sys.path.insert(0,str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np,skimage,skimagexx as xx
    public=importlib.import_module('skimage.draw');original=importlib.import_module('skimage.draw.draw');owned=importlib.import_module('skimagexx._draw_set_color')
    binary=Path(importlib.import_module('skimage.draw._draw').__file__);binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert binary_sha=={'1.24.4':'32673fec979236e2f83543ad289a8128059b96e6562a9ea30114c4e73c17139b','2.5.4':'ae81e3d9b252c264cac3236a22d96e7b5242d0f9819950dbe1f73464380869d4'}[np.__version__]
    raw=CONTRACT.read_bytes();contract=json.loads(raw);assert skimage.__version__=='0.26.0'
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_source_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    pools={key:os.environ[key]for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(pools.values())=={'1'}
    assert xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    native=xx.draw.set_color;upstream=public.set_color;assert native is owned.set_color
    assert str(inspect.signature(native))==str(inspect.signature(upstream))=='(image, coords, color, alpha=1)'
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();before=source_fingerprint(ROOT)
    policy,handler,buf=np.geterr(),np.geterrcall(),np.getbufsize();rows=[];getter=xx._core.get_num_threads;entry=xx._core.draw_set_color;origin_code=original.set_color.__code__
    for index,spec in enumerate(contract['cases']):
        call,buffers=fixture(spec,np);states=[(x.tobytes(),x.shape,x.strides,x.dtype,x.ctypes.data,x.flags.writeable)for x in buffers]
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');expected=call(upstream)
        assert not notices
        calls=[];accepted=[];gets=[]
        def denied(*args,**kwargs):raise AssertionError('native forwarded original algorithm')
        def profile(frame,event,value):
            assert not(event=='call'and frame.f_code is origin_code)
            if event=='c_return'and value is entry:calls.append('draw_set_color')
            if event=='c_return'and value is getter:gets.append('get_num_threads')
            if event=='return'and frame.f_code is native.__code__:accepted.append(frame.f_locals.get('native_result')is True)
        saved=public.set_color,original.set_color,owned.fallback;prior=sys.getprofile()
        try:
            public.set_color=original.set_color=owned.fallback=denied;sys.setprofile(profile)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');actual=call(native)
        finally:sys.setprofile(prior);public.set_color,original.set_color,owned.fallback=saved
        assert not notices and calls==['draw_set_color']and gets==['get_num_threads']and accepted==[True]
        assert actual is not expected and not np.shares_memory(actual,expected)
        assert actual.dtype.metadata is None and expected.dtype.metadata is None and actual.flags.writeable and actual.flags.aligned
        assert actual.dtype==expected.dtype==np.dtype(spec.get('dtype','float64'))and actual.shape==expected.shape and actual.strides==expected.strides
        assert actual.flags.owndata and expected.flags.owndata and actual.base is None and expected.base is None
        assert actual.tobytes()==expected.tobytes()and all(not np.shares_memory(actual,x)for x in buffers)
        assert states==[(x.tobytes(),x.shape,x.strides,x.dtype,x.ctypes.data,x.flags.writeable)for x in buffers]
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buf
        # Instrumentation removed before either timed arm; no repeated validators.
        timing=measure(lambda:call(upstream),lambda:call(native),7,100,journal=a.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert states==[(x.tobytes(),x.shape,x.strides,x.dtype,x.ctypes.data,x.flags.writeable)for x in buffers]
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buf and xx.get_num_threads()==1
        rows.append(dict(case_id=spec['id'],parameters=spec,bounds=source_safe(spec),native_accepted=True,actual_C_returns=calls,global_worker_snapshots=gets,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),dtype=actual.dtype.str,shape=actual.shape,strides=actual.strides,**timing))
    assert len(rows)==8 and sum(sum(map(len,x['samples_ms']))for x in rows)==112
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==raw
    assert xx.get_num_threads()==1 and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buf
    cpu=platform.processor()
    if Path('/proc/cpuinfo').is_file():cpu=next((s.partition(':')[2].strip()for s in Path('/proc/cpuinfo').read_text().splitlines()if s.startswith('model name')),cpu)
    report=dict(metadata=dict(**before,core_sha256=core_sha,original_draw_binary_sha256=binary_sha,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),python=sys.version,numpy=np.__version__,reference=skimage.__version__,cpu_model=cpu,logical_cpu_count=os.cpu_count(),avx2_available=bool(xx.has_avx2()),threads=1,reference_pool_limits=pools,repeat=7,target_ms=100,warmup_calibration_min_ms=250,raw_samples=112,deadline_utc=deadline,contract_sha256=hashlib.sha256(raw).hexdigest(),scope=contract['scope']),results=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
