"""Prospective allocation-inclusive find_boundaries; one worker, no work at import."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/find-boundaries-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']

def case_bounds(spec):
    assert spec['dtype']in('uint8','uint16','int32','bool')and spec['layout']in('C','F','reverse')
    assert spec['mode']in('thick','inner','outer','subpixel')and spec['pattern']in('cross','blocks','noncanonical-bool')
    shape=spec['shape'];assert 1<=len(shape)<=3 and all(type(n)is int and 2<=n<=517 for n in shape)
    assert type(spec['connectivity'])is int and 1<=spec['connectivity']<=len(shape)
    assert type(spec['background'])is int and spec['background']==0
    assert type(spec['readonly'])is bool
    pixels=math.prod(shape);output=math.prod([2*n-1 for n in shape]if spec['mode']=='subpixel'else shape)
    work=(output if spec['mode']=='subpixel'else pixels*(4 if spec['mode']=='outer'else 2))*3**len(shape)
    assert pixels<=262144 and output<=1000000 and work<=16000000
    if spec['pattern']=='cross':assert shape==[3,3]and spec['dtype']=='uint8'
    if spec['pattern']=='noncanonical-bool':assert spec['dtype']=='bool'and spec['layout']=='reverse'
    return dict(input_elements=pixels,output_elements=output,source_neighbor_work_bound=work)

def fixture(spec,np):
    bounds=case_bounds(spec)
    if spec['pattern']=='cross':image=np.array([0,0,0,0,1,2,0,2,2],dtype=np.uint8).reshape(3,3)
    else:
        values=(np.arange(bounds['input_elements'],dtype=np.uint32)//7)%4
        if spec['pattern']=='noncanonical-bool':
            values=np.where(values%3==0,0,np.where(values%3==1,2,255)).astype(np.uint8)
            image=values.reshape(spec['shape']).view(np.bool_)
        else:image=values.astype(spec['dtype']).reshape(spec['shape'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    elif spec['layout']=='reverse':image=image[...,::-1]
    if spec['readonly']:image.flags.writeable=False
    return image,dict(connectivity=spec['connectivity'],mode=spec['mode'],background=spec['background'])

def snapshot(array):
    return dict(type=[type(array).__module__,type(array).__qualname__],dtype=array.dtype.str,
        metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),shape=list(array.shape),strides=list(array.strides),
        own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),aligned=bool(array.flags.aligned),
        c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,
        raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc');parser.add_argument('--output',type=Path,default=Path('build/ci-find-boundaries.json'))
    args=parser.parse_args();raw_contract=CONTRACT.read_bytes();contract=json.loads(raw_contract)
    assert contract['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT','PROSPECTIVE: genuine input/root binding missing; no target imports or calls permitted'
    assert len(contract['cases'])==8 and all(case_bounds(s)for s in contract['cases'])
    for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    assert contract['shared_CMake_sha256']and hashlib.sha256((ROOT/'CMakeLists.txt').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['shared_CMake_sha256']
    assert contract['genuine_core_sha256']and contract['prepared_run']and contract['prepared_artifact']
    deadline=args.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat();assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists()and not journal.exists()
    sys.path[:0]=[str(ROOT/'tools')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimagexx import _compat
    original=importlib.import_module('skimage.segmentation.boundaries');public=importlib.import_module('skimage.segmentation')
    owned=importlib.import_module('skimagexx._find_boundaries');engine=importlib.import_module('scipy.ndimage._nd_image')
    upstream=public.find_boundaries;native=xx.segmentation.find_boundaries;entry=xx._core.find_boundaries_into
    assert native is owned.find_boundaries and upstream is original.find_boundaries
    assert str(inspect.signature(native))==str(inspect.signature(upstream))==contract['public_signature']
    assert inspect.isbuiltin(entry)and entry.__module__==xx._core.__name__=='skimagexx._core'
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();assert core_sha==contract['genuine_core_sha256']
    assert skimage.__version__=='0.26.0'
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_source_sha256']
    dependency=contract['runtime_dependencies'][np.__version__];assert scipy.__version__==dependency['scipy']
    assert hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()==dependency['ndimage_binary_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    limits={name:os.environ[name]for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}and xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    cpu_model=platform.processor();info=Path('/proc/cpuinfo')
    if info.is_file():cpu_model=next((line.partition(':')[2].strip()for line in info.read_text().splitlines()if line.startswith('model name')),cpu_model)
    cpu_count,avx2_available=os.cpu_count(),bool(xx._core.has_avx2())
    before=source_fingerprint(ROOT);policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    denied_codes=set()
    for function in(original.find_boundaries,original._find_boundaries_subpixel,original.dilation,original.erosion,original.ndi.generate_binary_structure,original.ndi.minimum_filter,original.ndi.maximum_filter,original.ndi.grey_erosion,original.ndi.grey_dilation):
        while True:
            if hasattr(function,'__code__'):denied_codes.add(function.__code__)
            if not hasattr(function,'__wrapped__'):break
            function=function.__wrapped__
    for index,spec in enumerate(contract['cases']):
        image,options=fixture(spec,np);frozen=snapshot(image);address=image.ctypes.data
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');expected=upstream(image,**options)
        assert not notices and snapshot(image)==frozen and image.ctypes.data==address
        calls=[];accepted=[];previous=sys.getprofile()
        def profile(frame,event,value):
            if event=='call'and frame.f_code in denied_codes:raise AssertionError('Native find_boundaries entered original/helper code')
            if event=='c_return'and value is entry:calls.append('find_boundaries_into')
            if event=='return'and frame.f_code is owned.find_boundaries.__code__:accepted.append(frame.f_locals.get('native_result')is True)
        try:
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');actual=native(image,**options)
        finally:sys.setprofile(previous)
        assert not notices and calls==['find_boundaries_into']and accepted==[True]
        assert type(actual)is type(expected)is np.ndarray and actual.dtype==expected.dtype==np.dtype(bool)
        assert actual is not expected and actual is not image and not np.shares_memory(actual,expected)and not np.shares_memory(actual,image)
        assert actual.flags.owndata and actual.base is None and snapshot(actual)==snapshot(expected)
        assert actual.size==case_bounds(spec)['output_elements']and snapshot(image)==frozen and image.ctypes.data==address
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        # Allocation and destruction of each public result remain inside the
        # unchanged sustained sampler. All guards/profile instrumentation ended.
        timing=measure(lambda:upstream(image,**options),lambda:native(image,**options),args.repeat,100,
            journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert snapshot(image)==frozen and image.ctypes.data==address and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer and not _compat._fallback_enabled
        rows.append(dict(case_id=spec['id'],api='segmentation.find_boundaries',parameters=spec,backend='native',threads=1,
            actual_C_returns=calls,native_accepted=True,input=frozen,output=snapshot(actual),max_abs_error=0.,**timing))
        print(f"{spec['id']}: {timing['speedup']:.9f}x",flush=True)
    assert sum(sum(len(sequence)for sequence in row['samples_ms'])for row in rows)==112
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==raw_contract
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,cpu_model=cpu_model,logical_cpu_count=cpu_count,
        avx2_available=avx2_available,**before,core_sha256=core_sha,threads=1,repeat=7,raw_samples=112,target_ms=100,
        warmup_calibration_min_ms=250,deadline_utc=deadline,reference_pool_limits=limits,contract_sha256=hashlib.sha256(raw_contract).hexdigest(),
        method_sha256=contract['method_sha256'],harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),scope=contract['scope']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
