"""Prospective partial RegionProperties benchmark; no work happens on import."""
import argparse
from datetime import datetime, timedelta, timezone
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
CONTRACT=ROOT/'coverage/contracts/regionprops-state-benchmark.json'
PROPERTIES=('label','slice','bbox','image','coords','num_pixels','area')


def case_specs():
    return json.loads(CONTRACT.read_bytes())['cases']


def case_bounds(spec):
    shape=spec['shape'];ndim=len(shape);pixels=math.prod(shape);regions=spec['max_label']
    assert ndim in (2,3)and all(type(n)is int and 0<n<=1024 for n in shape)
    assert 1<=regions<=8 and pixels<=400_000
    assert spec['operation']in('constructor','seven-properties')and spec['layout']in('C','F','reverse','transpose')
    assert spec['pattern']in('dense','sparse')and spec['dtype']in('int16','int32','uint16')and type(spec['cache'])is bool
    for key in ('spacing','offset'):
        if spec[key]is not None:
            assert len(spec[key])==ndim and all(type(v)in(int,float)and math.isfinite(v)and abs(v)<=4 for v in spec[key])
    if spec['spacing']is not None:assert all(v>0 for v in spec['spacing'])
    # Bounding each region by the entire image is intentionally conservative.
    # Validation retains two image/coords reads to check cache alias behavior.
    mask=pixels*regions;coords=mask*ndim
    return dict(input_elements=pixels,regions=regions,mask_elements=mask,coords_elements=coords,
                validation_payload_bytes=2*(mask+8*coords),input_bytes=4*pixels)


def fixture(spec,np):
    case_bounds(spec)
    indices=np.arange(math.prod(spec['shape']),dtype=np.int32).reshape(spec['shape'])
    if spec['pattern']=='dense':image=(indices%spec['max_label']+1).astype(spec['dtype'])
    else:image=np.where(indices%257==0,(indices//257)%spec['max_label']+1,0).astype(spec['dtype'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    elif spec['layout']=='reverse':image=image[::-1]
    elif spec['layout']=='transpose':image=image.T
    options=dict(cache=spec['cache'],spacing=spec['spacing'],offset=spec['offset'])
    return image,options


def invoke(call,image,options,spec):
    regions=call(image,**options)
    if spec['operation']=='constructor':return regions
    return [(r,tuple(getattr(r,name)for name in PROPERTIES))for r in regions]


def input_state(image):
    return (image.dtype.str,image.dtype.metadata,image.shape,image.strides,
            image.flags.writeable,image.flags.owndata,image.flags.aligned,
            image.flags.c_contiguous,image.flags.f_contiguous,type(image.base).__name__,image.ctypes.data)


def snapshot(payload,image,np,spec):
    def value(item):
        if isinstance(item,np.ndarray):
            base=item.base
            return dict(type=type(item).__name__,dtype=item.dtype.str,dtype_metadata=item.dtype.metadata,
                shape=list(item.shape),strides=list(item.strides),own=bool(item.flags.owndata),
                writeable=bool(item.flags.writeable),aligned=bool(item.flags.aligned),c=bool(item.flags.c_contiguous),f=bool(item.flags.f_contiguous),
                base_type=type(base).__name__,base_shape=None if not hasattr(base,'shape')else list(base.shape),
                base_strides=None if not hasattr(base,'strides')else list(base.strides),raw=item.tobytes(order='C'))
        if isinstance(item,np.generic):return dict(type=type(item).__name__,dtype=item.dtype.str,raw_scalar=item.tobytes().hex())
        if isinstance(item,slice):return dict(type='slice',start=item.start,stop=item.stop,step=item.step)
        if isinstance(item,(tuple,list)):return dict(container=type(item).__name__,items=[value(v)for v in item])
        assert item is None or type(item)in(int,float,bool,str)
        return item
    rows=[];types=[];array_elements=0;all_arrays=[]
    for entry in payload:
        if spec['operation']=='constructor':region=entry;properties=tuple(getattr(region,name)for name in PROPERTIES)
        else:region,properties=entry
        types.append([type(region).__module__,type(region).__qualname__])
        repeated=(region.image,region.coords)
        arrays=[v for v in properties if isinstance(v,np.ndarray)]+list(repeated)
        all_arrays.extend(arrays)
        array_elements+=sum(a.size for a in arrays)
        rows.append(dict(properties=value(properties),repeated=value(repeated),
            output_input_aliases=[[a is image,bool(np.shares_memory(a,image))]for a in arrays],
            output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in arrays for b in arrays]))
    bounds=case_bounds(spec);assert len(rows)<=bounds['regions']and array_elements<=2*(bounds['mask_elements']+bounds['coords_elements'])
    return dict(payload_container=type(payload).__name__,qualified_types=types,regions=rows,
                all_output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in all_arrays for b in all_arrays])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output',type=Path,default=Path('build/ci-regionprops-state.json'))
    args=parser.parse_args();deadline=args.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists()and not journal.exists()
    sys.path.insert(0,str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    owned=importlib.import_module('skimagexx._regionprops_state')
    original=importlib.import_module('skimage.measure._regionprops');public=importlib.import_module('skimage.measure');ndi=importlib.import_module('scipy.ndimage')
    raw_contract=CONTRACT.read_bytes();contract=json.loads(raw_contract)
    assert skimage.__version__=='0.26.0'
    for name,pin in contract['original_source_sha256'].items():
        module=importlib.import_module(name);assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    original_binary=importlib.import_module('scipy.ndimage._nd_image')
    original_binary_sha=hashlib.sha256(Path(original_binary.__file__).read_bytes()).hexdigest()
    assert original_binary_sha==contract['original_binary_sha256'][np.__version__]
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    limits={k:os.environ[k]for k in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}and xx.get_num_threads()==1
    xx.set_fallback_enabled(False)
    native=xx.measure.regionprops;upstream=public.regionprops
    assert native is owned.regionprops and str(inspect.signature(native))==str(inspect.signature(upstream))
    entries={getattr(xx._core,'regionprops_state_'+n):n for n in('prepare','image','population','coords')}
    assert all(inspect.isbuiltin(e)and e.__module__=='skimagexx._core'for e in entries)
    type_policy=contract['type_policy'];assert issubclass(owned.RegionProperties,original.RegionProperties)and owned.RegionProperties is not original.RegionProperties
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        cpu_model = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    cpu_count, avx2_available = os.cpu_count(), bool(xx._core.has_avx2())
    before=source_fingerprint(ROOT);core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    harness_sha=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    codes={owned.regionprops.__code__,owned.RegionProperties.image.fget.__wrapped__.__code__,owned.RegionProperties.num_pixels.fget.__wrapped__.__code__,owned.RegionProperties.coords.fget.__code__}
    for index,spec in enumerate(case_specs()):
        image,options=fixture(spec,np);input_bytes=image.tobytes(order='C');input_descriptor=input_state(image)
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');expected=snapshot(invoke(upstream,image,options,spec),image,np,spec)
        assert not notices
        calls=[];accepted=[]
        def denied(*a,**kw):raise AssertionError('owned region state forwarded to original constructor/image/count/coords algorithm')
        def profiler(frame,event,result):
            if event=='c_return'and result in entries:calls.append(entries[result])
            if event=='return'and frame.f_code in codes and 'native_result'in frame.f_locals:accepted.append(frame.f_locals['native_result']is not None)
        saved=[(public,'regionprops',public.regionprops),(original,'regionprops',original.regionprops),(ndi,'find_objects',ndi.find_objects),(owned,'_provider_regionprops',owned._provider_regionprops)]
        saved +=[(original.RegionProperties,n,getattr(original.RegionProperties,n))for n in('image','num_pixels','coords')]
        previous=sys.getprofile()
        try:
            for obj,name,old in saved:setattr(obj,name,property(denied)if isinstance(old,property)else denied)
            sys.setprofile(profiler)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');actual=snapshot(invoke(native,image,options,spec),image,np,spec)
        finally:
            sys.setprofile(previous)
            for obj,name,old in saved:setattr(obj,name,old)
        assert not notices and calls.count('prepare')==1 and set(calls)==set(entries.values())and accepted and all(accepted)
        for key in expected:
            if key!='qualified_types':assert actual[key]==expected[key],(spec['id'],key)
        assert expected['qualified_types']==[type_policy['original_qualified_type']for _ in expected['regions']]
        assert actual['qualified_types']==[type_policy['backend_qualified_type']for _ in actual['regions']]
        assert image.tobytes(order='C')==input_bytes and input_descriptor==input_state(image)
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        # Fresh output objects and all property allocations are inside each call;
        # the unchanged sampler del result includes destruction in every batch.
        timing=measure(lambda:invoke(upstream,image,options,spec),lambda:invoke(native,image,options,spec),args.repeat,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert image.tobytes(order='C')==input_bytes and input_descriptor==input_state(image)
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        # Preserve raw qualified types distinctly; only array payloads become hashes.
        pending=[actual]
        while pending:
            item=pending.pop()
            if isinstance(item,dict):
                if 'raw'in item:item['raw_sha256']=hashlib.sha256(item.pop('raw')).hexdigest()
                pending.extend(item.values())
            elif isinstance(item,list):pending.extend(item)
        rows.append(dict(case_id=spec['id'],parameters=spec,backend='partial-native-subclass',threads=1,actual_C_returns=calls,native_accepted=True,validation=actual,original_qualified_types=expected['qualified_types'],max_abs_error=0.0,**timing))
        print(spec['id']+': '+str(timing['speedup'])+'x',flush=True)
    assert sum(sum(map(len,r['samples_ms']))for r in rows)==contract['raw_samples']
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==raw_contract
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==harness_sha
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu_model=cpu_model,logical_cpu_count=cpu_count,avx2_available=avx2_available,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**before,core_path=str(core),core_sha256=core_sha,harness_sha256=harness_sha,original_binary_sha256=original_binary_sha,contract_sha256=hashlib.sha256(raw_contract).hexdigest(),method_sha256=contract['method_sha256'],threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,raw_samples=contract['raw_samples'],reference_pool_limits=limits,fallback_enabled=False,scope=contract['scope'],timed_boundary=contract['timed_boundary'],provider_frontier=contract['provider_frontier'],type_policy=type_policy,deadline_utc=deadline,command=sys.argv),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# RegionProperties partial state: initial one-thread comparison','',contract['scope'],'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    lines +=[f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |"for r in rows]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
