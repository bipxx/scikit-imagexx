"""Prospective public Region scalar workloads; no target work at import."""
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
CONTRACT=ROOT/'coverage/contracts/regionprops-performance-scalars-v2.json'
PROPERTIES=('area','area_bbox','extent')


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def case_bounds(spec):
    assert spec['operation'] in ('constructor','getters','pixel-cycle','slice-cycle')
    assert spec['layout'] in ('C','F') and type(spec['cache']) is bool
    assert spec['dtype']=='int16' and spec['max_label']==2
    assert len(spec['shape']) in (2,3) and all(type(n)is int and 4<=n<=128 for n in spec['shape'])
    count=math.prod(spec['shape']);assert count<=16384
    assert len(spec['spacing'])==len(spec['shape']) and all(type(v)is float and .25<=v<=2. for v in spec['spacing'])
    assert spec['operation']!='slice-cycle' or spec['cache'] is True
    return dict(input_elements=count,regions=2,mask_elements=2*count,scalar_outputs=6)


def fixture(spec,np):
    case_bounds(spec)
    image=(np.arange(math.prod(spec['shape']),dtype=np.int32)%2+1).astype(spec['dtype']).reshape(spec['shape'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    return image,dict(cache=spec['cache'],spacing=tuple(spec['spacing']))


def make_operation(call,image,options,spec):
    def read(regions):return [(r,tuple(getattr(r,n)for n in PROPERTIES))for r in regions]
    if spec['operation']=='getters':
        regions=call(image,**options)
        if spec['cache']:
            for r in regions:r.area
        return lambda:read(regions)
    if spec['operation']=='constructor':return lambda:read(call(image,**options))
    def cycle():
        regions=call(image,**options);region=regions[0];region.area
        if spec['operation']=='slice-cycle':
            region.slice=tuple(slice(1,n-1)for n in image.shape)
            return read([region])
        position=(0,)*image.ndim;old=image[position];image[position]=0
        try:return read([region])
        finally:image[position]=old
    return cycle


def snapshot(payload,image,np):
    rows=[];types=[]
    for region,values in payload:
        assert all(type(v)is np.float64 for v in values)
        types.append([type(region).__module__,type(region).__qualname__])
        mask=region.image;first,second=region.area,region.area
        rows.append(dict(label=int(region.label),slice=[[s.start,s.stop,s.step]for s in region.slice],
            scalars=[dict(type=type(v).__name__,dtype=v.dtype.str,raw=v.tobytes().hex())for v in values],
            area_identity_repeated=first is second,mask=dict(dtype=mask.dtype.str,shape=list(mask.shape),strides=list(mask.strides),
                own=bool(mask.flags.owndata),writeable=bool(mask.flags.writeable),aligned=bool(mask.flags.aligned),
                c=bool(mask.flags.c_contiguous),f=bool(mask.flags.f_contiguous),raw=mask.tobytes(order='C').hex(),
                input_alias=mask is image or bool(np.shares_memory(mask,image)))))
    return dict(qualified_types=types,rows=rows)


def zero_policy_preflight(scalar_class,core,np):
    # Actual typed helper + policy tail, using saved0/0 words/status, no extra
    # original or region constructor. Full publiczero sequences stay savedtests.
    zero,status=core.regionprops_state_scaled_count_f64(0,np.float64(1))
    assert status==0;events=[]
    with np.errstate(all='call',call=lambda message,flags:events.append([message,flags])):
        invalid=scalar_class._extent(zero,zero)
    assert type(zero)is type(invalid)is np.float64 and zero.tobytes().hex()=='0000000000000000'
    assert invalid.tobytes().hex()=='000000000000f8ff' and events==[['invalid value',8]]
    with np.errstate(all='raise'):
        try:scalar_class._extent(zero,zero)
        except FloatingPointError as error:assert str(error)=='invalid value encountered in scalar divide'
        else:raise AssertionError('saved scalar invalid policy did not raise')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--backend-role',choices=['native','prior-cpp'],default='native')
    parser.add_argument('--deadline-utc');parser.add_argument('--output',type=Path,default=Path('build/ci-regionprops-scalars.json'))
    args=parser.parse_args();raw_contract=CONTRACT.read_bytes();contract=json.loads(raw_contract)
    assert contract['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT','frontend closure remains pending; no calls permitted'
    assert all(type(pin)is str and len(pin)==64 for pin in contract['source_closure'].values())
    if args.backend_role=='native':
        for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    deadline=args.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists()and not journal.exists()
    sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    from regionprops_state import input_state
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimagexx import _compat
    owned=importlib.import_module('skimagexx._regionprops_state');original=importlib.import_module('skimage.measure._regionprops')
    public=importlib.import_module('skimage.measure');ndi=importlib.import_module('scipy.ndimage')
    assert skimage.__version__=='0.26.0'
    for name,pin in contract['original_source_sha256'].items():assert hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    binary=Path(importlib.import_module('scipy.ndimage._nd_image').__file__);binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert binary_sha==contract['original_binary_sha256'][np.__version__]
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    assert hashlib.sha256(inspect.getsource(input_state).encode()).hexdigest()==contract['input_state_method_sha256']
    pools={k:os.environ[k]for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(pools.values())=={'1'}and xx.get_num_threads()==1
    xx.set_fallback_enabled(False);native,upstream=xx.measure.regionprops,public.regionprops
    assert native is owned.regionprops and inspect.signature(native)==inspect.signature(upstream)
    before=source_fingerprint(ROOT)
    if args.backend_role=='prior-cpp':assert before['source_sha256']==contract['prior_CPP_controls']['source_fingerprint']
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();harness_sha=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    cpu_model=platform.processor();cpu_info=Path('/proc/cpuinfo')
    if cpu_info.is_file():cpu_model=next((line.partition(':')[2].strip()for line in cpu_info.read_text().splitlines()if line.startswith('model name')),cpu_model)
    cpu_count,avx2_available=os.cpu_count(),bool(xx._core.has_avx2())
    names=contract['native_entries']if args.backend_role=='native'else['regionprops_state_prepare','regionprops_state_image']
    entries={getattr(xx._core,name):name for name in names};assert all(inspect.isbuiltin(e)for e in entries)
    scalar_class=importlib.import_module('skimagexx._regionprops_state_class')if args.backend_role=='native'else None
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    for index,spec in enumerate(case_specs()):
        image,options=fixture(spec,np);pixels=image.tobytes(order='C');descriptor=input_state(image)
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');reference_operation=make_operation(upstream,image,options,spec);expected=snapshot(reference_operation(),image,np)
        assert not notices
        calls=[];accepted=[]
        def denied(*a,**kw):raise AssertionError('owned scalar arithmetic forwarded to provider')
        def trace(frame,event,value):
            if event=='c_return'and value in entries:calls.append(entries[value])
            if event=='return'and frame.f_code.co_filename.endswith(('_regionprops_state.py','_regionprops_state_class.py'))and'native_result'in frame.f_locals:accepted.append(frame.f_locals['native_result']is not None)
        saved=[(public,'regionprops',public.regionprops),(original,'regionprops',original.regionprops),(ndi,'find_objects',ndi.find_objects),(owned,'_provider_regionprops',owned._provider_regionprops)]
        if args.backend_role=='native':
            saved +=[(original.RegionProperties,name,getattr(original.RegionProperties,name))for name in ('image','area','area_bbox','extent')]
            saved.append((np,'sum',np.sum))
        previous=sys.getprofile()
        try:
            for obj,name,value in saved:setattr(obj,name,property(denied)if isinstance(value,property)else denied)
            sys.setprofile(trace)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');native_operation=make_operation(native,image,options,spec);actual=snapshot(native_operation(),image,np)
            if args.backend_role=='native'and index==0:zero_policy_preflight(scalar_class,xx._core,np)
        finally:
            sys.setprofile(previous)
            for obj,name,value in saved:setattr(obj,name,value)
        assert not notices and set(calls)==set(entries.values())and accepted and all(accepted)
        assert actual['rows']==expected['rows']
        types=contract['type_policy'];assert expected['qualified_types']==[types['original_qualified_type']for r in expected['rows']]
        assert actual['qualified_types']==[types['backend_qualified_type']for r in actual['rows']]
        assert image.tobytes(order='C')==pixels and input_state(image)==descriptor
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer and xx.get_num_threads()==1 and not _compat._fallback_enabled
        # Validation spies/denials are restored before the real sampler call.
        timing=measure(reference_operation,native_operation,7,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert image.tobytes(order='C')==pixels and input_state(image)==descriptor
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer and xx.get_num_threads()==1 and not _compat._fallback_enabled
        rows.append(dict(case_id=spec['id'],parameters=spec,backend_role=args.backend_role,actual_C_returns=calls,native_accepted=True,validation=actual,max_abs_error=0.0,**timing))
    assert sum(sum(map(len,r['samples_ms']))for r in rows)==112 and source_fingerprint(ROOT)==before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==raw_contract
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==harness_sha
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu_model=cpu_model,logical_cpu_count=cpu_count,avx2_available=avx2_available,
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**before,core_path=str(core),core_sha256=core_sha,harness_sha256=harness_sha,
        original_binary_sha256=binary_sha,contract_sha256=hashlib.sha256(raw_contract).hexdigest(),method_sha256=contract['method_sha256'],threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,
        raw_samples=112,reference_pool_limits=pools,fallback_enabled=False,backend_role=args.backend_role,scope=contract['scope'],timed_boundary=contract['timed_boundary'],provider_frontier=contract['provider_frontier'],type_policy=contract['type_policy'],deadline_utc=deadline,command=sys.argv),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Region scalar property workloads','',contract['scope'],'','| Case | Original ms | Backend ms | Original/backend |','|---|---:|---:|---:|']
    lines +=[f"| {r['case_id']} | {r['reference_ms']:.9f} | {r['cpp_ms']:.9f} | {r['speedup']:.9f}x |"for r in rows]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
