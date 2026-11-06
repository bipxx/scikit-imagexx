"""Source-bound array view intake/observer. CLI never invokes an original body."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
from array_views_contract import APIS,SIGNATURES,MAX_OUTPUT,fixture
from crop_observer import encode,retain_object_payloads,object_payload_relations
ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA256='fc9ab1bdc09fa464828b6ee3ce96e610d04606b90c13a3ed58f80138e65c876b'
SOURCE_FILES=('coverage/contracts/util-array-views-v1.json','tests/array_views_contract.py','tests/array_views_observer.py','tests/test_array_views_reference.py','tests/test_array_views_intake.py','tests/fixtures/array_views_upstream_v026.py.txt','tests/crop_contract.py','tests/crop_observer.py','tests/draw_coordinates_observer.py','tests/draw_coordinates_contract.py')


def source_revision():
    # Retained Windows-linked trees require native Windows Git even in WSL.
    executable='git'if os.name=='nt'else'/mnt/c/Program Files/Git/cmd/git.exe'
    root=str(ROOT)if os.name=='nt'else str(ROOT).replace('/mnt/c/','C:/',1)
    return subprocess.check_output([executable,'-C',root,'rev-parse','HEAD'],text=True).strip()


def validate_intake_binding(path):
    binding=json.loads(Path(path).read_bytes())
    assert set(binding)=={'schema','status','mode','source_revision','source_files_sha256','apis','original_calls'}
    assert binding['schema']=='array-views-intake-binding-v1'and binding['status']=='ROOT_BOUND_INTAKE_ONLY'
    assert binding['mode']=='intake-only'and type(binding['original_calls'])is int and binding['original_calls']==0
    assert binding['apis']==list(APIS)and binding['source_revision']==source_revision()
    assert set(binding['source_files_sha256'])==set(SOURCE_FILES)
    for name,h in binding['source_files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    return binding


def reference_runtime():
    # Explicit cached-safe import request is intentional; source guard intercepts
    # builtins.__import__, not only MetaPathFinder (which cached NumPy bypasses).
    import numpy as np
    module=importlib.import_module('skimage.util.shape')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA256
    for api in APIS:assert str(inspect.signature(getattr(module,api.split('.')[1])))==SIGNATURES[api]
    public=importlib.import_module('skimage.util')
    assert all(getattr(public,api.split('.')[1])is getattr(module,api.split('.')[1])for api in APIS)
    return np,{api:getattr(module,api.split('.')[1])for api in APIS}


def validate_result_storage(result,buffers,np):
    """Before reading a result, every logical element must name a live input slot."""
    assert type(result)is np.ndarray and result.size<=MAX_OUTPUT and result.ndim<=6
    array=buffers['input_array'];assert result.dtype==array.dtype
    input_offsets={sum(i*s for i,s in zip(index,array.strides))for index in np.ndindex(array.shape)}
    delta=int(result.ctypes.data-array.ctypes.data)
    offsets=[delta+sum(i*s for i,s in zip(index,result.strides))for index in np.ndindex(result.shape)]
    assert set(offsets)<=input_offsets,'Result points outside known live input slots'
    return dict(logical_elements=len(offsets),distinct_input_slots=len(set(offsets)),overlapping=len(offsets)!=len(set(offsets)),pointer_delta=delta)


def base_chain(result,buffers):
    chain=[];seen=set();current=result
    while current is not None:
        assert id(current)not in seen and len(chain)<12
        seen.add(id(current));chain.append(dict(module=type(current).__module__,type=type(current).__name__,is_buffers=[name for name,value in buffers.items()if current is value]))
        current=getattr(current,'base',None)
    return chain


def mutate_one_and_restore(result,buffers,np):
    if not result.size or not result.flags.writeable or result.dtype.kind not in 'biufc':return dict(performed=False)
    positions=list(np.ndindex(result.shape));first=positions[0];offset=lambda index:sum(i*s for i,s in zip(index,result.strides))
    aliases=[list(index)for index in positions if offset(index)==offset(first)]
    before=encode((result,buffers),np);old=result[first].copy()
    try:
        result[first]=1 if old==0 else 0
        changed=encode((result,buffers),np)
        observed=[encode(result[tuple(index)],np)for index in aliases]
    finally:result[first]=old
    assert encode((result,buffers),np)==before
    return dict(performed=True,write_index=list(first),same_slot_indices=aliases,alias_values_after=observed,before=before,after_one_write=changed,restored=True)


def observe(call,spec,np):
    args,options,events,buffers,bound=fixture(spec,np)
    before=encode((args,options,buffers),np);events.clear();retained=retain_object_payloads(buffers,np)
    old=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions();profile=sys.getprofile();pointers={n:a.ctypes.data for n,a in buffers.items()};FP=[]
    def handler(message,flags):FP.append(dict(message=message,flags=flags))
    value=None;error=None
    with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always')
        with np.errstate(all='warn',call=handler):
            policy=np.geterr()
            try:value=call(*args,**options)
            except Exception as exc:error=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc))
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old[2]
    checks=dict(FP_policy_stable=stable,NP_state_restored=old==(np.geterr(),np.geterrcall(),np.getbufsize()),print_options_stable=printing==np.get_printoptions(),profile_stable=profile is sys.getprofile(),pointers_stable=all(a.ctypes.data==pointers[n]for n,a in buffers.items()))
    assert all(checks.values())
    storage=None;chain=[];aliases={};mutation=dict(performed=False)
    if error is None:
        storage=validate_result_storage(value,buffers,np);chain=base_chain(value,buffers)
        aliases={name:dict(identity=value is a,base_identity=value.base is a,shares_memory=bool(np.shares_memory(value,a)))for name,a in buffers.items()}
        if spec['mutation']:mutation=mutate_one_and_restore(value,buffers,np)
    # Protocol events caused by the target end here, before codec observation.
    target_events=list(events)
    return dict(id=spec['id'],api=spec['api'],before=before,after=encode((args,options,buffers),np),result=encode(value,np),error=error,events=target_events,warnings=[dict(module=type(w.message).__module__,type=type(w.message).__name__,message=str(w.message))for w in caught],FP_events=FP,checks=checks,storage=storage,base_chain=chain,aliases=aliases,object_payload_identity=None if not retained or error is not None else object_payload_relations(value,retained,np),mutation=mutation,source_safety=bound)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Source-bound intake only; no reference-body execution')
    parser.add_argument('--root-binding',required=True);parser.add_argument('--intake-only',action='store_true',required=True)
    args=parser.parse_args(argv);validate_intake_binding(args.root_binding)
    np,functions=reference_runtime()
    print(json.dumps(dict(status='SOURCE_RUNTIME_INTAKE_ONLY_NO_ORIGINALS',numpy=np.__version__,apis=list(functions),original_calls=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
