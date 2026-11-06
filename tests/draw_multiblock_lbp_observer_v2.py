"""Typed LBP observation over admitted bounded fixtures and the crop codec.

Only the separately authenticated external controller may supply a real target.
This module imports no scientific package and never resolves a target itself.
"""
import json
import warnings
import draw_multiblock_lbp_contract as legacy
import draw_multiblock_lbp_overlap_contract as overlap
API=legacy.API
encode=legacy.encode
from crop_observer import retain_object_payloads,object_payload_relations
from mark_boundaries_exception_codec import encode_exception_args


def cases():return overlap.cases()

def select_fixture(spec):
    # v2.fixture deliberately accepts only the one new overlap recipe.
    if spec in legacy.cases():return legacy.fixture
    assert spec==overlap.added_case(),'Unknown or modified LBP recipe'
    return overlap.fixture


def observe(call,spec,np):
    args,options,events,buffers=select_fixture(spec)(spec,np)
    before=encode((args,options,buffers),np);assert events==[]
    retained=retain_object_payloads(buffers,np)
    pointers={name:a.ctypes.data for name,a in buffers.items()}
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();FP=[]
    def callback(message,flags):FP.append(dict(message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(**{'all':'warn',**spec['policy']},call=callback):
            expected_policy=np.geterr()
            try:result=call(*args,**options)
            except Exception as exc:
                error=dict(module=type(exc).__module__,type=type(exc).__name__,qualname=type(exc).__qualname__,message=str(exc),args=encode_exception_args(exc.args,np))
            stable=np.geterr()==expected_policy and np.geterrcall()is callback and np.getbufsize()==buffer
    # Snapshot actual target mutation/protocol effects before any other work.
    after=encode((args,options,buffers),np);source_events=list(events)
    aliases={}
    if isinstance(result,(np.ndarray,np.generic)):
        for name,a in buffers.items():
            sharing=bool(np.shares_memory(result,a))
            aliases[name]=dict(identity=result is a,base_is_input=getattr(result,'base',None)is a,shares_memory=sharing,pointer_delta=int(result.ctypes.data-a.ctypes.data)if isinstance(result,np.ndarray)else None,pointer_delta_role='same-storage-relative-offset'if sharing else'nonsharing-diagnostic-only')
    checks=dict(policy_unchanged=stable,state_restored=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer,buffer_pointers_stable=all(a.ctypes.data==pointers[name]for name,a in buffers.items()))
    assert all(checks.values())
    row=dict(id=spec['id'],api=API,before=before,after=after,result=encode(result,np),result_type=dict(module=type(result).__module__,name=type(result).__name__),result_is_arguments=[result is arg for arg in args],aliases=aliases,object_payload_identity=None if not retained else object_payload_relations(result,retained,np)if error is None else dict(slots=[[name,index]for name,index,_ in retained],kind='error'),events=source_events,warnings=[dict(module=type(w.message).__module__,type=type(w.message).__name__,message=str(w.message),category_module=w.category.__module__,category=w.category.__name__)for w in notices],error=error,FP_events=FP,checks=checks,supplied_call_attempts=1,mutation_policy='Actual input before/after preserved; observer performs no result or input mutation experiment')
    json.dumps(row,allow_nan=False);return row
