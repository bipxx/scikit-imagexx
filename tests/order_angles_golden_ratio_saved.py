"""Saved-only family parity; original raw bytes are never rewritten."""
import base64,ctypes,hashlib,json,zlib
from pathlib import Path
from order_angles_golden_ratio_observer_v2 import observe,counters
from util_views_fenv import FEnv


def load_saved(profile):
    data=json.loads((Path(__file__).parent/'fixtures/order_angles_golden_ratio_public_v1.json').read_bytes())['profiles'][profile]
    raw=zlib.decompress(base64.b64decode(data['raw_zlib_base64']))
    assert len(raw)==data['raw_bytes']and hashlib.sha256(raw).hexdigest()==data['raw_sha256']
    rows=json.loads(raw);assert len(rows)==45
    return {r['id']:r for r in rows}


def differences(actual,expected,path=''):
    if type(actual)is not type(expected):return [(path,'type',type(actual).__name__,type(expected).__name__)]
    out=[]
    if isinstance(actual,dict):
        if set(actual)!=set(expected):out.append((path,'keys',sorted(actual),sorted(expected)))
        for key in actual.keys()&expected.keys():
            # Current pointer identity is asserted independently; retained
            # addresses/instruction fields are diagnostic, not numeric gold.
            if key=='handler_id'and(path.endswith('numpy_before')or path.endswith('numpy_after')):continue
            if key=='raw32'and'immediate_fp'in path:continue
            out+=differences(actual[key],expected[key],path+'.'+key)
    elif isinstance(actual,list):
        if len(actual)!=len(expected):out.append((path,'length',len(actual),len(expected)))
        for i,(a,b)in enumerate(zip(actual,expected)):out+=differences(a,b,path+'['+str(i)+']')
    elif actual!=expected:out.append((path,'value',actual,expected))
    return out


def observe_saved(call,recipe,expected,np,libc,helper):
    def env():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):
        assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
        assert bytes(env())==bytes(value),'ACTUAL_CALLER32'
    caller=env();index=0;partial=[]
    def envelope(action,kind):
        nonlocal index
        gold=expected['actions'][index];assert gold['action']==kind
        saved=env();incoming=FEnv.from_buffer_copy(bytes(saved))
        # Select authentic incoming semantic fields BEFORE the supplied action.
        # Retain current diagnostic instruction/reserved words; never modify
        # outgoing status or normalize expected data to make a result pass.
        for field in('control','status','mxcsr'):setattr(incoming,field,gold['immediate_fp']['before'][field])
        restore(incoming);before=env();value=error=None
        try:
            try:value=action()
            except Exception as caught:error=caught
            finally:after=env()
        finally:restore(saved)
        index+=1
        state=lambda v:dict(raw32=bytes(v).hex(),control=v.control,status=v.status,mxcsr=v.mxcsr)
        return value,error,dict(before=state(before),after=state(after),caller_restore32=True)
    try:
        row=observe(call,recipe,np,envelope,partial);row['counters']=counters(row)
        assert index==len(expected['actions'])
        assert all(a['numpy_before']['handler_id']==a['numpy_after']['handler_id']==id(np.geterrcall())for a in row['actions'])
        return row,differences(row,expected)
    finally:restore(caller)
