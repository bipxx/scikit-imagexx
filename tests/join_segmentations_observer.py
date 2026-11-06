"""Bounded join observer. Caller supplies the target; runtime needs root GO."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
from join_segmentations_contract import API,SIGNATURE,fixture
from draw_coordinates_observer import encode,scientific_profile
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/join-segmentations-v1.json'


def reference_runtime(root_go):
    raw=CONTRACT.read_bytes();c=json.loads(raw)
    assert root_go.get('status')=='ROOT_AUTHORIZED_JOIN_REFERENCE_ONCE','NO_ROOT_GO: original imports/calls prohibited'
    assert root_go['api']==API and root_go['contract_sha256']==hashlib.sha256(raw).hexdigest()
    assert root_go['source_sha256']==c['source']['join_canonical_lf_sha256']
    assert 0<root_go['maximum_public_topcalls']<=len(c['cases'])
    assert root_go['native_calls']==root_go['timed_batches']==0
    np=scientific_profile();module=importlib.import_module('skimage.segmentation._join');maps=importlib.import_module('skimage.util._map_array')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==c['source']['join_canonical_lf_sha256']
    assert hashlib.sha256(Path(maps.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==c['source']['map_array_python_sha256']
    assert str(inspect.signature(module.join_segmentations))==SIGNATURE
    primitive=importlib.import_module('skimage.util._remap')
    assert hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest()==c['runtime_pins'][np.__version__]['remap_binary_sha256']
    return np,module.join_segmentations,dict(api=API,signature=SIGNATURE,numpy=np.__version__,source_sha256=c['source']['join_canonical_lf_sha256'],root_GO_contract_sha256=root_go['contract_sha256'],native_calls=0,timings=False)


def map_state(mapping,np):
    return dict(type=[type(mapping).__module__,type(mapping).__qualname__],
        keys=encode(mapping.in_values,np),values=encode(mapping.out_values,np),
        cache=encode(mapping._array,np),max_str_lines=mapping._max_str_lines,dtype=mapping.dtype.str)


def observe(call,spec,np):
    arguments,options,buffers,bounds=fixture(spec,np)
    before=encode((arguments,options,buffers),np);addresses={n:a.ctypes.data for n,a in buffers.items()}
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions()
    previous=sys.getprofile();target_code=getattr(call,'__code__',None);frame_state={};helper_calls={};target_returns=[]
    result=None;error=None;events=[];protocols=[]
    def profile(frame,event,value):
        if event=='call'and Path(frame.f_code.co_filename).name in('_join.py','_map_array.py'):
            key=Path(frame.f_code.co_filename).name+':'+frame.f_code.co_name;helper_calls[key]=helper_calls.get(key,0)+1
        if event=='return'and frame.f_code is target_code:
            target_returns.append(True)
            for key in('factor','s1_relabeled','s2_relabeled','j_initial','labels_j','labels_s1_relabeled','labels_s2_relabeled'):
                if key in frame.f_locals:frame_state[key]=encode(frame.f_locals[key],np)
    def fp(message,flags):events.append(dict(message=message,flags=flags))
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(all='warn',call=fp):
            active_policy=np.geterr().copy()
            try:
                sys.setprofile(profile)
                try:result=call(*arguments,**options)
                finally:sys.setprofile(previous)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            original_result=None;arrays=[];map_aliases=None
            if error is None:
                if type(result)is tuple:
                    assert len(result)==3
                    joined,one,two=result;assert type(joined)is np.ndarray
                    assert len(one.in_values)<=bounds['max_map_entries']and len(two.in_values)<=bounds['max_map_entries']
                    original_result=dict(container='tuple',joined=encode(joined,np),maps=[map_state(m,np)for m in(one,two)])
                    arrays=[joined,one.in_values,one.out_values,two.in_values,two.out_values]
                    map_aliases=dict(same_object=one is two,shared_in_values=one.in_values is two.in_values,shared_out_values=one.out_values is two.out_values)
                    if spec['map_protocols']:
                        # One join-specific protocol recipe only; general ArrayMap
                        # provider/dense contracts are reused rather than recaptured.
                        for name,action in [('getitem',lambda:one[joined]),('call',lambda:two(joined)),('scalar',lambda:one[0]),('slice',lambda:two[::2]),('bool',lambda:one[np.array([True,False,True])]),('dense',lambda:np.asarray(one)),('repr',lambda:repr(one)),('str',lambda:str(two))]:
                            try:protocols.append(dict(action=name,result=encode(action(),np),error=None))
                            except Exception as exc:protocols.append(dict(action=name,error=[type(exc).__module__,type(exc).__name__,str(exc)]))
                        try:
                            one[0]=1
                            protocols.append(dict(action='setitem-zero-to-one',maps=[map_state(m,np)for m in(one,two)],shared_in_values_after=one.in_values is two.in_values,error=None))
                        except Exception as exc:protocols.append(dict(action='setitem-zero-to-one',error=[type(exc).__module__,type(exc).__name__,str(exc)],maps=[map_state(m,np)for m in(one,two)]))
                else:
                    assert type(result)is np.ndarray and result.size<=bounds['input_elements_each'];arrays=[result];original_result=encode(result,np)
            stable=np.geterr()==active_policy and np.geterrcall()is fp and np.getbufsize()==buffer
    after=encode((arguments,options,buffers),np)
    checks=dict(inputs_unchanged=before==after,pointers_unchanged=all(a.ctypes.data==addresses[n]for n,a in buffers.items()),
        target_invoked_once=len(target_returns)==1,FP_policy_stable=stable,FP_state_restored=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer,
        print_options_unchanged=np.get_printoptions()==printing,profile_restored=sys.getprofile()is previous)
    assert all(checks.values())
    row=dict(id=spec['id'],api=API,input_before=before,input_after=after,result=original_result,error=error,
        source_frame=frame_state,original_python_helpers=helper_calls,returned_map_aliases=map_aliases,map_protocols=protocols,
        output_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in arrays for b in arrays],
        output_input_aliases=[[a is b,bool(np.shares_memory(a,b))]for a in arrays for b in buffers.values()],
        warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],FP_events=events,checks=checks)
    json.dumps(row,allow_nan=False);return row
