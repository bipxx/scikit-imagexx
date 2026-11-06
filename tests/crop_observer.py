"""Prospective exact crop observer; only an authorized external controller calls it."""
import base64
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import warnings
import zlib
from crop_contract import API,SIGNATURE,ArrayProvider,InterfaceProvider,IndexProbe,BoolProbe,SequenceProbe,fixture,MAX_INPUT_ELEMENTS
from draw_coordinates_observer import encode as encode_leaf,scientific_profile
SOURCE_SHA256='558df9e59105586d0b0e374a33836bd447919e9e5d91f21586127b7cafabfbac'


def reference_runtime():
    """Future source-pinned import only; no original body or fixture invocation."""
    np=scientific_profile();module=importlib.import_module('skimage.util.arraycrop')
    raw=Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
    assert str(inspect.signature(module.crop))==SIGNATURE
    public=importlib.import_module('skimage.util');assert public.crop is module.crop and 'crop'in public.__all__
    return np,module.crop,dict(api=API,numpy=np.__version__,skimage=importlib.metadata.version('scikit-image'),upstream_canonical_lf_sha256=SOURCE_SHA256,original_calls=0)


def encode(value,np):
    if isinstance(value,(ArrayProvider,InterfaceProvider)):
        return dict(protocol=type(value).__name__,array=encode(value.array,np))
    if isinstance(value,(IndexProbe,BoolProbe)):return dict(protocol=type(value).__name__,value=value.value)
    if isinstance(value,SequenceProbe):return dict(protocol='SequenceProbe',items=encode(value.items,np))
    if isinstance(value,memoryview):return dict(protocol='memoryview',format=value.format,shape=list(value.shape),strides=list(value.strides),readonly=value.readonly,bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,np.ndarray):
        if not value.dtype.hasobject and not value.dtype.fields and value.dtype.itemsize<=8:
            return encode_leaf(value,np)
        row=dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__,base_shape=None if not hasattr(value.base,'shape')else list(value.base.shape),base_strides=None if not hasattr(value.base,'strides')else list(value.base.strides))
        if value.dtype.hasobject:
            assert not value.dtype.fields
            row['object_payload']=encode(list(value.flat),np) # Builtin str/int/None only; never serialize object pointers.
        else:
            raw=np.ndarray.tobytes(value,order='C');row.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        return row
    if isinstance(value,np.generic):return dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),base_type=type(getattr(value,'base',None)).__name__,raw=value.tobytes().hex())
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(v,np)for v in value])
    if isinstance(value,dict):return {key:encode(v,np)for key,v in value.items()}
    return encode_leaf(value,np)


def retain_object_payloads(buffers,np):
    """Retain only the bounded builtin payload slots of controlled input arrays."""
    retained=[]
    for name,array in buffers.items():
        if array.dtype.kind=='O':
            assert not array.dtype.fields and array.size<=MAX_INPUT_ELEMENTS
            retained.extend((name,index,value)for index,value in enumerate(array.flat))
    assert len(retained)<=2*MAX_INPUT_ELEMENTS
    return retained


def object_payload_relations(result,retained,np):
    """Relational identity, never addresses or an arbitrary object graph."""
    row=dict(slots=[[name,index]for name,index,_ in retained])
    if isinstance(result,np.ndarray)and result.dtype.kind=='O':
        assert not result.dtype.fields and result.size<=MAX_INPUT_ELEMENTS
        row.update(kind='array',shape=list(result.shape),is_retained=[[value is original for _,_,original in retained]for value in result.flat])
    else:row.update(kind='scalar',is_retained=[result is original for _,_,original in retained])
    return row


def observe(call,spec,np):
    """One supplied-call observation; setup tests use refused/deterministic stand-ins only."""
    args,options,events,buffers=fixture(spec,np);before=encode((args,options,buffers),np);assert not events
    retained=retain_object_payloads(buffers,np)
    pointers={name:a.ctypes.data for name,a in buffers.items()};old=np.geterr(),np.geterrcall(),np.getbufsize();FP=[]
    def handler(message,flags):FP.append(dict(message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always')
        with np.errstate(all='warn',call=handler):
            policy=np.geterr()
            try:result=call(*args,**options)
            except Exception as exc:error=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc))
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old[2]
    checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old[0]and np.geterrcall()is old[1]and np.getbufsize()==old[2],buffer_pointers_stable=all(a.ctypes.data==pointers[name]for name,a in buffers.items()))
    assert all(checks.values())
    aliases={}
    if isinstance(result,(np.ndarray,np.generic)):
        for name,a in buffers.items():
            aliases[name]=dict(identity=result is a,base_is_input=getattr(result,'base',None)is a,shares_memory=bool(np.shares_memory(result,a)),pointer_delta=int(result.ctypes.data-a.ctypes.data)if isinstance(result,np.ndarray)else None)
    payload_identity=None if not retained else object_payload_relations(result,retained,np)if error is None else dict(slots=[[name,index]for name,index,_ in retained],kind='error')
    return dict(id=spec['id'],before=before,after=encode((args,options,buffers),np),result=encode(result,np),result_type=dict(module=type(result).__module__,name=type(result).__name__),result_is_argument=result is args[0],aliases=aliases,object_payload_identity=payload_identity,events=list(events),warnings=[dict(module=type(w.message).__module__,type=type(w.message).__name__,message=str(w.message))for w in caught],error=error,FP_events=FP,checks=checks,original_body_budget=1)
