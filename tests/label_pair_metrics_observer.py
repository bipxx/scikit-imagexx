"""Raw once-only reference observer; never captures implicitly on import."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import sys
import warnings
if __package__:
    from .draw_coordinates_observer import encode,scientific_profile
    from .label_pair_metrics_contract import APIS,fixture
else:
    from draw_coordinates_observer import encode,scientific_profile
    from label_pair_metrics_contract import APIS,fixture

SOURCE_PINS={
    'skimage.metrics._contingency_table':'e334f4ec16ac3c2bd9fb0c533e8ca454029c914c8649d5322b6f0cc5adbdbf62',
    'skimage.metrics._adapted_rand_error':'cdb0b245288e1b3dd00b36152dd9c20e91edbe1d99a07c8210a2907beb872640',
    'skimage.metrics._variation_of_information':'29fd36256b803934fa37963dbe4b22507152b927b3ac7bfb2b05a973c6552da8',
    'skimage._shared.utils':'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1',
}
SIGNATURES={
    'metrics.contingency_table':"(im_true, im_test, *, ignore_labels=None, normalize=False, sparse_type='matrix')",
    'metrics.adapted_rand_error':'(image_true=None, image_test=None, *, table=None, ignore_labels=(0,), alpha=0.5)',
    'metrics.variation_of_information':'(image0=None, image1=None, *, table=None, ignore_labels=())',
}


def reference_runtime():
    np=scientific_profile();sources={}
    for name,pin in SOURCE_PINS.items():
        module=importlib.import_module(name);path=Path(module.__file__)
        assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
        sources[name]=dict(path=str(path),canonical_lf_sha256=pin)
    public=importlib.import_module('skimage.metrics');functions={api:getattr(public,api.split('.')[1])for api in APIS}
    assert {api:str(inspect.signature(fn))for api,fn in functions.items()}==SIGNATURES
    return np,functions,dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),skimage='0.26.0',sources=sources,calls_per_spec=1,native_calls=0,timings=False)


@contextmanager
def without_reference_activity():
    """Deny actual metric bodies, including cached function references."""
    _,functions,_=reference_runtime();codes={fn.__code__ for fn in functions.values()};previous=sys.getprofile()
    def guard(frame,event,arg):
        if event=='call'and frame.f_code in codes:raise AssertionError('Original label-pair metric forbidden during intake')
    sys.setprofile(guard)
    try:yield
    finally:sys.setprofile(previous)


def observation(value,np):
    import scipy.sparse as sparse
    if sparse.issparse(value):
        assert value.format=='csr'
        return dict(type=type(value).__name__,module=type(value).__module__,format=value.format,shape=list(value.shape),dtype=value.dtype.str,
                    data=encode(value.data,np),indices=encode(value.indices,np),indptr=encode(value.indptr,np),nnz=len(value.data),
                    cached_sorted=value.__dict__.get('_has_sorted_indices'),cached_canonical=value.__dict__.get('_has_canonical_format'))
    if isinstance(value,dict):return {key:observation(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[observation(item,np)for item in value])
    return encode(value,np)


def observe(call,spec,np):
    args,options,buffers=fixture(spec,np);before=observation((args,options,buffers),np)
    pointers={key:value.ctypes.data for key,value in buffers.items()}
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();events=[];result=None;error=None
    def callback(message,flags):events.append(dict(event='FP-handler',message=message,flags=int(flags)))
    try:
        np.seterrcall(callback);np.seterr(**spec['policy'])
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            try:result=call(*args,**options)
            except Exception as exc:error=dict(type=type(exc).__name__,module=type(exc).__module__,message=str(exc))
        output=observation(result,np);after=observation((args,options,buffers),np)
        arrays=[]
        if hasattr(result,'format')and result.format=='csr':arrays=[result.data,result.indices,result.indptr]
        elif isinstance(result,np.ndarray):arrays=[result]
        elif isinstance(result,tuple):arrays=[v for v in result if isinstance(v,np.ndarray)]
        aliases={str(i):{key:bool(np.shares_memory(value,source))for key,source in buffers.items()}for i,value in enumerate(arrays)}
        row=dict(id=spec['id'],before=before,after=after,result=output,error=error,
                 warnings=[dict(category=type(w.message).__name__,message=str(w.message))for w in notices],events=events,
                 output_input_aliases=aliases,input_state_changed=before!=after,
                 checks=dict(input_buffer_addresses_stable=pointers=={key:value.ctypes.data for key,value in buffers.items()}))
    finally:np.seterr(**policy);np.seterrcall(handler);np.setbufsize(buffer)
    row['checks']['policy_restored']=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
    return row
