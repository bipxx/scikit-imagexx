"""Original-only prospective observer. Import is stdlib-only; calls need approval.

All numeric source products, reductions and exceptional NumPy semantics remain
the genuine 0.26 function. No candidate/core loader or tolerance fitting exists.
"""
import ast
import base64
import csv
import hashlib
import json
import inspect
from pathlib import Path
import struct
import sys
import warnings

SOURCE_CANONICAL_LF_SHA256='6cfe1b6e51a27005ffb47e597744cb980b865c1f139cedf47eb0543ee5316805'
UTILS_CANONICAL_LF_SHA256='b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1'
FUNCTIONS=('intersection_coeff','manders_coloc_coeff','manders_overlap_coeff')


def sha(data):return hashlib.sha256(data).hexdigest()


def authenticate(source, utils):
    """Read actual files and installed RECORD; never import an original module."""
    source=Path(source).resolve();utils=Path(utils).resolve()
    assert sha(source.read_bytes().replace(b'\r\n',b'\n'))==SOURCE_CANONICAL_LF_SHA256
    assert sha(utils.read_bytes().replace(b'\r\n',b'\n'))==UTILS_CANONICAL_LF_SHA256
    package=source.parents[1];site=package.parent
    records=list(site.glob('scikit_image-0.26.0.dist-info/RECORD'));assert len(records)==1
    installed={r[0]:r for r in csv.reader(records[0].read_text().splitlines())}
    result=[]
    for p in (source,utils):
        name=p.relative_to(site).as_posix();row=installed[name]
        raw=p.read_bytes();algorithm,digest=row[1].split('=',1)
        assert algorithm=='sha256' and base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')==digest
        assert len(raw)==int(row[2])
        result.append(dict(path=str(p),bytes=len(raw),sha256=sha(raw),RECORD_hash=row[1]))
    tree=ast.parse(source.read_bytes());signatures={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS:
            signatures[node.name]=dict(arguments=ast.unparse(node.args),AST_sha256=sha(ast.dump(node,include_attributes=False).encode()))
    assert set(signatures)==set(FUNCTIONS)
    return dict(version='0.26.0',source_files=result,signatures=signatures,
                RECORD=dict(path=str(records[0]),sha256=sha(records[0].read_bytes())))


def value(value,np):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:
            elements=[{'module':type(v).__module__,'type':type(v).__qualname__,
                       'value':str(v) if type(v) in (int,float,bool,str) else None} for v in value.flat]
            payload=dict(object_elements=elements,raw_pointer_words_excluded=True)
        else:payload=dict(raw_C_hex=value.tobytes(order='C').hex())
        return dict(kind='ndarray',dtype=value.dtype.str,shape=list(value.shape),strides=list(value.strides),
                    writeable=bool(value.flags.writeable),owndata=bool(value.flags.owndata),
                    aligned=bool(value.flags.aligned),C=bool(value.flags.c_contiguous),F=bool(value.flags.f_contiguous),**payload)
    if isinstance(value,np.generic):return dict(kind='numpy-scalar',type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if type(value) is float:return dict(kind='Python-float',raw_f64=struct.pack('=d',value).hex())
    if type(value) in (int,bool,str):return dict(kind='Python-'+type(value).__name__,value=value)
    return dict(kind='object',module=type(value).__module__,type=type(value).__qualname__)


def snapshots(arrays,np):
    rows=[]
    for array in arrays:
        chain=[];current=array;seen=set()
        while isinstance(current,np.ndarray) and id(current) not in seen:
            seen.add(id(current));chain.append(value(current,np));current=current.base
        rows.append(dict(array=value(array,np),owner_chain=chain,terminal_base_type=type(current).__name__))
    return rows


def observe(spec,factory,original,np):
    """One genuine original call; no image execution until root approves freeze."""
    assert original.__name__=='skimage.measure._colocalization'
    function=getattr(original,spec['api']);entry=function
    assert function.__module__==original.__name__
    source_path=Path(inspect.getsourcefile(function)).resolve()
    assert source_path==Path(original.__file__).resolve()
    auth=authenticate(source_path,inspect.getsourcefile(original.as_binary_ndarray))
    assert sha(ast.dump(ast.parse(inspect.getsource(function)).body[0],include_attributes=False).encode())==auth['signatures'][spec['api']]['AST_sha256']
    assert original.np is np
    events=[];args,kwargs,arrays=factory.fixture(spec,np,events)
    setup_events=list(events);events.clear()
    before=snapshots(arrays,np)
    old_policy=np.geterr();old_handler=np.geterrcall();old_buffer=np.getbufsize()
    old_np=original.np;old_binary=original.as_binary_ndarray;old_shape=original.check_shape_equality
    provider_names=('asarray','sum','square','multiply','count_nonzero','logical_and')
    providers={name:getattr(np,name) for name in provider_names}
    scalar_reads={'entry_calls':0,'entry_returns':0}
    old_profile=sys.getprofile()
    assert old_profile is None,'reference requires pristine passive-profile slot'
    code=function.__code__
    def profile(frame,event,arg):
        if frame.f_code is code:
            if event=='call':scalar_reads['entry_calls']+=1
            elif event=='return':scalar_reads['entry_returns']+=1
    class Proxy:
        def __getattr__(self,name):
            actual=getattr(np,name)
            if name not in provider_names:return actual
            def call(*a,**kw):
                events.append(dict(event='numpy-'+name,inputs=[dict(dtype=x.dtype.str,shape=list(x.shape)) for x in a if isinstance(x,np.ndarray)]))
                return actual(*a,**kw)
            return call
    def helper(name,actual):
        def call(*a,**kw):
            events.append(dict(event=name,variable_name=kw.get('variable_name')))
            return actual(*a,**kw)
        return call
    def handler(message,flag):
        events.append(dict(event='FP-handler',message=str(message),flag=int(flag)))
        if spec['group']=='protocol' and spec['name']=='warning_mutation' and arrays[0].size:
            arrays[0].flat[-1]=arrays[0].flat[0]
            events.append(dict(event='finite-inallocation-mutation',index=-1))
    result=None;error=None;recorded=[]
    try:
        if spec['policy']=='default':
            assert old_policy==dict(divide='warn',over='warn',under='ignore',invalid='warn')
        else:np.seterr(all=spec['policy'])
        np.seterrcall(handler if spec['policy']=='call' else old_handler)
        policy_before=np.geterr();handler_before=np.geterrcall()
        if spec['trace']:
            original.np=Proxy();original.as_binary_ndarray=helper('binary-conversion',old_binary)
            original.check_shape_equality=helper('shape-check',old_shape)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sys.setprofile(profile)
            try:
                result=function(*args,**kwargs)
            except Exception as exc:
                error=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc))
            finally:sys.setprofile(old_profile)
            recorded=[dict(module=type(w.message).__module__,category=w.category.__name__,message=str(w.message)) for w in caught]
        after=snapshots(arrays,np)
        state=dict(policy_before=policy_before,policy_after=np.geterr(),
                   handler_identity_unchanged=np.geterrcall() is handler_before,
                   buffer_before=old_buffer,buffer_after=np.getbufsize(),entry_identity_unchanged=getattr(original,spec['api']) is entry,
                   providers_unchanged=all(getattr(np,k) is v for k,v in providers.items()))
    finally:
        sys.setprofile(old_profile)
        original.np=old_np;original.as_binary_ndarray=old_binary;original.check_shape_equality=old_shape
        np.seterr(**old_policy);np.seterrcall(old_handler)
    assert np.geterr()==old_policy and np.geterrcall() is old_handler and np.getbufsize()==old_buffer
    assert original.np is old_np and original.as_binary_ndarray is old_binary and original.check_shape_equality is old_shape
    assert sys.getprofile() is old_profile and scalar_reads['entry_calls']==1 and scalar_reads['entry_returns']==1
    return dict(id=spec['id'],result=value(result,np) if error is None else None,error=error,warnings=recorded,
                setup_events=setup_events,events=events,input_before=before,input_after=after,
                result_aliases=[result is a for a in arrays],state=state,restored=True,
                source_entry=scalar_reads,native_calls=0)
