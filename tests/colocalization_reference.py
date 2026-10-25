"""Original-only prospective observer. Import is stdlib-only; calls need approval.

All numeric source products, reductions and exceptional NumPy semantics remain
the genuine 0.26 function. No candidate/core loader or tolerance fitting exists.
"""
import ast
import base64
import csv
import copy
import hashlib
import json
import inspect
from pathlib import Path
import struct
import sys
import traceback
import warnings
import zlib

SOURCE_CANONICAL_LF_SHA256='6cfe1b6e51a27005ffb47e597744cb980b865c1f139cedf47eb0543ee5316805'
UTILS_CANONICAL_LF_SHA256='b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1'
FUNCTIONS=('intersection_coeff','manders_coloc_coeff','manders_overlap_coeff')
CHUNK_BYTES=256*1024
TRACE_EVENTS=frozenset(('numpy-asarray','numpy-sum','numpy-square','numpy-multiply',
                        'numpy-count_nonzero','numpy-logical_and','binary-conversion','shape-check'))


def semantic_projection(record):
    """Only the eight declared extra trace-event kinds may be removed."""
    result=copy.deepcopy(record)
    result['events']=[e for e in result['events'] if e.get('event') not in TRACE_EVENTS]
    return result


class RawJournal:
    """Mandatory exclusive JSONL sink; every record is flushed before checks."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x',encoding='utf-8',newline='\n');self.records=[]

    def append(self,record):
        raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.stream.write(raw+'\n');self.stream.flush()
        ack=dict(sequence=len(self.records),sha256=sha(raw.encode()))
        self.records.append(ack)
        return ack

    def compare_pair(self,first,second):
        # Both exact raw observations must already have been flushed by THIS
        # journal before even evaluating the semantic comparison.
        for envelope in (first,second):
            ack=envelope['journal'];assert self.records[ack['sequence']]==ack
            expected=dict(kind='observation',actual=envelope['observation'])
            assert sha(json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False).encode())==ack['sha256']
        assert first['journal']['sequence']!=second['journal']['sequence']
        same=semantic_projection(first['observation'])==semantic_projection(second['observation'])
        self.append(dict(kind='trace-pristine-comparison',first=first['journal'],second=second['journal'],equal=same))
        if not same:raise AssertionError('retained trace/pristine semantic observations differ')
        return True

    def close(self):self.stream.close()


def retain_then_check(row,journal):
    """Record actual failure/state FIRST; no failed check loses the evidence."""
    ack=journal.append(dict(kind='observation',actual=row))
    if row.get('observer_error') is not None or not all(row['checks'].values()):
        failure=RuntimeError('observer/check failure after raw journal flush at '+str(row['phase']))
        failure.raw_journal_ack=ack
        raise failure
    return dict(observation=row,journal=ack)


class RawAssets:
    """Profile-private exclusive lossless chunks; never a full raster hex/JSON."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.max_raw_chunk=0

    def chunks(self,pieces):
        total=0;digest=hashlib.sha256();records=[]
        for piece in pieces:
            view=memoryview(piece).cast('B')
            for start in range(0,len(view),CHUNK_BYTES):
                raw=bytes(view[start:start+CHUNK_BYTES])
                self.max_raw_chunk=max(self.max_raw_chunk,len(raw))
                digest.update(raw);total+=len(raw);key=sha(raw)
                packed=zlib.compress(raw,6);path=self.root/(key+'.zlib')
                try:
                    with path.open('xb') as stream:stream.write(packed)
                except FileExistsError:
                    assert path.read_bytes()==packed,'immutable asset collision/corruption'
                records.append(dict(path=path.name,raw_bytes=len(raw),raw_sha256=key,
                                    bytes=len(packed),sha256=sha(packed)))
        return dict(encoding='raw-C-chunks+zlib',raw_bytes=total,raw_sha256=digest.hexdigest(),chunks=records)

    def array(self,array,np):
        assert not array.dtype.hasobject
        count=max(1,CHUNK_BYTES//array.dtype.itemsize)
        def pieces():
            iterator=np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],
                               op_flags=['readonly'],op_dtypes=[array.dtype],casting='no',
                               order='C',buffersize=count)
            for block in iterator:
                # Explicitly split even if a NumPy iterator returns a larger
                # contiguous inner loop; each tobytes allocation stays bounded.
                for start in range(0,block.size,count):
                    yield block[start:start+count].tobytes(order='C')
        return self.chunks(pieces())


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


def value(value,np,assets):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:
            elements=[]
            for v in value.flat:
                if type(v) is float:element=dict(type='float',raw_f64=struct.pack('=d',v).hex())
                elif type(v) in (int,bool,str):element=dict(type=type(v).__name__,value=v)
                else:
                    assert type(v).__name__=='Token' and type(v).__qualname__.endswith('fixture.<locals>.Token')
                    element=dict(type='opaque-Token',module=type(v).__module__,qualname=type(v).__qualname__,pointer_words_excluded=True)
                elements.append(element)
            payload=dict(object_elements=elements,encoding='typed-object-values-not-PyObject-addresses')
        else:payload=dict(raw_C=assets.array(value,np))
        return dict(kind='ndarray',module=type(value).__module__,type=type(value).__qualname__,dtype=value.dtype.str,shape=list(value.shape),strides=list(value.strides),
                    writeable=bool(value.flags.writeable),owndata=bool(value.flags.owndata),
                    aligned=bool(value.flags.aligned),C=bool(value.flags.c_contiguous),F=bool(value.flags.f_contiguous),**payload)
    if isinstance(value,np.generic):return dict(kind='numpy-scalar',module=type(value).__module__,type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if type(value) is float:return dict(kind='Python-float',raw_f64=struct.pack('=d',value).hex())
    if type(value) in (int,bool,str):return dict(kind='Python-'+type(value).__name__,value=value)
    return dict(kind='object',module=type(value).__module__,type=type(value).__qualname__)


def snapshots(arrays,np,assets):
    rows=[]
    for array in arrays:
        if isinstance(array,np.generic):
            rows.append(dict(scalar=value(array,np,assets),owner_chain=None,terminal_base=None))
            continue
        chain=[];current=array;seen=set()
        while isinstance(current,np.ndarray) and id(current) not in seen:
            seen.add(id(current));chain.append(value(current,np,assets));current=current.base
        terminal=dict(type=type(current).__name__)
        if isinstance(current,(bytearray,bytes,memoryview)):
            view=memoryview(current)
            assert view.contiguous,'unknown noncontiguous terminal buffer is not certified'
            terminal['byte_content']=assets.chunks((view,))
        else:assert current is None,'unknown terminal owner is not certified'
        rows.append(dict(array=value(array,np,assets),owner_chain=chain,terminal_base=terminal))
    return rows


def observe(spec,factory,original,np,assets,journal):
    """One genuine original call; no image execution until root approves freeze."""
    if not isinstance(journal,RawJournal):raise TypeError('mandatory RawJournal required before observer setup')
    journal.append(dict(kind='observer-start',id=spec['id']))
    row=dict(id=spec['id'],phase='authentication',result=None,error=None,observer_error=None,
             warnings=[],setup_events=[],events=[],input_before=None,input_after=None,
             result_aliases=None,state=None,restored=False,source_entry={'entry_calls':0,'entry_returns':0},
             original_call_attempted=False,native_calls=0,checks={})
    events=row['events'];scalar_reads=row['source_entry'];arrays=[];result=None
    old_policy=np.geterr();old_handler=np.geterrcall();old_buffer=np.getbufsize()
    old_np=original.np;old_binary=original.as_binary_ndarray;old_shape=original.check_shape_equality
    provider_names=('asarray','sum','square','multiply','count_nonzero','logical_and')
    providers={name:getattr(np,name) for name in provider_names}
    old_profile=sys.getprofile()
    function=getattr(original,spec['api']);entry=function;code=function.__code__
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
    try:
        assert sys.flags.optimize==0 and old_profile is None,'reference requires optimize0 and pristine passive-profile slot'
        assert original.__name__=='skimage.measure._colocalization' and function.__module__==original.__name__
        source_path=Path(inspect.getsourcefile(function)).resolve()
        assert source_path==Path(original.__file__).resolve()
        auth=authenticate(source_path,inspect.getsourcefile(original.as_binary_ndarray))
        assert sha(ast.dump(ast.parse(inspect.getsource(function)).body[0],include_attributes=False).encode())==auth['signatures'][spec['api']]['AST_sha256']
        assert original.np is np
        row['phase']='fixture'
        args,kwargs,arrays=factory.fixture(spec,np,events)
        row['setup_events']=list(events);events.clear()
        row['phase']='input-snapshot'
        row['input_before']=snapshots(arrays,np,assets)
        row['phase']='policy-setup'
        if spec['policy']=='default':
            assert old_policy==dict(divide='warn',over='warn',under='ignore',invalid='warn')
            assert old_handler is None,'default controls require the actual pristine handler'
        else:np.seterr(all=spec['policy'])
        if spec['policy']!='default':np.seterrcall(handler if spec['policy']=='call' else old_handler)
        policy_before=np.geterr();handler_before=np.geterrcall()
        if spec['trace']:
            original.np=Proxy();original.as_binary_ndarray=helper('binary-conversion',old_binary)
            original.check_shape_equality=helper('shape-check',old_shape)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sys.setprofile(profile)
            try:
                row['phase']='original-call';row['original_call_attempted']=True
                result=function(*args,**kwargs)
            except Exception as exc:
                row['error']=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc))
            finally:sys.setprofile(old_profile)
            row['warnings']=[dict(module=type(w.message).__module__,category=w.category.__name__,message=str(w.message)) for w in caught]
        row['phase']='result-snapshot'
        row['result']=value(result,np,assets) if row['error'] is None else None
        row['result_aliases']=[result is a for a in arrays]
        row['state']=dict(policy_before=policy_before,policy_after=np.geterr(),
                   handler_identity_unchanged=np.geterrcall() is handler_before,
                   buffer_before=old_buffer,buffer_after=np.getbufsize(),entry_identity_unchanged=getattr(original,spec['api']) is entry,
                   providers_unchanged=all(getattr(np,k) is v for k,v in providers.items()))
        row['phase']='post-snapshot'
        row['input_after']=snapshots(arrays,np,assets)
        row['phase']='complete'
    except Exception as exc:
        row['observer_error']=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        try:
            sys.setprofile(old_profile)
            original.np=old_np;original.as_binary_ndarray=old_binary;original.check_shape_equality=old_shape
            if spec['policy']!='default':np.seterr(**old_policy);np.seterrcall(old_handler)
            row['checks'].update(policy_restored=np.geterr()==old_policy,handler_restored=np.geterrcall() is old_handler,
                buffer_restored=np.getbufsize()==old_buffer,provider_module_restored=original.np is old_np,
                binary_helper_restored=original.as_binary_ndarray is old_binary,shape_helper_restored=original.check_shape_equality is old_shape,
                profiler_restored=sys.getprofile() is old_profile)
            row['restored']=all(row['checks'].values())
        except Exception as exc:
            row['restoration_error']=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            row['checks']['restoration_completed']=False
    row['checks'].update(original_entry_exact=scalar_reads==dict(entry_calls=1,entry_returns=1),
                        no_observer_error=row['observer_error'] is None,complete=row['phase']=='complete')
    if row['state'] is not None:
        state=row['state'];row['checks'].update(call_policy_unchanged=state['policy_before']==state['policy_after'],
            call_handler_unchanged=state['handler_identity_unchanged'],call_buffer_unchanged=state['buffer_before']==state['buffer_after'],
            call_entry_unchanged=state['entry_identity_unchanged'],call_providers_unchanged=state['providers_unchanged'])
    return retain_then_check(row,journal)
