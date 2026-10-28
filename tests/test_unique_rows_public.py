"""Original-first unique_rows contracts; no native registration yet."""
import argparse
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
import zlib
import pytest
if __package__:
    from .unique_rows_contract import API,UNASSESSED,ArrayProvider,cases,fixture
else:
    from unique_rows_contract import API,UNASSESSED,ArrayProvider,cases,fixture

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/unique-rows.json'
SOURCE_SHA='72d88a45b800ed9adae7ec80e7c80aba9d1f92c3ee3d21d8731fcefc753bd751'


def runtime():
    import numpy as np,scipy,skimage
    module=importlib.import_module('skimage.util.unique')
    assert skimage.__version__=='0.26.0'
    assert (np.__version__,scipy.__version__)in(('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    source=Path(module.__file__);raw=source.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    return np,module.unique_rows,dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,signature=str(inspect.signature(module.unique_rows)),source_path=str(source),source_raw_sha256=hashlib.sha256(raw).hexdigest(),source_canonical_lf_sha256=SOURCE_SHA,calls_per_spec=1,native_calls=0,timings=False)


def encode(value,np):
    if isinstance(value,ArrayProvider):return dict(protocol='ArrayProvider',dimension=value.dimension,array=encode(value.array,np))
    if isinstance(value,np.ndarray):
        result=dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=json.loads(json.dumps(value.dtype.descr)),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__)
        if value.dtype.hasobject:result['object_values']=encode(value.tolist(),np)
        else:
            raw=value.tobytes();result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        return result
    if isinstance(value,np.generic):
        if value.dtype.hasobject:return dict(type=type(value).__name__,value=encode(value.item(),np))
        return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    return value


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np);named=dict(ar=args[0]);before=encode((named,buffers),np)
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            state=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    original=args[0].array if isinstance(args[0],ArrayProvider)else args[0]
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((named,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=events,result_is_input=result is args[0],result_shares_input=isinstance(result,np.ndarray)and isinstance(original,np.ndarray)and bool(np.shares_memory(result,original)),checks=dict(policy_unchanged=state,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer))
    assert all(row['checks'].values()),spec['id']
    assert row['input_before']==row['input_after'],spec['id']
    if error is None:assert type(result)is np.ndarray and result.ndim==2 and not row['result_is_input']and not row['result_shares_input']
    return row


def test_unique_rows_frozen_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['cases']==CASES and matrix['unassessed']==list(UNASSESSED)
    assert matrix['native_implementation']=='pending'


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_unique_rows_saved_original_rows(profile):
    rows=json.loads(MATRIX.read_bytes())['original_observations'][profile]
    assert len(rows)==52 and [row['id']for row in rows]==[spec['id']for spec in CASES]
    for row in rows:
        assert all(row['checks'].values())and row['input_before']==row['input_after']
        if row['result']is not None:
            raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
            assert hashlib.sha256(raw).hexdigest()==row['result']['raw_sha256']


@pytest.mark.parametrize('spec',CASES,ids=[row['id']for row in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_unique_rows_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if API not in native_functions():pytest.skip('Prospective unique_rows contract; native registration pending')
    import numpy as np
    matrix=json.loads(MATRIX.read_bytes());expected=next(row for row in matrix['original_observations'][np.__version__]if row['id']==spec['id'])
    previous=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        assert str(inspect.signature(xx.util.unique_rows))==matrix['signature']
        assert observe(xx.util.unique_rows,spec,np)==expected
    finally:xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args();assert options.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    options.output.mkdir();np,original,record=runtime()
    for name,path in (('unique.py',Path(record['source_path'])),('scikit-image-LICENSE.txt',Path(__file__).parents[1]/'third_party/scikit-image-LICENSE.txt')):
        raw=path.read_bytes();(options.output/name).write_bytes(raw)
        if name.endswith('LICENSE.txt'):record['license_sha256']=hashlib.sha256(raw).hexdigest()
    (options.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    with(options.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            row=observe(original,spec,np);journal.write(json.dumps(row)+'\n');journal.flush();rows.append(row)
    assert not any(n=='skimagexx'or n.startswith('skimagexx.')for n in sys.modules)
    summary=dict(status='ORIGINAL_OBSERVATIONS_ONLY',cases=len(rows),calls=len(rows),native_calls=0,timings=False,returns=sum(r['error']is None for r in rows),errors=sum(r['error']is not None for r in rows))
    (options.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


if __name__=='__main__':main()
