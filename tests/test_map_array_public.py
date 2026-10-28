"""Original-first map_array contracts; native implementation remains pending."""
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
    from .map_array_contract import API,UNASSESSED,FloatBox,cases,fixture
else:
    from map_array_contract import API,UNASSESSED,FloatBox,cases,fixture

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/map-array.json'
SOURCE_SHA='aaca3e8ec753ab84d793a39e7aa71db991d5d39dd457cba96d11b4aba1f2cfe2'


def runtime():
    import numpy as np,scipy,skimage
    module=importlib.import_module('skimage.util._map_array');primitive=importlib.import_module('skimage.util._remap')
    assert skimage.__version__=='0.26.0'
    assert (np.__version__,scipy.__version__) in (('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    source=Path(module.__file__);raw=source.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    record=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,signature=str(inspect.signature(module.map_array)),source_path=str(source),source_raw_sha256=hashlib.sha256(raw).hexdigest(),source_canonical_lf_sha256=SOURCE_SHA,primitive_path=str(primitive.__file__),primitive_sha256=hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest(),calls_per_spec=1,native_calls=0,timings=False)
    return np,module.map_array,record


def encode(value,np):
    if isinstance(value,np.ndarray):
        result=dict(type=type(value).__name__,dtype=value.dtype.str,dtype_metadata=value.dtype.metadata,
                    shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),
                    writeable=bool(value.flags.writeable),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__)
        if value.dtype.hasobject:result['object_values']=[encode(item,np) for item in value.flat]
        else:
            raw=value.tobytes();result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        return result
    if isinstance(value,FloatBox):return dict(protocol='FloatBox',value=value.value)
    if isinstance(value,np.generic):return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,dict):return {key:encode(item,np) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np) for item in value])
    return value


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np);named=dict(zip(('input','keys','values'),args),**options)
    before=encode((named,buffers),np);old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            state=np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==old_buffer
    after=encode((named,buffers),np)
    identity=[name for name,value in named.items() if result is not None and result is value]
    shared=[name for name,value in named.items() if isinstance(result,np.ndarray) and isinstance(value,np.ndarray) and np.shares_memory(result,value)]
    row=dict(id=spec['id'],api=API,input_before=before,input_after=after,result=encode(result,np),error=error,
             warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices],events=events,
             result_identity=identity,result_shares_memory=shared,
             checks=dict(policy_unchanged=state,state_restored=np.geterr()==old_policy and np.geterrcall() is old_handler and np.getbufsize()==old_buffer))
    assert all(row['checks'].values()),spec['id']
    if error is None:
        assert isinstance(result,np.ndarray) and result.shape==args[0].shape
        if 'out' in options:assert result is options['out']
    return row


def test_map_array_frozen_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['cases']==CASES and matrix['unassessed']==list(UNASSESSED)
    assert matrix['native_implementation']=='pending'


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_map_array_saved_original_rows(profile):
    rows=json.loads(MATRIX.read_bytes())['original_observations'][profile]
    assert len(rows)==50 and [row['id'] for row in rows]==[spec['id'] for spec in CASES]
    for row in rows:
        assert all(row['checks'].values())
        if row['result'] is not None:
            raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
            assert hashlib.sha256(raw).hexdigest()==row['result']['raw_sha256']


@pytest.mark.parametrize('spec',CASES,ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_map_array_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if API not in native_functions():pytest.skip('Prospective original-first map_array contract; native registration pending')
    import numpy as np
    matrix=json.loads(MATRIX.read_bytes());expected=next(row for row in matrix['original_observations'][np.__version__] if row['id']==spec['id'])
    previous=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        assert str(inspect.signature(xx.util.map_array))==matrix['signature']
        assert observe(xx.util.map_array,spec,np)==expected
    finally:xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args();assert options.reference_only and sys.flags.optimize==0
    assert not any(name=='skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    options.output.mkdir();np,original,record=runtime()
    for name,path in (('_map_array.py',Path(record['source_path'])),('scikit-image-LICENSE.txt',Path(__file__).parents[1]/'third_party/scikit-image-LICENSE.txt')):
        raw=path.read_bytes();(options.output/name).write_bytes(raw)
        if name.endswith('LICENSE.txt'):record['license_sha256']=hashlib.sha256(raw).hexdigest()
    (options.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    with(options.output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            row=observe(original,spec,np);journal.write(json.dumps(row)+'\n');journal.flush();rows.append(row)
    assert not any(name=='skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    summary=dict(status='ORIGINAL_OBSERVATIONS_ONLY',cases=len(rows),calls=len(rows),native_calls=0,timings=False,returns=sum(row['error'] is None for row in rows),errors=sum(row['error'] is not None for row in rows))
    (options.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


if __name__=='__main__':main()
