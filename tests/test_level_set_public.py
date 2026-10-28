"""Original-first level-set contracts; no native completion is credited yet."""
import argparse
import base64
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import warnings
import zlib

import pytest
if __package__:
    from .level_set_contract import APIS, UNASSESSED, cases, fixture
    from .level_set_reference import reference_observation
else:
    from level_set_contract import APIS, UNASSESSED, cases, fixture
    from level_set_reference import reference_observation

CASES=cases()
MATRIX=Path(__file__).parents[1]/'coverage/contracts/level-set.json'
SOURCE_SHA='71ced09bbab3918523c38abeda1235a79ca34eb7467d76875e0ef5b13d2092d6'


def runtime():
    import numpy as np
    import scipy
    import skimage
    module=importlib.import_module('skimage.segmentation.morphsnakes')
    assert skimage.__version__=='0.26.0'
    assert (np.__version__,scipy.__version__) in (('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    source=Path(module.__file__);raw=source.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    functions={api:getattr(module,api.split('.')[-1]) for api in APIS}
    return np,functions,dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
        source_path=str(source),source_raw_sha256=hashlib.sha256(raw).hexdigest(),source_canonical_lf_sha256=SOURCE_SHA,
        signatures={api:str(inspect.signature(f)) for api,f in functions.items()},license='BSD-3-Clause; third_party/scikit-image-LICENSE.txt',
        calls_per_spec=1,scope='Only the two original level-set generators; no skimagexx imports, native calls or timings')


def encode(value,np):
    if isinstance(value,(np.ndarray,np.generic)):
        raw=value.tobytes()
        # The two frozen Linux/x86-64 profiles store x87 80-bit scalars in
        # 16-byte slots. Their trailing six ABI padding bytes are undefined.
        # This is a scoped input observer rule, not portable extended math.
        x87=(isinstance(value,np.generic) and value.dtype.kind=='f'
             and value.dtype.itemsize==16 and np.finfo(value.dtype).nmant==63)
        if x87:
            assert sys.byteorder=='little'
            raw=raw[:10]
        result=dict(type=type(value).__module__+'.'+type(value).__name__,dtype=value.dtype.str,dtype_metadata=value.dtype.metadata,
                    shape=list(value.shape),raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        if x87:result['raw_representation']='x87-80-little-endian; six undefined ABI padding bytes excluded'
        if isinstance(value,np.ndarray):result.update(strides=list(value.strides),own=bool(value.flags.owndata),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),writeable=bool(value.flags.writeable),base_type=type(value.base).__name__)
        return result
    if isinstance(value,dict):return {key:encode(item,np) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np) for item in value])
    if type(value).__name__ in ('ShapeProbe','IndexProbe','CenterProbe','generator'):return dict(protocol=type(value).__name__)
    if isinstance(value,complex):return dict(complex=[value.real,value.imag])
    if isinstance(value,float) and not math.isfinite(value):return dict(float=str(value))
    return value


def observe(call,spec,np):
    shape,options,events=fixture(spec,np)
    before=encode((shape,options),np);old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(shape,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            state=np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==old_buffer
    record=dict(id=spec['id'],api=spec['api'],input_before=before,input_after=encode((shape,options),np),result=encode(result,np),error=error,
                warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices],events=events,
                checks=dict(input_unchanged=encode((shape,options),np)==before,policy_unchanged=state,
                            state_restored=np.geterr()==old_policy and np.geterrcall() is old_handler and np.getbufsize()==old_buffer))
    assert all(record['checks'].values()),spec['id']
    if error is None:
        assert isinstance(result,(np.ndarray,np.int8)) and result.dtype==np.dtype(np.int8),spec['id']
        assert not np.any((result!=0)&(result!=1)),spec['id']
    return record


def test_level_set_frozen_manifest():
    record=json.loads(MATRIX.read_bytes())
    assert record['apis']==list(APIS) and record['cases']==CASES
    assert record['native_implementation']=='pending' and record['unassessed']==list(UNASSESSED)


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_level_set_saved_original_rows_have_exact_words_and_metadata(profile):
    record=json.loads(MATRIX.read_bytes())
    rows=record['original_observations'][profile]
    assert len(rows)==len(CASES) and [r['id'] for r in rows]==[s['id'] for s in CASES]
    for row in rows:
        assert all(row['checks'].values())
        if row['result'] is not None:
            if 'raw_zlib_base64' in row['result']:
                raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
                assert hashlib.sha256(raw).hexdigest()==row['result']['raw_sha256']
            else:
                assert next(s for s in CASES if s['id']==row['id'])['benchmark']
                assert len(row['result']['raw_sha256'])==64


@pytest.mark.parametrize('spec',CASES,ids=[r['id'] for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_level_set_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if spec['api'] not in native_functions():pytest.skip('Prospective contract only: no native level-set implementation registered')
    import numpy as np
    record=json.loads(MATRIX.read_bytes());expected=reference_observation(record,spec['id'],np)
    previous=xx.get_num_threads();xx.set_num_threads(workers);fallback=xx.set_fallback_enabled(False)
    try:
        candidate=getattr(xx.segmentation,spec['api'].split('.')[-1])
        assert str(inspect.signature(candidate))==record['signatures'][spec['api']]
        actual=observe(candidate,spec,np)
        if spec['benchmark'] and actual['result'] is not None:actual['result'].pop('raw_zlib_base64')
        assert actual==expected
    finally:xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert args.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    args.output.mkdir();np,functions,record=runtime()
    source=Path(record['source_path']);(args.output/'morphsnakes.py').write_bytes(source.read_bytes())
    license_path=Path(__file__).parents[1]/'third_party/scikit-image-LICENSE.txt'
    (args.output/'scikit-image-LICENSE.txt').write_bytes(license_path.read_bytes())
    record['license_sha256']=hashlib.sha256(license_path.read_bytes()).hexdigest()
    (args.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[]
    with(args.output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            row=observe(functions[spec['api']],spec,np)
            journal.write(json.dumps(row)+'\n');journal.flush();rows.append(row)
    assert not any(n=='skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    summary=dict(status='ORIGINAL_OBSERVATIONS_ONLY',cases=len(rows),calls=len(rows),native_calls=0,timings=False,
                 returns=sum(r['error'] is None for r in rows),errors=sum(r['error'] is not None for r in rows))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


if __name__=='__main__':main()
