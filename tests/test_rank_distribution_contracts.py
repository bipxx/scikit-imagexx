"""Pinned source contracts for rank distribution algorithms; forwarding is separate."""
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
from skimage.filters import rank as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

spec=importlib.util.spec_from_file_location('rank_v1_contracts',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
NAMES=('gradient','median','modal','entropy')
MATRIX=Path(__file__).parents[1]/'coverage/contracts/rank-distribution.json'
CASES=[(name+'/'+key.split('/',1)[1],name,dict(parameters)) for name in NAMES for key,_,parameters in base.CASES if key.startswith('minimum/') and parameters.get('out') not in ('image','view','overlap') and parameters.get('special')!='alias-converted']
for name in NAMES:
    for ndim in (2,3):
        for pattern in ('upper-tie','lowest-mode-tie','u8-fullrange','u16-fullrange','zero-window','disconnected','entropy-thirds','entropy-skew'):
            CASES.append((f'{name}/{ndim}d/distribution={pattern}',name,dict(ndim=ndim,distribution=pattern)))
    if name=='median':
        for ndim in (2,3):CASES.append((f'{name}/{ndim}d/default-footprint',name,dict(ndim=ndim,omitted=True)))


def profile():
    return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


def make_case(parameters):
    if 'distribution' not in parameters and not parameters.get('omitted'):
        return base.make_case(parameters)
    ndim=parameters['ndim'];shape=(5,7) if ndim==2 else (3,4,5)
    pattern=parameters.get('distribution')
    values=np.asarray([1,9,1,9,4,4,8],np.uint8)
    if pattern=='upper-tie':values=np.asarray([1,9],np.uint8)
    elif pattern=='lowest-mode-tie':values=np.asarray([2,2,7,7,9],np.uint8)
    elif pattern=='u8-fullrange':values=np.asarray([0,255,1,254,128],np.uint8)
    elif pattern=='u16-fullrange':values=np.asarray([0,65535,1,65534,32768],np.uint16)
    elif pattern=='entropy-thirds':values=np.asarray([0,1,2],np.uint8)
    elif pattern=='entropy-skew':values=np.asarray([0,0,0,0,1,1,2],np.uint8)
    image=np.resize(values,int(np.prod(shape))).reshape(shape)
    fp=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8)
    options={};owners={}
    if pattern=='zero-window':options['mask']=np.zeros(shape,np.uint8)
    if pattern=='disconnected':fp.flat[1:-1]=0
    tracked={'image':image,**({} if parameters.get('omitted') else {'footprint':fp}),**options,**owners}
    return (image,) if parameters.get('omitted') else (image,fp),options,tracked


def observe(function,parameters):
    args,options,tracked=make_case(parameters)
    inputs=[value for value in tracked.values() if isinstance(value,np.ndarray)]
    before={key:describe(value) for key,value in tracked.items()}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=describe(function(*args,**options),inputs),None
        except Exception as caught:result,error=None,dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={key:describe(value) for key,value in tracked.items()})


def describe(value,inputs=()):
    result=base.describe(value,inputs)
    if isinstance(value,np.ndarray) and not value.dtype.hasobject:
        result['raw_bytes_sha256']=hashlib.sha256(value.tobytes()).hexdigest()
    return result


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text(encoding='utf-8'))


def test_frozen_corpus_integrity(frozen):
    assert len(CASES)==926 and frozen['native_pending']==926 and frozen['native_checks']==0
    for snapshot in frozen['snapshots'].values():assert set(snapshot)=={key for key,_,_ in CASES}
    for digest,value in frozen['observations'].items():
        assert hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()==digest


@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_contract(case,frozen):
    key,name,parameters=case
    assert observe(getattr(reference,name),parameters)==frozen['observations'][frozen['snapshots'][profile()][key]],key


@pytest.mark.parametrize('name',NAMES)
def test_source_signatures_aliases_and_math_contract(name,frozen):
    generic=importlib.import_module('skimage.filters.rank.generic')
    function=getattr(reference,name);contract=frozen['functions'][name]
    assert str(inspect.signature(function))==contract['signature']
    assert hashlib.sha256(inspect.getsource(function).encode()).hexdigest()==contract['python_source_sha256']
    assert name in reference.__all__ and name in generic.__all__
    assert getattr(generic,name) is function
    assert contract['aliases']==['filters.rank.generic.'+name]
    assert contract['native_checks']==0 and contract['native_pending']==sum(c[1]==name for c in CASES)


@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
@pytest.mark.parametrize('namespace',['filters.rank','filters.rank.generic'])
@pytest.mark.parametrize('workers',[1,4])
def test_strict_pending_separately(case,namespace,workers):
    _,name,parameters=case
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        function=getattr(importlib.import_module('skimagexx.'+namespace),name)
        result=observe(function,parameters)
        if 'filters.rank.'+name not in native_functions():
            assert result['error']['type']=='NotImplementedError'
            assert 'fallback is disabled' in result['error']['message']
            assert result['result'] is None and not result['warnings'] and result['after']==result['before']
        else:assert result==observe(getattr(reference,name),parameters)
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(previous)
