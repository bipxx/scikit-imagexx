"""Shared Montage exports and frozen observations; no new originals."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import struct
import sys

import pytest

if __package__:
    from .montage_contract import cases
    from .test_montage_public import observe
else:
    from montage_contract import cases
    from test_montage_public import observe

ROOT=Path(__file__).parents[1]
CASES=cases()


def test_montage_benchmark_contract_metadata():
    contract=json.loads((ROOT/'coverage/contracts/montage-benchmark.json').read_bytes())
    assert contract['threads']==1 and contract['repeat']==7 and contract['raw_samples']==126
    specs=contract['cases'];assert len(specs)==len({s['case_id']for s in specs})==9
    for spec in specs:
        shape=spec['shape'];channels=shape[spec['channel_axis']]if spec['channel_axis']is not None else 1
        height,width=(shape[2],shape[3])if spec['layout']=='channels-first'else(shape[1],shape[2])
        padding=spec['padding'];rows,columns=spec['grid']
        assert shape[0]<=rows*columns
        assert ((height+padding)*rows+padding)*((width+padding)*columns+padding)*channels<=contract['max_output_elements']


def test_montage_independent_oracle_metadata():
    spec=importlib.util.spec_from_file_location('montage_benchmark_metadata',ROOT/'benchmarks/montage.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw,shape=module.oracle_tiles(bytes((1,2,3,4)),(2,1,2),dict(dtype='uint8',fill=9,grid=[2,2],padding=1,channel_axis=None))
    assert shape==[5,7] and raw==bytes([9]*7+[9,1,2,9,3,4,9]+[9]*21)
    raw,shape=module.oracle_tiles(struct.pack('=dd',1.,3.),(2,1,1),dict(dtype='float64',fill='mean',grid=[2,2],padding=0,channel_axis=None))
    assert shape==[2,2]and raw==struct.pack('=dddd',1.,3.,2.,2.)


@pytest.fixture(scope='module')
def product():
    import numpy as np,skimagexx as xx
    return np,xx,json.loads((ROOT/'coverage/contracts/montage.json').read_bytes())


def test_montage_public_exports(product):
    np,xx,matrix=product
    from skimagexx import _montage as implementation
    from skimagexx.coverage import native_functions
    assert xx.montage is xx.util.montage is implementation.montage
    assert str(inspect.signature(xx.util.montage))==matrix['signature']
    assert 'util.montage'in native_functions()
    assert inspect.isbuiltin(implementation._core.montage_apply)


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
def test_montage_registered_public_frozen_contract(product,spec,workers,monkeypatch,tmp_path):
    np,xx,matrix=product
    from skimagexx import _montage as implementation
    from .test_montage_pipeline import NATIVE_IDS
    import skimage.util as public
    import skimage.util._montage as original
    expected=next(r for r in matrix['original_observations'][np.__version__]if r['id']==spec['id'])
    entry=implementation._core.montage_apply
    body=inspect.unwrap(implementation.montage).__code__
    reference=inspect.unwrap(original.montage).__code__
    previous,prior=xx.get_num_threads(),sys.getprofile();calls=[];accepted=[]
    def reject(*a,**kw):raise AssertionError('registered montage forwarded to original whole API')
    def profile(frame,event,value):
        if event=='call'and frame.f_code is reference:reject()
        if event=='c_return'and value is entry:calls.append('montage_apply')
        if event=='return'and frame.f_code is body:accepted.append(frame.f_locals.get('native_result')is True)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(public,'montage',reject);patch.setattr(original,'montage',reject)
            sys.setprofile(profile)
            try:actual=observe(xx.util.montage,spec,np)
            finally:sys.setprofile(prior)
        assert actual==expected
        native_expected=spec['id']in NATIVE_IDS or (spec['id']=='fill-array' and np.__version__=='1.24.4')
        if native_expected:assert calls==['montage_apply']and accepted==[True]
        else:assert not any(accepted)
        assert xx.get_num_threads()==workers
        (tmp_path/'actual.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,observation=actual,C_returns=calls,native_accepted=accepted))+'\n')
    finally:
        sys.setprofile(prior);xx.set_num_threads(previous)


@pytest.mark.parametrize('shape',((1,1),(3,19),(519,521)))
@pytest.mark.parametrize('dtype',('uint8','float32','float64'))
@pytest.mark.parametrize('workers',(1,4))
def test_montage_default_adapter_exact_single_image(product,shape,dtype,workers):
    np,xx,_=product
    from skimagexx import _montage as implementation
    from skimagexx.coverage import default_call,montage_default_error,assert_default_parity
    image=np.arange(shape[0]*shape[1],dtype=dtype).reshape(shape)
    args,options=default_call('util.montage',image)
    assert args[0].shape==(1,*shape)and np.shares_memory(args[0],image)and not options
    # A one-tile, unpadded montage is byte-for-byte the input in a fresh 3D base.
    owner=np.empty((*shape,1),dtype=image.dtype);owner[...,0]=image;expected=owner[...,0]
    entry=implementation._core.montage_apply;body=inspect.unwrap(implementation.montage).__code__
    prior=sys.getprofile();previous=xx.get_num_threads();calls=[];accepted=[]
    def profile(frame,event,value):
        if event=='c_return'and value is entry:calls.append('montage_apply')
        if event=='return'and frame.f_code is body:accepted.append(frame.f_locals.get('native_result')is True)
    try:
        xx.set_num_threads(workers);sys.setprofile(profile)
        try:actual=xx.util.montage(*args,**options)
        finally:sys.setprofile(prior)
        assert calls==['montage_apply']and accepted==[True]
        assert montage_default_error(image,actual,expected)==0.
        assert_default_parity('util.montage',np.dtype(dtype),actual,expected,image=image)
    finally:
        sys.setprofile(prior);xx.set_num_threads(previous)
