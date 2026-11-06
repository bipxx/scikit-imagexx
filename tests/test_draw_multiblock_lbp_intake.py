"""Safe real NumPy construction/codec and actual REQUEST-refusal only."""
import base64
import builtins
import hashlib
import json
from pathlib import Path
import sys
import types
import zlib
import numpy as np
import pytest
import draw_multiblock_lbp_contract as contract


@pytest.mark.parametrize('row',contract.cases(),ids=lambda s:s['id'])
def test_actual_numpy_factory_and_reused_codec_no_target_request(row,monkeypatch):
    old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name.split('.')[0]in('skimage','skimagexx'):
            requests.append(name);raise AssertionError('TARGET_REQUEST_FORBIDDEN')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',refuse)
    args,options,events,buffers=contract.fixture(row,np)
    assert events==[]and all(v.size<=4096 for v in buffers.values())
    state=contract.encode((args,options,buffers),np);json.dumps(state,allow_nan=False)
    def visit(value):
        if isinstance(value,dict):
            if 'raw_zlib_base64'in value:
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(state)
    assert events==[]and not requests
    image=buffers['image']
    assert list(image.shape)==row['shape']and image.dtype==np.dtype(row['dtype'])
    if row['layout']=='F':assert image.flags.f_contiguous
    elif row['layout']=='reverse':assert image.strides[-1]<0
    elif row['layout']=='readonly':assert not image.flags.writeable
    elif row['layout']=='strided':assert image.strides[0]>=2*image.dtype.itemsize


def binding(tmp_path):
    raw=contract.CONTRACT.read_bytes();data=json.loads(raw)
    value=dict(schema='draw-multiblock-lbp-intake-v1',mode='intake-only',contract_sha256=hashlib.sha256(raw).hexdigest(),source_files_sha256=data['source_freeze'],original_calls=0)
    path=tmp_path/'binding.json';path.write_text(json.dumps(value));return path


@pytest.mark.parametrize('target',('numpy','skimage.feature'))
@pytest.mark.parametrize('cached',(False,True))
def test_actual_bound_CLI_import_REQUEST_refusal(tmp_path,monkeypatch,target,cached):
    path=binding(tmp_path);old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name==target:requests.append(name);raise RuntimeError('DRAW_MULTIBLOCK_REQUEST_REFUSED')
        return old(name,*a,**k)
    if cached:monkeypatch.setitem(sys.modules,target,types.ModuleType(target))
    else:monkeypatch.delitem(sys.modules,target,raising=False)
    monkeypatch.setattr(sys,'argv',[str(contract.__file__),'--intake-binding',str(path)])
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(RuntimeError,match='DRAW_MULTIBLOCK_REQUEST_REFUSED'):contract.main()
    assert requests==[target]


@pytest.mark.parametrize('fault',('budget','extra','contract','source'))
def test_bad_binding_stops_before_any_science_REQUEST(tmp_path,monkeypatch,fault):
    path=binding(tmp_path);value=json.loads(path.read_bytes())
    if fault=='budget':value['original_calls']=1
    elif fault=='extra':value['execute']=True
    elif fault=='contract':value['contract_sha256']='0'*64
    else:value['source_files_sha256']={}
    path.write_text(json.dumps(value));old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name.split('.')[0]in('numpy','skimage','skimagexx'):requests.append(name);raise AssertionError('SCIENCE_REACHED')
        return old(name,*a,**k)
    monkeypatch.setattr(sys,'argv',[str(contract.__file__),'--intake-binding',str(path)])
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(AssertionError):contract.main()
    assert requests==[]
