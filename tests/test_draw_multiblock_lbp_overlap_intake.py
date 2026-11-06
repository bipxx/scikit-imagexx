"""Only added factory/codec and cold/cached import REQUEST refusals."""
import base64
import builtins
import hashlib
import json
import sys
import types
import zlib
import numpy as np
import pytest
import draw_multiblock_lbp_overlap_contract as c


def test_added_factory_codec_no_target_request(monkeypatch):
    old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name.split('.')[0] in ('skimage','skimagexx'):
            requests.append(name);raise AssertionError('TARGET_REQUEST_FORBIDDEN')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',refuse)
    args,options,events,buffers=c.fixture(c.added_case(),np)
    image=args[0]
    assert image.shape==(3,3,3) and image.dtype==np.float32 and image.flags.owndata and image.flags.c_contiguous
    assert args[1:]==(-3,-3,2,2) and options['lbp_code']==170 and options['alpha']==0.3
    assert image.reshape(-1).tolist()==[i%7 for i in range(27)]
    before=image.tobytes();encoded=c.legacy.encode((args,options,buffers),np);json.dumps(encoded,allow_nan=False)
    def visit(value):
        if isinstance(value,dict):
            if 'raw_zlib_base64' in value:
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(encoded)
    assert image.tobytes()==before and events==[] and requests==[]


def binding(tmp_path):
    raw=c.CONTRACT.read_bytes();d=json.loads(raw)
    value=dict(schema='draw-multiblock-lbp-intake-v2',mode='intake-only',contract_sha256=hashlib.sha256(raw).hexdigest(),source_files_sha256=d['source_freeze'],original_calls=0)
    path=tmp_path/'binding.json';path.write_text(json.dumps(value));return path


@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('target',('numpy','skimage.feature'))
def test_real_bound_cli_request_refusal(tmp_path,monkeypatch,cached,target):
    path=binding(tmp_path);old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name==target:requests.append(name);raise RuntimeError('DRAW_MULTIBLOCK_REQUEST_REFUSED')
        return old(name,*a,**k)
    if cached:monkeypatch.setitem(sys.modules,target,types.ModuleType(target))
    else:monkeypatch.delitem(sys.modules,target,raising=False)
    monkeypatch.setattr(sys,'argv',[str(c.__file__),'--intake-binding',str(path)])
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(RuntimeError,match='DRAW_MULTIBLOCK_REQUEST_REFUSED'):c.main()
    assert requests==[target]


@pytest.mark.parametrize('fault',('budget','extra','contract','source'))
def test_bad_binding_refuses_before_science(tmp_path,monkeypatch,fault):
    path=binding(tmp_path);value=json.loads(path.read_bytes())
    if fault=='budget':value['original_calls']=1
    elif fault=='extra':value['execute']=True
    elif fault=='contract':value['contract_sha256']='0'*64
    else:value['source_files_sha256']={}
    path.write_text(json.dumps(value));old=builtins.__import__;requests=[]
    def refuse(name,*a,**k):
        if name.split('.')[0] in ('numpy','skimage','skimagexx'):requests.append(name);raise AssertionError('SCIENCE_REACHED')
        return old(name,*a,**k)
    monkeypatch.setattr(sys,'argv',[str(c.__file__),'--intake-binding',str(path)])
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(AssertionError):c.main()
    assert requests==[]
