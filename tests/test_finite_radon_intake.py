"""Safe NumPy factories/codec and actual cold/cached import REQUEST refusals."""
import builtins,hashlib,json,sys,types
import numpy as np
import pytest
import finite_radon_contract as c
from crop_observer import encode

@pytest.mark.parametrize('row',c.cases(),ids=lambda r:r['id'])
def test_safe_factory_only(row,monkeypatch):
 old=builtins.__import__
 def refuse(n,*a,**k):
  if n.split('.')[0]in('skimage','skimagexx'):raise AssertionError('TARGET_REQUEST')
  return old(n,*a,**k)
 monkeypatch.setattr(builtins,'__import__',refuse)
 args,kw,events,buffers=c.fixture(row,np);before=buffers['image'].tobytes();json.dumps(encode((args,kw),np),allow_nan=False)
 assert before==buffers['image'].tobytes()and not events and kw=={}

def binding(tmp_path):
 raw=c.CONTRACT.read_bytes();d=json.loads(raw);p=tmp_path/'binding.json';p.write_text(json.dumps(dict(schema='finite-radon-intake-v1',mode='intake-only',contract_sha256=hashlib.sha256(raw).hexdigest(),source_files_sha256=d['source_freeze'],original_calls=0)));return p

@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('target',('numpy','skimage.transform'))
def test_bound_real_cli_request_refusal(tmp_path,monkeypatch,cached,target):
 p=binding(tmp_path);old=builtins.__import__;requests=[]
 def refuse(n,*a,**kw):
  if n==target:requests.append(n);raise RuntimeError('FINITE_RADON_REQUEST_REFUSED')
  return old(n,*a,**kw)
 if cached:monkeypatch.setitem(sys.modules,target,types.ModuleType(target))
 else:monkeypatch.delitem(sys.modules,target,raising=False)
 monkeypatch.setattr(builtins,'__import__',refuse);monkeypatch.setattr(sys,'argv',['contract','--intake-binding',str(p)])
 with pytest.raises(RuntimeError,match='FINITE_RADON_REQUEST_REFUSED'):c.main()
 assert requests==[target]

@pytest.mark.parametrize('fault',('budget','extra','contract','source'))
def test_invalid_before_science(tmp_path,monkeypatch,fault):
 p=binding(tmp_path);d=json.loads(p.read_bytes())
 if fault=='budget':d['original_calls']=1
 elif fault=='extra':d['execute']=True
 elif fault=='contract':d['contract_sha256']='0'*64
 else:d['source_files_sha256']={}
 p.write_text(json.dumps(d));monkeypatch.setattr(c,'runtime_request',lambda:pytest.fail('runtime reached'));monkeypatch.setattr(sys,'argv',['contract','--intake-binding',str(p)])
 with pytest.raises(AssertionError):c.main()
