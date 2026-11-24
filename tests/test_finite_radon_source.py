import ast,hashlib,json,runpy
from pathlib import Path
import pytest
import finite_radon_contract as c

def test_source_contract_and_zero_budget():
 d=json.loads(c.CONTRACT.read_bytes());assert d['original_budget']==0 and d['upstream_sha256']==c.UPSTREAM_SHA
 assert d['signature']=={'frt2':'(a)','ifrt2':'(a)'}and not d['API_completion']
 for n,p in d['source_freeze'].items():assert hashlib.sha256((c.ROOT/n).read_bytes()).hexdigest()==p

def test_all_recipes_safe_unique_and_nonprime_not_rejected():
 rows=c.cases();assert len({r['id']for r in rows})==len(rows)
 for r in rows:assert c.source_safe(r)['original_budget']==0
 assert all(any(r['n']==4 and r['api']==api for r in rows)for api in('frt2','ifrt2'))

def test_observer_is_supplied_call_only_and_preserves_exception_descriptors():
 s=(c.ROOT/'tests/finite_radon_observer.py').read_text();tree=ast.parse(s)
 assert 'numpy.ufunc'in s and 'numpy.dtype'in s and 'qualname'in s
 assert 'skimage'not in s and 'fesetenv'not in s
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
 assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'for n in ast.walk(f))==1

def test_public_one_thread_case_method():
 for api in('frt2','ifrt2'):
  d=runpy.run_path(str(c.ROOT/'benchmarks'/f'{api}.py'));assert len(d['case_specs']())==4
  assert d['METHOD']==dict(threads=1,repeat=7,target_ms=100,warmup_min_ms=250,retain_all_losses=True,status='UNBOUND_NO_TIMING')

def test_missing_intake_refuses_before_runtime(monkeypatch):
 monkeypatch.setattr(c,'runtime_request',lambda:pytest.fail('runtime reached'));monkeypatch.setattr('sys.argv',['contract'])
 with pytest.raises(SystemExit,match='FINITE_RADON_UNBOUND'):c.main()


def test_real_public_star_exports_include_both_owned_Radon_entries(monkeypatch):
 import sys,types
 record=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v1.json').read_bytes())
 newest312_raw=(c.ROOT/record['source']).read_bytes()
 newest312_guard=Path(__file__).read_bytes()
 v6=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v6.json').read_bytes())
 assert v6['schema']=='finite-radon-public-exports-v6' and v6['original_calls']==0
 assert v6['previous_record']=='coverage/contracts/finite-radon-public-exports-v5.json'
 assert hashlib.sha256((c.ROOT/v6['previous_record']).read_bytes()).hexdigest()==v6['previous_record_raw_sha256']
 assert hashlib.sha256(newest312_raw).hexdigest()==v6['current_raw_sha256']
 assert hashlib.sha256(newest312_guard).hexdigest()==v6['guard_raw_sha256']
 v6_suffix=v6['source_suffix'].encode()
 assert v6_suffix and newest312_raw.endswith(v6_suffix) and newest312_raw.count(v6_suffix)==1
 newest_raw=newest312_raw.removesuffix(v6_suffix)
 newest_guard=newest312_guard
 for edit in v6['guard_inverse']:
  block=edit['current'].encode()
  assert block and newest_guard.count(block)==1
  newest_guard=newest_guard.replace(block,edit['previous'].encode(),1)
 v5=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v5.json').read_bytes())
 assert v5['schema']=='finite-radon-public-exports-v5' and v5['original_calls']==0
 assert v5['previous_record']=='coverage/contracts/finite-radon-public-exports-v4.json'
 assert hashlib.sha256((c.ROOT/v5['previous_record']).read_bytes()).hexdigest()==v5['previous_record_raw_sha256']
 assert hashlib.sha256(newest_raw).hexdigest()==v5['current_raw_sha256']
 assert hashlib.sha256(newest_guard).hexdigest()==v5['guard_raw_sha256']
 v5_suffix=v5['source_suffix'].encode()
 assert v5_suffix and newest_raw.endswith(v5_suffix) and newest_raw.count(v5_suffix)==1
 live_raw=newest_raw.removesuffix(v5_suffix)
 live_guard=newest_guard
 for edit in v5['guard_inverse']:
  block=edit['current'].encode()
  assert block and live_guard.count(block)==1
  live_guard=live_guard.replace(block,edit['previous'].encode(),1)
 v4=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v4.json').read_bytes())
 assert v4['schema']=='finite-radon-public-exports-v4' and v4['original_calls']==0
 assert v4['previous_record']=='coverage/contracts/finite-radon-public-exports-v3.json'
 assert hashlib.sha256((c.ROOT/v4['previous_record']).read_bytes()).hexdigest()==v4['previous_record_raw_sha256']
 assert hashlib.sha256(live_raw).hexdigest()==v4['current_raw_sha256']
 assert hashlib.sha256(live_guard).hexdigest()==v4['guard_raw_sha256']
 suffix=v4['source_suffix'].encode()
 assert suffix and live_raw.endswith(suffix) and live_raw.count(suffix)==1
 current_raw=live_raw.removesuffix(suffix)
 current_guard=live_guard
 for edit in v4['guard_inverse']:
  block=edit['current'].encode()
  assert block and current_guard.count(block)==1
  current_guard=current_guard.replace(block,edit['previous'].encode(),1)
 attached=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v3.json').read_bytes())
 assert attached['schema']=='finite-radon-public-exports-v3' and attached['original_calls']==0
 assert attached['previous_record']=='coverage/contracts/finite-radon-public-exports-v2.json'
 assert hashlib.sha256((c.ROOT/attached['previous_record']).read_bytes()).hexdigest()==attached['previous_record_raw_sha256']
 assert hashlib.sha256(current_raw).hexdigest()==attached['current_raw_sha256']
 assert hashlib.sha256(current_guard).hexdigest()==attached['guard_raw_sha256']
 insertion=attached['source_fragment'].encode()
 assert insertion and current_raw.count(insertion)==1
 raw=current_raw.replace(insertion,b'',1)
 guard=current_guard
 for edit in attached['guard_inverse']:
  block=edit['current'].encode()
  assert block and guard.count(block)==1
  guard=guard.replace(block,edit['previous'].encode(),1)
 successor=json.loads((c.ROOT/'coverage/contracts/finite-radon-public-exports-v2.json').read_bytes())
 assert successor['schema']=='finite-radon-public-exports-v2'
 assert successor['previous_record']=='coverage/contracts/finite-radon-public-exports-v1.json'
 assert hashlib.sha256((c.ROOT/successor['previous_record']).read_bytes()).hexdigest()==successor['previous_record_raw_sha256']
 assert hashlib.sha256(raw).hexdigest()==successor['current_raw_sha256']
 addition=successor['source_suffix'].encode()
 assert addition and raw.endswith(addition) and raw.count(addition)==1
 raw=raw.removesuffix(addition)
 assert hashlib.sha256(raw).hexdigest()==record['current_raw_sha256']
 fragment=record['inverse_fragment'].encode();assert raw.count(fragment)==1
 predecessor=raw.replace(fragment,b'',1)
 assert hashlib.sha256(predecessor).hexdigest()==record['predecessor_raw_sha256']
 assert hashlib.sha256(guard).hexdigest()==successor['guard_raw_sha256']
 current_block=successor['current_guard_block'].encode()
 previous_block=successor['previous_guard_block'].encode()
 assert current_block and guard.count(current_block)==1
 guard=guard.replace(current_block,previous_block,1)
 assert hashlib.sha256(guard).hexdigest()==record['guard_raw_sha256']
 # Execute only actual export/import declarations. Existing public names are
 # inert objects; no NumPy, frontend, core or provider body is imported/called.
 parent=types.ModuleType('skimagexx');parent.__path__=[]
 provider=types.ModuleType('skimagexx._finite_radon')
 provider.frt2=object();provider.ifrt2=object()
 monkeypatch.setitem(sys.modules,'skimagexx',parent)
 monkeypatch.setitem(sys.modules,'skimagexx._finite_radon',provider)
 for text,current in ((predecessor,False),(raw,True),(current_raw,True),(live_raw,True),(newest_raw,True),(newest312_raw,True)):
  tree=ast.parse(text);declarations=[]
  for node in tree.body:
   if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='__all__'for t in node.targets):declarations.append(node)
   elif isinstance(node,ast.AugAssign)and isinstance(node.target,ast.Name)and node.target.id=='__all__':declarations.append(node)
   elif isinstance(node,ast.ImportFrom)and node.level==1 and node.module=='_finite_radon':declarations.append(node)
  module=types.ModuleType('skimagexx.transform');module.__package__='skimagexx'
  exec(compile(ast.Module(body=declarations,type_ignores=[]),'<actual public export declarations>','exec'),module.__dict__)
  for name in module.__all__:module.__dict__.setdefault(name,object())
  monkeypatch.setitem(sys.modules,'skimagexx.transform',module);parent.transform=module
  exported={};exec('from skimagexx.transform import *',{},exported)
  for name in ('frt2','ifrt2'):
   assert getattr(module,name)is getattr(provider,name)
   if current:
    assert module.__all__.count(name)==1 and exported[name]is getattr(provider,name)
   else:assert name not in module.__all__ and name not in exported
