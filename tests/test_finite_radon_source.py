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
 raw=(c.ROOT/record['source']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==record['current_raw_sha256']
 fragment=record['inverse_fragment'].encode();assert raw.count(fragment)==1
 predecessor=raw.replace(fragment,b'',1)
 assert hashlib.sha256(predecessor).hexdigest()==record['predecessor_raw_sha256']
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==record['guard_raw_sha256']
 # Execute only actual export/import declarations. Existing public names are
 # inert objects; no NumPy, frontend, core or provider body is imported/called.
 parent=types.ModuleType('skimagexx');parent.__path__=[]
 provider=types.ModuleType('skimagexx._finite_radon')
 provider.frt2=object();provider.ifrt2=object()
 monkeypatch.setitem(sys.modules,'skimagexx',parent)
 monkeypatch.setitem(sys.modules,'skimagexx._finite_radon',provider)
 for text,current in ((predecessor,False),(raw,True)):
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
