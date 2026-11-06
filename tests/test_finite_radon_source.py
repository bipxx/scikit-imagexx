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
