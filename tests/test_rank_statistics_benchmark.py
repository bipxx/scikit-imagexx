"""Harness guards only. Delegating unit stubs are never native evidence."""
import inspect,json,time,zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import rank_statistics as h

def forbidden(*args,**kwargs):raise AssertionError('forbidden fixture/native/timer work')
def tiny(name='geometric_mean',label='unit-only',shape=(3,5)):
 f=dict(h.profiles(name in h.VECTOR)[0],shape=list(shape),out='default')
 return h.Case(name,label,f)
def manifest(monkeypatch,tmp_path,cases=None):
 cases=cases or [tiny()];m=json.loads(h.MANIFEST.read_text());m['case_inventory']=[h.inventory(c) for c in cases];m['case_contract_sha256']=h.digest(m['case_inventory']);m['reference_profiles']={h.runtime_profile():dict(case_spec_sha256=h.digest([h.spec(c) for c in cases]),observations={str(i):h.digest(h.observe_case(h.resolve(c.name,True),c)) for i,c in enumerate(cases)})}
 p=tmp_path/'unit.json';p.write_text(json.dumps(m));monkeypatch.setattr(h,'MANIFEST',p);monkeypatch.setattr(h,'corpus',lambda:iter(cases));return cases
def stubs(monkeypatch,wrong=None):
 originals={n:h.resolve(n,True) for n in h.NAMES};wrappers={}
 for name in h.NAMES:
  def call(*args,_name=name,**kwargs):return originals[_name](*args,**kwargs)
  call.__signature__=inspect.signature(originals[name]);wrappers[name]=call
 monkeypatch.setitem(call.__globals__,'_core',h.xx._core)
 def resolve(name,original=False,alias=False):
  if original:return originals[name]
  if wrong=='forward':return originals[name]
  if wrong=='alias' and alias:return lambda *a,**kw:None
  return wrappers[name]
 if wrong=='signature':wrappers[h.NAMES[0]].__signature__=inspect.Signature()
 monkeypatch.setattr(h,'resolve',resolve);monkeypatch.setattr(h,'native_functions',lambda:h.APIS)
 for key,value in dict(rank_statistics_interface='rank-statistics-v1',rank_statistics_capabilities=0xe000,rank_vector_interface='rank-vector-v1',rank_vector_capabilities=0x10000).items():monkeypatch.setattr(h.xx._core,key,value,raising=False)
 monkeypatch.setattr(h,'compiled_entries',lambda core:None) # UNIT STUB ONLY
 monkeypatch.setattr(h,'verify',lambda *a:{'unit_stub':True});return wrappers

def test_reference_only_never_calls_native_or_timers(monkeypatch,tmp_path):
 manifest(monkeypatch,tmp_path);real=h.resolve
 monkeypatch.setattr(h,'resolve',lambda n,original=False,alias=False:real(n,True,alias) if original else forbidden())
 monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
 out=tmp_path/'ref.json';assert h.main(['--reference-only','--check-only','--output',str(out)])==0
 r=json.loads(out.read_text());assert r['metadata']['native_checks']==0 and r['metadata']['validation_scope']=='reference-only/native0' and not r['results'] and len(r['alias_checks'])==4

def test_missing_registry_blocks_every_fixture_capsule_and_timer(monkeypatch,tmp_path):
 monkeypatch.setattr(h,'native_functions',lambda:());monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'corpus',forbidden);monkeypatch.setattr(h,'verify',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'missing.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert 'not registered' in r['failures'][0]['message'] and not r['checks'] and not r['alias_checks'] and not r['results']

@pytest.mark.parametrize('field,value',[('rank_statistics_capabilities',8191),('rank_statistics_capabilities',-1),('rank_statistics_capabilities',True),('rank_statistics_interface','rank-histogram-v3'),('rank_vector_capabilities',0),('rank_vector_capabilities',-1),('rank_vector_interface','rank-vector-unknown')])
def test_old_or_malformed_capabilities_precede_all_fixtures(monkeypatch,tmp_path,field,value):
 stubs(monkeypatch);monkeypatch.setattr(h.xx._core,field,value);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'verify',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'caps.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert 'required' in r['failures'][0]['message'] and not r['checks']

@pytest.mark.parametrize('wrong,message',[('forward','upstream-forwarded'),('alias','alias differs'),('signature','signature differs')])
def test_registered_bad_wrappers_rejected_before_fixture(monkeypatch,tmp_path,wrong,message):
 stubs(monkeypatch,wrong);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'bad.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert message in r['failures'][0]['message'] and not r['checks']

def test_compiled_entry_admission_rejects_missing_python_or_wrong_core():
 class Fake:__name__='fake';rank_statistics_run=lambda *args:None
 with pytest.raises(AssertionError,match='compiled R4 entry'):h.compiled_entries(Fake())

def test_check_only_stub_exercises_workers_aliases_and_restores_state(monkeypatch,tmp_path):
 manifest(monkeypatch,tmp_path);stubs(monkeypatch);monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
 workers=h.xx.get_num_threads();policy=h.xx.set_fallback_enabled(True)
 try:
  out=tmp_path/'stub.json';assert h.main(['--check-only','--threads','1','4','--capsule',str(tmp_path),'--output',str(out)])==0
  r=json.loads(out.read_text());assert set(r['checks'][0]['actual_by_workers'])=={'1','4'} and not r['results'];assert h.xx.get_num_threads()==workers and h.xx.set_fallback_enabled(False) is True
 finally:h.xx.set_fallback_enabled(policy)

def test_late_parity_failure_blocks_all_measurement(monkeypatch,tmp_path):
 cases=[tiny(),tiny(label='late-unit',shape=(3,7))];manifest(monkeypatch,tmp_path,cases);wrappers=stubs(monkeypatch);original=wrappers['geometric_mean']
 def wrong(image,*args,**kwargs):
  result=original(image,*args,**kwargs)
  if image.shape==(3,7):result.flat[0]=int(result.flat[0])^1
  return result
 wrong.__signature__=inspect.signature(h.resolve('geometric_mean',True));wrappers['geometric_mean']=wrong
 monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setenv('GITHUB_ACTIONS','true')
 out=tmp_path/'late.json';assert h.main(['--capsule',str(tmp_path),'--output',str(out)])==1
 r=json.loads(out.read_text());assert r['status']=='failed-parity' and len(r['checks'])==2 and r['checks'][0]['status']=='passed' and not r['results'],r['failures']

@pytest.mark.parametrize('patch',[{'depth':0},{'depth':65537},{'shape':[4096,4096]},{'dtype':'u1','bins':1024,'depth':512}])
def test_vector_resource_bounds_precede_all_allocations(monkeypatch,patch):
 case=tiny('windowed_histogram');case=h.Case(case.name,case.label,dict(case.fixture,**patch))
 monkeypatch.setattr(h.np,'arange',forbidden);monkeypatch.setattr(h.np,'full',forbidden)
 with pytest.raises(AssertionError):h.make_case(case)

def test_every_frozen995_id_and_balanced_candidate_retained():
 cases=list(h.corpus());contracts=[c for c in cases if c.frozen_id]
 assert len(cases)==1379 and {c.frozen_id for c in contracts}=={key for key,_,_ in h.qualified.CASES} and len(contracts)==995
 assert all(not c.timed for c in contracts)
 assert sum(c.timed and c.name in h.SCALAR for c in cases)==288 and sum(c.timed and c.name in h.VECTOR for c in cases)==96
 for case in cases:
  if case.timed:
   args,options,tracked=h.make_case(case);assert all(a.nbytes<=h.LIMIT for a in tracked.values() if isinstance(a,np.ndarray))
   if case.name in h.VECTOR:assert args[0].ndim==2 and 1<=case.fixture['depth']<=64

def test_output_reset_destruction_scope_and_bounded_vector_owner():
 c=tiny('windowed_histogram');c=h.Case(c.name,c.label,dict(c.fixture,out='f4'))
 call,tracked=h.timing_callback(h.resolve(c.name,True),c);out=tracked['out'];first=call().copy();out.fill(13);second=call()
 assert second is out and np.array_equal(first,second) and out.nbytes<=h.LIMIT

@pytest.mark.parametrize('arguments',[['--reference-only'],['--reference-only','--check-only','--capsule','unused'],['--threads','0','--check-only'],['--threads','1','1','--check-only'],['--repeat','0','--check-only'],['--target-ms','nan','--check-only']])
def test_invalid_modes_reject_without_fixture_or_timer(monkeypatch,arguments):
 monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 with pytest.raises(SystemExit) as error:h.main(arguments)
 assert error.value.code==2

def test_local_timing_forbidden(monkeypatch,tmp_path):
 monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(h,'make_case',forbidden)
 with pytest.raises(SystemExit):h.main(['--capsule',str(tmp_path),'--output',str(tmp_path/'x.json')])

def test_future_workflow_all_prepares_barrier_and_offline_measurement():
 text=(h.ROOT/'.github/workflows/rank-statistics-benchmark.yml').read_text();measure=text.split('  measure:',1)[1]
 assert 'needs: prepare' in measure and '--no-index' in measure and '--require-hashes' in measure and '--repeat 7' in measure
 assert 'pip wheel' not in measure and 'pytest' not in measure and 'cmake --build' not in measure

def test_record_and_capsule_extra_bytes_are_authenticated(tmp_path):
 wheel=tmp_path/'bad.whl'
 with zipfile.ZipFile(wheel,'w') as z:
  z.writestr('package.py','changed');z.writestr('unit.dist-info/RECORD','package.py,,\nunit.dist-info/RECORD,,\n')
 with pytest.raises(AssertionError,match='unhashed wheel member'):h.verify_record(wheel)
 capsule=tmp_path/'capsule';capsule.mkdir();(capsule/'input').write_bytes(b'frozen');before=h.capsule_tree(capsule)
 (capsule/'extra').write_bytes(b'added');assert h.capsule_tree(capsule)!=before
