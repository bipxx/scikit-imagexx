"""Bounded harness blockers; all delegating stubs are tooling, never native proof."""
import inspect,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import rank_percentile as h


def forbidden(*args,**kwargs):raise AssertionError('forbidden fixture/native/timer work')

def tiny(name='mean_percentile',label='unit-only',shape=(3,5)):
 return h.Case(name,label,dict(h.profiles()[0],shape=list(shape),out='default'))

def manifest(monkeypatch,tmp_path,cases=None):
 cases=cases or [tiny()];m=json.loads(h.MANIFEST.read_text());m['case_inventory']=[h.inventory(c) for c in cases];m['case_contract_sha256']=h.digest(m['case_inventory'])
 m['reference_profiles']={h.runtime_profile():dict(case_spec_sha256=h.digest([h.spec(c) for c in cases]),observations={str(i):h.digest(h.observe_case(h.resolve(c.name,True),c)) for i,c in enumerate(cases)},alias_sha256=h.digest([]),provider_sha256=h.digest([]),policy_sha256=h.digest([]))}
 p=tmp_path/'unit-only.json';p.write_text(json.dumps(m));monkeypatch.setattr(h,'MANIFEST',p);monkeypatch.setattr(h,'corpus',lambda:iter(cases))
 for name in ('check_aliases','check_providers','check_policies'):monkeypatch.setattr(h,name,lambda *args:[])
 return cases

def stubs(monkeypatch,wrong=None):
 originals={n:h.resolve(n,True) for n in h.NAMES};wrappers={}
 for name in h.NAMES:
  def call(*args,_name=name,**kwargs):return originals[_name](*args,**kwargs)
  call.__signature__=inspect.signature(originals[name]);wrappers[name]=call
 monkeypatch.setitem(call.__globals__,'_core',h.xx._core)
 def resolve(name,original=False,alias=False):
  if original or wrong=='forward':return originals[name]
  return wrappers[name]
 if wrong=='signature':wrappers[h.NAMES[0]].__signature__=inspect.Signature()
 if wrong=='core':monkeypatch.setitem(call.__globals__,'_core',object())
 monkeypatch.setattr(h,'resolve',resolve);monkeypatch.setattr(h,'native_functions',lambda:h.CANONICAL)
 monkeypatch.setattr(h.xx._core,'rank_percentile_interface','rank-percentile-v1',raising=False);monkeypatch.setattr(h.xx._core,'rank_percentile_capabilities',0x3fe0000,raising=False)
 monkeypatch.setattr(h,'compiled_entries',lambda core:None);monkeypatch.setattr(h,'verify',lambda capsule:{'scope':'UNIT-STUB-NOT-NATIVE'})
 # Delegation exists only in this explicit tooling stub, never a CLI mode.
 monkeypatch.setattr(h,'native_observe',lambda observer,*args:observer(*args))
 return wrappers


def test_reference_only_never_calls_native_or_timer(monkeypatch,tmp_path):
 manifest(monkeypatch,tmp_path);real=h.resolve;monkeypatch.setattr(h,'resolve',lambda n,original=False,alias=False:real(n,True,alias) if original else forbidden())
 monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
 out=tmp_path/'ref.json';assert h.main(['--reference-only','--check-only','--output',str(out)])==0
 r=json.loads(out.read_text());assert r['metadata']['native_checks']==0 and r['metadata']['validation_scope']=='reference-only/native0' and not r['results']


def test_missing_registry_blocks_fixtures_capsule_and_timers(monkeypatch,tmp_path):
 monkeypatch.setattr(h,'native_functions',lambda:());monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'corpus',forbidden);monkeypatch.setattr(h,'verify',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'missing.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert 'not registered' in r['failures'][0]['message'] and not r['checks'] and not r['alias_checks'] and not r['provider_checks'] and not r['results']


@pytest.mark.parametrize('field,value',[('rank_percentile_capabilities',8191),('rank_percentile_capabilities',0x1fe0000),('rank_percentile_capabilities',-1),('rank_percentile_capabilities',True),('rank_percentile_interface','rank-histogram-v3')])
def test_incomplete_interface_blocks_all_fixtures(monkeypatch,tmp_path,field,value):
 stubs(monkeypatch);monkeypatch.setattr(h.xx._core,field,value);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'corpus',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'caps.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert 'required' in r['failures'][0]['message'] and not r['checks']


@pytest.mark.parametrize('wrong,message',[('forward','upstream-forwarded'),('signature','signature differs'),('core','another core')])
def test_registered_wrong_wrappers_rejected_before_fixture(monkeypatch,tmp_path,wrong,message):
 stubs(monkeypatch,wrong);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 out=tmp_path/'wrong.json';assert h.main(['--check-only','--output',str(out)])==1
 r=json.loads(out.read_text());assert message in r['failures'][0]['message'] and not r['checks']


def test_python_entry_cannot_pose_as_compiled():
 class Fake:__name__='fake';rank_percentile_run=lambda *args:None
 with pytest.raises(AssertionError,match='compiled R5 entry'):h.compiled_entries(Fake())


def test_original_python_forwarding_rejected_and_profile_restored():
 saved=sys.getprofile()
 def previous(frame,event,arg):pass
 try:
  sys.setprofile(previous)
  result=h.native_observe(h.observe_case,h.resolve('mean_percentile',True),tiny())
  assert result['error']['type']=='AssertionError' and 'upstream original rank' in result['error']['message']
  assert sys.getprofile() is previous
 finally:sys.setprofile(saved)


def test_exact_cython_lookup_guard_restores_entry_after_failure():
 module=h.importlib.import_module('skimage.filters.rank._percentile');entry=module.percentile_cy._mean;before=sys.getprofile()
 def dynamic_lookup():return module.percentile_cy._mean()
 with pytest.raises(AssertionError,match='upstream original rank'):h.native_observe(dynamic_lookup)
 assert module.percentile_cy._mean is entry and sys.getprofile() is before


def test_provider_module_name_alone_is_not_an_original_entry():
 def provider():return abs(-6)
 provider.__module__='skimage.filters.rank.metadata_provider'
 assert h.native_observe(provider)==6


def test_check_only_tooling_stub_checks_workers_and_restores_policy(monkeypatch,tmp_path):
 manifest(monkeypatch,tmp_path);stubs(monkeypatch);monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
 workers=h.xx.get_num_threads();policy=h.xx.set_fallback_enabled(True)
 try:
  out=tmp_path/'stub.json';assert h.main(['--check-only','--threads','1','4','--capsule',str(tmp_path),'--output',str(out)])==0
  r=json.loads(out.read_text());assert set(r['checks'][0]['actual_by_workers'])=={'1','4'} and not r['results'] and r['metadata']['canonical_paths']==list(h.CANONICAL)
  assert h.xx.get_num_threads()==workers and h.xx.set_fallback_enabled(False) is True
 finally:h.xx.set_fallback_enabled(policy)


def test_late_parity_failure_blocks_every_timer(monkeypatch,tmp_path):
 cases=[tiny(),tiny(label='late',shape=(3,7))];manifest(monkeypatch,tmp_path,cases);wrappers=stubs(monkeypatch);old=wrappers['mean_percentile']
 def wrong(image,*args,**kwargs):
  result=old(image,*args,**kwargs)
  if image.shape==(3,7):result.flat[0]=int(result.flat[0])^1
  return result
 wrong.__signature__=inspect.signature(h.resolve('mean_percentile',True));wrappers['mean_percentile']=wrong
 monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setenv('GITHUB_ACTIONS','true')
 out=tmp_path/'late.json';assert h.main(['--capsule',str(tmp_path),'--output',str(out)])==1
 r=json.loads(out.read_text());assert r['status']=='failed-parity' and len(r['checks'])==2 and r['checks'][0]['status']=='passed' and not r['results']


@pytest.mark.parametrize('auxiliary',['check_providers','check_policies','check_aliases'])
def test_auxiliary_failure_blocks_measurement(monkeypatch,tmp_path,auxiliary):
 manifest(monkeypatch,tmp_path);stubs(monkeypatch);monkeypatch.setattr(h,auxiliary,forbidden);monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setenv('GITHUB_ACTIONS','true')
 out=tmp_path/'aux.json';assert h.main(['--capsule',str(tmp_path),'--output',str(out)])==1
 r=json.loads(out.read_text());assert r['status']=='failed' and not r['results']


@pytest.mark.parametrize('patch',[{'shape':[4096,4096]},{'dtype':'f4'},{'bins':65537},{'dtype':'u1','bins':1024},{'regime':'nonfinite-unsafe'}])
def test_resource_and_domain_bounds_precede_allocation(monkeypatch,patch):
 c=tiny();c=h.Case(c.name,c.label,dict(c.fixture,**patch));monkeypatch.setattr(h.np,'arange',forbidden);monkeypatch.setattr(h.np,'full',forbidden)
 with pytest.raises(AssertionError):h.make_case(c)


def test_every_frozen1544_id_and_all_candidates_preserved():
 cases=list(h.corpus());old=[c for c in cases if c.frozen_id]
 assert len(cases)==2552 and len(old)==1544 and {c.frozen_id for c in old}=={key for key,_,_ in h.qualified.CASES}
 assert all(not c.timed for c in old) and sum(c.timed for c in cases)==1008 and all(c.name in h.SCALAR for c in cases)
 for c in cases:
  if c.timed:
   args,options,tracked=h.make_case(c);assert args[0].ndim==2 and all(a.nbytes<=h.LIMIT for a in tracked.values() if isinstance(a,np.ndarray))


def test_actual_namespace_paths_not_invented_generic_algorithms():
 assert len(h.NAMES)==len(h.CANONICAL)==len(h.APIS)==9 and len(h.ALIASES)==9
 for n in h.NAMES:assert h.resolve(n,True) is h.resolve(n,True,True)
 assert all('._percentile.' in path for path in h.ALIASES) and not any('.generic.' in path for path in h.APIS)


def test_output_reset_is_inside_public_call_scope():
 c=tiny();c=h.Case(c.name,c.label,dict(c.fixture,out='same'));original=h.resolve(c.name,True);seen=[]
 def observe_reset(*args,**options):
  seen.append(options['out'].copy());return original(*args,**options)
 call,tracked=h.timing_callback(observe_reset,c);out=tracked['out'];first=call().copy();out.fill(13);second=call()
 assert all(np.array_equal(value,np.full(out.shape,99,out.dtype)) for value in seen)
 # Pinned percentile wrappers return a reshape view of supplied storage.
 assert second is not out and np.shares_memory(second,out) and np.array_equal(first,second)


@pytest.mark.parametrize('arguments',[['--reference-only'],['--reference-only','--check-only','--capsule','unused'],['--threads','0','--check-only'],['--threads','4','--check-only'],['--threads','1','1','--check-only'],['--repeat','0','--check-only'],['--repeat','3','--check-only'],['--target-ms','nan','--check-only']])
def test_invalid_modes_precede_fixture_and_timer(monkeypatch,arguments):
 monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
 with pytest.raises(SystemExit) as e:h.main(arguments)
 assert e.value.code==2


def test_local_measurement_rejected(monkeypatch,tmp_path):
 monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(h,'make_case',forbidden)
 with pytest.raises(SystemExit):h.main(['--capsule',str(tmp_path),'--output',str(tmp_path/'x.json')])


def test_future_barrier_artifact_only_hosts_and_fixed_checks():
 text=(h.ROOT/'.github/workflows/rank-percentile-benchmark.yml').read_text();measure=text.split('  measure:',1)[1]
 assert 'needs: prepare' in measure and '--no-index' in measure and '--require-hashes' in measure and '--repeat 7' in measure
 assert 'pip wheel' not in measure and 'pytest' not in measure and 'cmake --build' not in measure
 assert text.count('--check-only --threads 1 4')==2


def test_record_and_capsule_extra_members_authenticated(tmp_path):
 wheel=tmp_path/'bad.whl'
 with zipfile.ZipFile(wheel,'w') as z:z.writestr('package.py','changed');z.writestr('unit.dist-info/RECORD','package.py,,\nunit.dist-info/RECORD,,\n')
 with pytest.raises(AssertionError,match='unhashed wheel member'):h.verify_record(wheel)
 capsule=tmp_path/'capsule';capsule.mkdir();(capsule/'input').write_bytes(b'frozen');old=h.capsule_tree(capsule);(capsule/'extra').write_bytes(b'added');assert h.capsule_tree(capsule)!=old
