"""Pure authority controls; no scientific imports, helper or target bodies."""
import copy,hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
import finite_radon_capture_controller as C
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def test_no_GO_and_bad_GO_digest_before_runtime(tmp_path):
 with pytest.raises(AssertionError,match='NO_ROOT_GO'):C.validate_go(SimpleNamespace(root_go=None,root_go_sha256=None),{})
 p=tmp_path/'go.json';p.write_text('{}')
 with pytest.raises(AssertionError,match='DIGEST'):C.validate_go(SimpleNamespace(root_go=p,root_go_sha256='0'*64),{})

@pytest.mark.parametrize('fault',('none','source','binding','controller','case-order','api-order','profile-budget','total-budget','calls-bool','native-budget','profile','deadline','journal','readiness','prefix-pin','prefix-duplicate','prefix-body','prefix-helper','prefix-counts','prefix-restore','runtime'))
def test_exact_fixed_pool_authority(tmp_path,monkeypatch,fault):
 monkeypatch.setattr(C,'R',tmp_path)
 ids=['transform.'+api+'/'+str(i)for api in('frt2','ifrt2')for i in range(54)]
 q=dict(source='a'*40,controller_raw_sha256='b'*64,source_files_raw_sha256={'fixed':'c'*64},cases_sha256='d'*64,case_ids=ids,profile_case_ids={'minimum':ids,'modern':ids},api_case_ids={'frt2':ids[:54],'ifrt2':ids[54:]},helper={'synthetic':'only'})
 review=tmp_path/'review.json';review.write_text(json.dumps(dict(passed=True,source=q['source'],controller_sha256=q['controller_raw_sha256'],binding_sha256='e'*64)))
 prefixes=[]
 for api in('frt2','ifrt2'):
  for cache in('cold','cached'):
   value=dict(status='PASS_ACTUAL_FINITE_RADON_CONTROLLER_ZERO_BODY_PREFIX',profile='minimum',source=q['source'],controller_raw_sha256=q['controller_raw_sha256'],binding_sha256='e'*64,recipe_observations=0,journal_created=False,full_initial_state_restored=True,counts={'target_body_lines':0,'provider_body_lines':0,'target_attempted':2,'target_refused':2,'provider_attempted':2,'provider_refused':2,'compiled_provider_attempted':1,'compiled_provider_refused':1},controls=[{'caller_restore':{'exact32':True}}]*6,observer_target_immediate=[{'synthetic':True}],module_request={'was_cached':cache=='cached'},case_ids=ids,full_recipe_pool_count=108,runtime_identity={'synthetic':True},helper=q['helper'],codec_metadata_controls=[{'caller_restore':{'exact32':True}}]*2,prefix_api=api,cache_state=cache)
   path=tmp_path/(api+'-'+cache+'.json');path.write_text(json.dumps(value));prefixes.append(dict(path=path.name,sha256=H(path)))
 now=datetime.now(timezone.utc)
 go=dict(status='ROOT_AUTHORIZED_FINITE_RADON_REFERENCE_ONCE',source=q['source'],binding_sha256='e'*64,controller_sha256=q['controller_raw_sha256'],source_files_raw_sha256=q['source_files_raw_sha256'],cases_sha256=q['cases_sha256'],case_ids=ids,api_case_ids=q['api_case_ids'],profiles=['minimum','modern'],profile='minimum',maximum_public_topcalls_per_profile=108,maximum_total_public_topcalls=216,calls_per_recipe=1,native_calls=0,sampler_bodies=0,timed_batches=0,builds=0,journal_directory=str(tmp_path/'never-created'),no_retry=True,not_before_utc=(now-timedelta(minutes=1)).isoformat(),deadline_utc=(now+timedelta(minutes=1)).isoformat(),independent_readiness=dict(path=review.name,sha256=H(review)),prefixes=prefixes,runtime_identity={'synthetic':True})
 if fault in('source','binding','controller'):go[{'source':'source','binding':'binding_sha256','controller':'controller_sha256'}[fault]]='0'*64
 elif fault=='case-order':go['case_ids']=list(reversed(ids))
 elif fault=='api-order':go['api_case_ids']=dict(q['api_case_ids'],frt2=list(reversed(ids[:54])))
 elif fault=='profile-budget':go['maximum_public_topcalls_per_profile']=109
 elif fault=='total-budget':go['maximum_total_public_topcalls']=217
 elif fault=='calls-bool':go['calls_per_recipe']=True
 elif fault=='native-budget':go['native_calls']=1
 elif fault=='profile':go['profile']='modern'
 elif fault=='deadline':go['deadline_utc']=(now-timedelta(seconds=1)).isoformat()
 elif fault=='journal':go['journal_directory']=str(tmp_path/'other')
 elif fault=='readiness':review.write_text('{}')
 elif fault=='prefix-pin':(tmp_path/prefixes[0]['path']).write_text('{}')
 elif fault=='prefix-duplicate':go['prefixes']=[prefixes[0]]*4
 elif fault in('prefix-body','prefix-helper','prefix-counts','prefix-restore'):
  p=tmp_path/prefixes[0]['path'];v=json.loads(p.read_bytes())
  if fault=='prefix-body':v['counts']['target_body_lines']=1
  elif fault=='prefix-helper':v['helper']={}
  elif fault=='prefix-counts':v['counts']['target_attempted']=0
  else:v['controls'][0]['caller_restore']['exact32']=False
  p.write_text(json.dumps(v));go['prefixes'][0]['sha256']=H(p)
 elif fault=='runtime':go['runtime_identity']={}
 path=tmp_path/'go.json';path.write_text(json.dumps(go));args=SimpleNamespace(root_go=path,root_go_sha256=H(path),binding_sha256='e'*64,profile='minimum',output=tmp_path/'never-created')
 if fault=='none':assert C.validate_go(args,q)==go
 else:
  with pytest.raises(AssertionError):C.validate_go(args,q)
 assert not args.output.exists()
