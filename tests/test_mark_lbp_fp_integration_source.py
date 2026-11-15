"""Exact integrated source/SDK/public collector gates; no scientific execution."""
import ast,hashlib,json,runpy,shlex
from pathlib import Path
import pytest
from tools.mark_lbp_fp_context import predecessor_bytes,source_successor_predecessor_bytes
ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'coverage/contracts/mark-lbp-fp-integration-v2.json').read_bytes())
H=lambda b:hashlib.sha256(b).hexdigest()

@pytest.mark.parametrize('name',tuple(C['predecessors']))
def test_exact_once_raw_and_canonical_predecessors(name):
 row=C['predecessors'][name];current=(ROOT/name).read_bytes();raw=source_successor_predecessor_bytes(ROOT,name,current)
 assert H(raw)==row['current_raw_sha256']
 for canonical in (False,True):
  old=predecessor_bytes(ROOT,name,current,canonical)
  assert H(old)==row['predecessor_canonical_sha256'if canonical else'predecessor_raw_sha256']
  with pytest.raises(AssertionError):predecessor_bytes(ROOT,name,old,canonical)
  with pytest.raises(AssertionError):predecessor_bytes(ROOT,name,current+b'\n// tamper',canonical)
 assert predecessor_bytes(ROOT,name,current.replace(b'\n',b'\r\n'),True)==row['predecessor'].encode().replace(b'\r\n',b'\n')

def test_real_installed_sdk_uses_complete_client_and_authenticated_host_helper():
 cm=(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text();client=(ROOT/'tests/cmake_consumer/mark_boundaries_consumer.cpp').read_text()
 assert '#include "mark_wide_outer_consumer.cpp"'in client and 'return mark_wide_saved_main();'in client
 assert 'target_link_libraries(mark_boundaries_consumer PRIVATE "${SKIMAGEXX_FENV_HELPER}")'in cm
 q=(ROOT/'tools/public_qualification.py').read_text()
 assert "util_views_fenv_environment(out,state)\n        configure += ['-DSKIMAGEXX_FENV_HELPER='"in q
 assert 'public-mark-wide-saved-linux'in q
 assert "'tests/fixtures/mark_wide_outer_sdk_v1.hpp')):"in q


def test_active_current_find_contract_and_bound_provenance_separation():
 d=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'));current=json.loads(d['CONTRACT'].read_bytes());old=json.loads((ROOT/'coverage/contracts/find-boundaries-benchmark-binding-v5.json').read_bytes())
 assert current['cases']==old['cases']and current['method_sha256']==old['method_sha256']
 d['validate_source_closure'](current)
 bound=dict(current,state='FROZEN_READY_FOR_ROOT_FIRST_COHORT',shared_CMake_sha256='a'*64,genuine_core_sha256='b'*64,prepared_run=1,prepared_artifact=2)
 d['validate_source_closure'](bound)
 broken=dict(bound,cases=[])
 with pytest.raises(AssertionError):d['validate_source_closure'](broken)


def test_actual_source_selector_and_strict_genuine_consumer_closure():
 q=runpy.run_path(str(ROOT/'tools/public_qualification.py'));groups=q['applicable_public_groups'](q['groups_for_source'](ROOT),'minimum')
 assert groups['public-mark-wide-saved-linux']==('test_mark_wide_outer_public.py',)
 assert 'public-mark-wide-saved-linux'not in q['applicable_public_groups'](q['groups_for_source'](ROOT),'windows')
 text=(ROOT/'tests/test_mark_wide_outer_public.py').read_text();ast.parse(text)
 for token in ['find._core is mark._core is core','return functions[spec[','sys.settrace(oldtrace);sys.setprofile(oldprofile)','restore(caller)','not assess(expected,actual,immediate[0])','ORIGINAL_PROVIDER_BODY_FORBIDDEN']:assert token in text
 line=next(l for l in(ROOT/'.github/workflows/public-qualification.yml').read_text().splitlines()if'python -m pytest'in l and'qualification-tool-guards.log'in l)
 left,right=line.split('2>&1 |');assert shlex.split(right)==['tee','build/qualification-tool-guards.log']
 assert 'tests/test_mark_lbp_fp_integration_source.py'in shlex.split(left)
 assert 'tests/test_mark_wide_outer_public.py'in q['DRAW_SOURCE_PATHS']

@pytest.mark.parametrize('old,new',[('find-boundaries-frontend-v4.json','find-boundaries-frontend-v5.json'),('find-boundaries-sdk-v3.json','find-boundaries-sdk-v4.json')])
def test_new_current_metadata_preserves_every_old_recipe_and_checks_current_bytes(old,new):
 before=json.loads((ROOT/'coverage/contracts'/old).read_bytes());current=json.loads((ROOT/'coverage/contracts'/new).read_bytes())
 assert {k:v for k,v in current.items()if k!='source_canonical_pins'}=={k:v for k,v in before.items()if k!='source_canonical_pins'}
 assert set(current['source_canonical_pins'])==set(before['source_canonical_pins'])
 for name,pin in current['source_canonical_pins'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name

def test_actual_sdk_copy_keeps_new_named_client_and_raw_word_fixture(tmp_path,monkeypatch):
 import importlib.util,zipfile
 spec=importlib.util.spec_from_file_location('wide_sdk_copy_only',ROOT/'tools/public_qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 names=('tests/cmake_consumer/mark_wide_outer_consumer.cpp','tests/fixtures/mark_wide_outer_sdk_v1.hpp')
 members={name:(ROOT/name).read_bytes()for name in names}
 (tmp_path/'package.json').write_text(json.dumps(dict(product_scope=q.CURRENT_PRODUCT_SCOPE,SDK_sha256={})))
 with zipfile.ZipFile(tmp_path/'source.zip','w')as archive:
  for name,raw in members.items():archive.writestr(name,raw)
  archive.writestr('tests/unrelated_private.hpp',b'refused unrelated fixture')
 class CopyComplete(Exception):pass
 def stop_before_tools(*args):
  for name,raw in members.items():assert(tmp_path/'consumer source'/name).read_bytes()==raw
  assert not(tmp_path/'consumer source/tests/unrelated_private.hpp').exists()
  raise CopyComplete
 monkeypatch.setattr(q,'tool_in',stop_before_tools)
 with pytest.raises(CopyComplete):q.sdk(tmp_path,{},object())

@pytest.mark.parametrize('name',('CMakeLists.txt','src/find_boundaries.cpp','src/draw_multiblock_lbp.cpp','tools/join_source_context.py','tools/public_qualification.py'))
@pytest.mark.parametrize('crlf',(False,True))
def test_actual_common_entries_equal_authenticated_DDA_readers_and_refuse_double(name,crlf):
 from tools import join_source_context as current
 code=C['predecessors']['tools/join_source_context.py']
 old=code['predecessor'].encode();assert H(old)==code['predecessor_raw_sha256']
 baseline={};exec(compile(old,'<authenticated DDA metadata reader>','exec'),baseline)
 row=C['predecessors'][name];raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
 if crlf:raw=raw.replace(b'\n',b'\r\n')
 for entry in ('rolling_predecessor_bytes','rolling_predecessor_raw_bytes'):
  expected=baseline[entry](ROOT,name,row['predecessor'].encode())
  assert getattr(current,entry)(ROOT,name,raw)==expected
  with pytest.raises(AssertionError):getattr(current,entry)(ROOT,name,row['predecessor'].encode())
  with pytest.raises(AssertionError):getattr(current,entry)(ROOT,name,raw+b'tamper')

def test_linux_group_metadata_successor_retains_exact_frozen_parent_and_only_named_changes():
 parent=C['predecessor_context'];raw=(ROOT/parent['path']).read_bytes();assert H(raw)==parent['sha256']
 old=json.loads(raw)
 assert set(C['predecessors'])==set(old['predecessors'])|{'tests/test_util_metadata_source.py'}
 for name,row in old['predecessors'].items():
  if name!='tools/public_qualification.py':assert C['predecessors'][name]==row
  else:
   current=C['predecessors'][name]
   assert current['predecessor']==row['predecessor']and current['predecessor_raw_sha256']==row['predecessor_raw_sha256']
 assert C['predecessor_context']['path']=='coverage/contracts/mark-lbp-fp-integration-v1.json'
