"""Current enabled rolling-ball source/collector closure, no target execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/rolling-ball-enabled-context-v1.json'
EXPECTED_CONTEXT='a9f317747114b1766467f9c3624c3c7bce9b7ebbbb209bd0b1539d5b13c0b373'
H=lambda b:hashlib.sha256(b).hexdigest()
from tools.rolling_ball_sentinel_context import predecessor_bytes as sentinel_predecessor
def historical(name):return sentinel_predecessor(ROOT,name,(ROOT/name).read_bytes())


def qualifier():
    spec=importlib.util.spec_from_file_location('_rolling_enabled_qualifier',ROOT/'tools/public_qualification.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);return q


def test_exact_shared_and_controller_predecessors_and_unchanged_family():
    assert H(CONTEXT.read_bytes())==EXPECTED_CONTEXT
    c=json.loads(CONTEXT.read_bytes())
    from tools.join_source_context import rolling_predecessor_bytes
    for name,row in c['sources'].items():
        raw=(ROOT/name).read_bytes()
        assert H(rolling_predecessor_bytes(ROOT,name,raw))==row['predecessor_canonical_lf_sha256'],name
    for name,row in {**c['controller_sources'],**c['family_sources']}.items():
        raw=historical(name).replace(b'\r\n',b'\n')
        assert H(raw)==row['current_canonical_lf_sha256'],name
        for f in row['inverse_fragments']:
            assert raw.count(f['current'].encode())==1
            raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
        assert H(raw)==row['predecessor_canonical_lf_sha256'],name
    for name,pin in c['unchanged_imports'].items():assert H(historical(name))==pin,name
    for name,row in c['SDK_source_imports'].items():assert H((ROOT/name).read_bytes())==row['sha256'],name
    assert c['actual_genuine_authority']is c['actual_measurement_authority']is None


@pytest.mark.parametrize('fault',('current','inverse'))
def test_shared_projection_refuses_changed_current_or_inverse(tmp_path,fault):
    from tools.join_source_context import rolling_predecessor_bytes
    c=json.loads(CONTEXT.read_bytes());name='CMakeLists.txt';raw=(ROOT/name).read_bytes()
    if fault=='current':raw+=b'\nchanged'
    else:c['sources'][name]['inverse_fragments'][0]['predecessor']+='changed'
    path=tmp_path/'coverage/contracts';path.mkdir(parents=True)
    (path/CONTEXT.name).write_text(json.dumps(c))
    with pytest.raises(AssertionError):rolling_predecessor_bytes(tmp_path,name,raw)


@pytest.mark.parametrize('profile',('minimum','modern','debug','windows'))
def test_actual_current_census_and_explicit_saved_platform(profile):
    q=qualifier();p=q.plan(ROOT,profile,281)
    assert not p['blockers'],p['blockers']
    assert (p['registry']['image_count'],p['registry']['registered_count'],p['product_members'],p['SDK_members'],len(p['SDK_families']),p['SDK_checks'])==(281,284,228,108,38,76)
    assert 'test_rolling_ball_integration.py'in p['public_groups']['public-post203']
    group='public-rolling-ball-saved-linux'
    if profile=='windows':assert group not in p['public_groups']
    else:assert p['public_groups'][group]==('test_rolling_ball_public.py',)
    assert 'rolling_ball'in p['SDK_families']
    from rolling_ball_sdk_dtype_context import historical_sdk_source
    old=json.loads((ROOT/'coverage/contracts/rolling-ball-native-v1.json').read_bytes())
    for name in ('include/skimagexx/rolling_ball.hpp','src/rolling_ball.cpp'):assert H(historical_sdk_source(name,(ROOT/name).read_bytes()))==old['source_files_raw_sha256'][name]


def test_actual_collector_sdk_copy_and_F1_entry_before_child():
    q=qualifier()
    for name in ('tests/fixtures/rolling_ball_sdk_v1.bin','tests/test_rolling_ball_sdk.cpp','tests/cmake_consumer/rolling_ball_consumer.cpp','tests/util_views_fenv_restore.c','src/half.hpp','include/skimagexx/convert.hpp','tests/cmake_consumer/rolling_ball_storage_consumer.cpp','coverage/contracts/rolling-ball-measurement-v2.json',CONTEXT.relative_to(ROOT).as_posix()):
        assert (ROOT/name).is_file()and name in q.DRAW_SOURCE_PATHS,name
    tree=ast.parse((ROOT/'tools/public_qualification.py').read_bytes());sdk=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sdk')
    predicate=next(n.test for n in ast.walk(sdk)if isinstance(n,ast.If)and 'tests/fixtures/bezier_curve_originals_v1.bin'in ast.unparse(n.test))
    code=compile(ast.Expression(predicate),'<actual SDK extraction>','eval')
    for name in ('tests/fixtures/rolling_ball_sdk_v1.bin','tests/test_rolling_ball_sdk.cpp','tests/cmake_consumer/rolling_ball_consumer.cpp'):assert eval(code,{'name':name})
    assert not eval(code,{'name':'tests/fixtures/unadmitted.bin'})
    child=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='child')
    text=ast.get_source_segment((ROOT/'tools/public_qualification.py').read_text(),child)
    branch=text[text.index("if group=='public-rolling-ball-saved-linux':"):].split("if group=='public-flood-saved-linux':")[0]
    assert "state['profile']!='windows'"in branch and "not os.environ.get('ROLLING_BALL_CORE_EXTENSION')"in branch and 'util_views_fenv_environment(out,state)'in branch


def test_benchmark_context_successor_only_changes_source_projection():
    old=json.loads((ROOT/'coverage/contracts/rolling-ball-measurement-v1.json').read_bytes())
    current=json.loads((ROOT/'coverage/contracts/rolling-ball-measurement-v2.json').read_bytes())
    for key in ('cases','execution_scope','helper'):assert old[key]==current[key]
    assert old['families']['rolling_ball_native']['case_ids']==current['families']['rolling_ball_native']['case_ids']
    assert current['families']['rolling_ball_native']['contract']=='coverage/contracts/rolling-ball-benchmark-execution-v2.json'
    assert current['current_interface']=={**old['current_interface'],'driver_contract':'coverage/contracts/rolling-ball-benchmark-execution-v2.json'}
    a=dict(old['first_cohort']);b=dict(current['first_cohort']);a.pop('source_files_raw_sha256');b.pop('source_files_raw_sha256');assert a==b
    changed={n for n in old['benchmark_source_sha256']if old['benchmark_source_sha256'][n]!=current['benchmark_source_sha256'][n]}
    assert changed=={'CMakeLists.txt','benchmarks/next_util_execution.py','benchmarks/rolling_ball_execution.py','tests/test_rolling_ball_benchmark.py','include/skimagexx/rolling_ball.hpp','src/rolling_ball.cpp','tests/test_rolling_ball_native_source.py'}
    assert set(current['benchmark_source_sha256'])-set(old['benchmark_source_sha256'])==(set(json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v2.json').read_bytes())['source_files_canonical_sha256'])|{'coverage/contracts/rolling-ball-benchmark-execution-v2.json'})-set(old['benchmark_source_sha256'])
    context=json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v2.json').read_bytes())
    predecessor=json.loads((ROOT/'coverage/contracts/rolling-ball-benchmark-execution-v1.json').read_bytes())
    assert context['cases']==predecessor['cases']and context['method']==predecessor['method']and context['route']==predecessor['route']
    row=context['current_executor_projection'];raw=historical(row['path']).replace(b'\r\n',b'\n')
    assert H(raw)==row['current_canonical_sha256']
    for f in row['inverse_fragments']:
        assert raw.count(f['current'].encode())==1;raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
    assert H(raw)==row['predecessor_canonical_sha256']==predecessor['reused_executor_canonical_sha256']
    assert all(current[k]is None for k in ('prepared_source','SDK_members','core_sha256','runtime_sha256','deadline_utc','benchmark_fingerprint'))


def test_actual_active_shared_benchmark_literal_dependencies():
    # These are the exact current driver contexts; older contexts remain history.
    for name in ('rolling-ball-benchmark-execution-v2.json','mark-boundaries-benchmark-execution-v6.json','draw-multiblock-lbp-benchmark-execution-v5.json','variation-of-information-benchmark-execution-v5.json'):
        c=json.loads((ROOT/'coverage/contracts'/name).read_bytes())
        for source,pin in c['source_files_canonical_sha256'].items():assert H(historical(source).replace(b'\r\n',b'\n'))==pin,(name,source)
        for field,source in (('common_boundary_canonical_sha256','benchmarks/mark_boundaries_execution.py'),('reused_executor_canonical_sha256','benchmarks/next_util_execution.py'),('metadata_fixture_canonical_sha256','tests/test_mark_boundaries_benchmark_execution.py')):
            if field in c:assert H((ROOT/source).read_bytes().replace(b'\r\n',b'\n'))==c[field],(name,field)
        assert all(value is None for value in c['unbound'].values())

    import runpy
    adapter=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_execution.py'))
    path='coverage/contracts/variation-of-information-benchmark-execution-v5.json'
    current=json.loads((ROOT/path).read_bytes())
    for name,row in current['historical_sources'].items():
        assert H(adapter['historical_bytes'](ROOT,name,context_path=path))==row['predecessor_canonical_sha256']


@pytest.mark.parametrize('fault',('unreviewed-current','already-predecessor'))
def test_sdk_family_assertion_projection_is_strict_and_once(fault):
    from tools.join_source_context import sdk_family_assertion_predecessor_bytes
    project=lambda raw:sdk_family_assertion_predecessor_bytes(ROOT,'tests/test_public_qualification.py',raw)
    raw=(ROOT/'tests/test_public_qualification.py').read_bytes()
    if fault=='unreviewed-current':raw+=b'changed'
    else:raw=project(raw)
    with pytest.raises(AssertionError):project(raw)
