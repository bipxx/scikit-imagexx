"""Current enabled IGG integration/collector context; no numerical execution."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/inverse-gaussian-gradient-enabled-context-v3.json'
H=lambda raw:hashlib.sha256(raw).hexdigest()


def test_exact_enabled_source_inverse_and_admitted_family_projection():
    from tools.join_source_context import igg_predecessor_bytes
    assert H(CONTEXT.read_bytes())=='c28f075b86a5e6c80e826b1d798858b4375357ecee4d88b5a33c990dba1a2a89'
    c=json.loads(CONTEXT.read_bytes())
    assert c['predecessor_source']=='dda9cc83d9434a10616f84cf1e1125d93f5f36fd'
    assert c['family_source']=='065cc3a79391e4c5343511b53bbbaac992a4c185'
    for name,row in c['sources'].items():assert H(igg_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==row['predecessor_canonical_lf_sha256'],name
    for name,pin in c['unchanged_family_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    assert c['actual_genuine_authority']is c['actual_benchmark_authority']is None


@pytest.mark.parametrize('fault',('current','inverse'))
def test_enabled_projection_refuses_unreviewed_current_and_inverse(tmp_path,fault):
    from tools.join_source_context import igg_predecessor_bytes
    c=json.loads(CONTEXT.read_bytes());name='CMakeLists.txt';raw=(ROOT/name).read_bytes()
    if fault=='current':raw+=b'changed'
    else:c['sources'][name]['inverse_fragments'][0]['predecessor']+='changed'
    target=tmp_path/'coverage/contracts';target.mkdir(parents=True);(target/CONTEXT.name).write_text(json.dumps(c))
    with pytest.raises(AssertionError):igg_predecessor_bytes(tmp_path,name,raw)


@pytest.mark.parametrize('profile',('minimum','modern','debug','windows'))
def test_actual_current_source_census_and_explicit_linux_envelope(profile):
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'));p=q['plan'](ROOT,profile,282)
    assert not p['blockers'],p['blockers']
    assert (p['registry']['image_count'],p['registry']['registered_count'],p['product_members'],p['SDK_members'],len(p['SDK_families']),p['SDK_checks'])==(282,285,230,109,39,78)
    group='public-inverse-gaussian-gradient-saved-linux'
    if profile=='windows':assert group not in p['public_groups']and 'public-rolling-ball-saved-linux'not in p['public_groups']
    else:assert p['public_groups'][group]==('test_inverse_gaussian_gradient_public.py','test_inverse_gaussian_gradient_policy.py')
    assert 'test_inverse_gaussian_gradient_integration.py'in p['public_groups']['public-post203']
    assert 'inverse_gaussian_gradient'in p['SDK_families']
    assert 'include/skimagexx/inverse_gaussian_gradient.hpp'in json.loads((ROOT/q['CURRENT_SCOPE_CONTRACT']).read_bytes())['required_client_headers']


def test_actual_sdk_fixture_extraction_and_F1_before_genuine_child():
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'));text=(ROOT/'tools/public_qualification.py').read_text();tree=ast.parse(text)
    for name in ('tests/fixtures/inverse_gaussian_gradient_sdk_v1.bin','tests/test_inverse_gaussian_gradient_sdk.cpp','tests/cmake_consumer/inverse_gaussian_gradient_consumer.cpp','tests/inverse_gaussian_gradient_exception_codec.py','tests/inverse_gaussian_gradient_observer.py','tests/sdk/inverse_gaussian_gradient/CMakeLists.txt',CONTEXT.relative_to(ROOT).as_posix()):assert name in q['DRAW_SOURCE_PATHS']and(ROOT/name).is_file(),name
    sdk=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sdk');predicate=next(n.test for n in ast.walk(sdk)if isinstance(n,ast.If)and 'tests/fixtures/bezier_curve_originals_v1.bin'in ast.unparse(n.test));code=compile(ast.Expression(predicate),'<actualSDK>','eval')
    for name in ('tests/fixtures/inverse_gaussian_gradient_sdk_v1.bin','tests/test_inverse_gaussian_gradient_sdk.cpp','tests/cmake_consumer/inverse_gaussian_gradient_consumer.cpp'):assert eval(code,{'name':name})
    assert not eval(code,{'name':'tests/fixtures/unadmitted.bin'})
    child=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='child');body=ast.get_source_segment(text,child);branch=body[body.index("if group=='public-inverse-gaussian-gradient-saved-linux':"):].split("if group=='public-rolling-ball-saved-linux':")[0]
    assert "state['profile']!='windows'"in branch and "not os.environ.get('IGG_CORE_EXTENSION')"in branch and 'util_views_fenv_environment(out,state)'in branch
    assert "row['tests']==297"in text


def test_current_benchmark_contexts_are_exact_unbound_and_keep_cases_methods():
    c=json.loads(CONTEXT.read_bytes())
    for old_path,new_path in c['benchmark_successors'].items():
        old=json.loads((ROOT/old_path).read_bytes());new=json.loads((ROOT/new_path).read_bytes())
        assert old['cases']==new['cases']and old['method']==new['method']
        for name,pin in new['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,(new_path,name)
        for field,name in (('common_boundary_canonical_sha256','benchmarks/mark_boundaries_execution.py'),('reused_executor_canonical_sha256','benchmarks/next_util_execution.py'),('metadata_fixture_canonical_sha256','tests/test_mark_boundaries_benchmark_execution.py')):
            if field in new:assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==new[field]
        assert all(value is None for value in new['unbound'].values())
