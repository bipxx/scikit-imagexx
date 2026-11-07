"""Scoped cumulative wiring/collector checks; no scientific or target imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zipfile
import pytest
ROOT=Path(__file__).resolve().parents[1]


def qualifier():
    spec=importlib.util.spec_from_file_location('_abi_vi_qualification_source',ROOT/'tools/public_qualification.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('profile',('minimum','modern','debug','windows'))
def test_actual_source_census_and_profile_groups(profile):
    q=qualifier();data=q.plan(ROOT,profile,281)
    assert data['registry']['image_count']==281 and data['registry']['registered_count']==284
    assert data['product_scope']=='current281'and data['product_members']==228 and data['SDK_members']==108
    assert len(data['SDK_families'])==38 and data['SDK_checks']==76
    groups=data['public_groups']
    if profile=='windows':
        assert not{'public-VI-saved-linux','public-radon-ABI-policy-linux','public-radon-saved-linux','public-flood-saved-linux'}&groups.keys()
    else:
        assert groups['public-VI-saved-linux']==('test_variation_of_information_public.py','test_variation_of_information_current_counts_public.py')
        assert groups['public-radon-ABI-policy-linux']==('test_finite_radon_numpy_abi_public.py',)
        assert groups['public-radon-saved-linux']==('test_finite_radon_saved_public.py','test_finite_radon_exceptional_saved_public.py')
        assert groups['public-flood-saved-linux']==('test_flood_fill_public.py',)


def test_admitted_kernel_and_shared_global_source_projection():
    ctx=json.loads((ROOT/'coverage/contracts/abi-vi-cumulative-integration-v1.json').read_bytes())
    from tools.join_source_context import correction_predecessor_bytes,enabled_predecessor_bytes
    # This historical slice stays exact; the new reduction guard checks current math.
    for name,pin in ctx['unchanged_kernel_source_raw_sha256'].items():
        raw=correction_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==pin,name
    for name,row in ctx['sources'].items():
        value=correction_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes())
        for fragment in row['inverse_fragments']:
            assert value.count(fragment['current'].encode())==1
            value=value.replace(fragment['current'].encode(),fragment['predecessor'].encode(),1)
        assert hashlib.sha256(value).hexdigest()==row['predecessor_canonical_lf_sha256'],name
        assert isinstance(enabled_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()),bytes)
    binder=(ROOT/'src/bind_finite_radon.cpp').read_text()
    assert '#define NPY_TARGET_VERSION NPY_1_23_API_VERSION'in binder
    assert 'PyArray_GetNDArrayCFeatureVersion'in binder and 'PyUFunc_API[46]'in binder


def test_actual_sdk_fixture_copy_stops_before_any_tools(tmp_path,monkeypatch):
    q=qualifier();out=tmp_path/'qualification';out.mkdir()
    (out/'package.json').write_text(json.dumps(dict(product_scope=q.CURRENT_PRODUCT_SCOPE,SDK_sha256={})))
    names=('tests/fixtures/find_boundaries_originals_v1.hpp','tests/label_pair_metrics_native_controls.hpp','tests/fixtures/label_pair_metrics_originals_v1.hpp','tests/fixtures/label_pair_rand_originals_v1.hpp','tests/contingency_preweighted_controls.hpp','tests/fixtures/finite_radon_saved_v2.hpp','tests/fixtures/finite_radon_exceptional_saved_v1.hpp','tests/fixtures/variation_of_information_saved_v1.hpp','tests/variation_of_information_current_counts_controls.hpp','tests/variation_of_information_large_controls.hpp','tests/cmake_consumer/variation_of_information_enabled_consumer.cpp','tests/fixtures/flood_fill_sdk_original_v1.inc')
    with zipfile.ZipFile(out/'source.zip','w')as archive:
        for name in names:archive.writestr(name,(ROOT/name).read_bytes())
        archive.writestr('tests/unrelated.hpp','MUST_NOT_COPY')
    class Stop(BaseException):pass
    def stop(*a,**k):raise Stop('BEFORE_TOOLS')
    monkeypatch.setattr(q,'tool_in',stop)
    with pytest.raises(Stop,match='BEFORE_TOOLS'):q.sdk(out,dict(profile_config=dict(config='Release')),None)
    copied=out/'consumer source'
    assert all((copied/name).read_bytes()==(ROOT/name).read_bytes()for name in names)
    assert not(copied/'tests/unrelated.hpp').exists()


def test_workflow_actual_pytest_selectors_precede_only_tee_output():
    source=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    line=next(x.strip()for x in source.splitlines()if x.strip().startswith('python -m pytest -q --capture=sys'))
    argv=shlex.split(line);pipe=argv.index('|')
    assert argv[pipe-1]=='2>&1'and argv[pipe:]==['|','tee','build/qualification-tool-guards.log']
    selectors=[x for x in argv[:pipe-1]if x.startswith('tests/')]
    assert len(selectors)==len(set(selectors))
    assert 'tests/test_finite_radon_numpy_abi_source.py'in selectors
    assert 'tests/test_variation_of_information_benchmark_execution.py'in selectors
    assert 'tests/test_abi_vi_integration_source.py'in selectors
    for selector in selectors:assert (ROOT/selector.split('::')[0]).is_file(),selector


def test_vi_default_factory_and_checker_are_strict_without_runtime():
    source=(ROOT/'python/skimagexx/coverage.py').read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='variation_of_information_default_error')
    text=ast.get_source_segment(source,fn)
    assert 'actual.tobytes()==expected.tobytes()'in text and 'np.allclose'not in text
    assert 'actual.shape==expected.shape==(2,)'in text and 'actual.base is expected.base is None'in text
    api=(ROOT/'tests/test_api.py').read_text();assert "if api=='metrics.variation_of_information':"in api
    assert 'float VI labels reached integer native counts'in api and 'args,options=default_call(api,a);before=observation(args,np)'in api


def test_genuine_collector_transport_and_scope_are_fail_closed():
    q=(ROOT/'tools/public_qualification.py').read_text()
    for fragment in("row['tests']==224","row['tests']==18","row['tests']==40","len(callers)==40","len(policy_rows)==18","'VI_CALLER_RECORD_DIR'","'FINITE_RADON_ABI_POLICY_RECORDS'"):
        assert fragment in q,fragment
    public=(ROOT/'tests/test_variation_of_information_public.py').read_text()
    assert "if not os.environ.get('VI_COMPONENT_DIR')"in public
    assert 'xx.metrics.variation_of_information is module.variation_of_information'in public
    assert 'helper.reference_fenv_restore(ctypes.byref(saved),32)'in public
    assert 'bytes(actual)==bytes(saved)'in public
    assert 'Caller restoration only; old VI rows contain no original hardware-FP witness'in public



@pytest.mark.parametrize('group,count', [('public-radon-ABI-policy-linux',18),('public-VI-saved-linux',40),('public-VI-large-saved-linux',8),('public-flood-saved-linux',100)])
def test_actual_new_group_parser_and_seal_count(group,count,monkeypatch):
    q=qualifier();calls=[]
    monkeypatch.setattr(q,'child',lambda out,name:calls.append(name)or 0)
    assert q.main(['_child','--output',str(ROOT/'build/NOT_CREATED_PARSER_ONLY'),'--group',group])==0
    assert calls==[group]
    tree=ast.parse((ROOT/'tools/public_qualification.py').read_bytes())
    seal=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='seal')
    nodes=[n for n in ast.walk(seal)if isinstance(n,ast.If)and ast.unparse(n.test)==f"group == '{group}'"]
    node=next(n for n in nodes if isinstance(n.body[0],ast.Expr)and isinstance(n.body[0].value,ast.Call)and isinstance(n.body[0].value.func,ast.Name)and n.body[0].value.func.id=='require')
    # VI's first clause has the count then transport; execute the count only.
    clause=ast.Module(body=[ast.If(test=node.test,body=node.body[:1],orelse=[])],type_ignores=[])
    code=compile(ast.fix_missing_locations(clause),'<actual group cardinality>','exec')
    exec(code,dict(group=group,row=dict(tests=count),require=q.require))
    for wrong in(0,count-1,count+1):
        with pytest.raises(q.QualificationError):exec(code,dict(group=group,row=dict(tests=wrong),require=q.require))
