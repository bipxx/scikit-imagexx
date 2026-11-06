"""Affected shared JOIN source gates; no scientific imports or target calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_join_single_target_and_existing_global_worker_wiring():
    cmake=(ROOT/'CMakeLists.txt').read_text();bindings=(ROOT/'src/bindings.cpp').read_text()
    assert cmake.count('target_sources(skimagexx PRIVATE src/join_segmentations.cpp)')==1
    assert cmake.count('target_sources(_core PRIVATE src/bind_join_segmentations.cpp)')==1
    assert cmake.count('set_source_files_properties(src/join_segmentations.cpp src/bind_join_segmentations.cpp')==2
    assert bindings.count('void bind_join_segmentations(pybind11::module_&);')==bindings.count('    bind_join_segmentations(m);')==1
    leaf=(ROOT/'src/join_segmentations.cpp').read_text()
    assert 'detail::active_workers(n)'in leaf and 'std::atomic'not in leaf
    assert 'PYBIND11_MODULE'not in (ROOT/'src/bind_join_segmentations.cpp').read_text()
    public=(ROOT/'python/skimagexx/segmentation.py').read_text()
    assert 'from ._join_segmentations import join_segmentations'in public


def test_join_binary_registry_default_and_exact_checker_source():
    coverage=load('join_shared_coverage','python/skimagexx/coverage.py')
    assert 'segmentation.join_segmentations'in coverage.native_functions()
    assert coverage.native_dtypes('segmentation.join_segmentations')==('int8','uint8','int16','uint16','int32','uint32','int64','uint64')
    source=(ROOT/'python/skimagexx/coverage.py').read_text()
    assert "return (np.array([0,0,1,1,2,2,3,3],dtype=image.dtype).reshape(2,4),"in source
    assert "np.array([0,1,1,0,0,1,1,0],dtype=image.dtype).reshape(2,4)),{}"in source
    assert 'def join_segmentations_default_error(actual,expected):'in source
    assert "actual.tobytes(order='C')==expected.tobytes(order='C')"in source
    smoke=(ROOT/'tests/test_api.py').read_text()
    branch=smoke[smoke.index("        if api == 'segmentation.join_segmentations':"):]
    branch=branch[:branch.index("        if api == 'metrics.contingency_table':")]
    assert 'default_call(api,a)'in branch and '(*args,**options)'in branch and '(a)'not in branch
    assert 'getattr(original,name)'not in branch and 'join_reference_rows(dtype)'in branch


@pytest.mark.parametrize('profile',('minimum','modern','windows','debug'))
def test_join_current_closed_source_plan_and_collector(profile):
    q=load('join_source_qualification_'+profile,'tools/public_qualification.py');plan=q.plan(ROOT,profile,266)
    assert not plan['blockers']and plan['readiness_only']
    assert (plan['registry']['image_count'],plan['registry']['registered_count'])==(266,269)
    assert (plan['product_scope'],plan['product_members'],plan['SDK_members'],len(plan['SDK_families']),plan['SDK_checks'])==('current266',208,98,32,64)
    required={'test_join_segmentations_public.py','test_join_segmentations_integration.py','test_join_segmentations_native_contract.py','test_join_segmentations_benchmark.py'}
    assert required<=set(plan['public_groups']['public-post203'])
    current=q.current_contract(lambda name:(ROOT/name).read_bytes())
    assert current['post203_contracts']['segmentation.join_segmentations']=='tests/test_join_segmentations_integration.py'
    assert 'include/skimagexx/join_segmentations.hpp'in current['required_client_headers']


def test_join_existing_SDK_composition_and_real_source_copy_predicate():
    client=(ROOT/'tests/test_component_consumers_native.cpp').read_text()
    assert '#include "cmake_consumer/join_segmentations_consumer.hpp"'in client
    assert 'require(skimagexx_test::join_segmentations_consumer(workers)>0);'in client
    qtree=ast.parse((ROOT/'tools/public_qualification.py').read_bytes())
    sdk=next(n for n in qtree.body if isinstance(n,ast.FunctionDef)and n.name=='sdk')
    predicate=next(n.test for n in ast.walk(sdk)if isinstance(n,ast.If)and 'tests/fixtures/bezier_curve_originals_v1.bin'in ast.unparse(n.test))
    expression=compile(ast.Expression(predicate),'<actual SDK copy>','eval')
    for name in ('tests/cmake_consumer/join_segmentations_consumer.hpp','tests/fixtures/join_segmentations_originals_v1.hpp','tests/test_component_consumers_native.cpp'):
        assert (ROOT/name).is_file()and eval(expression,{'name':name})
    assert not eval(expression,{'name':'tests/private_unrelated.hpp'})


def test_join_strict_additive_context_preserves_all_old_source_hashes():
    helper=load('join_strict_source_context','tools/join_source_context.py')
    context=json.loads((ROOT/'coverage/contracts/join-additive-source-context-v1.json').read_bytes())
    for name,row in context['sources'].items():
        original=helper.predecessor_bytes(ROOT,name,(ROOT/name).read_bytes())
        assert hashlib.sha256(original).hexdigest()==row['predecessor_canonical_lf_sha256']
    old=json.loads((ROOT/'coverage/contracts/join-segmentations-native-v1.json').read_bytes())
    current=json.loads((ROOT/'coverage/contracts/join-segmentations-native-v2.json').read_bytes())
    assert hashlib.sha256((ROOT/current['predecessor']['contract']).read_bytes()).hexdigest()==current['predecessor']['raw_sha256']
    for name in ('src/join_segmentations.cpp','src/bind_join_segmentations.cpp','include/skimagexx/join_segmentations.hpp'):
        assert old['source_raw_sha256'][name]==current['source_raw_sha256'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


@pytest.mark.parametrize('name',('CMakeLists.txt','src/bindings.cpp','python/skimagexx/coverage.py','benchmarks/draw_polygon.py'))
def test_join_additive_context_rejects_unrelated_mutation(name):
    helper=load('join_mutation_source_context','tools/join_source_context.py')
    with pytest.raises(AssertionError):helper.predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()+b'\n# unrelated mutation\n')


def test_join_genuine_guards_and_unmet_domains_are_explicit():
    genuine=(ROOT/'tests/test_join_segmentations_integration.py').read_text()
    assert 'owned._core is consumers._core is xx._core'in genuine
    assert 'pipeline(xx)'in genuine and "entry.__module__=='skimagexx._core'"in genuine
    assert "with pytest.raises(TypeError):imported(args[0])"in genuine
    assert "('bool','float16','float32','float64')"in genuine
    assert '_join_segmentations_core'not in genuine and 'spec_from_file_location'not in genuine
    scope=json.loads((ROOT/'coverage/contracts/join-segmentations-native-v2.json').read_bytes())
    assert scope['API_completion']is False
    assert any('ArrayMap class identity'in x for x in scope['frontiers'])
    assert 'nonpositive'in scope['standalone_SDK']['unmet_domains'][0]
    assert 'floating'in scope['standalone_SDK']['unmet_domains'][1]



def test_join_shared_public_fixture_has_no_private_test_body_imports():
    source=(ROOT/'tests/test_join_segmentations_public.py').read_text()
    tree=ast.parse(source)
    for function in tree.body:
        if isinstance(function,ast.FunctionDef)and function.name.startswith('test_'):
            assert "'_join_test_package"not in ast.get_source_segment(source,function)
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='target')
    first=target.body[0]
    assert isinstance(first,ast.If)and 'JOIN_COMPONENT_DIR'in ast.unparse(first.test)
    assert 'skimagexx._join_segmentations'in ast.get_source_segment(source,first)
