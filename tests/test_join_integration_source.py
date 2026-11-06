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
    assert cmake.count('target_sources(skimagexx PRIVATE src/join_segmentations.cpp src/crop.cpp src/regular_grid.cpp)')==1
    assert cmake.count('target_sources(_core PRIVATE src/bind_join_segmentations.cpp src/bind_crop.cpp src/bind_regular_grid.cpp)')==1
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
    import textwrap
    parsed=ast.parse(textwrap.dedent(branch))
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Call)and isinstance(n.func.func,ast.Name)and n.func.func.id=='getattr'and isinstance(n.func.args[0],ast.Name)and n.func.args[0].id=='original'for n in ast.walk(parsed))
    assert 'join_reference_rows(args,options,inspect.getsourcefile(getattr(original,name)))'in branch


@pytest.mark.parametrize('profile',('minimum','modern','windows','debug'))
def test_join_current_closed_source_plan_and_collector(profile):
    q=load('join_source_qualification_'+profile,'tools/public_qualification.py');plan=q.plan(ROOT,profile,268)
    assert not plan['blockers']and plan['readiness_only']
    assert (plan['registry']['image_count'],plan['registry']['registered_count'])==(268,271)
    assert (plan['product_scope'],plan['product_members'],plan['SDK_members'],len(plan['SDK_families']),plan['SDK_checks'])==('current268',212,100,32,64)
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


def source_check_argv(text):
    import shlex
    lines=[line.strip() for line in text.splitlines() if 'python -m pytest' in line and 'qualification-tool-guards.log' in line]
    assert len(lines)==1
    line=lines[0]
    assert line.count('|')==1 and line.count('2>&1 |')==1
    before,after=line.split('2>&1 |')
    assert shlex.split(after)==['tee','build/qualification-tool-guards.log']
    argv=shlex.split(before)
    assert argv[:3]==['python','-m','pytest']
    for name in ('tests/test_join_integration_source.py','tests/test_join_segmentations_native_contract.py','tests/test_join_segmentations_benchmark.py'):
        assert argv.count(name)==1
    return argv


def test_join_actual_pytest_argv_and_sole_tee_target():
    source_check_argv((ROOT/'.github/workflows/public-qualification.yml').read_text())


def test_join_cache_authenticates_actual_inputs_options_and_upstream_source():
    source=(ROOT/'tests/test_api.py').read_text()
    tree=ast.parse(source)
    fixture=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='join_reference_rows')
    reference=next(n for n in fixture.body if isinstance(n,ast.FunctionDef)and n.name=='reference')
    assert [a.arg for a in reference.args.args]==['args','options','source_path']
    assert 'inspect.getsourcefile(getattr(original,name))' in source
    assert 'SOURCE_DERIVED_EARLY_ERROR_EQUIVALENCE' in ast.get_source_segment(source,fixture)
    assert 'encode(args,np)' in ast.get_source_segment(source,fixture)


@pytest.mark.parametrize('mutation',('extra-tee-output','missing-selector','second-pipe'))
def test_join_workflow_guard_rejects_shell_boundary_mutations(mutation):
    text=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    if mutation=='extra-tee-output':text=text.replace('tee build/qualification-tool-guards.log','tee build/qualification-tool-guards.log tests/test_join_integration_source.py')
    elif mutation=='missing-selector':text=text.replace(' tests/test_join_segmentations_benchmark.py','')
    else:text=text.replace('2>&1 | tee','2>&1 | cat | tee')
    with pytest.raises(AssertionError):source_check_argv(text)



def test_join_unsupported_cache_equivalence_pins_complete_pre_error_dependency():
    source=(ROOT/'tests/fixtures/join_segmentations_upstream_v026.py.txt').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(source).hexdigest()=='8d1da1717bf35818f57156f88a94cd84a9527f48aff16d60e250b7d8045cd8be'
    tree=ast.parse(source)
    join=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='join_segmentations')
    relabel=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='relabel_sequential')
    assert ast.unparse(join.body[1].test)=='s1.shape != s2.shape'
    assert ast.unparse(join.body[2].value)=='relabel_sequential(s1)'
    assert ast.unparse(join.body[3].value)=='relabel_sequential(s2)'
    assert [a.arg for a in relabel.args.args]==['label_field','offset']and relabel.args.defaults[0].value==1
    prefix=ast.Module(body=relabel.body[1:8],type_ignores=[])
    expected=ast.parse('''if offset <= 0:
    raise ValueError("Offset must be strictly positive.")
if np.min(label_field) < 0:
    raise ValueError("Cannot relabel array that contains negative values.")
offset = int(offset)
in_vals = np.unique(label_field)
if in_vals[0] == 0:
    out_vals = np.concatenate([[0], np.arange(offset, offset + len(in_vals) - 1)])
else:
    out_vals = np.arange(offset, offset + len(in_vals))
input_type = label_field.dtype
if input_type.kind not in "iu":
    raise TypeError("label_field must have an integer dtype")''')
    assert ast.dump(prefix)==ast.dump(expected)
    # No source invocation. Both prefixes parsed on this interpreter; full
    # module canonical-byte pin authenticates signature/default/remaining code.
    assert all(n.id not in ('s2','return_mapping','map_array','ArrayMap')for n in ast.walk(prefix)if isinstance(n,ast.Name))
