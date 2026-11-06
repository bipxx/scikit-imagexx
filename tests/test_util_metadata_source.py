"""Shared crop/grid source-only closure; never import scientific modules."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import pytest
ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_util_metadata_build_namespace_and_single_core_source():
    cmake=(ROOT/'CMakeLists.txt').read_text();binding=(ROOT/'src/bindings.cpp').read_text();util=(ROOT/'python/skimagexx/util.py').read_text()
    for name in ('crop','regular_grid'):
        assert cmake.count('src/'+name+'.cpp')==3 and cmake.count('src/bind_'+name+'.cpp')==3
        assert binding.count('void bind_'+name+'(')==1 and binding.count('bind_'+name+'(m);')==1
        assert 'from ._'+name+' import '+name in util
    assert 'crop._core is grid._core is xx._core'in(ROOT/'tests/test_util_metadata_integration.py').read_text()
    assert '_regular_grid_core'not in(ROOT/'tests/test_util_metadata_integration.py').read_text()
    assert hashlib.sha256((ROOT/'python/skimagexx/_regular_seeds.py').read_bytes()).hexdigest()=='22d05626e51af3917fa1869ff6c754ef7163f10d605e181bb02e99a3f4715a37'


def test_util_metadata_source_bound_reference_and_checker():
    coverage=load('util_metadata_source_coverage','python/skimagexx/coverage.py')
    assert set(('util.crop','util.regular_grid'))<=set(coverage.native_functions())
    helper=(ROOT/'tests/util_metadata_reference.py').read_text()
    assert 'source_path' in helper and 'hashlib.sha256(Path(source_path).read_bytes()'in helper
    assert 'IDENTICAL_CAPTURED_CROP_REQUEST'in helper and 'SOURCE_DERIVED_TYPED_GRID_METADATA_EQUIVALENCE'in helper
    assert 'type(args[0])is np.ndarray'in helper and "dict(dtype.metadata)=={'units':'linear','source':'parity'}"in helper
    smoke=(ROOT/'tests/test_api.py').read_text();assert 'reference(api,args,options,inspect.getsourcefile(getattr(original,name)),np)'in smoke
    tree=ast.parse(smoke);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_every_native_function')
    branch=next(n for n in fn.body if isinstance(n,ast.If)and "('util.crop', 'util.regular_grid')"in ast.unparse(n.test))
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Call)and ast.unparse(n.func).startswith('getattr(original')for n in ast.walk(branch))


def test_util_metadata_real_pytest_pipeline_and_SDK_closure():
    line=next(l for l in(ROOT/'.github/workflows/public-qualification.yml').read_text().splitlines()if 'qualification-tool-guards.log' in l and 'python -m pytest' in l)
    left,right=line.split('2>&1 |');assert shlex.split(right)==['tee','build/qualification-tool-guards.log']
    args=shlex.split(left)
    for f in ('test_util_metadata_source.py','test_crop_benchmark.py','test_regular_grid_native_contract.py','test_regular_grid_benchmark.py'):assert args.count('tests/'+f)==1
    consumer=(ROOT/'tests/test_component_consumers_native.cpp').read_text()
    for name in ('crop_component','regular_grid'):
        assert '#include "cmake_consumer/'+name+'_consumer.hpp"'in consumer
        assert 'require(skimagexx_test::'+name+'_consumer(workers)>0);'in consumer
    qualifier=(ROOT/'tools/public_qualification.py').read_text()
    assert "'tests/fixtures/regular_grid_originals_v1.hpp'"in qualifier
    crop=(ROOT/'tests/cmake_consumer/crop_component_consumer.hpp').read_text()
    assert 'set_num_threads(workers)'in crop and '~Restore(){set_num_threads(n);}'in crop
    for name in ('crop.hpp','regular_grid.hpp'):assert 'include/skimagexx/'+name in qualifier


@pytest.mark.parametrize('profile',('minimum','modern','windows','debug'))
def test_util_metadata_current_source_plan(profile):
    q=load('util_metadata_qualifier_'+profile,'tools/public_qualification.py');result=q.plan(ROOT,profile,268)
    assert not result['blockers'],result['blockers']
    assert result['registry']['registered_count']==271 and result['registry']['image_count']==268
    assert result['product_members']==212 and result['SDK_members']==100
    assert len(q.CURRENT_FAMILIES)==32


def test_util_metadata_descriptor_consumers_keep_source_bound_whole_results():
    for name,symbol in (('test_dtype_aliases.py','test_every_native_default_accepts_equivalent_descriptor'),('test_dtype_metadata.py','calls')):
        text=(ROOT/'tests'/name).read_text();tree=ast.parse(text)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==symbol)
        branch=next(n for n in fn.body if isinstance(n,ast.If)and "('util.crop', 'util.regular_grid')"in ast.unparse(n.test))
        body=ast.get_source_segment(text,branch)
        assert 'saved_reference(api,arguments,options,inspect.getsourcefile('in body
        assert 'actual=native(*arguments,**options)'in body
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in ('original','reference')for n in ast.walk(branch))
        assert ('tagged=True'in body)==(name=='test_dtype_metadata.py')
    text=(ROOT/'tests/test_dtype_metadata.py').read_text();tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compare')
    first=fn.body[0];assert "('util.crop', 'util.regular_grid')"in ast.unparse(first.test)
    assert 'assert_default_parity(api,np.dtype(dtype),actual,expected)'in ast.get_source_segment(text,first)
