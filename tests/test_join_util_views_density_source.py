"""Current compatible bundle source-only shared wiring and exact contexts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shlex
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_three_view_APIs_single_core_build_registry_namespace_defaults():
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    apis=('util.slice_along_axes','util.view_as_blocks','util.view_as_windows')
    assert all(a in coverage['native_functions']()for a in apis)
    assert all(len(coverage['native_dtypes'](a))==14 for a in apis)
    cmake=(ROOT/'CMakeLists.txt').read_text();bindings=(ROOT/'src/bindings.cpp').read_text();util=(ROOT/'python/skimagexx/util.py').read_text()
    assert cmake.count('src/util_views.cpp')==3 and cmake.count('src/bind_util_views.cpp')==3
    assert bindings.count('void bind_util_views(')==1 and bindings.count('bind_util_views(m);')==1
    assert 'from ._array_views import view_as_blocks, view_as_windows'in util and 'from ._slice_along_axes import slice_along_axes'in util
    assert 'arrays._core is slicing._core is xx._core'in(ROOT/'tests/test_util_views_integration.py').read_text()
    assert "('util.slice_along_axes','util.view_as_blocks','util.view_as_windows')"in(ROOT/'tests/test_api.py').read_text()
    for name in ('_array_views.py','_slice_along_axes.py'):
        for node in ast.walk(ast.parse((ROOT/'python/skimagexx'/name).read_bytes())):
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0]in('skimage','scipy')for a in node.names)
            if isinstance(node,ast.ImportFrom):assert(node.module or'').split('.')[0]not in('skimage','scipy')
    reference=(ROOT/'tests/util_views_reference.py').read_text()
    assert 'Path(source_path).read_bytes()'in reference and 'IDENTICAL_CAPTURED_UTIL_VIEW_REQUEST'in reference
    assert "dict(dtype.metadata)=={'units':'linear','source':'parity'}"in reference
    assert "encode((physical,options),np)=="in reference


def test_compatible_shared_density_context_is_exact_and_tamper_refuses():
    context=json.loads((ROOT/'coverage/contracts/util-views-density-additive-context-v1.json').read_bytes())
    spec=importlib.util.spec_from_file_location('combined_current_source',ROOT/'tools/join_source_context.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name,row in context['sources'].items():
        actual=(ROOT/name).read_bytes().replace(b'\r\n',b'\n');raw=m.enabled_predecessor_bytes(ROOT,name,actual);assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256']
        projected=raw
        for f in row['inverse_fragments']:
            assert projected.count(f['current'].encode())==1;projected=projected.replace(f['current'].encode(),f['predecessor'].encode(),1)
        assert hashlib.sha256(projected).hexdigest()==row['predecessor_canonical_lf_sha256']
        m.predecessor_bytes(ROOT,name,actual)
        with pytest.raises(AssertionError):m.predecessor_bytes(ROOT,name,actual+b'\n// UNADMITTED_CHANGE\n')
    metric=runpy.run_path(str(ROOT/'tools/label_pair_benchmark_context.py'))
    metric['validate_source_closure'](ROOT,json.loads((ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json').read_bytes()))


def test_f_boundary_case_method_and_source_successor():
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'));c=json.loads(driver['CONTRACT'].read_bytes());driver['validate_source_closure'](c)
    old=json.loads((ROOT/c['f_row_successor']['contract']).read_bytes())
    assert c['cases']==old['cases']and c['sampler']==old['sampler']and c['method_sha256']==old['method_sha256']and len(c['cases'])==8
    assert all(c[k]is None for k in ('shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'))


def test_shared_SDK_complete_operations_and_source_pipeline_selectors():
    sdk=(ROOT/'tests/test_component_consumers_native.cpp').read_text()
    assert '#include "cmake_consumer/util_views_consumer.hpp"'in sdk and 'util_views_consumer();'in sdk
    assert 'if(argc==1){require(get_num_threads()==1);util_views_consumer();require(get_num_threads()==1);}'in sdk
    body=(ROOT/'tests/cmake_consumer/util_views_consumer.hpp').read_text()
    for token in ('view_as_blocks(', 'view_as_windows(', 'slice_along_axes(', 'Kind::owned','std::move(', 'assert(saved==1)'):
        assert token in body
    line=next(l for l in(ROOT/'.github/workflows/public-qualification.yml').read_text().splitlines()if 'qualification-tool-guards.log'in l and 'python -m pytest'in l)
    left,right=line.split('2>&1 |');assert shlex.split(right)==['tee','build/qualification-tool-guards.log']
    assert shlex.split(left).count('tests/test_join_util_views_density_source.py')==1


@pytest.mark.parametrize('profile',('minimum','modern','debug','windows'))
def test_saved_view_profile_and_actual_helper_prepare_seal_transport(profile):
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'))
    base=q['groups_for_source'](ROOT);groups=q['applicable_public_groups'](base,profile)
    assert all(groups[n]==v for n,v in base.items())
    if profile=='windows':assert groups==base
    else:assert groups['public-util-views-saved-linux']==('test_util_views_public.py',)and groups['public-regular-grid-saved-linux']==('test_regular_grid_public.py',)
    for name in ('tests/util_views_fenv_restore.c','tests/util_views_fenv_build.py','tests/util_views_fenv.py','tests/util_views_fenv_probe.py','coverage/contracts/util-views-fenv-v1.json'):
        assert q['DRAW_SOURCE_PATHS'].count(name)==1
    qualifier=(ROOT/'tools/public_qualification.py').read_text()
    for token in ('prepare-util-views-fenv','--control-python',"util_views_fenv_environment(out,state)","row['tests']==553"):
        assert token in qualifier
    stage=(ROOT/'tools/rank_statistics_products.py').read_text()
    assert "qualified FENV bundle file set differs"in stage and "staged qualified state differs"in stage
    assert "selected.update(util_views_fenv_files(qout,q,state.get('util_views_fenv')))"in stage


@pytest.mark.parametrize('fault',('none','missing','windows','path','digest','legacy'))
def test_actual_genuine_helper_environment_authenticates_external_receipt(tmp_path,monkeypatch,fault):
    import importlib.util,os
    from test_util_views_fenv import synthetic_bundle
    stage=tmp_path/'stage';bundle=stage/'util-views-fenv';bundle.mkdir(parents=True)
    path,record=synthetic_bundle(bundle)
    binding=dict(receipt='util-views-fenv/receipt.json',receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    state=dict(profile='minimum',util_views_fenv=binding)
    if fault=='missing':state.pop('util_views_fenv')
    elif fault=='windows':state['profile']='windows'
    elif fault=='path':binding['receipt']='../receipt.json'
    elif fault=='digest':binding['receipt_sha256']='0'*64
    elif fault=='legacy':monkeypatch.setenv('UTIL_VIEWS_FENV_HELPER','UNQUALIFIED_OLD_LIBRARY')
    else:monkeypatch.delenv('UTIL_VIEWS_FENV_HELPER',raising=False)
    spec=importlib.util.spec_from_file_location('qualified_actual_helper_env',ROOT/'tools/public_qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    if fault=='none':
        env=q.util_views_fenv_environment(stage,state,toolchain=False)
        assert env==dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(path.resolve()),UTIL_VIEWS_FENV_RECEIPT_SHA256=binding['receipt_sha256'])
    else:
        with pytest.raises((AssertionError,q.QualificationError)):q.util_views_fenv_environment(stage,state,toolchain=False)
    # Synthetic byte library never loaded, no actual target/helper/science call.


@pytest.mark.parametrize('fault',('none','missing','tamper','extra-record','bad-path','receipt-digest'))
def test_actual_stager_helper_collection_is_closed_and_tamper_refuses(tmp_path,fault):
    from test_util_views_fenv import synthetic_bundle
    directory=tmp_path/'stage';bundle=directory/'util-views-fenv';bundle.mkdir(parents=True)
    receipt,manifest=synthetic_bundle(bundle)
    binding=dict(receipt='util-views-fenv/receipt.json',receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    records={p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in bundle.iterdir()if p.is_file()}
    qualification=dict(records=records)
    if fault=='missing':(bundle/'controls.json').unlink()
    elif fault=='tamper':(bundle/'libutil_views_fenv_restore.so').write_bytes(b'UNQUALIFIED')
    elif fault=='extra-record':records['util-views-fenv/extra.json']=dict(bytes=0,sha256='0'*64)
    elif fault=='receipt-digest':binding['receipt_sha256']='0'*64
    elif fault=='bad-path':
        manifest['files']['../escape']='0'*64;receipt.write_text(json.dumps(manifest));binding['receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest();records[binding['receipt']]=dict(bytes=receipt.stat().st_size,sha256=binding['receipt_sha256']);records['util-views-fenv/../escape']=dict(bytes=0,sha256='0'*64)
    spec=importlib.util.spec_from_file_location('actual_stager_helper_collection',ROOT/'tools/rank_statistics_products.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if fault=='none':
        files=module.util_views_fenv_files(directory,qualification,binding)
        assert set(files)==set(records)and all(p.is_file()for p in files.values())
    else:
        with pytest.raises((RuntimeError,OSError,AssertionError)):module.util_views_fenv_files(directory,qualification,binding)
    # Synthetic metadata only: fake library is never loaded or built.
