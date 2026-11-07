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
    q=load('util_metadata_qualifier_'+profile,'tools/public_qualification.py');result=q.plan(ROOT,profile,280)
    assert not result['blockers'],result['blockers']
    assert result['registry']['registered_count']==283 and result['registry']['image_count']==280
    assert result['product_members']==226 and result['SDK_members']==107
    assert len(q.CURRENT_FAMILIES)==37


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
    first=next(n for n in fn.body if isinstance(n,ast.If)and "('util.crop', 'util.regular_grid')"in ast.unparse(n.test));assert "('util.crop', 'util.regular_grid')"in ast.unparse(first.test)
    assert 'assert_default_parity(api,np.dtype(dtype),actual,expected)'in ast.get_source_segment(text,first)


def test_regular_grid_installed_adapter_preserves_full49_and_all_assertions():
    raw=(ROOT/'tests/test_regular_grid_public.py').read_bytes()
    context=json.loads((ROOT/'coverage/contracts/regular-grid-public-integration-context-v1.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==context['current_raw_sha256']
    previous=raw
    for fragment in context['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and previous.count(current)==1
        previous=previous.replace(current,old,1)
    assert hashlib.sha256(previous).hexdigest()==context['predecessor_raw_sha256']
    old_tree,new_tree=ast.parse(previous),ast.parse(raw)
    old={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
    assert {name for name in old if name.startswith('test_')}=={name for name in new if name.startswith('test_')}
    for name in old:
        if name.startswith('test_')and name!='test_saved_rows':assert ast.dump(old[name])==ast.dump(new[name])
    for node in ast.walk(old['test_saved_rows']):
        if isinstance(node,(ast.Assert,ast.Try,ast.For)):
            assert ast.dump(node)in{ast.dump(n)for n in ast.walk(new['test_saved_rows'])if isinstance(n,type(node))}
    text=raw.decode();fixture=ast.get_source_segment(text,new['target'])
    assert "directory=os.environ.get('REGULAR_GRID_COMPONENT_DIR')"in fixture
    assert 'import skimagexx as xx'in fixture and 'xx.util.regular_grid is module.regular_grid and module._core is core'in fixture
    assert "entry.__module__=='skimagexx._core'"in fixture
    assert 'core.get_num_threads()==1'in fixture and 'yield module,core'in fixture
    assert 'finally:core.set_num_threads(1);compat.set_fallback_enabled(previous)'in fixture
    assert 'pytest.skip'not in text and 'pytest.mark.skip'not in text


@pytest.mark.parametrize('profile',('minimum','modern','windows','debug'))
def test_regular_grid_full_saved_group_has_explicit_profile_scope(profile):
    q=load('grid_public_profile_'+profile,'tools/public_qualification.py')
    original=q.groups_for_source(ROOT);groups=q.applicable_public_groups(original,profile)
    assert all(groups[name]==value for name,value in original.items())
    if profile=='windows':assert groups==original
    else:assert groups['public-regular-grid-saved-linux']==('test_regular_grid_public.py',)and len(groups)==len(original)+10
    plan=q.plan(ROOT,profile,280);assert not plan['blockers']
    assert plan['public_groups']==groups
    assert 'noWindows' in (ROOT/'tools/public_qualification.py').read_text()
    import argparse
    tree=ast.parse((ROOT/'tools/public_qualification.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    argument=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='add_argument'and n.args and isinstance(n.args[0],ast.Constant)and n.args[0].value=='--group')
    parser=argparse.ArgumentParser();scope=dict(parser=parser,GROUPS=q.GROUPS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=argument)],type_ignores=[])),'<actual-group-argument>','exec'),scope)
    assert parser.parse_args(['--group','public-regular-grid-saved-linux']).group=='public-regular-grid-saved-linux'


@pytest.mark.parametrize('profile',('minimum','windows'))
def test_regular_grid_actual_dispatch_keeps_global_groups_and_adds_linux49(tmp_path,monkeypatch,profile):
    q=load('grid_dispatch_'+profile,'tools/public_qualification.py')
    out=tmp_path/'qualification';out.mkdir();(out/'state.json').write_text(json.dumps(dict(source={},profile=profile)))
    calls=[];import types,sys
    monkeypatch.setattr(q,'source_state',lambda _: {})
    monkeypatch.setattr(q,'Journal',lambda _:types.SimpleNamespace(run=lambda label,argv,env:calls.append((label,list(map(str,argv))))))
    monkeypatch.setattr(q,'python_in',lambda _:Path(sys.executable));monkeypatch.setattr(q,'GROUPS',{'public-post203':('test_util_metadata_integration.py',)})
    monkeypatch.setattr(q,'CORPORA',{});monkeypatch.setattr(q,'environment',lambda package=None,scalar=False:{'scalar':scalar});monkeypatch.setattr(q,'sdk',lambda *_:None)
    q.qualify(out)
    assert all(any(label==mode+'-public-post203'for label,_ in calls)for mode in ('normal','scalar'))
    selected=[argv for label,argv in calls if label.endswith('-public-regular-grid-saved-linux')]
    assert len(selected)==(0 if profile=='windows'else 2)
    for argv in selected:assert argv[-2:]==['--group','public-regular-grid-saved-linux']
    text=(ROOT/'tools/public_qualification.py').read_text()
    assert "row['tests']==49,'Full49 grid saved-domain nodes required; no partial selector'"in text
    assert "require(not os.environ.get('REGULAR_GRID_COMPONENT_DIR')"in text


@pytest.mark.parametrize('count',(48,50))
def test_regular_grid_seal_refuses_incomplete_or_extra_saved_nodes(tmp_path,monkeypatch,count):
    q=load('grid_seal_'+str(count),'tools/public_qualification.py')
    q.GROUPS={'public-post203':('test_util_metadata_integration.py',)};q.CORPORA={}
    state=dict(source={},profile='minimum',build_identity={},interpreter_provenance={},expected={'image_count':271},product_wheel='inert.whl')
    (tmp_path/'state.json').write_text(json.dumps(state))
    labels={'prepare-util-views-fenv','initial-public-admission','actual-inventory','sdk-configure','sdk-build','sdk-list','sdk-normal','sdk-scalar','gather-build-identity-facts','capture-build-identity'}
    for mode in ('normal','scalar'):
        labels.update(mode+'-'+name for name in ('admission',*q.applicable_public_groups(q.GROUPS,'minimum'),'returned-maps','metrics-corpus'))
        for group in q.applicable_public_groups(q.GROUPS,'minimum'):
            tests=count if group=='public-regular-grid-saved-linux'else 1
            (tmp_path/(mode+'-'+group+'.xml')).write_text('<testsuite tests="'+str(tests)+'" errors="0" failures="0" skipped="0"/>')
    (tmp_path/'commands.json').write_text(json.dumps([dict(label=name,state='terminal',returncode=0)for name in labels]))
    (tmp_path/'product-wheels').mkdir();wheel=tmp_path/'product-wheels/inert.whl';wheel.write_bytes(b'INERT_UNIT_ONLY_NOT_A_WHEEL')
    (tmp_path/'package.json').write_text(json.dumps(dict(members_sha256={},wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())))
    monkeypatch.setattr(q,'source_state',lambda _:{});monkeypatch.setattr(q,'validate_build_identity',lambda *_:None)
    monkeypatch.setattr(q,'validate_interpreter_provenance',lambda *_:None);monkeypatch.setattr(q,'verify_tree',lambda *_:None)
    with pytest.raises(q.QualificationError,match='Full49'):q.seal(tmp_path)
    assert not(tmp_path/'manifest.json').exists()
