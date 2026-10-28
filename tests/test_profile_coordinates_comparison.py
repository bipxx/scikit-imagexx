"""Coordinate ABBA metadata guards; reuse the proven decision, never time code."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/profile-coordinates-same-host.yml').read_text(encoding='utf-8')
OLD=(ROOT/'.github/workflows/profile-sampling-same-host.yml').read_text(encoding='utf-8')
def scripts(text):
    return [textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",text,re.S)]
SCRIPTS=scripts(WORKFLOW)
PAIR=next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'"and any(isinstance(x,ast.Assign)and ast.unparse(x.targets[0])=='(old, new)'for x in n.body))
COMMON=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='common'))
MUTABLE=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='mutable'))
ADDED=ast.literal_eval(next(n.value for n in PAIR.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='added'))


def test_same_proven_decision_AST_except_frozen_contract_and_deadline():
    assert len(SCRIPTS)==4
    for source in SCRIPTS:compile(source,'workflow-inline','exec')
    def normalized(source):
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant)and node.value in ('2026-10-28T08:08:50.185599+00:00','2026-10-28T08:53:40.730139+00:00'):node.value='FROZEN_DEADLINE'
            if isinstance(node,ast.Constant)and node.value in ('05c198d0f01adfc37ef1d31beec9f46f30b9fe8e39735b851f6b64912fc2017f','85d2536b85ecfd6f4d75c9eee3694b081fb909073cb8106061877da9b50c4a70'):node.value='FROZEN_CONTRACT'
        return ast.dump(tree)
    assert normalized(SCRIPTS[3])==normalized(scripts(OLD)[3])
    assert '2026-10-28T08:53:40.730139+00:00'in SCRIPTS[3]
    assert '85d2536b85ecfd6f4d75c9eee3694b081fb909073cb8106061877da9b50c4a70'in SCRIPTS[3]


def test_fixed_products_runtime_normalization_and_isolated_ABBA():
    assert "default: '40645607032'"in WORKFLOW
    assert "default: 'e190a76500ebd33f3a2d3cb2aee26668a486b72f'"in WORKFLOW
    assert "'0e8b9de6ed849f12b23330c50b374cd4caa8de63'"in SCRIPTS[0]
    assert "item['id'] == 12798011912"in SCRIPTS[0]
    assert '8aa52e2ac8f1466c4be0f7fe59f5d457367f77becdbbd9d45276b4c0a7135e88'in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure the fixed18',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline; do'in timed
    assert 'benchmarks/profile_line.py --threads 1 --repeat 7'in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel',timed)
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD'in timed
    assert 'name: profile-coordinates-same-host-minimum'in WORKFLOW and 'if: always()'in WORKFLOW
    assert '--force-reinstall'in WORKFLOW and '-r "snapshots/$role/build/measurement-v2/pytest-lock.txt"'in WORKFLOW
    assert "updated['runtime']['versions']['pytest'] = '9.2.0'"in SCRIPTS[2]


def manifest(candidate=False):
    version='9.3.0'if candidate else'9.2.0'
    runtime={k:'same'for k in ('python','implementation','cache_tag','machine','system')}
    runtime.update(dependency_libraries_sha256={'numpy':{'blas.so':'a'}},versions={'numpy':'1.24.4','pytest':version,'scikit-imagexx':'0.1.0'},packages_sha256={k:{'source.py':'a'}for k in ('numpy','scipy','skimage')})
    paths=COMMON|(MUTABLE if candidate else MUTABLE-ADDED)|{'src/convert.cpp','pyproject.toml'}
    wheels={f'dep{i}.whl':dict(name=f'dep{i}',version='1',sha256='a'*64)for i in range(12)}
    digest='c'*64 if candidate else'07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588'
    wheels['pytest-'+version+'-py3-none-any.whl']=dict(name='pytest',version=version,sha256=digest)
    wheels['native.whl']=dict(name='scikit-imagexx',version='0.1.0',sha256='b'*64 if candidate else'a'*64)
    return dict(schema='ci-offline-measurement-v1',source=dict(revision='b'*40,files_sha256=dict.fromkeys(paths,'a')),runtime=runtime,wheels=wheels)


@pytest.mark.parametrize('bad',[None,'harness','old-public','other-kernel','added-missing','other-file','science-wheel','science-runtime','baseline-pytest','pytest-runtime'])
def test_pair_allows_coordinate_files_and_pytest_only(bad):
    old,new=manifest(),manifest(True)
    for name in MUTABLE:new['source']['files_sha256'][name]='b'
    paths=new['source']['files_sha256']
    if bad=='harness':paths['benchmarks/profile_line.py']='changed'
    elif bad=='old-public':paths['tests/profile_line_cases.py']='changed'
    elif bad=='other-kernel':paths['src/convert.cpp']='changed'
    elif bad=='added-missing':del paths['tests/profile_coordinates_cases.py']
    elif bad=='other-file':paths['new-source.cpp']='changed'
    elif bad=='science-wheel':new['wheels']['dep0.whl']['sha256']='changed'
    elif bad=='science-runtime':new['runtime']['versions']['numpy']='changed'
    elif bad=='baseline-pytest':old['wheels']['pytest-9.2.0-py3-none-any.whl']['sha256']='changed'
    elif bad=='pytest-runtime':new['runtime']['versions']['pytest']='other'
    code=compile(ast.Module(body=PAIR.body[1:-1],type_ignores=[]),'pair-equality','exec')
    namespace=dict(old=old,new=new,re=re)
    if bad:
        with pytest.raises(AssertionError):exec(code,namespace)
    else:exec(code,namespace)


def write_preparation(folder,candidate=False):
    script=ast.parse(SCRIPTS[1]).body
    cohorts=next(ast.literal_eval(n.iter)for n in script if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple))
    for family,count in [*cohorts,*([('profile-coordinates',51)]if candidate else[])]:
        for mode in ('normal','scalar'):
            tree=ET.Element('testsuite')
            for i in range(count):ET.SubElement(tree,'testcase',name=str(i))
            (folder/(family+'-'+mode+'.xml')).write_bytes(ET.tostring(tree))
    guards=next(ast.literal_eval(n.iter)for n in script if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and 'polygon-intake'in ast.unparse(n.iter))
    for family,count in guards:
        tree=ET.Element('testsuite')
        for i in range(count):ET.SubElement(tree,'testcase',name=str(i))
        (folder/(family+'.xml')).write_bytes(ET.tostring(tree))
    smoke=dict(status='passed',exit_code=0,outcomes={'passed':1413},results=[{}]*1413,implementations={'native-partial':235,'metadata':3})
    (folder/'feature-smoke.json').write_text(json.dumps(smoke))
    (folder/'measurement.json').write_text(json.dumps(manifest(candidate)))


@pytest.mark.parametrize('candidate,bad',[(False,None),(True,None),(True,'missing'),(True,'short'),(True,'failure')])
def test_additive_51_coordinate_preparation_is_required_only_for_candidate(tmp_path,monkeypatch,candidate,bad):
    write_preparation(tmp_path,candidate)
    monkeypatch.setenv('BASELINE_REVISION','b'*40);monkeypatch.setenv('CANDIDATE_REVISION','b'*40)
    if bad:
        p=tmp_path/'profile-coordinates-scalar.xml'
        if bad=='missing':p.unlink()
        else:
            tree=ET.fromstring(p.read_bytes())
            if bad=='short':tree.remove(tree[-1])
            else:ET.SubElement(tree[0],'failure')
            p.write_bytes(ET.tostring(tree))
    body=ast.parse(SCRIPTS[1]).body
    start=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='manifest')
    stop=next(i for i,n in enumerate(body)if n is PAIR)if any(n is PAIR for n in body)else next(i for i,n in enumerate(body)if isinstance(n,ast.If)and any(isinstance(x,ast.Assign)and ast.unparse(x.targets[0])=='(old, new)'for x in n.body))
    import os
    namespace=dict(prepared=tmp_path,role='candidate'if candidate else'baseline',json=json,ET=ET,os=os)
    code=compile(ast.Module(body=body[start:stop],type_ignores=[]),'preparation-counts','exec')
    if bad:
        with pytest.raises((AssertionError,FileNotFoundError)):exec(code,namespace)
    else:exec(code,namespace)


def source_fixture(tmp_path):
    old={
        'src/resample.cpp':'namespace x {\nbool profile_sample(){return true;}\n}\n',
        'src/simd_resample_avx2.cpp':'namespace skimagexx::simd {\nint zoom_sample(){return 0;}\n} // namespace skimagexx::simd\n',
        'src/resample_detail.hpp':'#pragma once\nbool profile_sample();\n',
        'src/bind_profile_line.cpp':'void bind_profile_line(){\n    module.def("profile_sample", []{});\n}\n',
        'python/skimagexx/_profile_line.py':'def _line_profile_coordinates(src,dst,linewidth=1):\n    line_row=old_prelude()\n    return old_stack()\ndef profile_line(image):\n    return old_sampler(image)\n',
        'CMakeLists.txt':'old_compile_flags()\n    add_executable(test_polygon_native old.cpp)\n',
        '.github/workflows/ci.yml':'      - name: Next\n        run: old_command\n'}
    new=deepcopy(old)
    new['src/resample.cpp']=old['src/resample.cpp'].rsplit('\n}',1)[0]+'\nbool profile_coordinates_into(){return true;}\n}\n'
    new['src/simd_resample_avx2.cpp']=old['src/simd_resample_avx2.cpp'].rsplit('\n} //',1)[0]+'\nstd::size_t profile_coordinate_row(){return 0;}\n} // namespace skimagexx::simd\n'
    new['src/resample_detail.hpp']=old['src/resample_detail.hpp'].rstrip('\n')+'\nstd::size_t profile_coordinate_row(double start);\nstd::size_t profile_coordinate_single(double radius);\n// Prepared C float64 center lines; fresh output. False restores\n// FP state and requests fallback.\nbool profile_coordinates_into(double* rows);\n'
    new['src/bind_profile_line.cpp']=old['src/bind_profile_line.cpp'].replace('    module.def("profile_sample"','    module.def("profile_coordinates", []{});\n    module.def("profile_sample"')
    new['python/skimagexx/_profile_line.py']=old['python/skimagexx/_profile_line.py'].replace('    return old_stack()','    if type(linewidth) is int:\n        native_coordinates=owned_grid()\n        if native_coordinates is not None:return native_coordinates\n    return old_stack()')
    new['CMakeLists.txt']=old['CMakeLists.txt'].replace('    add_executable(test_polygon_native','    add_executable(test_profile_coordinates_native new.cpp)\n    new_test_only_options()\n    add_executable(test_polygon_native')
    new['.github/workflows/ci.yml']='      - name: Profile coordinate exact words and protocol parity\n        run: new_normal\n      - name: Profile coordinate exact words and protocol parity with SIMD disabled\n        run: new_scalar\n        env:\n          SKIMAGEXX_DISABLE_SIMD: 1\n'+old['.github/workflows/ci.yml']
    for role,values in (('baseline',old),('candidate',new)):
        for name,text in values.items():
            p=tmp_path/'snapshots'/role/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)


@pytest.mark.parametrize('bad',[None,'sampler','zoom','header','binder','prelude','reducer','build','ci'])
def test_old_sampler_zoom_and_literal_public_behavior_cannot_change(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path);source_fixture(tmp_path)
    edits={'sampler':('src/resample.cpp','return true','return false'),'zoom':('src/simd_resample_avx2.cpp','return 0','return 1'),
           'header':('src/resample_detail.hpp','bool profile_sample()','int profile_sample()'),
           'binder':('src/bind_profile_line.cpp','"profile_sample"','"changed_sample"'),
           'prelude':('python/skimagexx/_profile_line.py','old_prelude()','changed_prelude()'),
           'reducer':('python/skimagexx/_profile_line.py','old_sampler(image)','changed_sampler(image)'),
           'build':('CMakeLists.txt','old_compile_flags','changed_flags'),'ci':('.github/workflows/ci.yml','old_command','changed_command')}
    if bad:
        name,old,new=edits[bad];p=tmp_path/'snapshots/candidate'/name;p.write_text(p.read_text().replace(old,new,1))
    function=next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n,ast.FunctionDef)and n.name=='assert_existing_sources')
    namespace=dict(ast=ast,re=re,Path=Path);exec(compile(ast.Module(body=[function],type_ignores=[]),'source-preservation','exec'),namespace)
    if bad:
        with pytest.raises((AssertionError,ValueError)):namespace['assert_existing_sources']()
    else:namespace['assert_existing_sources']()


@pytest.mark.parametrize('corrupt',[False,True])
def test_normalization_preserves_originals_and_changes_only_pytest(tmp_path,monkeypatch,corrupt):
    import shutil
    monkeypatch.chdir(tmp_path);(tmp_path/'results').mkdir();originals={}
    for role in ('baseline','candidate'):
        prepared=tmp_path/'snapshots'/role/'build/prepared';(prepared/'wheels').mkdir(parents=True);(prepared/'ci-artifacts').mkdir()
        value=manifest(role=='candidate');value['lock_sha256']='original'
        for name,item in value['wheels'].items():
            data=name.encode();(prepared/'wheels'/name).write_bytes(data);item['sha256']=hashlib.sha256(data).hexdigest()
        raw=json.dumps(value).encode();(prepared/'ci-artifacts/measurement.json').write_bytes(raw);originals[role]=raw
    baseline=tmp_path/'snapshots/baseline/build/prepared';value=json.loads(originals['baseline']);filename,record=next((n,v)for n,v in value['wheels'].items()if v['name']=='pytest')
    if corrupt:(tmp_path/'snapshots/candidate/build/prepared/wheels/dep0.whl').write_bytes(b'corrupt')
    body=ast.parse(SCRIPTS[2]).body;start=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='normalized')
    code=compile(ast.Module(body=body[start:],type_ignores=[]),'normalize-test-runner','exec');namespace=dict(baseline=baseline,filename=filename,record=record,deepcopy=deepcopy,hashlib=hashlib,json=json,shutil=shutil,Path=Path)
    if corrupt:
        with pytest.raises(AssertionError):exec(code,namespace)
        return
    exec(code,namespace)
    for role in ('baseline','candidate'):
        prepared=tmp_path/'snapshots'/role/'build/prepared';target=prepared.parent/'measurement-v2';before=json.loads(originals[role]);after=json.loads((target/'measurement.json').read_bytes())
        assert(prepared/'ci-artifacts/measurement.json').read_bytes()==originals[role]
        before['wheels']={n:v for n,v in before['wheels'].items()if v['name']!='pytest'};before['wheels'][filename]=record;before['runtime']['versions']['pytest']='9.2.0';before['lock_sha256']=hashlib.sha256((target/'runtime-lock.txt').read_bytes()).hexdigest()
        assert after==before
        assert set(p.name for p in(target/'wheels').iterdir())==set(after['wheels'])
        assert(target/'pytest-lock.txt').read_text()=='pytest==9.2.0 --hash=sha256:'+record['sha256']+'\n'
