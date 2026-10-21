"""Baseline evidence must identify the actual measured source and binary."""
import hashlib
import json
import subprocess
import sys
import zipfile
import pytest
from benchmarks.provenance import source_fingerprint
from tools import preserve_baseline
from test_benchmark_provenance import make_tree


@pytest.fixture
def package(tmp_path):
    source=tmp_path/'source'
    make_tree(source)
    def git(*args):
        subprocess.run(['git','-C',str(source),*args],check=True,capture_output=True)
    git('init','-q')
    git('add','.')
    git('-c','user.name=Baseline fixture','-c','user.email=baseline@example.invalid',
        'commit','-qm','Measured product source')
    installed=tmp_path/'installed'/'skimagexx'
    installed.mkdir(parents=True)
    (installed/'__init__.py').write_bytes((source/'python/skimagexx/__init__.py').read_bytes())
    binary=b'the measured native binary'
    name='_core.fixture.pyd'
    (installed/name).write_bytes(binary)
    (installed/'__pycache__').mkdir()
    (installed/'__pycache__/old.pyc').write_bytes(b'obsolete bytecode')
    measurement=tmp_path/'measured.json'
    measurement.write_text(json.dumps(dict(metadata=dict(**source_fingerprint(source),
        current_binary_sha256={name:hashlib.sha256(binary).hexdigest()}),results=[])),encoding='utf-8')
    return source,installed,measurement,tmp_path/'preserved',name,binary


def run(monkeypatch, package, wheel=None):
    source,installed,measurement,output,_,_=package
    args=['preserve_baseline','--source-root',str(source),'--output',str(output)]
    if wheel is None:
        args+=['--installed-package',str(installed),'--measurement',str(measurement)]
    else: args+=['--wheel',str(wheel)]
    monkeypatch.setattr(sys,'argv',args)
    preserve_baseline.main()


def test_exact_installed_package_and_measurement_are_preserved(monkeypatch, package):
    run(monkeypatch,package)
    _,installed,measurement,output,name,binary=package
    manifest=json.loads((output/'baseline-source.json').read_text(encoding='utf-8'))
    assert manifest['binary_sha256']=={name:hashlib.sha256(binary).hexdigest()}
    assert manifest['measurement_sha256']==hashlib.sha256(measurement.read_bytes()).hexdigest()
    assert (output/'initial-measurement.json').read_bytes()==measurement.read_bytes()
    assert (output/'_baseline'/name).read_bytes()==binary
    assert (output/'_baseline/__init__.py').read_bytes()==(installed/'__init__.py').read_bytes()
    assert not (output/'_baseline/__pycache__').exists()


@pytest.mark.parametrize('mismatch',['source','binary','python','extra_python'])
def test_rejects_unrelated_evidence_without_creating_snapshot(monkeypatch,package,mismatch):
    _,installed,measurement,output,name,_=package
    data=json.loads(measurement.read_text(encoding='utf-8'))
    if mismatch=='source':data['metadata']['source_sha256']='0'*64
    elif mismatch=='binary':data['metadata']['current_binary_sha256'][name]='0'*64
    elif mismatch=='python':(installed/'__init__.py').write_text('VALUE = 2\n',encoding='utf-8')
    else:(installed/'stale.py').write_text('VALUE = 2\n',encoding='utf-8')
    measurement.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='differs|differ'):
        run(monkeypatch,package)
    assert not output.exists()


def test_rejects_dirty_product_source(monkeypatch,package):
    source,_,_,output,_,_=package
    (source/'src/example.cpp').write_text('int value() { return 2; }\n',encoding='utf-8')
    with pytest.raises(SystemExit) as error:run(monkeypatch,package)
    assert error.value.code==2
    assert not output.exists()


def test_uncommitted_tests_do_not_mislabel_measured_product(monkeypatch,package):
    source,_,_,output,_,_=package
    (source/'tests').mkdir()
    (source/'tests/new_contract.py').write_text('def test_contract(): pass\n',encoding='utf-8')
    run(monkeypatch,package)
    manifest=json.loads((output/'baseline-source.json').read_text(encoding='utf-8'))
    assert manifest['source_sha256']==source_fingerprint(source)['source_sha256']


def test_preserved_evidence_cannot_be_overwritten(monkeypatch,package):
    run(monkeypatch,package)
    output=package[3]
    manifest=(output/'baseline-source.json').read_bytes()
    with pytest.raises(SystemExit) as error:run(monkeypatch,package)
    assert error.value.code==2
    assert (output/'baseline-source.json').read_bytes()==manifest


@pytest.mark.parametrize('escape',[False,True])
def test_wheel_mode_and_member_boundaries(monkeypatch,package,tmp_path,escape):
    _,installed,_,output,name,binary=package
    wheel=tmp_path/'baseline.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('skimagexx/__init__.py',(installed/'__init__.py').read_bytes())
        archive.writestr('skimagexx/'+name,binary)
        if escape:archive.writestr('skimagexx/../../escape.py',b'must not be written')
    if escape:
        with pytest.raises(ValueError,match='escapes'):
            run(monkeypatch,package,wheel)
        assert not output.exists()
        assert not (tmp_path/'escape.py').exists()
    else:
        run(monkeypatch,package,wheel)
        assert (output/'_baseline'/name).read_bytes()==binary
