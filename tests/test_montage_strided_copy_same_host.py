"""Inert fixed-artifact transport and decision tests; no scientific calls."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import zlib
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/montage-strided-copy-same-host.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREP=ast.parse(SCRIPTS[1]);PAIR=next(n for n in PREP.body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'")
CONTRACT='coverage/contracts/montage-strided-copy-candidate.json'
PIN='a677ff44b69c8c25285b5c702c698fc99f2998c688848b96e84f12724d75debc'


def test_fixed_transport_and_isolated_measurement():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    for pin in ('40740008474','40746039506','12840295821','12842995489','894edfd7519404f3c4b341aa0124c96976b83486','6025439512b27a0a594ee6df1fc27aa7a609ada5c50ab439398ea0fa4e09f951','0511f01031b29b8cfa7e53e0a548c3f9c207cad62edb73f52c209b2dc22655cc'):assert pin in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/montage.py --threads 1 --repeat 7' in timed and timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py',timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert 'no normalization authorized' in SCRIPTS[1]and 'name: montage-strided-copy-same-host-minimum'in WORKFLOW
    assert 'if: always()'in WORKFLOW
    loops=[ast.literal_eval(n.iter)for n in PREP.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and isinstance(n.iter,ast.Tuple)]
    assert sum(count*2 for name,count in loops[0]if name!='color-transforms-regression')==7778
    assert len(loops[0])==34 and ('montage',322)in loops[0]
    assert "{'passed': 1439}"in SCRIPTS[1]and '== 170'in SCRIPTS[1]and '== 80'in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',versions={'numpy':'1.24.4','pytest':'9.2.0','scikit-imagexx':'same'},dependency_libraries_sha256={'blas':'a'},packages_sha256={name:{'module':'a'}for name in('numpy','scipy','skimage')},sdk_sha256={'include/skimagexx/montage.hpp':'a','lib/libskimagexx.a':'a'})
    runtime['packages_sha256']['skimagexx']={'_montage.py':'a','_core.so':'a'}
    runtime['sdk_sha256'].update({f'include/skimagexx/p{i}.hpp':'a'for i in range(78)})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a')for i in range(13)};wheels['product']=dict(name='scikit-imagexx',sha256='a')
    files=dict.fromkeys(('src/montage.cpp','src/montage.hpp','src/bind_montage.cpp','src/color.cpp','include/skimagexx/montage.hpp','python/skimagexx/_montage.py','CMakeLists.txt','pyproject.toml','benchmarks/montage.py'),'a')
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=files));new=deepcopy(old)
    new['source']['files_sha256'].update({'src/montage.cpp':'b',CONTRACT:PIN,'tests/test_montage_native.cpp':'b','docs/progress.md':'b'})
    new['runtime']['sdk_sha256']['lib/libskimagexx.a']='b';new['runtime']['packages_sha256']['skimagexx']['_core.so']='b';new['wheels']['product']['sha256']='b'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','header','binder','color','cmake','added-production','dependency','pytest','numpy-package','runtime','SDK-header','SDK-count','Python-installed','missing-leaf','contract'])
def test_actual_paired_source_runtime_closure(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path);(tmp_path/'results').mkdir();old,new=manifests()
    mapping={'frontend':'python/skimagexx/_montage.py','header':'include/skimagexx/montage.hpp','binder':'src/bind_montage.cpp','color':'src/color.cpp','cmake':'CMakeLists.txt','added-production':'src/other.cpp'}
    if bad in mapping:new['source']['files_sha256'][mapping[bad]]='b'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='b'
    elif bad=='pytest':new['runtime']['versions']['pytest']='bad'
    elif bad=='numpy-package':new['runtime']['packages_sha256']['numpy']['module']='bad'
    elif bad=='runtime':new['runtime']['python']='bad'
    elif bad=='SDK-header':new['runtime']['sdk_sha256']['include/skimagexx/montage.hpp']='b'
    elif bad=='SDK-count':del new['runtime']['sdk_sha256']['include/skimagexx/p0.hpp']
    elif bad=='Python-installed':new['runtime']['packages_sha256']['skimagexx']['_montage.py']='b'
    elif bad=='missing-leaf':del new['source']['files_sha256']['src/montage.cpp']
    elif bad=='contract':new['source']['files_sha256'][CONTRACT]='bad'
    body=[n for n in PAIR.body[1:]if not(isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-source-closure','exec')
    namespace=dict(old=old,new=new,Path=Path,json=json)
    if bad:
        with pytest.raises(AssertionError):exec(code,namespace)
    else:exec(code,namespace)


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime
    real=datetime.datetime
    class FixedDatetime(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,21 if scenario=='expired'else 19,tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime,'datetime',FixedDatetime);monkeypatch.chdir(tmp_path)
    original=json.loads((ROOT/'benchmarks/montage-minimum-initial.json').read_bytes())['raw_benchmark']
    contract=tmp_path/'snapshots/baseline/coverage/contracts/montage-benchmark.json';contract.parent.mkdir(parents=True);contract.write_bytes((ROOT/'coverage/contracts/montage-benchmark.json').read_bytes())
    identities={};fingerprints={}
    for role in('baseline','candidate'):
        source=tmp_path/'snapshots'/role;prepared=source/'build/prepared/ci-artifacts';prepared.mkdir(parents=True);(source/'benchmarks').mkdir()
        value=dict(source=dict(revision=role[0]*40),runtime=dict(python=original['metadata']['python'],packages_sha256={'skimagexx':{'_core.so':role*8}}));raw=json.dumps(value).encode();(prepared/'measurement.json').write_bytes(raw)
        fingerprints[role]=dict(source_sha256=role,source_hash_schema='fixture',source_hash_scope='fixture');(source/'benchmarks/provenance.py').write_text('def source_fingerprint(root):return '+repr(fingerprints[role])+'\n')
        identities[role]=dict(status='verify',revision=value['source']['revision'],wheel_count=14,artifact_sha256=hashlib.sha256(raw).hexdigest())
    for name in('1-baseline','2-candidate','3-candidate','4-baseline'):
        role=name.split('-')[1];candidate=role=='candidate';folder=tmp_path/'results'/name;folder.mkdir(parents=True);report=deepcopy(original)
        report['metadata'].update(fingerprints[role]);report['metadata']['source_before']=fingerprints[role];report['metadata']['core_sha256']=role*8
        for row in report['results']:
            cpp=2.0 if candidate else 10.0;ref=10.0;target=row['case_id']=='gray-strided-reverse-f32';mean=row['case_id']=='gray-mean-f64'
            if scenario=='target-repeat'and name=='3-candidate'and target:cpp=10/2.99
            if scenario=='target-original'and target:ref=1.0
            if scenario=='preserved-loss'and row['case_id']=='gray-small-u8':ref=1.0
            if scenario=='remaining-loss'and mean:ref=1.0
            if scenario=='other-direct-floor'and candidate and mean:cpp=10/.94
            if scenario=='other-normalized-floor'and candidate and mean:ref=cpp*.94
            if scenario=='global-direct'and candidate:cpp=10/(3.1 if target else .8)
            if scenario=='global-normalized'and candidate:ref=cpp*(3.1 if target else .8)
            if scenario=='geometric-repeat':cpp=3.0 if candidate else(8.0 if name=='1-baseline'else 12.5)
            row.update(reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp,samples_ms=[[ref]*7,[cpp]*7])
        if name=='3-candidate':
            row=report['results'][0]
            if scenario=='output-drift':row['output_sha256']='f'*64
            elif scenario=='layout-drift':row['output_metadata']['strides'][0]+=1
            elif scenario=='wrong-C-entry':row['actual_C_returns']=['fake']
            elif scenario=='wrong-owner':row['native_entry_module']='fake'
            elif scenario=='wrong-core':report['metadata']['core_sha256']='f'*64
            elif scenario=='incomplete-samples':row['samples_ms'][0].pop()
        (folder/'montage.json').write_text(json.dumps(report))
        for when in('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-comparison-decision','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('remaining-loss','KEEP'),('target-repeat','DISCARD'),('target-original','DISCARD'),('preserved-loss','DISCARD'),('other-direct-floor','DISCARD'),('other-normalized-floor','DISCARD'),('global-direct','DISCARD'),('global-normalized','DISCARD'),('expired','DISCARD')])
def test_actual_frozen_decision(tmp_path,monkeypatch,scenario,expected):
    result=run_decision(tmp_path,monkeypatch,scenario)
    assert result['status']==expected and result['raw_samples']==504 and len(result['rows'])==9
    assert len(result['initial_original_losses'])==3
    if scenario=='remaining-loss':assert result['balanced_original_losses']==['gray-mean-f64']


@pytest.mark.parametrize('scenario',['output-drift','layout-drift','wrong-C-entry','wrong-owner','wrong-core','incomplete-samples'])
def test_actual_raw_report_refusals(tmp_path,monkeypatch,scenario):
    with pytest.raises(AssertionError):run_decision(tmp_path,monkeypatch,scenario)


@pytest.mark.parametrize('mutate',[False,True])
def test_actual_candidate_contract_bytes(tmp_path,monkeypatch,mutate):
    monkeypatch.chdir(tmp_path);path=tmp_path/'snapshots/candidate'/CONTRACT;path.parent.mkdir(parents=True)
    assert hashlib.sha256(CONTRACT_RAW).hexdigest()==PIN
    assert CONTRACT_RAW.replace(b'\r\n',b'\n')==CONTRACT_RAW
    path.write_bytes(CONTRACT_RAW+(b'changed'if mutate else b''))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-byte-pin','exec')
    if mutate:
        with pytest.raises(AssertionError):exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))
    else:exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))

# Exact frozen candidate contract, inert data; no scientific calls.
CONTRACT_RAW=zlib.decompress(base64.b64decode('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'))
