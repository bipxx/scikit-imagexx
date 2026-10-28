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
WORKFLOW=(ROOT/'.github/workflows/montage-wrapper-simd-same-host.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREP=ast.parse(SCRIPTS[1]);PAIR=next(n for n in PREP.body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'"and any(isinstance(item,ast.Assign)and isinstance(item.targets[0],ast.Tuple)for item in n.body))
CONTRACT='coverage/contracts/montage-wrapper-simd-candidate.json'
PIN='f39a341c447b872e1d2ec9b77d50fc5fb6a79d8d374ce5cd6574f4bfc975d3df'
CANONICAL_PIN='f39a341c447b872e1d2ec9b77d50fc5fb6a79d8d374ce5cd6574f4bfc975d3df'


def test_fixed_transport_and_isolated_measurement():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    for pin in ('40758210302','40766081804','12846458475','12849749460','61b4f79821b0c7fd0d224e2a12d4263d14308cb7','d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89','c4420f7d4e95d4cb85ffcbd66e24ac7ee8c0f5684b8619b061663a2b98697989'):assert pin in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/montage.py --threads 1 --repeat 7' in timed and timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py',timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert 'no normalization authorized' in SCRIPTS[1]and 'name: montage-wrapper-simd-same-host-minimum'in WORKFLOW
    assert 'if: always()'in WORKFLOW
    loops=[ast.literal_eval(n.iter)for n in PREP.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and isinstance(n.iter,ast.Tuple)]
    assert sum(count*2 for name,count in loops[0]if name!='color-transforms-regression')==8238
    assert len(loops[0])==35 and ('montage',322)in loops[0]
    assert "{'passed': 1443}"in SCRIPTS[1]and '== 172'in SCRIPTS[1]and '== 81'in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',versions={'numpy':'1.24.4','pytest':'9.2.0','scikit-imagexx':'same'},dependency_libraries_sha256={'blas':'a'},packages_sha256={name:{'module':'a'}for name in('numpy','scipy','skimage')},sdk_sha256={'include/skimagexx/montage.hpp':'a','lib/libskimagexx.a':'a'})
    runtime['packages_sha256']['skimagexx']={'_montage.py':'a','_core.so':'a'}
    runtime['sdk_sha256'].update({f'include/skimagexx/p{i}.hpp':'a'for i in range(79)})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a')for i in range(13)};wheels['product']=dict(name='scikit-imagexx',sha256='a')
    files=dict.fromkeys(('src/montage.cpp','src/montage.hpp','src/bind_montage.cpp','src/color.cpp','include/skimagexx/montage.hpp','python/skimagexx/_montage.py','CMakeLists.txt','pyproject.toml','benchmarks/montage.py'),'a')
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=files));new=deepcopy(old)
    old['source']['files_sha256']['tests/test_montage_native.cpp']='a'
    new=deepcopy(old)
    new['source']['files_sha256'].update({'src/montage.cpp':'b','CMakeLists.txt':'b',CONTRACT:CANONICAL_PIN,'tests/test_montage_native.cpp':'b','src/montage_simd.hpp':'b','src/simd_montage_avx2.cpp':'b','src/montage_wrapper.cpp':'b'})
    new['runtime']['sdk_sha256']['lib/libskimagexx.a']='b';new['runtime']['packages_sha256']['skimagexx']['_core.so']='b';new['wheels']['product']['sha256']='b'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','header','binder','color','cmake-extra','added-production','docs-extra','removed-source','missing-new-simd','SDK-other-object','SDK-missing-new-object','missing-new-wrapper','generic-body-extra','generic-register','dependency','pytest','numpy-package','runtime','SDK-header','SDK-count','Python-installed','missing-leaf','contract'])
def test_actual_paired_source_runtime_closure(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path);(tmp_path/'results').mkdir();old,new=manifests()
    mapping={'frontend':'python/skimagexx/_montage.py','header':'include/skimagexx/montage.hpp','binder':'src/bind_montage.cpp','color':'src/color.cpp','docs-extra':'docs/progress.md','added-production':'src/other.cpp'}
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
    for role in ('baseline','candidate'):
        folder=tmp_path/'snapshots'/role;folder.mkdir(parents=True)
        cmake=b'baseline\n'
        if role=='candidate':cmake+=b'target_sources(simd_montage_avx2.cpp)\nmsvc(simd_montage_avx2.cpp)\ngcc(simd_montage_avx2.cpp)\nwrapper(montage_wrapper.cpp)\n'
        if bad=='cmake-extra'and role=='candidate':cmake+=b'extra production change\n'
        (folder/'CMakeLists.txt').write_bytes(cmake)
        source=folder/'src/montage.cpp';source.parent.mkdir()
        generic=b'void montage_into(){original_body();}\n'
        if role=='candidate':generic=generic.replace(b'void montage_into(',b'void montage_generic_into(')
        if bad=='generic-body-extra'and role=='candidate':generic+=b'extra body edit\n'
        source.write_bytes(generic)
        code='file-header\nDisassembly of section .text:\n0000000000000000 <montage_into-copy>:\n0: mov %rax,%rbx\n0000000000000010 <montage_into-fill>:\n10: ret\n0000000000000020 <montage_into>:\n20: ret\n'
        if role=='candidate':code=code.replace('montage_into','montage_generic_into')
        if bad=='generic-register'and role=='candidate':code=code.replace('%rax','%rcx')
        (tmp_path/'results'/(role+'-generic-disassembly.txt')).write_text(code)
        objects={'image.cpp.o':'a','montage.cpp.o':'a'}
        if role=='candidate':objects.update({'montage.cpp.o':'b','simd_montage_avx2.cpp.o':'b','montage_wrapper.cpp.o':'b'})
        if bad=='SDK-other-object'and role=='candidate':objects['image.cpp.o']='b'
        if bad=='SDK-missing-new-object'and role=='candidate':objects.pop('simd_montage_avx2.cpp.o')
        (tmp_path/'results'/(role+'-sdk-objects.json')).write_text(json.dumps(objects))
    if bad=='removed-source':del new['source']['files_sha256']['src/color.cpp']
    if bad=='missing-new-wrapper':del new['source']['files_sha256']['src/montage_wrapper.cpp']
    if bad=='missing-new-simd':del new['source']['files_sha256']['src/simd_montage_avx2.cpp']
    body=[n for n in PAIR.body[1:]if not(isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-source-closure','exec')
    namespace=dict(old=old,new=new,Path=Path,json=json,re=re)
    if bad:
        with pytest.raises(AssertionError):exec(code,namespace)
    else:exec(code,namespace)


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime
    real=datetime.datetime
    class FixedDatetime(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,23 if scenario=='expired'else 21,tzinfo=datetime.timezone.utc)
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
            if scenario=='target-repeat'and name=='3-candidate'and target:cpp=10/1.49
            if scenario=='target-original'and target:ref=1.0
            if scenario=='preserved-loss'and row['case_id']=='gray-small-u8':ref=1.0
            if scenario=='remaining-loss'and mean:ref=1.0
            if scenario=='other-direct-floor'and candidate and mean:cpp=10/.96
            if scenario=='other-normalized-floor'and candidate and mean:ref=cpp*.96
            if scenario=='global-direct'and candidate:cpp=10/(3.1 if target else .8)
            if scenario=='global-normalized'and candidate:ref=cpp*(3.1 if target else .8)
            if scenario=='geometric-repeat':cpp=3.0 if candidate else(8.0 if name=='1-baseline'else 12.5)
            if scenario=='rgb-direct-repeat'and row['case_id']=='rgb-first-f32':
                ref=40.0
                if name=='3-candidate':cpp=10/.96;ref=cpp*8
            if scenario=='rgb-normalized-repeat'and row['case_id']=='rgb-first-f32':
                ref=40.0
                if name=='3-candidate':ref=cpp*3.84
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


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('remaining-loss','KEEP'),('target-repeat','DISCARD'),('target-original','DISCARD'),('preserved-loss','DISCARD'),('other-direct-floor','DISCARD'),('other-normalized-floor','DISCARD'),('global-direct','DISCARD'),('global-normalized','DISCARD'),('expired','DISCARD'),('rgb-direct-repeat','DISCARD'),('rgb-normalized-repeat','DISCARD')])
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
    assert hashlib.sha256(CONTRACT_RAW.replace(b'\r\n',b'\n')).hexdigest()==CANONICAL_PIN
    assert CONTRACT_RAW.count(b'\r\n')==0
    path.write_bytes(CONTRACT_RAW+(b'changed'if mutate else b''))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-byte-pin','exec')
    if mutate:
        with pytest.raises(AssertionError):exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))
    else:exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))

# Exact frozen LF contract, embedded inert data only.
CONTRACT_RAW=zlib.decompress(base64.b64decode('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'))


def test_exact_committed_contract_normalization_and_fixed_clock():
    contract=json.loads(CONTRACT_RAW)
    assert contract['source_base']=='390dff892ad13c1f9b8cee0e76dce9a016f14c01'
    assert contract['deadline_utc']=='2026-10-28T22:35:00.116019+00:00'
    assert contract['variant_limit']==1 and contract['raw_samples']==504
    assert CANONICAL_PIN in SCRIPTS[1]and PIN in SCRIPTS[1]
    assert "== 27"in SCRIPTS[0]and "step['conclusion'] == 'skipped'"in SCRIPTS[0]
    assert "('rolling-ball-kernels', 230)"in SCRIPTS[1]
    assert "simd_montage_avx2.cpp.o"in SCRIPTS[1]
    assert "93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd"in SCRIPTS[2]
    assert "434d7540206cd2adcf3253bc097e36db7131a15b0c6054fa06b2c97fd5da3d68"in SCRIPTS[2]


def test_actual_sdk_archive_member_selector_uses_root_path():
    from types import SimpleNamespace
    node=next(n for n in ast.walk(PREP)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='static_names'for t in n.targets))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-sdk-archive-selector','exec')
    namespace={'product':SimpleNamespace(namelist=lambda:['include/skimagexx/montage.hpp','lib/libskimagexx.a','foreign/lib/libskimagexx.a'])}
    exec(code,namespace)
    assert namespace['static_names']==['lib/libskimagexx.a']


def test_strict_genuine_generic_validator_allows_only_symbol_rename():
    node=next(n for n in ast.walk(PAIR)if isinstance(n,ast.FunctionDef)and n.name=='validate_generic_code')
    namespace={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'strict-genuine-validator','exec'),namespace)
    before='baseline.o\nDisassembly of section .text:\n0000000000000000 <montage_into-copy>:\n0: mov 0xf8(%rsp),%rax\n0000000000000010 <montage_into-fill>:\n10: ret\n0000000000000020 <montage_into>:\n20: ret\n'
    candidate=before.replace('baseline.o','candidate.o').replace('montage_into','montage_generic_into')
    assert namespace['validate_generic_code'](before,candidate)['status']=='PASS_STRICT_GENUINE_GENERIC_CODEGEN'
    for wrong in (candidate.replace('0xf8','0x100'),candidate.replace('%rax','%rbx'),candidate.replace('mov','add'),candidate.replace('20: ret','20: jmp 0'),candidate+'extra instruction\n'):
        with pytest.raises(AssertionError):namespace['validate_generic_code'](before,wrong)


def test_new_rgb_repeat_controls_are_separate_from_balanced_floor(tmp_path,monkeypatch):
    result=run_decision(tmp_path,monkeypatch,'rgb-normalized-repeat')
    row=next(r for r in result['rows']if r['case_id']=='rgb-first-f32')
    assert row['direct_gain']>.97 and row['normalized_gain']>.97 and row['original_repeat']
    assert result['rgb_first_each_pass_direct_floor']and not result['rgb_first_each_pass_normalized_floor']
