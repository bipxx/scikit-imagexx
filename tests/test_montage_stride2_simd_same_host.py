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
WORKFLOW=(ROOT/'.github/workflows/montage-stride2-simd-same-host.yml').read_text()
SCRIPTS=[textwrap.dedent(s)for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREP=ast.parse(SCRIPTS[1]);PAIR=next(n for n in PREP.body if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'candidate'")
CONTRACT='coverage/contracts/montage-stride2-simd-candidate.json'
PIN='9e5a4b0727ba0629742431234f1ff385c9d3c1a648238bf398db0035439854d9'
CANONICAL_PIN='4ca2b78afa6fb886c49b1d15bd25baafc855c73ccbf1d6ddcd8ef13d318b8efe'


def test_fixed_transport_and_isolated_measurement():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    for pin in ('40758210302','40758988448','12846458475','12848346695','e66770d55ea7c342ef763b9c842bb4fa1939f6ad','d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89','0ecf50ec8eb8bfdcc60b914e9ce65b0b8888d6e6c251be71566e5a299b0a6126'):assert pin in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/montage.py --threads 1 --repeat 7' in timed and timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py',timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert 'no normalization authorized' in SCRIPTS[1]and 'name: montage-stride2-simd-same-host-minimum'in WORKFLOW
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
    new['source']['files_sha256'].update({'src/montage.cpp':'b','CMakeLists.txt':'b',CONTRACT:CANONICAL_PIN,'tests/test_montage_native.cpp':'b','src/montage_simd.hpp':'b','src/simd_montage_avx2.cpp':'b'})
    new['runtime']['sdk_sha256']['lib/libskimagexx.a']='b';new['runtime']['packages_sha256']['skimagexx']['_core.so']='b';new['wheels']['product']['sha256']='b'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','header','binder','color','cmake-extra','added-production','docs-extra','removed-source','missing-new-simd','SDK-other-object','SDK-missing-new-object','dependency','pytest','numpy-package','runtime','SDK-header','SDK-count','Python-installed','missing-leaf','contract'])
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
        if role=='candidate':cmake+=b'target_sources(simd_montage_avx2.cpp)\nmsvc(simd_montage_avx2.cpp)\ngcc(simd_montage_avx2.cpp)\n'
        if bad=='cmake-extra'and role=='candidate':cmake+=b'extra production change\n'
        (folder/'CMakeLists.txt').write_bytes(cmake)
        objects={'image.cpp.o':'a','montage.cpp.o':'a'}
        if role=='candidate':objects.update({'montage.cpp.o':'b','simd_montage_avx2.cpp.o':'b'})
        if bad=='SDK-other-object'and role=='candidate':objects['image.cpp.o']='b'
        if bad=='SDK-missing-new-object'and role=='candidate':objects.pop('simd_montage_avx2.cpp.o')
        (tmp_path/'results'/(role+'-sdk-objects.json')).write_text(json.dumps(objects))
    if bad=='removed-source':del new['source']['files_sha256']['src/color.cpp']
    if bad=='missing-new-simd':del new['source']['files_sha256']['src/simd_montage_avx2.cpp']
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
        def now(cls,tz=None):return real(2026,10,28,22 if scenario=='expired'else 20,tzinfo=datetime.timezone.utc)
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
    assert hashlib.sha256(CONTRACT_RAW.replace(b'\r\n',b'\n')).hexdigest()==CANONICAL_PIN
    assert CONTRACT_RAW.count(b'\r\n')==68
    path.write_bytes(CONTRACT_RAW+(b'changed'if mutate else b''))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert)and 'Path('in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-byte-pin','exec')
    if mutate:
        with pytest.raises(AssertionError):exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))
    else:exec(code,dict(Path=Path,hashlib=hashlib,contract=CONTRACT))

# Exact committed contract bytes, including CRLF; canonical source hashes use LF.
CONTRACT_RAW=zlib.decompress(base64.b64decode('eJy9VtuO2zYQfQ+QfyD2tStbku+7TYGklyBA0xZJ0ZeiEGhyJDGhRZek7N0E+fee0cXxbrbpDejLejWcGc45c+P7x4+EuAiqpp28uBIXO9dEWVESojea8iSYnU6UbLTRMlJyyC4ue4soYxvYYu9d2JOK5kDJlkrnYexaryghbWJIlNvtjZXRuCZx/syXa+O+jWFwWHqid1S0UbHTPM2XSZYm+frnbHO1mF9li8lytkrz+RdpepWmg5Emqa1pHjLLsz8327a6oljUCJMh5L30IL2RTSys2ZkIcTYg7cAUWxmIr5htUl2W600udTZTWbnZrhVRSqulVrSRabYss7lKR57YrIuwd/NfPIBo3SryhW8b+Jmnq8U6z9JZmt9TlHpnQgDfXXoIfDfVtdBONC4K05TkRWiVojByL611R9JDjEVpLDEvv/IhM+DVdKiLidrve6NBzvVRDIeFPNzkn2iMh6w5qc8OI4UYpvz35KCRXEZ3XHz9Ur6l7w1UJ/EmXrD0tz5qBSMvVSz2MtYMVbkDebiZjidh+vlynrwJIKn3VkvfgJEi1DJfLAtbssfNTKZaLVW61nI9l6t0sV6nelWmGa0WmZyvyiyjLF/laylnOl+rbJNluZYLnc+WW60H30fn31on9T3v89lcrxbzNE+XSqMaVDnLF7OtSjcrmi31dpXNMpkttqlapot5KdPlNlebVakXGrct14N3hbwXRp9nrPLyNgk75DVp1ycqO6lpuCEtoQHvH2lqAiUI4q7YSg8G22x5V7wj2STlcn6S+mqb7KXWpM89s7Q0PsRPHfdJ0Ykn5O2hmzFMkMf+lrPERw4oFmroyM/7wnyiQP6A8nbeVKaRtjia5n9k6+/wco6u9phq4eMA8rQnyQNpdQf+rlPZ9DLnNfkzSOMsOF13qvnPSE5GZ9F4eSyCZAr4ukU678UV212J94Nl13jWFtp4bILi+cui6iboJF1c3lNpnAfT5h3y8ZDaAI6kqotThOhwdOYhYAhb2SiYnmbdcOPg51+7OYvqH7g6FdQIY7TplF2sMazJVHX8eOMpXqYvnWxWf21yJ7YHzR4u8QdDvh8ppwSN00QTb6Hh+QcWxWmzevq9RcwaZtG3xGYfxq1R4bpY77gJf2zsrUCxiHmyvY0kAnaOpUTVsmnIir0Lhke76Pt0Lbw7Cm4tTGksKvg+OtEicFM1pAVmZLI1UXRDc9QTaO1GYJZCFJ3oWBI86k3VujaI/j3RK0zEL6DZebG1Tr0VA4ggvnqywcdOmgbxwX1llER05oZsuBQBH9KLKI0VXz5ZT8SrFt53JF6/ePmNQKVUZmssKBK1w0ZCoA4ZuhYVNeSNEgPaKRaonTJC3kwBuPigIj0RP1DVrTjGPzU7rKaBkTtaL5pIFba026NnmB+AA78wB+wbjETBG1t1R5NxCzjPlcU7jPPx1NpVJkrv3oGzYduXrR1IAtbTdcJZLfrFK1BIpbkRKAAk8EbsHAK77vjAKgVnqEMR9rIJ01OupsPE1Qwp9JjC9Luf+jxMSyurMOUFCKXscj4R396AOmZ/326tUdO9NweUp+CREobkACUSPkLjHRD22AUM7Dk1LepSHGtnOWB+E0UEObayGJ9JYnj5iNNzqMN16gcGi4KnjlohywgT7/BCAiBDx2uQR6EWJjg8XgEPU5CS2oUonj579hS1wbragI6o6s7JNR5YgvNiBaoGV4fptjVWc/YGiWAHzAHgZTOhiYOkRt0CD6EGe+q7Z5hQ1oXW0ynB+ESH4wUOFl5z6AzC0xtkHeHRDYoFtdowAGSM2+i88BzEbSAhcYfA7mbX52ygV2DXcl+MN4IfkA4B3/iqc8qFtxG8eVFC+B8LQQzrofvuiokxNHQEFQF6KJAKTHICoNMldkybGCaPgdY4cK77zOKRxt7A6iWzGvgVh3vbAHKVlWY3wZb68PjRHxH0E6Y='))


def test_exact_committed_contract_normalization_and_fixed_clock():
    contract=json.loads(CONTRACT_RAW)
    assert contract['source_base']=='390dff892ad13c1f9b8cee0e76dce9a016f14c01'
    assert contract['deadline_utc']=='2026-10-28T21:54:15.637024+00:00'
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
