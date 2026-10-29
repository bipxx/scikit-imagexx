"""Private word-copy transport guards; no scientific imports or calls."""
import ast,hashlib,json,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'coverage/contracts/montage-word-copy-measurement.json'
WORKFLOW=ROOT/'.github/workflows/montage-sustained-same-host.yml'
BASE='4f69eba77a262056459eaab1e0a39e53610e9305'
CANDIDATE='9e1edffc12a6c290f521194fcb2386753f8845f8'


def test_frozen_gates_method_and_workloads_are_exact_old_sustained_design():
    old=json.loads((ROOT/'coverage/contracts/montage-sustained-measurement.json').read_bytes())
    new=json.loads(PROTOCOL.read_bytes())
    for key in ('case_ids','target_case','preserved_original_wins','gate','measurement_variance_rule',
                'measurement','driver_sha256','method_sha256','workloads_sha256','frozen_harness_sha256'):
        assert new[key]==old[key],key
    assert new['measurement']['raw_samples']==9*2*7*8==1008
    assert hashlib.sha256((ROOT/new['driver_path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==new['driver_sha256']
    assert new['products']['A']['revision']==BASE and new['products']['B']['revision']==CANDIDATE
    assert new['variant_limit']==1 and new['automatic_reruns']==0


def test_exact_candidate_delta_pins_and_raw_vs_canonical_contract():
    c=json.loads(PROTOCOL.read_bytes())
    names=subprocess.check_output(['git','diff','--name-only',BASE,CANDIDATE],text=True).splitlines()
    assert set(names)==set(c['source_delta_sha256']) and len(names)==7
    for name,pin in c['source_delta_sha256'].items():
        raw=subprocess.check_output(['git','show',CANDIDATE+':'+name])
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==pin
    raw=subprocess.check_output(['git','show',CANDIDATE+':'+c['acceptance_contract_path']])
    assert hashlib.sha256(raw).hexdigest()==c['acceptance_contract_sha256']
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==c['acceptance_contract_canonical_sha256']


@pytest.mark.parametrize('fault',(None,'foreign-source','changed-native','missing-new-source'))
def test_actual_inline_source_closure_rejects_any_extra_or_changed_byte(tmp_path,monkeypatch,fault):
    c=json.loads(PROTOCOL.read_bytes());text=WORKFLOW.read_text()
    begin=text.index("              protocol = json.loads(Path('controller/coverage/contracts/montage-word-copy-measurement.json')")
    end=text.index('              old_cmake, new_cmake',begin)
    code='\n'.join(line[14:]for line in text[begin:end].splitlines())
    before={'python/skimagexx/_montage.py':'unchanged','src/montage.cpp':'old',
            'include/skimagexx/montage.hpp':'old','CMakeLists.txt':'old','tests/test_montage_native.cpp':'old'}
    after=before|c['source_delta_sha256']
    for name in c['source_delta_sha256']:
        target=tmp_path/'snapshots/candidate'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','show',CANDIDATE+':'+name]))
    target=tmp_path/'controller'/PROTOCOL.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(PROTOCOL.read_bytes())
    if fault=='foreign-source':after['python/skimagexx/_montage.py']='foreign-wrapper'
    elif fault=='changed-native':(tmp_path/'snapshots/candidate/src/montage.cpp').write_bytes(b'changed')
    elif fault=='missing-new-source':after.pop('src/montage_simd.hpp')
    monkeypatch.chdir(tmp_path)
    namespace=dict(json=json,hashlib=hashlib,Path=Path,old={'source':{'files_sha256':before}},new={'source':{'files_sha256':after}})
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'actual-inline-source-closure','exec'),namespace)
    else:exec(compile(code,'actual-inline-source-closure','exec'),namespace)


def test_inline_transport_is_syntax_valid_artifact_only_and_keeps_journals():
    text=WORKFLOW.read_text();lines=text.splitlines();count=0
    for i,line in enumerate(lines):
        if line.strip().endswith("python - <<'PY'"):
            end=lines.index('          PY',i+1);ast.parse('\n'.join(row[10:]for row in lines[i+1:end]));count+=1
    assert count==4
    start=next(i for i,line in enumerate(lines)if line.strip()=="cat > results/launch.py <<'PY'")
    end=lines.index('          PY',start+1);ast.parse('\n'.join(row[10:]for row in lines[start+1:end]))
    for forbidden in ('g++','cmake --build','pip wheel','run.measure=','snapshots/parent'):
        assert forbidden not in text
    c=json.loads(PROTOCOL.read_bytes())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() in text
    assert "all(event['elapsed_ns']>=100000000 for event in accepted)" in text
    assert "if: always()" in text and "assert len(set(pids))==8" in text
    assert "all13 runtime dependency wheels exact; no normalization authorized" in text
    assert "['montage.cpp.o']" in text and "{'simd_montage_avx2.cpp.o'}" in text



def test_admitted_candidate_pins_are_concrete_and_transport_enforced():
    c=json.loads(PROTOCOL.read_bytes());b=c['products']['B'];text=WORKFLOW.read_text()
    assert b['run']==40810420960 and b['artifact']==12870128574
    assert b['sha256']=='57b947a2a0e5c840fd30d0d34ddb22cbc7488be24e82f480448793e9803cf333'
    assert b['core_sha256']=='6b1a17576756e58f76c41a8c6d9e5f2255a972b24676b3b5e85f4f14aaccc023'
    assert b['manifest_sha256']=='15fafa44a107e70185d172bd6ead08eda2cf600c20b956e867a3d843d04440e9'
    assert 'TBD_' not in PROTOCOL.read_text() and 'TBD_' not in text
    assert "cores==[product['core_sha256']]" in text
    assert "item['id']==product['artifact'] and item['digest']=='sha256:'+product['sha256']" in text


@pytest.mark.parametrize('fault',(None,'manifest','core'))
def test_actual_manifest_core_guard_rejects_unadmitted_identity(tmp_path,monkeypatch,fault):
    text=WORKFLOW.read_text();start=text.index("          if role == 'candidate':\n              product=")
    end=text.index("          source = Path('snapshots',role)",start)
    code='\n'.join(line[10:]for line in text[start:end].splitlines())
    manifest={'runtime':{'packages_sha256':{'skimagexx':{'_core.synthetic.so':'expected-core'}}}}
    raw=json.dumps(manifest).encode();prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'measurement.json').write_bytes(raw)
    protocol={'products':{'B':{'core_sha256':'expected-core','manifest_sha256':hashlib.sha256(raw).hexdigest()}}}
    target=tmp_path/'controller/coverage/contracts/montage-word-copy-measurement.json';target.parent.mkdir(parents=True);target.write_text(json.dumps(protocol))
    if fault=='manifest':(prepared/'measurement.json').write_bytes(raw+b' ')
    elif fault=='core':manifest['runtime']['packages_sha256']['skimagexx']['_core.synthetic.so']='foreign-core'
    monkeypatch.chdir(tmp_path);ns=dict(role='candidate',json=json,Path=Path,hashlib=hashlib,manifest=manifest,prepared=prepared)
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'actual-inline-manifest-guard','exec'),ns)
    else:exec(compile(code,'actual-inline-manifest-guard','exec'),ns)
