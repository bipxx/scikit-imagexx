"""Actual admitted metadata binding only; no host helper or target execution."""
import ast,copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH='coverage/contracts/next-metrics-measurement-v3.json'
DATA=json.loads((ROOT/PATH).read_bytes())
CONTEXT=json.loads((ROOT/'coverage/contracts/next-metrics-controller-binding-v3.json').read_bytes())
spec=importlib.util.spec_from_file_location('bound_controller_v3',ROOT/'tools/next_metrics_first_cohort_probe.py')
PROBE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROBE)
H=lambda b:hashlib.sha256(b).hexdigest()


def test_actual_bound_root_source_runtime_core_SDK_metadata():
    path,m=PROBE.read_manifest(ROOT)
    assert path==ROOT/PATH and m==DATA
    assert m['producer']==CONTEXT['producer']==dict(run='42262940227',revision='a6bae9ac86dbba362537758784edf06c973e9c53',artifact='13528655843',sha256='73276b66ee5c5537628c26ce1d19def8d8e41410d388667808a5e22a198c383e')
    assert m['runtime_sha256']==CONTEXT['runtime_sha256']=='6d41b7036debf39346a0439a28c81df242aa489299663905a3ca57a1ec5e1e5f'
    assert len(CONTEXT['SDK_source_map'])==m['SDK_members']==102
    assert m['benchmark_fingerprint']==CONTEXT['benchmark_fingerprint']
    for n,pin in CONTEXT['root_receipts'].items():assert H((ROOT/n).read_bytes())==pin
    assert len(m['benchmark_source_sha256'])==len(m['first_cohort']['source_files_raw_sha256'])==67
    assert all(m['prepared_source']['files_sha256'][n]==pin for n,pin in m['benchmark_source_sha256'].items())
    assert len(m['helper']['source_sha256'])==7 and m['helper']['prepare_once_on_actual_host']is True
    assert m['helper']['receipt_sha256']is m['helper']['library_sha256']is m['helper']['root_GO']is None


def test_frozen_v2_cases_methods_native_C1_and_literal_successor():
    old=json.loads((ROOT/'coverage/contracts/next-metrics-measurement-v2.json').read_bytes())
    assert old['status']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'
    assert H((ROOT/'coverage/contracts/next-metrics-measurement-v2.json').read_bytes())==CONTEXT['predecessor_manifest_raw_sha256']
    for key in ('families','execution_scope','benchmark_source_sha256','helper'):assert DATA[key]==old[key],key
    for key in ('routes','source_files_raw_sha256'):assert DATA['first_cohort'][key]==old['first_cohort'][key]
    text=(ROOT/'tools/next_metrics_first_cohort_probe.py').read_text();tree=ast.parse(text)
    functions={n.name:ast.get_source_segment(text,n)for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name,pin in CONTEXT['predecessor_functions_canonical_sha256'].items():
        raw=functions[name]
        if name in CONTEXT['literal_path_successors']:raw=raw.replace('next-metrics-measurement-v3.json','next-metrics-measurement-v2.json')
        if name=='read_manifest':raw=raw.replace('    validate_bound_manifest(controller,manifest)\n','')
        assert H(raw.encode())==pin,name
    # Including every original function covers inner native ownership/FP,
    # mandatory entry stop, C1 poststop and postbatch verifier, timing and loss
    # validators. New logic is solely the small root metadata trust boundary.
    assert set(functions)==set(CONTEXT['predecessor_functions_canonical_sha256'])|{'validate_bound_manifest'}
    workflow=(ROOT/'.github/workflows/next-metrics-initial.yml').read_text()
    assert workflow.count('coverage/contracts/next-metrics-measurement-v3.json')==2
    assert 'next-metrics-measurement-v2.json'not in workflow
    assert workflow.count("probe['prepare_host_helper']")==1
    assert workflow.index("probe['prepare_host_helper']")<workflow.index("probe['run_cohorts']")<workflow.index("probe['run_measurements']")


@pytest.mark.parametrize('tamper',('none','source','prepared','run','artifact','archive','admission-digest','stager-digest','core','core-member','SDK','fingerprint','source-pin','source-count','probe','routes','case-budget','sample-budget','worker','losses','helper-receipt','helper-library','helper-GO','runtime-format'))
def test_actual_binding_metadata_tamper_before_any_body(tmp_path,monkeypatch,tamper):
    m=copy.deepcopy(DATA);first=m['first_cohort']
    for n in (first['root_admission_path'],first['root_stager_path'],first['probe_path']):
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    if tamper=='source':m['producer']['revision']='0'*40
    elif tamper=='prepared':m['prepared_source']['revision']='0'*40
    elif tamper=='run':m['producer']['run']='1'
    elif tamper=='artifact':m['producer']['artifact']='1'
    elif tamper=='archive':m['producer']['sha256']='0'*64
    elif tamper=='admission-digest':first['root_admission_sha256']='0'*64
    elif tamper=='stager-digest':first['root_stager_sha256']='0'*64
    elif tamper=='core':m['core_sha256']='0'*64
    elif tamper=='core-member':first['core_member']='wrong.so'
    elif tamper=='SDK':m['SDK_members']=97
    elif tamper=='fingerprint':m['benchmark_fingerprint']['source_sha256']='0'*64
    elif tamper=='source-pin':m['prepared_source']['files_sha256']['CMakeLists.txt']='0'*64
    elif tamper=='source-count':m['prepared_source']['files_sha256'].pop('AGENTS.md')
    elif tamper=='probe':first['probe_sha256']='0'*64
    elif tamper=='routes':first['routes'].pop('blocks')
    elif tamper=='case-budget':m['execution_scope']['canonical_cases']=55
    elif tamper=='sample-budget':m['execution_scope']['canonical_raw_samples']=1
    elif tamper=='worker':m['execution_scope']['workers']=4
    elif tamper=='losses':m['execution_scope']['retain_all_losses']=False
    elif tamper=='helper-receipt':m['helper']['receipt_sha256']='0'*64
    elif tamper=='helper-library':m['helper']['library_sha256']='0'*64
    elif tamper=='helper-GO':m['helper']['root_GO']={}
    elif tamper=='runtime-format':m['runtime_sha256']='unknown'
    p=tmp_path/PATH;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m))
    calls=[]
    def forbidden(*a,**kw):calls.append(a);raise AssertionError('BODY_OR_PROCESS_REACHED')
    monkeypatch.setattr(PROBE.subprocess,'run',forbidden)
    if tamper=='none':assert PROBE.read_manifest(tmp_path)[1]==m
    else:
        with pytest.raises(AssertionError):PROBE.read_manifest(tmp_path)
    assert not calls
