"""Pure manual IGG3 workflow authority, inline-source and collector guards."""
import ast,datetime,hashlib,json,os,re,textwrap
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/'.github/workflows/igg-native-three-diagnostic.yml'


def run_blocks():
    lines=WORKFLOW.read_text().splitlines();blocks=[]
    for i,line in enumerate(lines):
        if line.strip()!='run: |':continue
        indent=len(line)-len(line.lstrip());body=[]
        for next_line in lines[i+1:]:
            if next_line.strip()and len(next_line)-len(next_line.lstrip())<=indent:break
            body.append(next_line)
        blocks.append(textwrap.dedent('\n'.join(body)))
    return blocks


def programs():
    return [p for block in run_blocks()for p in re.findall(r"python3? - <<'PY'\n(.*?)\nPY",block,re.S)]


def authority():
    return dict(status='ROOT_AUTHORIZED_HOSTED_IGG_THREE_ONCE',diagnostic_source='a'*40,source='852cc1882fbd757f55bc9f92eb783282e2f5addb',case_ids=['dtype-float64','dtype-complex128','image-list'],owned_topcalls=3,max_image_core_calls=15,original_calls=0,timing=0,script_sha256=hashlib.sha256((ROOT/'tools/igg_host_native_diagnostic.py').read_bytes()).hexdigest(),contract_sha256=hashlib.sha256((ROOT/'coverage/contracts/igg-host-native-diagnostic-v1.json').read_bytes()).hexdigest(),core_sha256='ddcc9e141d3b6b9a2a5c4979ce6bc1877c7391b261cece3673bee4c7eae4732a',deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=30)).isoformat(),allow_fresh_host_helper=True)


def test_manual_default_and_exact_scope_precede_any_checkout_fetch_or_algorithm():
    text=WORKFLOW.read_text();blocks=run_blocks()
    assert 'workflow_dispatch:'in text and 'default: UNBOUND'in text
    assert 'push:'not in text and 'pull_request:'not in text and 'schedule:'not in text
    assert text.index("raw=os.environ['ROOT_AUTHORIZATION_JSON']")<text.index('uses: actions/checkout@v5')<text.index('uses: actions/setup-python@v6')<text.index(' fetch --output')
    assert 'runs-on: ubuntu-24.04'in text and "python-version: '3.11.17'"in text
    assert 'ref: ${{ github.sha }}'in text and 'persist-credentials: false'in text
    assert 'GITHUB_RUN_ATTEMPT'in blocks[0]and "=='1'"in blocks[0]
    assert all(b.startswith('set -euo pipefail')for b in blocks)
    assert 'continue-on-error'not in text and '|| true'not in text
    assert text.count('tools/igg_host_native_diagnostic.py run ')==1
    assert ' --go build/igg-host-diag/root-GO.json'in text
    assert blocks[-1].rstrip().endswith('--output build/igg-host-diag/result')


def test_all_inline_python_compiles_on_this_interpreter_without_execution():
    source=programs();assert len(source)==5 and len(run_blocks())==7
    for i,program in enumerate(source):compile(program,'<actual workflow Python '+str(i)+'>','exec')


@pytest.mark.parametrize('fault',('none','UNBOUND','expired','nonUTC','source','dispatch','case','topcalls','imagecalls','original','timing','boolbudget','script','contract','core','helper','extra','attempt'))
def test_actual_before_checkout_gate_accepts_only_one_scoped_UTC_authority(tmp_path,monkeypatch,fault):
    a=authority()
    if fault=='expired':a['deadline_utc']='2000-01-01T00:00:00+00:00'
    if fault=='nonUTC':a['deadline_utc']=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2)))+datetime.timedelta(minutes=30)).isoformat()
    for f,key in(('source','source'),('dispatch','diagnostic_source'),('script','script_sha256'),('contract','contract_sha256'),('core','core_sha256')):
        if fault==f:a[key]='changed'
    if fault=='case':a['case_ids']=['unknown']
    for f,key in(('topcalls','owned_topcalls'),('imagecalls','max_image_core_calls'),('original','original_calls'),('timing','timing')):
        if fault==f:a[key]+=1
    if fault=='boolbudget':a['original_calls']=False
    if fault=='helper':a['allow_fresh_host_helper']=False
    if fault=='extra':a['future_budget']=100
    monkeypatch.setenv('ROOT_AUTHORIZATION_JSON','UNBOUND'if fault=='UNBOUND'else json.dumps(a))
    monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setenv('GITHUB_SHA','a'*40)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2'if fault=='attempt'else'1');monkeypatch.setenv('RUNNER_TEMP',str(tmp_path))
    code=compile(programs()[0],'<actual before-checkout authority>','exec')
    if fault=='none':
        exec(code,{})
        assert json.loads((tmp_path/'igg-three-authorization.json').read_bytes())==a
    else:
        with pytest.raises(AssertionError):exec(code,{})
        assert not(tmp_path/'igg-three-authorization.json').exists()


def test_actual_gate_does_not_overwrite_an_already_retained_authorization(tmp_path,monkeypatch):
    monkeypatch.setenv('ROOT_AUTHORIZATION_JSON',json.dumps(authority()));monkeypatch.setenv('GITHUB_ACTIONS','true');monkeypatch.setenv('GITHUB_SHA','a'*40);monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1');monkeypatch.setenv('RUNNER_TEMP',str(tmp_path))
    p=tmp_path/'igg-three-authorization.json';p.write_text('old')
    with pytest.raises(FileExistsError):exec(compile(programs()[0],'<actual gate>','exec'),{})
    assert p.read_text()=='old'


def test_actual_offline_install_and_fresh_helper_binding_preserve_admitted_interfaces():
    text=WORKFLOW.read_text();source=programs()
    assert "if row['name']!='scikit-imagexx'"in source[2]
    assert "'--no-index','--no-deps','--no-compile','--target'"in source[2]
    assert "len(c['wheels'])==14 and len(selected)==13"in source[2]
    assert 'venv'not in text and 'cmake'not in text and '--build'not in text
    assert '--stager build/igg-host-diag/downloads/13565257857.zip'in text
    assert '--qualified build/igg-host-diag/downloads/13569191569.zip'in text
    assert text.index('python tests/util_views_fenv_build.py')<text.index('go.update(helper_receipt=')<text.index('tools/igg_host_native_diagnostic.py run ')
    assert "record['schema']=='util-views-fenv-build-v1'"in source[4]
    assert "record['status']=='PASS_SOURCE_BUILT_FENV32'"in source[4]
    assert "record=dict(schema='util-views-fenv-build-v1',status='PASS_SOURCE_BUILT_FENV32'"in(ROOT/'tests/util_views_fenv_build.py').read_text()
    assert "record['source_context_sha256']"in source[4]
    assert "helper_receipt_sha256=hashlib.sha256(raw).hexdigest()"in source[4]
    assert "UTIL_VIEWS_FENV_MODE='source-built-v1'"in source[4]
    assert 'unset UTIL_VIEWS_FENV_HELPER LD_PRELOAD LD_AUDIT NPY_DISABLE_CPU_FEATURES OPENBLAS_CORETYPE SKIMAGEXX_DISABLE_SIMD'in run_blocks()[-1]
    assert 'tests/test_igg_host_native_diagnostic.py tests/test_igg_native_three_workflow.py'in text


def test_actual_collector_keeps_small_receipts_helper_results_and_excludes_payload_copies():
    text=WORKFLOW.read_text();assert 'if: always()'in text
    paths=text.split('          path: |\n',1)[1].splitlines();paths=[x.strip()for x in paths if x.strip()]
    assert 'build/igg-host-diag/downloads/*.json'in paths
    assert 'build/igg-host-diag/inputs/intake.json'in paths and 'build/igg-host-diag/inputs/runtime.json'in paths
    assert 'build/igg-host-diag/fenv/'in paths and 'build/igg-host-diag/result/'in paths
    assert 'build/igg-host-diag/root-GO.json'in paths and '${{ runner.temp }}/igg-three-authorization.json'in paths
    assert all(not any(bad in p for bad in('.zip','.whl','inputs/core','inputs/wheels','runtime-prefix','SDK'))for p in paths)
    assert 'build/igg-host-diag/'not in paths and 'build/igg-host-diag/**'not in paths


def test_checkout_source_identity_and_remaining_UTC_guard_precede_fetch():
    code=programs()[1]
    assert "['git','rev-parse','HEAD']"in code
    assert "==a['diagnostic_source']==os.environ['GITHUB_SHA']"in code
    assert "['git','status','--porcelain']"in code
    assert code.index('datetime.datetime.now(')<code.index('shutil.disk_usage(')
    assert "'script_sha256'"in code and "'contract_sha256'"in code


@pytest.mark.parametrize('fault',('none','schema','status','context','permission','source','attempt','expired'))
def test_actual_GO_synthesis_uses_fresh_receipt_without_budget_or_source_override(tmp_path,monkeypatch,fault):
    monkeypatch.chdir(tmp_path)
    a=authority();out=tmp_path/'build/igg-host-diag';bundle=out/'fenv';bundle.mkdir(parents=True)
    source=tmp_path/'coverage/contracts/util-views-fenv-v1.json';source.parent.mkdir(parents=True);source.write_text('fixed-test-context')
    record=dict(schema='util-views-fenv-build-v1',status='PASS_SOURCE_BUILT_FENV32',source_context_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if fault=='schema':record['schema']='legacy'
    if fault=='status':record['status']='failed'
    if fault=='context':record['source_context_sha256']='changed'
    if fault=='permission':a['allow_fresh_host_helper']=False
    if fault=='source':a['diagnostic_source']='changed'
    if fault=='expired':a['deadline_utc']='2000-01-01T00:00:00+00:00'
    (bundle/'receipt.json').write_text(json.dumps(record))
    (out/'authorization.json').write_text(json.dumps(a))
    monkeypatch.setenv('GITHUB_SHA','a'*40);monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2'if fault=='attempt'else'1');monkeypatch.setenv('GITHUB_ENV',str(tmp_path/'job-env'))
    code=compile(programs()[4],'<actual scoped leafGO binding>','exec')
    if fault=='none':
        exec(code,{})
        go=json.loads((out/'root-GO.json').read_bytes())
        assert go['diagnostic_source']==a['diagnostic_source']and go['source']==a['source']
        assert go['owned_topcalls']==3 and go['max_image_core_calls']==15 and go['original_calls']==go['timing']==0
        assert go['helper_receipt']==str((bundle/'receipt.json').resolve())
        assert go['helper_receipt_sha256']==hashlib.sha256((bundle/'receipt.json').read_bytes()).hexdigest()
        assert 'allow_fresh_host_helper'not in go
        env=(tmp_path/'job-env').read_text();assert 'UTIL_VIEWS_FENV_MODE=source-built-v1'in env and 'UTIL_VIEWS_FENV_HELPER='not in env
    else:
        with pytest.raises(AssertionError):exec(code,{})
        assert not(out/'root-GO.json').exists()and not(tmp_path/'job-env').exists()
