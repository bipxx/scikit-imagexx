"""TV harness metadata guards; synthetic modules never qualify native code."""
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import skimagexx as xx
from benchmarks import tv_chambolle as h


def manifest():return json.loads(h.MANIFEST.read_text())


def test_fixed400_ids_and330_source_observations_are_not_trimmed():
    m=manifest();cases=h.corpus();source=json.loads(h.SOURCE_CONTRACT.read_text())
    assert len(cases)==400 and sum(c.timed for c in cases)==70
    assert [c.fixture for c in cases if c.source_id]==source['cases']
    assert all(not c.timed for c in cases if c.source_id)
    assert [h.inventory(c) for c in cases]==m['case_inventory']
    assert h.digest(m['case_inventory'])==m['case_contract_sha256']
    assert h.text_sha(h.SOURCE_CONTRACT)==m['source_contract_sha256']
    assert all(h.text_sha(h.ROOT/p)==sha for p,sha in m['fixture_source_sha256'].items())
    assert m['accepted_public'] is None and all(v is None for v in m['accepted_products'].values())


@pytest.mark.parametrize('case',[c for c in h.corpus() if c.timed],ids=lambda c:c.id)
def test_timing_fixture_storage_data_hash_and_workspace_certificate(case):
    passed,image,owner,options,events=h.fixture(case)
    other=h.fixture(case)[1]
    assert h.contracts().array(image)==h.contracts().array(other)
    assert not events and image.size<=257*259
    spatial=image.ndim-(1 if 'channel_axis' in options else 0)
    assert image.size*8*(4*spatial+7)<=h.LIMIT
    assert np.isfinite(image).all()
    assert case.source_id is None and case.fixture['cohort'] in ('tiny','small','raster','volume','channels','layout')


@pytest.mark.parametrize('mode',['unregistered','stage-only','missing-products'])
def test_native_pending_rejects_before_fixtures_capsules_or_timers(monkeypatch,mode):
    m=manifest()
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture import reached before admission'))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    coverage=__import__('skimagexx.coverage',fromlist=['native_functions'])
    monkeypatch.setattr(coverage,'native_functions',lambda:() if mode=='unregistered' else (h.API,))
    if mode=='missing-products':m['accepted_public']={'stage':'not sufficient'}
    with pytest.raises(RuntimeError,match='unregistered|public TV admission pending|all3'):h.public_admission(m)


@pytest.mark.parametrize('argv',[
    ['--reference-only'],['--reference-only','--check-only','--capsule','missing'],
    ['--check-only','--threads','4'],['--check-only','--repeat','2'],
    ['--check-only','--threads','1','1'],['--prepare-capsule','--check-only','--threads','1'],
    [],['--threads','1','4','--output','ignored','--capsule','missing'],
])
def test_bad_cli_modes_reject_without_numerical_calls(monkeypatch,argv):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture creation reached'))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('timer reached'))
    with pytest.raises(SystemExit):h.main(argv)


def test_reference_check_never_times_and_restores_fallback_workers(monkeypatch,tmp_path):
    case=h.corpus()[0];m=manifest();m['case_inventory']=[h.inventory(case)];m['case_contract_sha256']=h.digest(m['case_inventory']);m['reference_profiles']={}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m));monkeypatch.setattr(h,'MANIFEST',path);monkeypatch.setattr(h,'corpus',lambda:(case,))
    monkeypatch.setattr(h,'measure',lambda *_:pytest.fail('check-only timer reached'))
    monkeypatch.setattr(h,'seal',lambda *_:pytest.fail('reference capsule reached'));monkeypatch.setattr(h,'verify',lambda *_:pytest.fail('reference capsule reached'))
    workers=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_fallback_enabled(fallback)
    output=tmp_path/'report.json'
    assert h.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text());assert report['status']=='complete-reference-check' and report['metadata']['native_checks']==0 and not report['results']
    assert xx.get_num_threads()==workers
    assert xx.set_fallback_enabled(fallback)==fallback


@pytest.mark.parametrize('forward',['public','helper'])
def test_exact_original_forwarding_is_caught_and_prior_profile_restored(forward):
    case=h.corpus()[0];module=__import__('skimage.restoration._denoise',fromlist=['_denoise_tv_chambolle_nd'])
    def existing(frame,event,arg):pass
    before=sys.getprofile();sys.setprofile(existing)
    try:
        fn=h.resolve(True) if forward=='public' else lambda image,**options:module._denoise_tv_chambolle_nd(image)
        actual=h.native_observe(fn,case)
        assert actual['error']['type']=='AssertionError' and 'forwarding detected' in actual['error']['message']
        assert sys.getprofile() is existing
    finally:sys.setprofile(before)


def test_whole_call_callback_excludes_fixture_but_keeps_output_allocation():
    case=next(c for c in h.corpus() if c.timed);calls=[]
    def function(image,**options):calls.append((image,options));return image.copy()
    callback,unchanged=h.timing_callback(function,case)
    a=callback();b=callback();assert a is not b and calls[0][0] is calls[1][0] and unchanged()
    assert a.shape==b.shape and a.tobytes()==b.tobytes()


def test_missing_capsule_and_source_tamper_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):h.verify(tmp_path)
    (tmp_path/'tv-chambolle-artifacts.json').write_text(json.dumps(dict(schema='tv-chambolle-exact-offline-v1',files_sha256={},corpus_manifest_sha256='0'*64,product_source={})))
    with pytest.raises(AssertionError):h.verify(tmp_path)


def test_checks_complete_before_only_measurement_entry_and_seven_rounds():
    tree=ast.parse(Path(h.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timers=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure'];assert len(timers)==1
    text=ast.unparse(main);assert text.index('if args.check_only:')<text.index('timing = measure(')
    assert "if report['failures']:" in text and text.index("if report['failures']:")<text.index('timing = measure(')
    assert ast.literal_eval(timers[0].args[2])==7
    assert "args.threads != [1]" in text and 'slower_than_original' in text


def test_source330_fixture_fullfile_pin_remains_strict():
    m=manifest();assert set(m['fixture_source_sha256'])=={'tests/test_tv_chambolle_contracts.py'}
    assert m['counts']==dict(source_untimed=330,additional_timed_candidates=70,total=400,algorithms=1,public_paths=1,aliases=0)


def test_workflow_has_all_three_preparation_barrier_and_fresh_artifact_measurement():
    text=(h.ROOT/'.github/workflows/tv-chambolle-benchmark.yml').read_text()
    prepare,measure=text.split('  measure:\n',1)
    assert 'needs: prepare' in measure and 'matrix:' in prepare and 'matrix:' in measure
    assert prepare.count("python: '3.13'")==2 and prepare.count("python: '3.11'")==1
    assert '--prepare-capsule --check-only --threads 1 4' in prepare
    assert '--check-only --threads 1 4' in prepare and "SKIMAGEXX_DISABLE_SIMD: '1'" in prepare
    assert 'python -m pytest' in prepare and 'pip wheel' in prepare
    assert 'pytest' not in measure and 'pip wheel' not in measure and 'cmake' not in measure
    assert '--no-index --no-deps --require-hashes' in measure and '--threads 1 --repeat 7' in measure
    assert 'actions/download-artifact@v4' in measure and 'if: always()' in measure
    assert 'admit_native' not in text and 'public_admission' in prepare


def test_timing_profiles_have_no_protocol_nonfinite_or_error_reclassification():
    rows=h.timing_profiles()
    assert {p['dtype'] for p in rows}=={'u1','f4','f8'}
    assert {len(p['shape']) for p in rows}=={2,3}
    assert {p.get('axis') for p in rows}=={None,0,-1}
    assert {p['iterations'] for p in rows}=={1,2,8,200}
    assert all(p['pattern'] in ('ramp','constant') for p in rows)
    assert not any(any(k in p for k in ('intake','parameter','warning_hook','fp_action','allocation','policy')) for p in rows)


@pytest.mark.parametrize('fault',('version','float-cap','bool-cap','entry-name','spoof-entry','other-core','default4','signature','source-pin','binary-pin'))
def test_public_guard_metadata_stubs_reject_before_any_fixture(tmp_path,monkeypatch,fault):
    """Harness-only metadata stubs: never a public/native acceptance proof."""
    import math
    binary=tmp_path/'tooling-core-not-a-binary';binary.write_bytes(b'harness-only')
    core=types.SimpleNamespace(__name__='math',__file__=str(binary),interface='stub-interface',cap=7,sin=math.sin)
    wrapper=types.ModuleType('skimagexx._tv_harness_only');wrapper._core=core
    exec('def denoise_tv_chambolle(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None):\n    raise AssertionError("harness stub must never execute")',wrapper.__dict__)
    function=wrapper.denoise_tv_chambolle
    workers=[1]
    def get():return workers[0]
    def set_workers(n):workers[0]=n
    core.get_num_threads=get;core.set_num_threads=set_workers
    monkeypatch.setattr(xx,'_core',core);monkeypatch.setattr(xx,'get_num_threads',get);monkeypatch.setattr(xx,'set_num_threads',set_workers);monkeypatch.setattr(xx,'set_fallback_enabled',lambda _:False)
    coverage=__import__('skimagexx.coverage',fromlist=['native_functions']);monkeypatch.setattr(coverage,'native_functions',lambda:(h.API,))
    importer=h.importlib.import_module
    monkeypatch.setattr(h.importlib,'import_module',lambda name:wrapper if name==wrapper.__name__ else importer(name))
    original=h.resolve(True);monkeypatch.setattr(h,'resolve',lambda original_flag=False:original if original_flag else function)
    monkeypatch.setattr(h,'source_fingerprint',lambda _:dict(source_sha256='c'*64))
    monkeypatch.setattr(h,'contracts',lambda:pytest.fail('fixture import reached'))
    m=manifest();m['accepted_public']=dict(wrapper_module=wrapper.__name__,interface_attribute='interface',interface='stub-interface',capability_attribute='cap',capability=7,entries=['sin'],qualified_source='a'*40,qualified_proof_sha256='b'*64)
    pin=dict(revision='a'*40,qualification_sha256='b'*64,wheel_sha256='d'*64,core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256='c'*64)
    m['accepted_products']={p:dict(pin) for p in ('windows','modern','minimum')}
    if fault=='version':core.interface='wrong'
    elif fault=='float-cap':core.cap=7.
    elif fault=='bool-cap':core.cap=True
    elif fault=='entry-name':core.sin=math.cos
    elif fault=='spoof-entry':core.sin=types.SimpleNamespace(__module__='math',__name__='sin')
    elif fault=='other-core':wrapper._core=object()
    elif fault=='default4':workers[0]=4
    elif fault=='signature':function.__signature__=__import__('inspect').Signature()
    elif fault=='source-pin':m['accepted_products'][h.profile()]['source_sha256']='f'*64
    else:m['accepted_products'][h.profile()]['core_sha256']='e'*64
    with pytest.raises(AssertionError):h.public_admission(m)
