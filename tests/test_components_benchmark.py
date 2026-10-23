"""Harness guards only: delegated original unit stubs never prove native parity."""
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import components as harness


def no_timing(*args,**kwargs):raise AssertionError('untimed/failing component run reached timing')


def tiny_corpus():yield harness.Case('measure.label','unit-stub-only',(np.array([0,1,1,0,2]),),dict(return_num=True),'tiny')


def install_stub(monkeypatch,callback=None):
    def unit_stub(*args,**kwargs):return (harness.reference_measure.label if callback is None else callback)(*args,**kwargs)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    for namespace in [harness.xx.measure,harness.xx.morphology]:monkeypatch.setattr(namespace,'label',unit_stub,raising=False)
    return unit_stub


def test_reference_check_never_calls_native_or_timer(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny_corpus)
    monkeypatch.setattr(harness,'measure',no_timing);monkeypatch.setattr(time,'perf_counter_ns',no_timing)
    monkeypatch.setattr(harness.xx.measure,'label',no_timing,raising=False)
    output=tmp_path/'reference.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text())
    assert report['status']=='complete-reference-check' and not report['results'] and not report['failures']
    assert report['metadata']['validation_scope']=='reference-only; native not exercised'
    assert report['checks'][0]['expected_output']['structure']['items'][1]['value']==2


def test_registry_gate_precedes_every_fixture_and_timer(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_functions',lambda:())
    monkeypatch.setattr(harness,'corpus',no_timing);monkeypatch.setattr(harness,'measure',no_timing)
    output=tmp_path/'missing.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text())
    assert 'public native APIs are not registered' in report['failures'][0]['message']
    assert not report['checks'] and not report['results']


def test_registered_original_forward_is_rejected_before_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    for namespace in [harness.xx.measure,harness.xx.morphology]:monkeypatch.setattr(namespace,'label',harness.reference_measure.label,raising=False)
    monkeypatch.setattr(harness,'corpus',no_timing)
    output=tmp_path/'forward.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    assert 'native registry leaked original' in json.loads(output.read_text())['failures'][0]['message']


def test_unit_stub_workers_restore_policy_without_timing(monkeypatch,tmp_path):
    install_stub(monkeypatch);monkeypatch.setattr(harness,'corpus',tiny_corpus)
    monkeypatch.setattr(harness,'measure',no_timing);monkeypatch.setattr(time,'perf_counter_ns',no_timing)
    old_workers=harness.xx.get_num_threads();old_fallback=harness.xx.set_fallback_enabled(True)
    try:
        output=tmp_path/'stub.json'
        assert harness.main(['--check-only','--threads','1','4','--output',str(output)])==0
        report=json.loads(output.read_text())
        assert set(report['checks'][0]['actual_output_by_workers'])=={'1','4'}
        assert report['status']=='complete-parity-check' and not report['results']
        assert harness.xx.get_num_threads()==old_workers and harness.xx.set_fallback_enabled(False) is True
    finally:harness.xx.set_fallback_enabled(old_fallback)


def test_late_parity_failure_prevents_all_timing(monkeypatch,tmp_path):
    def cases():
        yield from tiny_corpus()
        yield harness.Case('morphology.label','bad-unit-stub',(np.array([0,2,2]),),{},'tiny')
    def stub(image,**options):
        result=harness.reference_measure.label(image,**options)
        return result if options.get('return_num') else result+1
    install_stub(monkeypatch,stub);monkeypatch.setattr(harness,'corpus',cases)
    monkeypatch.setattr(harness,'measure',no_timing)
    monkeypatch.setattr(harness,'verify_prepared',lambda *args:{'unit_stub':True});monkeypatch.setattr(harness,'verify_sealed',lambda *args:{'unit_stub':True})
    monkeypatch.setenv('GITHUB_ACTIONS','true');output=tmp_path/'failure.json'
    assert harness.main(['--prepared-source',str(tmp_path/'stub.json'),'--output',str(output)])==1
    report=json.loads(output.read_text())
    assert report['status']=='failed-parity' and len(report['checks'])==2 and not report['results']
    assert report['checks'][0]['status']=='passed' and report['failures'][0]['case']=='bad-unit-stub'


def test_expected_error_is_verified_and_never_timed(monkeypatch,tmp_path):
    install_stub(monkeypatch)
    monkeypatch.setattr(harness,'corpus',lambda:iter([harness.Case('measure.label','bad-connectivity',(np.ones((2,3),bool),),dict(connectivity=0),expect_error=True)]))
    monkeypatch.setattr(harness,'measure',no_timing);output=tmp_path/'error.json'
    assert harness.main(['--check-only','--output',str(output)])==0
    report=json.loads(output.read_text())
    assert report['checks'][0]['expected_exception']['type']=='ValueError'
    assert report['checks'][0]['error_parity_workers']==[1]
    assert report['metadata']['timing_eligible_case_count']==0 and len(report['untimed_cases'])==1
    assert not report['results']


def test_exact_array_storage_scalar_types_and_nan_specs():
    array=np.ones((2,3),np.intp);view=array.T
    assert harness.output_spec(view,[array])!=harness.output_spec(view.copy(),[array])
    assert harness.output_spec((array,1),[])!=harness.output_spec((array,np.int64(1)),[])
    tagged=array.view(np.dtype(np.intp,metadata={'unit':'labels'}))
    assert harness.output_spec(tagged,[])!=harness.output_spec(array,[])
    json.dumps(harness.serial(dict(nan=np.nan,inf=np.inf)),allow_nan=False)


def test_fixed_corpus_unique_bounded_and_deterministic():
    cases=list(harness.corpus());specs=[harness.spec(c) for c in cases]
    assert {c.api for c in cases}==set(harness.APIS)
    assert len({(c.api,c.label) for c in cases})==len(cases)
    assert len(cases)==954 and sum(not c.expect_error for c in cases)==802
    assert {'raster','tiny','empty','cast_warning'}=={c.work_kind for c in cases}
    assert any(c.expect_error for c in cases) and any('validation-precedence' in c.label for c in cases)
    assert all(a.size<=524289 for c in cases for a in c.args if isinstance(a,np.ndarray))
    for layout in ['F','reverse','step','endian','broadcast','readonly','unaligned','tagged','memmap','singleton']:
        assert any(layout in c.label for c in cases),layout
    json.dumps(specs,allow_nan=False)
    # Array fixture paths are deliberately absent from contract hashes.
    assert all('components-benchmark-fixtures' not in json.dumps(s) for s in specs)


def test_local_timings_and_missing_one_worker_are_rejected(monkeypatch,tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(harness,'measure',no_timing)
    for args in [[],['--check-only','--threads','4'],['--reference-only']]:
        with pytest.raises(SystemExit) as caught:harness.main(args+['--prepared-source',str(tmp_path/'unused.json'),'--output',str(tmp_path/'unused-output.json')])
        assert caught.value.code==2


@pytest.mark.parametrize('options',[['--threads','0','1'],['--threads','1','1'],['--repeat','0'],['--target-ms','nan'],['--target-ms','inf']])
def test_invalid_numeric_configuration_fails_before_corpus(monkeypatch,options):
    monkeypatch.setattr(harness,'corpus',no_timing);monkeypatch.setattr(harness,'measure',no_timing)
    with pytest.raises(SystemExit) as caught:harness.main(['--check-only']+options)
    assert caught.value.code==2


def test_artifact_seal_rejects_sdk_and_wheel_tampering(monkeypatch,tmp_path):
    installed=tmp_path/'site'
    package=installed/'skimagexx'
    prepared=tmp_path/'artifacts/exact-package/baseline-source.json'
    prepared.parent.mkdir(parents=True);prepared.write_text('{}')
    wheel=prepared.parent.parent/'wheels/unit-stub.whl';wheel.parent.mkdir()
    members={'skimagexx/__init__.py':b'# unit stub\n','skimagexx/_core.so':b'unit stub binary',
             'include/skimagexx/image.hpp':b'unit stub header',
             'lib/libskimagexx.a':b'unit stub SDK archive',
             'lib/cmake/skimagexx/skimagexxConfig.cmake':b'# unit stub cmake',
             'share/licenses/scikit-imagexx/LICENSE':b'unit stub notice'}
    with zipfile.ZipFile(wheel,'w') as archive:
        for name,data in members.items():
            archive.writestr(name,data)
            target=installed/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    monkeypatch.setattr(harness.xx,'__file__',str(package/'__init__.py'))
    sealed=harness.seal_artifacts(prepared,wheel,source_root=installed)
    assert sealed['sdk_sha256']['include/skimagexx/image.hpp']
    header=installed/'include/skimagexx/image.hpp';header.write_bytes(b'changed')
    with pytest.raises(AssertionError,match='SDK differs'):
        harness.verify_sealed(prepared)
    header.write_bytes(members['include/skimagexx/image.hpp'])
    wheel.write_bytes(wheel.read_bytes()+b'changed')
    with pytest.raises(AssertionError,match='wheel changed'):
        harness.verify_sealed(prepared)


def test_workflow_prepares_all_profiles_before_fresh_measurement():
    workflow=(Path(harness.__file__).parents[1]/'.github/workflows/components-benchmark.yml').read_text()
    preparation,measurement=workflow.split('\n  measure:',1)
    assert 'needs: prepare' in measurement and 'workflow_dispatch:' in workflow
    assert 'pip wheel' in preparation and '--seal-wheel' in preparation
    assert 'pip wheel' not in measurement and 'pytest' not in '\n'.join(
        line for line in measurement.splitlines() if not line.lstrip().startswith('#'))
    assert '--no-deps build/prepared/wheels/*.whl' in measurement
    assert 'tests/test_connected_components_contracts.py tests/test_components_pipeline.py tests/test_components_benchmark.py' in preparation
