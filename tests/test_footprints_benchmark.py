"""Tooling guards only. Patched original functions are unit stubs, not native proof."""
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import footprints as harness


def no_timing(*args, **kwargs):
    raise AssertionError('untimed/failing footprint run reached timing')


def tiny_corpus():
    yield harness.Case('disk', 'unit-test-only', (2,), {})


def test_reference_check_never_calls_native_or_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, 'corpus', tiny_corpus)
    monkeypatch.setattr(harness, 'measure', no_timing)
    monkeypatch.setattr(time, 'perf_counter_ns', no_timing)
    monkeypatch.setattr(harness.xx.morphology, 'disk', no_timing, raising=False)
    output=tmp_path/'reference.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text())
    assert report['status']=='complete-reference-check'
    assert report['metadata']['validation_scope']=='reference-only; native not exercised'
    assert report['checks'][0]['expected_output']['structure']['shape']==[5,5]
    assert not report['results'] and not report['failures']


def test_registry_gate_is_mandatory(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_functions',lambda:())
    monkeypatch.setattr(harness,'measure',no_timing)
    output=tmp_path/'missing.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text())
    assert 'public native APIs are not registered' in report['failures'][0]['message']
    assert not report['checks'] and not report['results']


def test_stub_check_restores_policy_and_never_times(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny_corpus)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.morphology,'disk',harness.reference.disk,raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    monkeypatch.setattr(time,'perf_counter_ns',no_timing)
    previous_workers=harness.xx.get_num_threads();previous_fallback=harness.xx.set_fallback_enabled(True)
    try:
        output=tmp_path/'stub.json'
        assert harness.main(['--check-only','--threads','1','4','--output',str(output)])==0
        report=json.loads(output.read_text())
        assert report['status']=='complete-parity-check'
        assert set(report['checks'][0]['actual_output_by_workers'])=={'1','4'}
        assert not report['results'] and harness.xx.get_num_threads()==previous_workers
        assert harness.xx.set_fallback_enabled(False) is True
    finally:
        harness.xx.set_fallback_enabled(previous_fallback)


def test_late_parity_failure_blocks_every_timing(monkeypatch,tmp_path):
    def cases():
        yield harness.Case('disk','good-unit-stub',(2,),{})
        yield harness.Case('mirror_footprint','bad-unit-stub',(np.ones((3,5)),),{},'view_metadata')
    monkeypatch.setattr(harness,'corpus',cases)
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.morphology,'disk',harness.reference.disk,raising=False)
    monkeypatch.setattr(harness.xx.morphology,'mirror_footprint',lambda a:a.copy(),raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    monkeypatch.setattr(harness,'verify_prepared',lambda *args:{'unit_test_stub':True})
    monkeypatch.setattr(harness,'verify_sealed',lambda *args:{'unit_test_stub':True})
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    output=tmp_path/'failed.json'
    assert harness.main(['--prepared-source',str(tmp_path/'stub.json'),'--output',str(output)])==1
    report=json.loads(output.read_text())
    assert report['status']=='failed-parity' and len(report['checks'])==2
    assert report['checks'][0]['status']=='passed'
    assert report['failures'][0]['case']=='bad-unit-stub' and not report['results']


def test_expected_errors_are_checked_and_never_timed(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',lambda:iter([harness.Case('disk','bad-dtype',(2,),dict(dtype='invalid'),expect_error=True)]))
    monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx.morphology,'disk',harness.reference.disk,raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    output=tmp_path/'errors.json'
    assert harness.main(['--check-only','--output',str(output)])==0
    report=json.loads(output.read_text())
    assert report['checks'][0]['expected_exception']['type']=='TypeError'
    assert report['checks'][0]['error_parity_workers']==[1]
    assert report['metadata']['timing_eligible_case_count']==0
    assert len(report['untimed_cases'])==1 and not report['results']


def test_exact_tuple_scalar_and_alias_descriptions():
    array=np.ones((3,3),dtype=np.uint8)
    view=array[::-1]
    assert harness.output_spec(((array,1),),[])!=harness.output_spec(((array,np.uint8(1)),),[])
    assert harness.output_spec(((array,1),(view,1)),[])!=harness.output_spec(((array,1),(view.copy(),1)),[])
    assert harness.output_spec(view,[array])!=harness.output_spec(view.copy(),[array])


def test_fixed_corpus_is_unique_and_honestly_classified():
    cases=list(harness.corpus()); specs=[harness.spec(c) for c in cases]
    assert {c.api for c in cases}==set(harness.NAMES)
    assert len({(c.api,c.label) for c in cases})==len(cases)
    assert {'raster','decomposition_plan','view_metadata'}=={c.work_kind for c in cases}
    assert all(c.work_kind=='view_metadata' for c in cases if c.api=='mirror_footprint')
    assert all(c.options.get('decomposition')=='sequence' for c in cases if c.label.startswith('table='))
    assert {'sequence','separable','crosses'}<={c.options.get('decomposition') for c in cases}
    assert any(c.expect_error and c.label=='nested-sequence' for c in cases)
    assert any('readonly' in c.label for c in cases) and any('>i2' in c.label for c in cases)
    json.dumps(specs,allow_nan=False)


def test_local_timings_and_missing_one_worker_are_rejected(monkeypatch,tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    monkeypatch.setattr(harness,'measure',no_timing)
    for args in [[],['--check-only','--threads','4'],['--reference-only']]:
        with pytest.raises(SystemExit) as caught:
            harness.main(args+['--prepared-source',str(tmp_path/'unused.json'),'--output',str(tmp_path/'unused-output.json')])
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
    workflow=(Path(harness.__file__).parents[1]/'.github/workflows/footprints-benchmark.yml').read_text()
    preparation,measurement=workflow.split('\n  measure:',1)
    assert 'needs: prepare' in measurement and 'workflow_dispatch:' in workflow
    assert 'pip wheel' in preparation and '--seal-wheel' in preparation
    assert 'pip wheel' not in measurement and 'pytest' not in '\n'.join(
        line for line in measurement.splitlines() if not line.lstrip().startswith('#'))
    assert '--no-deps build/prepared/wheels/*.whl' in measurement
    assert 'tests/test_footprint_contracts.py tests/test_footprints_benchmark.py' in preparation
