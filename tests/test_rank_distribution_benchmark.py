"""Harness-only guards. Delegated original unit stubs never prove native work."""
import json
import hashlib
import time
import zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import rank_distribution as harness


def forbidden(*args,**kwargs):raise AssertionError('untimed/failing rank run reached forbidden work')


def tiny_cases():
    yield harness.Case('gradient','unit-only',dict(shape=(3,5),dtype='u1',bins=16,pattern='ramp',footprint='dense3',mask='none',layout='C',out='u1',out_layout='C'))


def install_unit_stubs(monkeypatch,wrong=None):
    originals={n:harness.resolve(n,True) for n in harness.NAMES};stubs={}
    for name in harness.NAMES:
        def unit_stub(*args,_name=name,**kwargs):return originals[_name](*args,**kwargs)
        stubs[name]=unit_stub
    def resolve(name,original=False,alias=False):
        if original:return originals[name]
        if wrong=='alias' and alias:return lambda *a,**k:None
        if wrong=='forward':return originals[name]
        return stubs[name]
    monkeypatch.setattr(harness,'resolve',resolve);monkeypatch.setattr(harness,'native_functions',lambda:harness.APIS)
    monkeypatch.setattr(harness.xx._core,'rank_histogram_capabilities',511,raising=False)
    return stubs


def test_reference_only_never_calls_native_or_timer(monkeypatch,tmp_path):
    real=harness.resolve
    monkeypatch.setattr(harness,'resolve',lambda name,original=False,alias=False:real(name,True,alias) if original else forbidden())
    monkeypatch.setattr(harness,'corpus',tiny_cases);monkeypatch.setattr(harness,'measure',forbidden)
    monkeypatch.setattr(time,'perf_counter_ns',forbidden);output=tmp_path/'reference.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==0
    r=json.loads(output.read_text());assert r['status']=='complete-reference-check' and not r['results'] and not r['failures']
    assert r['metadata']['validation_scope']=='reference-only; native not exercised'
    assert len(r['alias_checks'])==4 and all(c['native_identity'] is None and not c['timed'] for c in r['alias_checks'])
    assert len(r['policy_checks'])==24 and all(not c['timed'] and 'actual_by_workers' not in c for c in r['policy_checks'])


def test_missing_registry_precedes_fixtures_and_timers(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'native_functions',lambda:());monkeypatch.setattr(harness,'make_case',forbidden)
    monkeypatch.setattr(harness,'measure',forbidden);output=tmp_path/'missing.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    r=json.loads(output.read_text());assert 'not registered' in r['failures'][0]['message']
    assert not r['checks'] and not r['alias_checks'] and not r['results']


def test_missing_v2_capability_precedes_fixtures_and_timers(monkeypatch,tmp_path):
    install_unit_stubs(monkeypatch)
    monkeypatch.setattr(harness.xx._core,'rank_histogram_capabilities',31)
    monkeypatch.setattr(harness,'make_case',forbidden);monkeypatch.setattr(harness,'measure',forbidden)
    output=tmp_path/'old-core.json'
    assert harness.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text())
    assert 'capability511' in report['failures'][0]['message']
    assert not report['checks'] and not report['alias_checks'] and not report['policy_checks'] and not report['results']


@pytest.mark.parametrize('wrong',['forward','alias'])
def test_registered_bad_paths_rejected_before_image_checks(monkeypatch,tmp_path,wrong):
    install_unit_stubs(monkeypatch,wrong);monkeypatch.setattr(harness,'make_case',forbidden)
    monkeypatch.setattr(harness,'measure',forbidden);output=tmp_path/(wrong+'.json')
    assert harness.main(['--check-only','--output',str(output)])==1
    r=json.loads(output.read_text());assert not r['checks'] and not r['results']
    assert ('upstream-forwarded' if wrong=='forward' else 'alias differs') in r['failures'][0]['message']


def test_unit_stub_workers_policy_and_aliases_are_separate(monkeypatch,tmp_path):
    install_unit_stubs(monkeypatch);monkeypatch.setattr(harness,'corpus',tiny_cases)
    monkeypatch.setattr(harness,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
    old=harness.xx.get_num_threads();policy=harness.xx.set_fallback_enabled(True)
    try:
        output=tmp_path/'stub.json';assert harness.main(['--check-only','--threads','1','4','--output',str(output)])==0
        r=json.loads(output.read_text());assert len(r['checks'])==1 and len(r['alias_checks'])==4
        assert set(r['checks'][0]['actual_by_workers'])=={'1','4'}
        assert r['metadata']['algorithm_count']==4 and r['metadata']['aliases_timed'] is False
        assert not r['results'] and harness.xx.get_num_threads()==old
        assert harness.xx.set_fallback_enabled(False) is True
    finally:harness.xx.set_fallback_enabled(policy)


def test_late_bad_case_blocks_every_timer(monkeypatch,tmp_path):
    stubs=install_unit_stubs(monkeypatch);original=stubs['gradient']
    def bad(image,*args,**kwargs):
        result=original(image,*args,**kwargs)
        if image.shape==(3,7):result.flat[0]=255
        return result
    stubs['gradient']=bad
    def cases():
        first=next(tiny_cases());yield first
        yield harness.Case('gradient','late-bad-unit-stub',dict(first.fixture,shape=(3,7)))
    monkeypatch.setattr(harness,'corpus',cases);monkeypatch.setattr(harness,'measure',forbidden)
    monkeypatch.setattr(harness,'verify_prepared',lambda *a:{'unit_stub':True});monkeypatch.setattr(harness,'verify_sealed',lambda *a:{'unit_stub':True})
    monkeypatch.setenv('GITHUB_ACTIONS','true');output=tmp_path/'failed.json'
    assert harness.main(['--prepared-source',str(tmp_path/'stub.json'),'--output',str(output)])==1
    r=json.loads(output.read_text());assert r['status']=='failed-parity' and len(r['checks'])==2
    assert r['checks'][0]['status']=='passed' and not r['results']


def test_fixed_corpus_balanced_and_aliases_never_timed():
    cases=list(harness.corpus());timed=[c for c in cases if c.timed];profiles=list(harness.profiles())
    assert len(profiles)==104 and len(cases)==648 and len(timed)==384
    assert {c.name for c in cases}==set(harness.NAMES)
    assert len({(c.name,c.label) for c in cases})==len(cases)
    assert len({c.fixture['id'] for c in timed})==96
    assert {tuple(p['shape']) for p in profiles}=={(17,19),(512,513),(7,11,13),(17,129,131)}
    assert {p['bins'] for p in profiles if p['dtype']=='u2'}=={4,256,1024,65536}
    assert {p['out'] for p in profiles}=={'u1','u2','f4','f8'}
    assert all(np.prod(p['shape'])<=1001 for p in profiles if p['bins']==65536)
    assert all(c.frozen_id is None for c in timed)
    assert all('filters.rank.generic' not in harness.spec(c)['api'] for c in [cases[0],cases[-1]])
    assert sum(c.frozen_id is not None for c in cases)==232


def test_selected_contracts_are_exact_immutable_926_source_rows():
    manifest=json.loads(harness.MANIFEST.read_text())
    source=json.loads(harness.SOURCE_CONTRACT.read_text())
    assert manifest['source_contract_hash_schema']=='sha256-posix-lf-v1'
    assert manifest['source_contract_sha256']==hashlib.sha256(harness.SOURCE_CONTRACT.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert len(manifest['selected_contracts'])==232
    assert len({row['id'] for row in manifest['selected_contracts']})==232
    for row in manifest['selected_contracts']:
        for tag,profile in [('numpy1','minimum'),('numpy2','windows')]:
            expected=source['observations'][source['snapshots'][profile][row['id']]]
            actual=row['profiles'][tag]
            assert actual['error']==expected['error'] and actual['warnings']==expected['warnings']
            if actual['result'] is not None:
                assert actual['result']=={key:expected['result'][key] for key in actual['result']}


def test_changed_source_contract_blocks_fixtures_and_timers(monkeypatch,tmp_path):
    changed=tmp_path/'changed.json';changed.write_text('{}')
    monkeypatch.setattr(harness,'SOURCE_CONTRACT',changed)
    monkeypatch.setattr(harness,'make_case',forbidden);monkeypatch.setattr(harness,'measure',forbidden)
    output=tmp_path/'changed-source.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==1
    report=json.loads(output.read_text())
    assert 'source contract changed' in report['failures'][0]['message']
    assert not report['checks'] and not report['alias_checks'] and not report['policy_checks'] and not report['results']


def test_entropy_reference_metadata_includes_raw_bits_provider_and_untimed_policies(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'corpus',tiny_cases);monkeypatch.setattr(harness,'measure',forbidden)
    output=tmp_path/'provider.json'
    assert harness.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text())
    assert len(report['metadata']['entropy_math_provider']['reference_extension_sha256'])==64
    assert 'exact full output raw bytes' in report['metadata']['entropy_math_provider']['comparison']
    assert len(report['policy_checks'])==24 and not report['results']
    assert all(not row['timed'] for row in report['policy_checks'])


def test_all_overlap_specs_are_excluded_and_rejected_without_reference_calls():
    assert all(c.fixture.get('out') not in ('image','view','overlap') and c.fixture.get('special')!='alias-converted' for c in harness.corpus())
    for out in ('image','view','overlap'):
        with pytest.raises(AssertionError,match='excludes all overlapping'):
            harness.contract_fixture(dict(ndim=2,out=out))


def test_provided_output_tracks_owner_sentinels_and_bits():
    profile=dict(next(harness.profiles()),out='f4',out_layout='step')
    case=harness.Case('gradient','strided-output',profile)
    args,options,tracked=harness.make_case(case);owner=tracked['out_owner'];before=owner[...,1::2].copy()
    with pytest.raises(ValueError,match='not C-contiguous'):harness.resolve('gradient',True)(*args,**options)
    np.testing.assert_array_equal(owner[...,1::2],before)
    assert np.all(owner==99),'defined buffer-admission failure must preserve owner'


@pytest.mark.parametrize('args',[[],['--check-only','--threads','4'],['--check-only','--threads','1','1'],['--reference-only'],['--check-only','--repeat','0'],['--check-only','--target-ms','nan']])
def test_local_timings_and_invalid_configuration_rejected(monkeypatch,tmp_path,args):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(harness,'measure',forbidden)
    with pytest.raises(SystemExit) as error:harness.main(args+['--prepared-source',str(tmp_path/'unused.json'),'--output',str(tmp_path/'unused-output.json')])
    assert error.value.code==2


def test_exact_wheel_package_sdk_seal_rejects_tampering(monkeypatch,tmp_path):
    installed=tmp_path/'site';package=installed/'skimagexx';prepared=tmp_path/'artifact/exact-package/baseline-source.json'
    prepared.parent.mkdir(parents=True);prepared.write_text('{}');wheel=prepared.parent.parent/'wheels/stub.whl';wheel.parent.mkdir()
    members={'skimagexx/__init__.py':b'# stub','skimagexx/_core.so':b'stub binary','include/skimagexx/image.hpp':b'image stub',
             'include/skimagexx/rank.hpp':b'rank stub','lib/libskimagexx.a':b'stub archive',
             'lib/cmake/skimagexx/skimagexxConfig.cmake':b'# stub','share/licenses/scikit-imagexx/LICENSE':b'stub license'}
    with zipfile.ZipFile(wheel,'w') as archive:
        for name,data in members.items():
            archive.writestr(name,data);path=installed/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(harness.xx,'__file__',str(package/'__init__.py'))
    harness.seal_artifacts(prepared,wheel,source_root=installed)
    extra=package/'unexpected.py';extra.write_text('# extra')
    with pytest.raises(AssertionError,match='complete package changed'):harness.verify_sealed(prepared)
    extra.unlink()
    header=installed/'include/skimagexx/rank.hpp';header.write_bytes(b'changed')
    with pytest.raises(AssertionError,match='SDK differs'):harness.verify_sealed(prepared)
    header.write_bytes(members['include/skimagexx/rank.hpp']);wheel.write_bytes(wheel.read_bytes()+b'changed')
    with pytest.raises(AssertionError,match='wheel changed'):harness.verify_sealed(prepared)


def test_workflow_all_preparation_before_fresh_measurement():
    text=(harness.ROOT/'.github/workflows/rank-distribution-benchmark.yml').read_text();preparation,measurement=text.split('\n  measure:',1)
    assert 'needs: prepare' in measurement and 'workflow_dispatch:' in text
    assert 'pip wheel' in preparation and '--seal-wheel' in preparation
    assert 'pip wheel' not in measurement and 'pytest' not in '\n'.join(l for l in measurement.splitlines() if not l.lstrip().startswith('#'))
    assert '--no-deps build/prepared/wheels/*.whl' in measurement
    assert 'tests/test_rank_distribution_contracts.py tests/test_rank_distribution_pipeline.py tests/test_rank_distribution_benchmark.py' in preparation
    assert all(path in preparation for path in ('tests/test_rank_public.py','tests/test_rank_public_admission.py','tests/test_rank_integer_warning.py'))
