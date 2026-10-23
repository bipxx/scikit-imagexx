"""Harness-only guards; original-delegating unit stubs prove no native algorithms."""
import inspect
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks import rank_intensity as h


def forbidden(*args,**kwargs):raise AssertionError('untimed/failing harness reached forbidden work')


def tiny_cases():
    yield h.Case('autolevel','unit-only',dict(shape=(3,5),dtype='u1',bins=16,pattern='ramp',footprint='dense3',mask='none',layout='C',out='u1',out_layout='C'))


def unit_manifest(monkeypatch,tmp_path,cases=None):
    cases=list(tiny_cases() if cases is None else cases)
    manifest=json.loads(h.MANIFEST.read_text());manifest['scope']='HARNESS UNIT ONLY; never actual public acceptance'
    manifest['selected_contracts']=[];manifest['case_inventory']=[h.contract_inventory(c) for c in cases]
    manifest['case_contract_sha256']=h.digest(manifest['case_inventory'])
    manifest['reference_profiles']={h.runtime_profile():dict(case_spec_sha256=h.digest([h.spec(c) for c in cases]),policy_sha256=h.digest(h.policy_reference_rows(h.check_policies([1],True))),
        observations={str(i):h.digest(h.observe_case(h.resolve(c.name,True),c)) for i,c in enumerate(cases)})}
    path=tmp_path/'unit-manifest.json';path.write_text(json.dumps(manifest));monkeypatch.setattr(h,'MANIFEST',path)
    monkeypatch.setattr(h,'corpus',lambda:iter(cases));return cases


def unit_stubs(monkeypatch,wrong=None):
    originals={name:h.resolve(name,True) for name in h.NAMES};stubs={}
    for name in h.NAMES:
        def delegate(*args,_name=name,**kwargs):return originals[_name](*args,**kwargs)
        delegate.__signature__=inspect.signature(originals[name]);stubs[name]=delegate
    def resolve(name,original=False,alias=False):
        if original:return originals[name]
        if wrong=='forward':return originals[name]
        if wrong=='alias' and alias:return lambda *args,**kwargs:None
        return stubs[name]
    if wrong=='signature':stubs[h.NAMES[0]].__signature__=inspect.Signature()
    monkeypatch.setattr(h,'resolve',resolve);monkeypatch.setattr(h,'native_functions',lambda:h.APIS)
    monkeypatch.setattr(h.xx._core,'rank_histogram_capabilities',8191,raising=False)
    monkeypatch.setattr(h,'verify',lambda *args:{'unit_stub':True});return stubs


def test_reference_only_never_calls_native_or_timers(monkeypatch,tmp_path):
    unit_manifest(monkeypatch,tmp_path);real=h.resolve
    monkeypatch.setattr(h,'resolve',lambda name,original=False,alias=False:real(name,True,alias) if original else forbidden())
    monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setattr(time,'perf_counter_ns',forbidden)
    output=tmp_path/'reference.json';assert h.main(['--reference-only','--check-only','--output',str(output)])==0
    report=json.loads(output.read_text());assert report['status']=='complete-reference-check' and not report['results'] and not report['failures']
    assert report['metadata']['validation_scope']=='reference-only; native image paths not exercised'
    assert len(report['alias_checks'])==4 and all(row['native_identity'] is None and not row['timed'] for row in report['alias_checks'])
    assert len(report['policy_checks'])==24 and all(not row['timed'] and 'actual_by_workers' not in row for row in report['policy_checks'])


def test_missing_registry_precedes_every_fixture_capsule_and_timer(monkeypatch,tmp_path):
    monkeypatch.setattr(h,'native_functions',lambda:());monkeypatch.setattr(h,'make_case',forbidden)
    monkeypatch.setattr(h,'verify',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'missing.json';assert h.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text());assert 'not registered' in report['failures'][0]['message']
    assert not report['checks'] and not report['alias_checks'] and not report['results']


def test_r2_capabilities_and_alias_flags_cannot_admit_r3(monkeypatch,tmp_path):
    unit_stubs(monkeypatch);monkeypatch.setattr(h.xx._core,'rank_histogram_capabilities',511)
    monkeypatch.setattr(h.xx._core,'rank_alias_interface','rank-alias-v1',raising=False)
    monkeypatch.setattr(h.xx._core,'rank_alias_capabilities',480,raising=False)
    monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'r2-only.json';assert h.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text());assert 'capability8191' in report['failures'][0]['message'] and not report['checks']


@pytest.mark.parametrize('wrong,expected',[('forward','upstream-forwarded'),('alias','alias differs'),('signature','signature differs')])
def test_bad_registered_paths_rejected_before_image_calls(monkeypatch,tmp_path,wrong,expected):
    unit_stubs(monkeypatch,wrong);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'bad.json';assert h.main(['--check-only','--output',str(output)])==1
    report=json.loads(output.read_text());assert expected in report['failures'][0]['message'] and not report['checks']


def test_unit_stub_workers_aliases_policies_and_state_restore(monkeypatch,tmp_path):
    unit_manifest(monkeypatch,tmp_path);unit_stubs(monkeypatch);monkeypatch.setattr(h,'measure',forbidden)
    monkeypatch.setattr(time,'perf_counter_ns',forbidden)
    threads=h.xx.get_num_threads();policy=h.xx.set_fallback_enabled(True);previous=np.geterr();handler=np.geterrcall()
    try:
        output=tmp_path/'unit-stub.json'
        assert h.main(['--check-only','--threads','1','4','--capsule',str(tmp_path/'unit-capsule'),'--output',str(output)])==0
        report=json.loads(output.read_text());assert len(report['checks'])==1 and len(report['alias_checks'])==4 and len(report['policy_checks'])==24
        assert set(report['checks'][0]['actual_by_workers'])=={'1','4'} and not report['results']
        assert report['metadata']['algorithm_count']==4 and report['metadata']['aliases_timed'] is False
        assert h.xx.get_num_threads()==threads and h.xx.set_fallback_enabled(False) is True
        assert np.geterr()==previous and np.geterrcall() is handler
    finally:h.xx.set_fallback_enabled(policy)


def test_late_failed_case_prevents_all_timer_calls(monkeypatch,tmp_path):
    first=next(tiny_cases());cases=[first,h.Case('autolevel','late-bad-unit',dict(first.fixture,shape=(3,7)))]
    unit_manifest(monkeypatch,tmp_path,cases);stubs=unit_stubs(monkeypatch);original=stubs['autolevel']
    def wrong(image,*args,**kwargs):
        output=original(image,*args,**kwargs)
        if image.shape==(3,7):output.flat[0]=255
        return output
    wrong.__signature__=inspect.signature(h.resolve('autolevel',True));stubs['autolevel']=wrong
    monkeypatch.setattr(h,'measure',forbidden);monkeypatch.setenv('GITHUB_ACTIONS','true')
    output=tmp_path/'late-failure.json'
    assert h.main(['--capsule',str(tmp_path/'unit-capsule'),'--output',str(output)])==1
    report=json.loads(output.read_text());assert report['status']=='failed-parity' and len(report['checks'])==2
    assert report['checks'][0]['status']=='passed' and report['checks'][1]['status']=='failed' and not report['results']


def test_case_and_fixture_identity_mutation_rejected_before_fixture_work(monkeypatch,tmp_path):
    unit_manifest(monkeypatch,tmp_path);original=next(h.corpus())
    monkeypatch.setattr(h,'corpus',lambda:iter([h.Case(original.name,'changed-id',original.fixture)]))
    monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'changed-case.json'
    assert h.main(['--reference-only','--check-only','--output',str(output)])==1
    assert 'case IDs/fixtures changed' in json.loads(output.read_text())['failures'][0]['message']


def test_changed_source_contract_rejected_before_fixtures(monkeypatch,tmp_path):
    path=tmp_path/'wrong-source.json';path.write_text('{}');monkeypatch.setattr(h,'SOURCE_CONTRACT',path)
    monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'changed-source.json';assert h.main(['--reference-only','--check-only','--output',str(output)])==1
    assert 'source contract changed' in json.loads(output.read_text())['failures'][0]['message']


def test_source_text_hashes_canonicalize_lf_and_binary_hashes_do_not(tmp_path):
    lf=tmp_path/'lf.py';crlf=tmp_path/'crlf.py';lf.write_bytes(b'one\ntwo\n');crlf.write_bytes(b'one\r\ntwo\r\n')
    assert h.source_text_sha(lf)==h.source_text_sha(crlf) and h.artifacts.sha(lf)!=h.artifacts.sha(crlf)


def test_crlf_source_contract_has_identical_recorded_source_hash(monkeypatch,tmp_path):
    unit_manifest(monkeypatch,tmp_path);source=tmp_path/'crlf-source.json'
    source.write_bytes(h.SOURCE_CONTRACT.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    monkeypatch.setattr(h,'SOURCE_CONTRACT',source);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'canonical-source.json';assert h.main(['--reference-only','--check-only','--output',str(output)])==0


def test_fixed_corpus_all_safe_contract_rows_and_four_algorithms():
    cases=list(h.corpus());profiles=list(h.profiles());manifest=json.loads(h.MANIFEST.read_text())
    assert len(cases)==1428 and len(profiles)==104 and sum(c.timed for c in cases)==384
    assert len({(c.name,c.label) for c in cases})==1428
    assert [h.contract_inventory(c) for c in cases]==manifest['case_inventory']
    assert h.digest(manifest['case_inventory'])==manifest['case_contract_sha256']
    assert manifest['operation_ids']==dict(zip(h.NAMES,range(9,13)))
    assert sum(c.frozen_id is not None for c in cases)==1012
    assert {tuple(p['shape']) for p in profiles}=={(17,19),(512,513),(7,11,13),(17,129,131)}
    assert {p['out'] for p in profiles}=={'default','u1','u2','f4','f8'}
    assert all(np.prod(p['shape'])<=1001 for p in profiles if p['bins']==65536)
    assert all('generic' not in h.spec(c)['api'] for c in (cases[0],cases[-1]))


def test_selected_source_projections_exactly_match_sealed1012_rows():
    manifest=json.loads(h.MANIFEST.read_text());source=json.loads(h.SOURCE_CONTRACT.read_text())
    assert manifest['source_contract_sha256']==h.source_text_sha(h.SOURCE_CONTRACT)
    assert len(manifest['selected_contracts'])==1012
    for row in manifest['selected_contracts']:
        for tag in ('windows','modern','minimum'):
            original=source['observations'][source['snapshots'][tag][row['id']]];actual=row['profiles'][tag]
            assert actual['error']==original['error'] and actual['warnings']==original['warnings']
            if actual['result'] is not None:assert actual['result']=={k:original['result'][k] for k in actual['result']}


def test_overlap_specs_rejected_without_original_calls():
    assert all(c.fixture.get('out') not in ('image','view','overlap') and c.fixture.get('special')!='alias-converted' for c in h.corpus())
    for out in ('image','view','overlap'):
        with pytest.raises(AssertionError,match='excludes all overlapping'):h.contract_fixture(dict(ndim=2,out=out))


def test_supplied_output_reset_is_inside_each_whole_call():
    case=next(tiny_cases());seen=[]
    def call(image,footprint,**options):
        out=options['out'];assert np.all(out==99);out.fill(7);seen.append(out);return out
    callback,tracked=h.timing_callback(call,case)
    assert np.all(tracked['out']==99)
    for _ in range(3):assert callback() is tracked['out']
    assert len(seen)==3 and all(value is tracked['out'] for value in seen)


def test_default_output_allocation_is_left_inside_library_call():
    case=next(tiny_cases());case=h.Case(case.name,case.label,dict(case.fixture,out='default'))
    def allocate(image,footprint,**options):assert 'out' not in options;return np.empty_like(image)
    callback,tracked=h.timing_callback(allocate,case);left=callback();right=callback()
    assert left is not right and left.flags.owndata and not np.shares_memory(left,tracked['image']) and 'out' not in tracked


def test_rejected_strided_output_preserves_padding_and_payload():
    spec=dict(next(h.profiles()),out='f4',out_layout='step');case=h.Case('autolevel','defined-output-error',spec,False)
    args,options,tracked=h.make_case(case);before=tracked['out_owner'].copy()
    with pytest.raises(ValueError,match='not C-contiguous'):h.resolve('autolevel',True)(*args,**options)
    np.testing.assert_array_equal(tracked['out_owner'],before)


@pytest.mark.parametrize('args',[[],['--reference-only'],['--check-only','--threads','4'],['--check-only','--threads','1','1'],['--check-only','--threads','0','1'],['--check-only','--threads','1','1025'],['--check-only','--repeat','0'],['--check-only','--target-ms','nan'],['--check-only','--target-ms','inf'],['--check-only','--prepare-capsule']])
def test_local_measurements_and_invalid_configuration_rejected(monkeypatch,tmp_path,args):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False);monkeypatch.setattr(h,'measure',forbidden)
    with pytest.raises(SystemExit) as error:h.main(args+['--output',str(tmp_path/'never.json')])
    assert error.value.code==2


def test_bad_capsule_blocks_all_fixtures_and_timers(monkeypatch,tmp_path):
    unit_manifest(monkeypatch,tmp_path);unit_stubs(monkeypatch)
    def tamper(*args):raise AssertionError('unit corrupted capsule')
    monkeypatch.setattr(h,'verify',tamper);monkeypatch.setattr(h,'make_case',forbidden);monkeypatch.setattr(h,'measure',forbidden)
    output=tmp_path/'bad-capsule.json';assert h.main(['--check-only','--capsule',str(tmp_path/'unit'),'--output',str(output)])==1
    report=json.loads(output.read_text());assert 'corrupted capsule' in report['failures'][0]['message'] and not report['checks'] and not report['alias_checks']


def fake_capsule(monkeypatch,tmp_path,extra_header=False,extra_python=False):
    root=tmp_path/'source';capsule=tmp_path/'capsule';(capsule/'wheels').mkdir(parents=True)
    files={'include/skimagexx/image.hpp':b'image\n','include/skimagexx/rank.hpp':b'rank\n','python/skimagexx/__init__.py':b'# unit\n'}
    if extra_header:files['include/skimagexx/omitted.hpp']=b'omitted header\n'
    if extra_python:files['python/skimagexx/omitted.py']=b'# omitted Python\n'
    for name,data in files.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    manifest=root/'manifest.json';manifest.write_text('{}');files['manifest.json']=manifest.read_bytes()
    runtime=capsule/'rank-intensity-runtime.json';runtime.write_text('{}')
    with zipfile.ZipFile(capsule/'source.zip','w') as z:
        for name,data in files.items():z.writestr(name,data)
    with zipfile.ZipFile(capsule/'wheels/unit.whl','w') as z:
        for name,data in files.items():
            if 'omitted' in name or name=='manifest.json':continue
            z.writestr(name.removeprefix('python/'),data)
        z.writestr('lib/libskimagexx.a',b'unit archive')
    monkeypatch.setattr(h,'ROOT',root);monkeypatch.setattr(h,'MANIFEST',manifest);monkeypatch.setattr(h,'source_fingerprint',lambda *args:{'unit_only':True})
    state={'source':{'files_sha256':{name:h.source_text_sha(root/name) for name in files}},'runtime':{'sdk_sha256':{'include/skimagexx/image.hpp':'unit','include/skimagexx/rank.hpp':'unit','lib/libskimagexx.a':'unit'}}}
    monkeypatch.setattr(h.artifacts,'verify',lambda *args:state)
    info=dict(schema='rank-intensity-exact-offline-v1',runtime_manifest_sha256=h.artifacts.sha(runtime),corpus_manifest_sha256=h.artifacts.sha(manifest),product_source={'unit_only':True},source_archive_sha256=h.artifacts.sha(capsule/'source.zip'))
    (capsule/'rank-intensity-artifacts.json').write_text(json.dumps(info));return capsule


@pytest.mark.parametrize('kind,message',[('runtime','runtime manifest changed'),('source','source archive changed'),('corpus','corpus manifest changed')])
def test_capsule_metadata_and_source_bytes_reject_tampering(monkeypatch,tmp_path,kind,message):
    capsule=fake_capsule(monkeypatch,tmp_path);assert h.verify(capsule)
    target=capsule/'rank-intensity-runtime.json' if kind=='runtime' else capsule/'source.zip' if kind=='source' else h.MANIFEST
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(AssertionError,match=message):h.verify(capsule)


def test_header_omitted_from_wheel_and_installed_set_still_rejected(monkeypatch,tmp_path):
    capsule=fake_capsule(monkeypatch,tmp_path,extra_header=True)
    with pytest.raises(AssertionError,match='complete SDK header set'):h.verify(capsule)


def test_python_omitted_from_wheel_and_installed_set_still_rejected(monkeypatch,tmp_path):
    capsule=fake_capsule(monkeypatch,tmp_path,extra_python=True)
    with pytest.raises(AssertionError,match='complete Python source'):h.verify(capsule)


def test_workflow_all_three_preparations_before_fresh_offline_measurement():
    text=(h.ROOT/'.github/workflows/rank-intensity-benchmark.yml').read_text();preparation,measurement=text.split('\n  measure:',1)
    assert 'needs: prepare' in measurement and 'workflow_dispatch:' in text
    assert '--prepare-capsule --check-only' in preparation and 'pip wheel' in preparation
    assert 'tests/test_rank_intensity_public.py' in preparation and 'tests/test_rank_intensity_benchmark.py' in preparation
    commands='\n'.join(line for line in measurement.splitlines() if not line.lstrip().startswith('#'))
    assert all(word not in commands for word in ('pip wheel','pytest','cmake','setup.py','pip install .'))
    assert '--no-index --no-deps --require-hashes' in measurement and '--repeat 7' in measurement
    assert 'runtime-lock.txt' in measurement and 'build/prepared/capsule' in measurement
    assert 'rank-intensity-performance.json' in measurement


@pytest.mark.parametrize('kind',['wheel','lock','runtime','source'])
def test_real_offline_authority_rejects_changed_frozen_inputs(monkeypatch,tmp_path,kind):
    # Synthetic metadata wheels exercise the shared authority, not native images.
    wheels=tmp_path/'wheels';wheels.mkdir();artifact=tmp_path/'rank-runtime.json'
    for name in ('scikit-imagexx','numpy','scipy','scikit-image'):
        with zipfile.ZipFile(wheels/(name+'.whl'),'w') as archive:
            archive.writestr(name+'.dist-info/METADATA','Name: '+name+'\nVersion: 1\n')
    state={'source':'unit-original'};runtime={'packages_sha256':{'skimagexx':{'_core.so':'unit-original'}}}
    monkeypatch.setattr(h.artifacts,'source_state',lambda *args:dict(state))
    monkeypatch.setattr(h.artifacts,'installed_state',lambda *args:json.loads(json.dumps(runtime)))
    h.artifacts.freeze(tmp_path,wheels,artifact);assert h.artifacts.verify(tmp_path,wheels,artifact)
    if kind=='wheel':
        wheel=next(wheels.glob('*.whl'));wheel.write_bytes(wheel.read_bytes()+b'changed')
    elif kind=='lock':artifact.with_name('runtime-lock.txt').write_text('changed')
    elif kind=='runtime':runtime['packages_sha256']['skimagexx']['_core.so']='unit-changed'
    else:state['source']='unit-changed'
    with pytest.raises(AssertionError):h.artifacts.verify(tmp_path,wheels,artifact)


def test_recorded_source_paths_are_portable_posix_names():
    manifest=json.loads(h.MANIFEST.read_text())
    assert all('\\' not in name and not Path(name).is_absolute() for name in manifest['fixture_source_sha256'])
    assert all((h.ROOT/name).is_file() for name in manifest['fixture_source_sha256'])
