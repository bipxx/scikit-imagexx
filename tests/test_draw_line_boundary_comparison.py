"""Prospective comparison and actual workflow gates; no scientific imports/calls."""
import ast
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import subprocess
import pytest
if __package__:
    from . import test_draw_coordinates_initial_workflow as initial
else:
    import test_draw_coordinates_initial_workflow as initial

ROOT=Path(__file__).parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-line-boundary-comparison.json'
PIN='75529f3a4377f95d10be0374c8880eed349ea079931fe854d25dafac3e82a0df'


def test_comparison_is_prospective_and_reuses_all_frozen_workloads():
    raw=CONTRACT.read_bytes();contract=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==PIN
    old=json.loads((ROOT/contract['execution_contract']).read_bytes())
    assert contract['case_ids']==[s['id']for s in old['cases']]
    assert len(contract['control_case_ids'])==11
    assert contract['target_case']=='line/tiny'
    assert contract['raw_samples_total']==8*12*2*7==1344
    assert contract['warmup_groups_total']==8*12*2==192
    assert contract['sequence']==['1-baseline','2-candidate','3-candidate','4-baseline','5-candidate','6-baseline','7-baseline','8-candidate']
    assert contract['candidate']==dict(run='41856678370',revision='0755818b0b8416947a47a8ba074b8e2388af27a5',artifact='13337554237',sha256='eea9aaeb9396ab16a0432b11455de9241b3ad2101a06629351245cdff4adced5')
    assert contract['variation_policy']['separate_variance_estimator']is False
    assert 'All16'in contract['variation_policy']['guard']
    for path,key in ((contract['driver'],'driver_sha256'),(contract['execution_contract'],'execution_contract_sha256'),(contract['reference_contract'],'reference_contract_sha256')):
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract[key]
    # The controller may contain baseline math. Candidate pins are checked
    # against its authenticated artifact and checked-out source by the workflow.
    assert set(contract['candidate_production_sha256'])=={'python/skimagexx/_draw_coordinates.py','src/bind_draw_coordinates.cpp'}
    assert all(re.fullmatch('[0-9a-f]{64}',digest)for digest in contract['candidate_production_sha256'].values())
    assert contract['gates']==dict(target_balanced_direct_min=1.20,target_balanced_normalized_min=1.20,target_all_pairs_direct_min=1.10,target_all_pairs_normalized_min=1.10,controls_all_pairs_direct_min=0.97,controls_all_pairs_normalized_min=0.97,preserved_original_wins_each_candidate_min_exclusive=1.0)


@pytest.mark.parametrize('fault',(None,'partial','source','artifact','timing-ran','missing-job','contract','workflow-failed','unsealed-run','unsealed-artifact','unsealed-digest','artifact-kind'))
def test_actual_comparison_authentication(tmp_path,monkeypatch,fault):
    monkeypatch.chdir(tmp_path);Path('results').mkdir()
    dest=Path('controller/coverage/contracts');dest.mkdir(parents=True)
    raw=CONTRACT.read_bytes()
    if fault=='contract':raw+=b' '
    (dest/CONTRACT.name).write_bytes(raw)
    contract=json.loads(CONTRACT.read_bytes());old=contract['baseline'];candidate=contract['candidate']
    env=dict(PRODUCER_RUN=old['run'],PRODUCER_REVISION=old['revision'],ARTIFACT_ID=old['artifact'],ARTIFACT_SHA256=old['sha256'],GITHUB_REPOSITORY='bipxx/scikit-imagexx',EXECUTION_CREATED_UTC='2026-10-29T00:00:00+00:00',DEADLINE_UTC='2026-10-29T02:00:00+00:00',CANDIDATE_RUN=candidate['run'],CANDIDATE_REVISION=candidate['revision'],CANDIDATE_ARTIFACT_ID=candidate['artifact'],CANDIDATE_ARTIFACT_SHA256=candidate['sha256'])
    if fault=='partial':env['CANDIDATE_ARTIFACT_ID']=''
    elif fault=='unsealed-run':env['CANDIDATE_RUN']='987'
    elif fault=='unsealed-artifact':env['CANDIDATE_ARTIFACT_ID']='654'
    elif fault=='unsealed-digest':env['CANDIDATE_ARTIFACT_SHA256']='a'*64
    for key,value in env.items():monkeypatch.setenv(key,value)
    def records(run,revision,artifact,digest):
        record=dict(id=int(run),head_sha=revision,head_repository=dict(full_name=env['GITHUB_REPOSITORY']),path='.github/workflows/ci.yml',status='completed',conclusion='success')
        item=dict(id=int(artifact),name='ci-performance-minimum',digest='sha256:'+digest,expired=False,size_in_bytes=42,workflow_run=dict(id=int(run),head_sha=revision))
        jobs=[dict(name='numpy-minimum',conclusion='success',steps=[]),dict(name='measure-minimum',conclusion='success',steps=[dict(name='Run python benchmarks/'+str(i),conclusion='skipped')for i in range(29)])]
        return record,dict(total_count=1,artifacts=[item]),dict(total_count=2,jobs=jobs)
    replies=[*records(old['run'],old['revision'],old['artifact'],old['sha256']),*records(candidate['run'],candidate['revision'],candidate['artifact'],candidate['sha256'])]
    if fault=='source':replies[3]['head_sha']='b'*40
    elif fault=='artifact':replies[4]['artifacts'][0]['digest']='sha256:'+'b'*64
    elif fault=='artifact-kind':replies[4]['artifacts'][0]['name']='numpy-minimum-ciede2000'
    elif fault=='timing-ran':replies[5]['jobs'][1]['steps'][0]['conclusion']='success'
    elif fault=='missing-job':replies[5]['jobs'].pop();replies[5]['total_count']=1
    elif fault=='workflow-failed':
        replies[3]['conclusion']='failure';replies[5]['jobs'][1]['conclusion']='failure'
        replies[4]=replies[5];replies.pop()
    monkeypatch.setattr(subprocess,'check_output',lambda args:json.dumps(replies.pop(0)).encode())
    script=initial.SCRIPTS[0].replace('from datetime import datetime, timezone','from datetime import datetime as real_datetime, timezone\nclass datetime(real_datetime):\n    @classmethod\n    def now(cls, tz): return cls(2026,10,29,1,tzinfo=tz)')
    if fault:
        with pytest.raises(AssertionError):exec(script,{})
        if fault=='workflow-failed':
            status=json.loads(Path('results/candidate-producer-status.json').read_bytes())
            assert status['workflow_conclusion']=='failure'
            assert status['source_producer'][0]['conclusion']=='success'and status['offline_verifier'][0]['conclusion']=='failure'
    else:
        exec(script,{})
        assert not replies
        assert Path('results/selection.json').is_file()and Path('results/candidate-selection.json').is_file()
        assert Path('results/comparison-contract.json').read_bytes()==CONTRACT.read_bytes()


def actual_decision():
    tree=ast.parse(initial.SCRIPTS[2])
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='comparison_decision')
    ns=dict(math=math,statistics=statistics)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-comparison-decision','exec'),ns)
    return ns['comparison_decision']


def reports():
    contract=json.loads(CONTRACT.read_bytes());result={}
    for name in contract['sequence']:
        candidate=name.endswith('-candidate');rows=[]
        for case in contract['case_ids']:
            native=1.5 if candidate and case=='line/tiny'else 2.0
            original=1.0 if case=='line/tiny'else 1.98 if case=='rectangle/2d'else 4.0
            rows.append(dict(case_id=case,cpp_ms=native,reference_ms=original,speedup=original/native))
        result[name]=dict(results=rows)
    return result,contract


@pytest.mark.parametrize('fault',(None,'target-balanced-direct','target-balanced-normalized','target-pair-direct','target-pair-normalized','control-direct','control-normalized','original-win','order','missing-pass','case-order'))
def test_actual_decision_applies_all_strict_gates_and_retains_original_losses(fault):
    data,contract=reports()
    def setrow(name,index,native=None,original=None):
        row=data[name]['results'][index]
        if native is not None:row['cpp_ms']=native
        if original is not None:row['reference_ms']=original
        row['speedup']=row['reference_ms']/row['cpp_ms']
    candidates=[n for n in data if n.endswith('-candidate')]
    if fault=='target-balanced-direct':
        for n in candidates:setrow(n,0,native=1.75,original=1.75*.75)
    elif fault=='target-balanced-normalized':
        for n in candidates:setrow(n,0,original=.8)
    elif fault=='target-pair-direct':setrow(candidates[0],0,native=1.9,original=1.9*.75)
    elif fault=='target-pair-normalized':setrow(candidates[0],0,original=.7)
    elif fault=='control-direct':setrow(candidates[0],1,native=2.1,original=4.2)
    elif fault=='control-normalized':setrow(candidates[0],1,original=3.8)
    elif fault=='original-win':
        for n in data:setrow(n,1,original=1.0)
    elif fault=='order':data=dict(reversed(list(data.items())))
    elif fault=='missing-pass':data.pop(candidates[0])
    elif fault=='case-order':data[candidates[0]]['results'].reverse()
    if fault in ('order','missing-pass','case-order'):
        with pytest.raises(AssertionError):actual_decision()(data,contract)
    else:
        summary=actual_decision()(data,contract)
        assert summary['decision']==('DISCARD'if fault else'KEEP')
        if not fault:
            assert all(summary['gates'].values())
            assert max(summary['cases'][0]['candidate_original_speedups'])<1
            assert max(summary['cases'][3]['candidate_original_speedups'])<1
            assert summary['raw_samples']==1344
            assert summary['variation_policy']['separate_variance_estimator']is False
        else:
            key={'target-balanced-direct':'target_balanced_direct','target-balanced-normalized':'target_balanced_normalized','target-pair-direct':'target_all_pairs_direct','target-pair-normalized':'target_all_pairs_normalized','control-direct':'controls_all_pairs_direct','control-normalized':'controls_all_pairs_normalized','original-win':'preserved_original_wins'}[fault]
            assert summary['gates'][key]is False


def test_real_optional_recipe_uses_two_products_and_same_existing_sampler():
    workflow=initial.WORKFLOW
    assert workflow.count('      candidate_run:')==1 and 'candidate_snapshot' in workflow
    assert len(initial.SCRIPTS)==3
    assert 'passes="1-baseline 2-candidate 3-candidate 4-baseline 5-candidate 6-baseline 7-baseline 8-candidate"'in workflow
    assert "role == 'candidate'" in initial.SCRIPTS[1]
    assert "'sdk_sha256'" in initial.SCRIPTS[1]and "'candidate_production_sha256'"in initial.SCRIPTS[1]
    assert 'assert focused_count == 10258' in initial.SCRIPTS[1]and '== 456' in initial.SCRIPTS[1]
    assert "'test_draw_coordinates_integration'] = 185"in initial.SCRIPTS[1]
    assert 'pass_index=1' in initial.SCRIPTS[2]and 'i+index+pass_index' in initial.SCRIPTS[2]
    assert 'UNRESOLVED_MEASUREMENT' in initial.SCRIPTS[2]
    assert not any(word in workflow for word in ('cmake --build','python -m pytest','pip install numpy','pip install scipy'))


@pytest.mark.parametrize('fault',(None,'science-wheel','science-package','library','SDK','production','package','checked-out-source'))
def test_actual_product_pair_boundary_rejects_changed_dependencies(tmp_path,monkeypatch,fault):
    monkeypatch.chdir(tmp_path)
    root=Path('results');root.mkdir()
    before_dir=Path('snapshot/build/prepared/ci-artifacts');before_dir.mkdir(parents=True)
    snapshot=Path('candidate_snapshot');snapshot.mkdir()
    # Small synthetic envelopes exercise the actual post-layout pair boundary;
    # the workflow separately authenticates real178/84 members and453 source files.
    contract=json.loads(CONTRACT.read_bytes());contract['production_scope']['count']=3
    files={'python/skimagexx/_draw_coordinates.py':b'new wrapper','src/bind_draw_coordinates.cpp':b'new binder','include/skimagexx/image.hpp':b'unchanged'}
    hashes={n:hashlib.sha256(raw).hexdigest()for n,raw in files.items()}
    contract['candidate_production_sha256']={n:hashes[n]for n in list(files)[:2]}
    old_hashes={**hashes,**{n:'old'for n in list(files)[:2]}}
    wheels={str(i):dict(name='dependency'+str(i),sha256='pin')for i in range(13)}
    wheels['product']=dict(name='scikit-imagexx',sha256='old')
    runtime=dict(python='python',implementation='CPython',cache_tag='tag',machine='x86_64',system='Linux',dependency_libraries_sha256={'lib':'pin'},sdk_sha256={'header':'pin'},versions={'numpy':'1.24.4'},packages_sha256={name:{'file':'pin'}for name in ('numpy','scipy','skimage')})
    runtime['packages_sha256']['skimagexx']={'_draw_coordinates.py':'old','_core.fixture.so':'old','__init__.py':'same'}
    before=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=old_hashes))
    manifest=deepcopy(before);manifest['source']['files_sha256']=hashes.copy()
    manifest['wheels']['product']['sha256']='new'
    manifest['runtime']['packages_sha256']['skimagexx'].update({'_draw_coordinates.py':'new','_core.fixture.so':'new'})
    if fault=='science-wheel':manifest['wheels']['0']['sha256']='changed'
    elif fault=='science-package':manifest['runtime']['packages_sha256']['numpy']['file']='changed'
    elif fault=='library':manifest['runtime']['dependency_libraries_sha256']['lib']='changed'
    elif fault=='SDK':manifest['runtime']['sdk_sha256']['header']='changed'
    elif fault=='production':manifest['source']['files_sha256']['include/skimagexx/image.hpp']='changed'
    elif fault=='package':manifest['runtime']['packages_sha256']['skimagexx']['__init__.py']='changed'
    elif fault=='checked-out-source':files['src/bind_draw_coordinates.cpp']=b'foreign'
    for name,raw in files.items():
        path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (before_dir/'measurement.json').write_text(json.dumps(before))
    (root/'comparison-contract.json').write_text(json.dumps(contract))
    tree=ast.parse(initial.SCRIPTS[1])
    node=next(n for n in tree.body if isinstance(n,ast.If)and any(isinstance(x,ast.Name)and x.id=='production'for x in ast.walk(n)))
    ns=dict(role='candidate',snapshot=snapshot,manifest=manifest,json=json,Path=Path,hashlib=hashlib)
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-pair-boundary','exec')
    if fault:
        with pytest.raises(AssertionError):exec(code,ns)
    else:
        exec(code,ns)
        assert json.loads((root/'product-pair-authenticated.json').read_bytes())['production_files']==3


@pytest.mark.parametrize('missing',('clock','contract','pair'))
def test_actual_finalizer_marks_incomplete_comparison_unresolved(tmp_path,monkeypatch,missing):
    monkeypatch.chdir(tmp_path);Path('results').mkdir()
    monkeypatch.setenv('CANDIDATE_RUN','987')
    if missing!='clock':
        Path('results/execution-clock.json').write_text(json.dumps(dict(created_utc='2098-12-31T23:00:00+00:00',deadline_utc='2099-01-01T00:00:00+00:00')))
    if missing=='pair':Path('results/comparison-contract.json').write_bytes(CONTRACT.read_bytes())
    with pytest.raises((AssertionError,FileNotFoundError)):
        exec(initial.SCRIPTS[2],{})
    summary=json.loads(Path('results/summary.json').read_bytes())
    assert summary['decision']=='UNRESOLVED_MEASUREMENT'
    assert 'gates'not in summary


def test_prospective_metadata_does_not_require_candidate_math_in_controller(monkeypatch):
    original=Path.read_bytes
    def metadata_only(path):
        assert not path.is_relative_to(ROOT/'python')and not path.is_relative_to(ROOT/'src')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',metadata_only)
    test_comparison_is_prospective_and_reuses_all_frozen_workloads()
