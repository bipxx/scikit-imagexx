"""Inert controller/schema/selector checks; no algorithm or timer execution."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import pytest

ROOT=Path(__file__).parents[1]

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

C=module('tools/region_stain_compare.py','region_stain_controller')
P=json.loads((ROOT/'coverage/contracts/region-stain-same-host-v1.json').read_bytes())


def test_imports_refuse_scientific_algorithms(monkeypatch):
    old=builtins.__import__
    def refused(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx'),name
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refused)
    module('tools/region_stain_compare.py','inert_controller')
    module('benchmarks/stain_f32_holdouts.py','inert_holdouts')


def test_fixed_contract_workflow_sources_and_pool_scope():
    assert P['passes']==['baseline','candidate','candidate','baseline'] and P['pairs']==[[0,1],[3,2]]
    assert [len(P[k]) for k in ('region_cases','half_cases','holdout_cases')]==[12,4,4]
    assert len({s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]})==20
    assert P['sampling']==dict(threads=1,repeat=7,target_ms=100,warmup_ms=250,cohorts=20,samples_per_pass=280,samples_total=1120,deadline='required immutable UTC deadline; checked before every actualbatch')
    g=P['fixed_gates'];assert g['all20_candidate_over_prior_max']==1.05 and g['region_targets_candidate_over_prior_max']==.5 and g['stain_targets_candidate_over_prior_max']==.8
    context=json.loads((ROOT/'coverage/contracts/region-stain-controller-v1-context.json').read_bytes())['entries']
    for name,pin in P['controller_source_sha256'].items():
        raw=context[name]['utf8'].encode()if name in context else(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==pin
    workflow=context['.github/workflows/region-stain-same-host.yml']['utf8']
    assert "python-version: '3.11.17'" in workflow and 'if: always()' in workflow
    assert 'eea321b2a334ba51c475d7733ad7547fa24c553b' in workflow
    assert 'candidate_run:' in workflow and 'candidate_source:' in workflow and 'candidate_artifact:' in workflow and 'deadline_utc:' in workflow
    for name in C.POOLS:assert name+": '1'" in workflow
    tree=ast.parse((ROOT/'tools/region_stain_compare.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    source=ast.get_source_segment((ROOT/'tools/region_stain_compare.py').read_text(),execute)
    assert source.index('manifests_equal')<source.index('all-runtimes-preflight-complete.json')<source.index('enumerate(protocol[\'passes\']')
    assert 'ci_measurement_artifacts.py' in source and 'verify' in source and "'--require-hashes'" in source and "'--no-index'" in source
    assert 'CONTRACT=' not in source and 'write_text' not in source.split('for index,role in enumerate')[1].split('result=acceptance')[0]


def selection():
    b=P['baseline'];r=dict(id=int(b['run']),head_sha=b['source'],status='completed',conclusion='success',head_repository={'full_name':'bipxx/qupathxx'},path='.github/workflows/ci.yml',event='workflow_dispatch')
    a=dict(id=b['artifact'],name='ci-performance-minimum',expired=False,size_in_bytes=b['bytes'],digest='sha256:'+b['sha256'],workflow_run={'id':int(b['run']),'head_sha':b['source']})
    return r,{'total_count':1,'artifacts':[a]}


def test_exact_baseline_selection():
    r,a=selection();b=P['baseline'];assert C.validate_selection(r,a,'baseline',b['source'],b['run'],b['artifact'],'bipxx/qupathxx',P)==a['artifacts'][0]


@pytest.mark.parametrize('fault',('failed','wrong-kind','wrong-digest','wrong-source','expired','wrong-repo','pagination'))
def test_artifact_authentication_negative(fault):
    r,a=selection();b=P['baseline']
    if fault=='failed':r['conclusion']='failure'
    if fault=='wrong-kind':a['artifacts'][0]['name']='numpy-minimum-ciede2000'
    if fault=='wrong-digest':a['artifacts'][0]['digest']='sha256:'+'0'*64
    if fault=='wrong-source':a['artifacts'][0]['workflow_run']['head_sha']='0'*40
    if fault=='expired':a['artifacts'][0]['expired']=True
    if fault=='wrong-repo':r['head_repository']['full_name']='other/repo'
    if fault=='pagination':a['total_count']=2
    with pytest.raises(AssertionError):C.validate_selection(r,a,'baseline',b['source'],b['run'],b['artifact'],'bipxx/qupathxx',P)


@pytest.mark.parametrize('timed', (False,True,'ciede-no-repeat'))
def test_prepare_only_actual_step_admission(timed):
    names=['Install exact offline hash-locked runtime','Verify source, wheel, runtime, package and SDK before measurements','Verify measurement inputs remained immutable']
    steps=[{'name':n,'conclusion':'success'} for n in names]
    steps += [{'name':f'Run python benchmarks/closed_{index}.py --repeat 7','conclusion':'success' if timed is True and index==0 else 'skipped'} for index in range(34)]
    steps += [{'name':'Run python benchmarks/ciede2000.py --threads 1 --output build/ci-ciede2000.json','conclusion':'success' if timed=='ciede-no-repeat' else 'skipped'}]
    assert len([step for step in steps if step['name'].startswith('Run python benchmarks/')])==35
    jobs={'total_count':1,'jobs':[{'conclusion':'success','steps':steps}]}
    if timed:
        with pytest.raises(AssertionError):C.validate_prepare_jobs(jobs)
    else:C.validate_prepare_jobs(jobs)


def passes():
    ids=[s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]]
    old={cid:{'original':10.,'cpp':10.} for cid in ids};new=copy.deepcopy(old)
    for cid in P['fixed_gates']['region_targets']:new[cid]['cpp']=4.
    for cid in P['fixed_gates']['stain_targets']:new[cid]['cpp']=7.
    return [copy.deepcopy(old),copy.deepcopy(new),copy.deepcopy(new),copy.deepcopy(old)]


def test_both_pairs_and_all40_fixed_gates():
    result=C.acceptance(passes(),P);assert result['decision']=='KEEP' and len(result['checks'])==40


@pytest.mark.parametrize('case',[s['id'] for k in ('region_cases','half_cases','holdout_cases') for s in P[k]])
@pytest.mark.parametrize('pair',(1,2))
def test_every_cohort_each_pair_rejects_regression(case,pair):
    data=passes();data[pair][case]['cpp']=11.
    assert C.acceptance(data,P)['decision']=='DISCARD'


def test_original_normalized_gate_cannot_be_hidden_by_direct_gain():
    data=passes();data[1][P['holdout_cases'][0]['id']]['original']=1.
    assert C.acceptance(data,P)['decision']=='DISCARD'


def sample_report(tmp_path):
    events=[];samples=[[],[]];counts=[[],[]];records=[]
    def emit(f,phase,r,o,ns):
        common=dict(case_index=0,case_id='case',function=f,round=r,ordinal=o,loops=1)
        before=dict(common,event='batch_before',requested_phase=phase)
        after=dict(common,event='batch_after',phase=phase,elapsed_ns=ns,complete=True,accepted=phase=='sample',failure=None)
        events.extend([before,after]);records.append(after)
        if phase=='sample':samples[f].append(ns/1e6);counts[f].append(1)
    for f in (0,1):emit(f,'warmup',-1,f,250_000_000)
    orders=[]
    for r in range(7):
        order=[0,1] if (r+1)%2==0 else [1,0];orders.append(order)
        for o,f in enumerate(order):emit(f,'sample',r,o,100_000_000)
    row=dict(case_id='case',threads=1,native_accepted=True,actual_C_returns=['entry'],samples_ms=samples,loops=counts,batch_records=records,warmup_elapsed_ns=[250_000_000]*2,sample_order=orders,reference_ms=100.,cpp_ms=100.,speedup=1.)
    events.append(dict(event='measure_returned',case_index=0,case_id='case',timing=copy.deepcopy(row)))
    report={'metadata':dict(threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,reference_pool_limits={k:'1' for k in C.POOLS},fallback_enabled=False,raw_samples=14),'results':[row]}
    path=tmp_path/'journal.jsonl';path.write_text('\n'.join(map(json.dumps,events))+'\n')
    return report,path,events


def test_raw_journal_medians_warmup_sample_order(tmp_path):
    report,path,events=sample_report(tmp_path);assert C.validate_report(report,path,['case'])=={'case':{'original':100.,'cpp':100.}}


@pytest.mark.parametrize('fault',('short-sample','warm','median','failure','extra','order','loops','missing-witness'))
def test_raw_journal_negative(tmp_path,fault):
    report,path,events=sample_report(tmp_path)
    if fault=='short-sample':events[5]['elapsed_ns']=99_000_000
    if fault=='warm':events[1]['elapsed_ns']=1
    if fault=='median':report['results'][0]['cpp_ms']=90.
    if fault=='failure':events[5]['complete']=False
    if fault=='extra':events+=events[:2]
    if fault=='order':events[5]['ordinal']=9
    if fault=='loops':events[5]['loops']=0
    if fault=='missing-witness':report['results'][0]['actual_C_returns']=[]
    path.write_text('\n'.join(map(json.dumps,events))+'\n')
    with pytest.raises(AssertionError):C.validate_report(report,path,['case'])


@pytest.mark.parametrize('spec',P['holdout_cases'],ids=[s['id'] for s in P['holdout_cases']])
def test_source_safe_holdout_factory_original_refused(spec):
    import numpy as np
    h=module('benchmarks/stain_f32_holdouts.py','factory_intake')
    reference=types.SimpleNamespace(rgb_from_hed=np.array([[.65,.7,.29],[.07,.99,.11],[.27,.57,.78]],dtype=np.float64))
    def denied(*a,**k):pytest.fail('no original or native algorithm authorized during fixture intake')
    reference.combine_stains=reference.separate_stains=denied
    image,matrix=h.fixture(spec,np,reference)
    assert image.dtype==np.dtype('float32') and image.shape==(512,512,3)
    assert image.flags.owndata and image.flags.c_contiguous and 0<=image.min()<=image.max()<=1
    assert matrix.shape==(3,spec['k']) and np.isfinite(matrix).all() and np.abs(matrix).max()<=1


@pytest.mark.parametrize('deadline',('2026-11-04T12:00:00','2026-11-04T12:00:00+01:00'))
def test_controller_rejects_non_UTC_before_any_execution(deadline,monkeypatch):
    monkeypatch.setattr(C,'execute',lambda args:pytest.fail('invaliddeadline reached controller execution'))
    with pytest.raises(AssertionError):
        C.main(['execute','--repository','bipxx/qupathxx','--candidate-source','0'*40,'--candidate-run','1','--candidate-artifact','1','--snapshots','unused','--results','unused','--deadline-utc',deadline])


V3=json.loads((ROOT/'coverage/contracts/regionprops-performance-compare-v3.json').read_bytes())


def test_v3_fixed_cf0_baseline_unchanged_cases_and_method():
    oldpath=ROOT/V3['predecessor_contract']['path'];old=json.loads(oldpath.read_bytes())
    assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==V3['predecessor_contract']['sha256']
    assert V3['baseline']==dict(source='cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b',run='42002481826',artifact=13403507497,bytes=85519557,sha256='e213203d7ebfe3353dcf2c6972c7b1b4c1b8d0ebb576052379ca713bf6c57e0e')
    changed={'baseline','fixed_gates','scope','next_gate','controller_source_sha256','prospective_root_contract_sha256','comparison_revision','protocol_parser_version_note','predecessor_contract','workflow_readiness'}
    assert {k:v for k,v in V3.items()if k not in changed}=={k:v for k,v in old.items()if k not in changed}
    assert V3['optimization_version']==2 and V3['comparison_revision']==3
    ids=[s['id']for k in ('region_cases','scalar_cases','half_cases','holdout_cases')for s in V3[k]]
    assert len(ids)==len(set(ids))==28 and V3['sampling']['samples_total']==1568
    limits=V3['fixed_gates']['per_case_limits'];assert set(limits)==set(ids)
    targets={'regionprops-scalars/tiny-warm-c-on','regionprops-scalars/3d-warm-f-on'}
    assert limits=={cid:.95 if cid in targets else 1.05 for cid in ids}
    assert V3['fixed_gates']['policy']==old['fixed_gates']['policy']
    assert isinstance(V3['candidate'],str) and 'required successful minimum prepare-only' in V3['candidate']


def test_v3_controller_pointer_only_inverse_and_pending_workflow():
    old=json.loads((ROOT/V3['predecessor_contract']['path']).read_bytes())
    source=(ROOT/'tools/region_stain_compare.py').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(source).hexdigest()==V3['controller_source_sha256']['tools/region_stain_compare.py']
    inverse=source
    for new,previous in [(b"CONTRACT=ROOT/'coverage/contracts/regionprops-performance-compare-v3.json'",b"CONTRACT=ROOT/'coverage/contracts/region-stain-same-host-v1.json'"),(b"active=load(ROOT/'coverage/contracts/regionprops-performance-compare-v3.json')",b"active=load(ROOT/'coverage/contracts/regionprops-performance-compare-v2.json')")]:
        assert inverse.count(new)==1;inverse=inverse.replace(new,previous)
    assert hashlib.sha256(inverse).hexdigest()==old['controller_source_sha256']['tools/region_stain_compare.py']
    assert C.CONTRACT==ROOT/'coverage/contracts/regionprops-performance-compare-v3.json'
    # The shared integrator owns this exact two-pointer workflow successor.
    assert V3['workflow_readiness'].startswith('FROZEN_SHARED_WORKFLOW_PROJECTION:')
    for name,pin in V3['controller_source_sha256'].items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if name=='.github/workflows/region-stain-same-host.yml':
            if hashlib.sha256(raw).hexdigest()==old['controller_source_sha256'][name]:
                raw=raw.replace(b'ref: 6a2b33a2f78606b9315dd0fcc50777b6e85ac38f',b'ref: cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b').replace(b'--contract controller/coverage/contracts/regionprops-performance-compare-v2.json',b'--contract controller/coverage/contracts/regionprops-performance-compare-v3.json')
            else:
                predecessor=raw.replace(b'ref: cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b',b'ref: 6a2b33a2f78606b9315dd0fcc50777b6e85ac38f').replace(b'--contract controller/coverage/contracts/regionprops-performance-compare-v3.json',b'--contract controller/coverage/contracts/regionprops-performance-compare-v2.json')
                assert hashlib.sha256(predecessor).hexdigest()==old['controller_source_sha256'][name]
        assert hashlib.sha256(raw).hexdigest()==pin


def test_v3_exact_baseline_selection_and_wrong_predecessor_refusal():
    b=V3['baseline'];record,listing=selection()
    record.update(id=int(b['run']),head_sha=b['source'])
    item=listing['artifacts'][0];item.update(id=b['artifact'],size_in_bytes=b['bytes'],digest='sha256:'+b['sha256'],workflow_run=dict(id=int(b['run']),head_sha=b['source']))
    assert C.validate_selection(record,listing,'baseline',b['source'],b['run'],b['artifact'],'bipxx/qupathxx',V3)==item
    with pytest.raises(AssertionError):C.validate_selection(*selection(),'baseline',P['baseline']['source'],P['baseline']['run'],P['baseline']['artifact'],'bipxx/qupathxx',V3)


@pytest.mark.parametrize('candidate_pass',(1,2))
def test_v3_every_direct_and_normalized_gate_remains_strict(candidate_pass):
    limits=V3['fixed_gates']['per_case_limits']
    data=[{cid:dict(original=1.,cpp=1.)for cid in limits}for _ in range(4)]
    for index in (1,2):
        for cid,limit in limits.items():data[index][cid]['cpp']=min(limit,1.)
    result=C.acceptance(data,V3);assert result['decision']=='KEEP' and len(result['checks'])==56
    for cid,limit in limits.items():
        fault=copy.deepcopy(data);fault[candidate_pass][cid]['cpp']=limit*1.000001
        rejected=C.acceptance(fault,V3);assert rejected['decision']=='DISCARD'
        row=next(r for r in rejected['checks']if r['case_id']==cid and r['candidate_pass']==candidate_pass+1)
        assert row['direct']>limit and not row['passed']
        fault=copy.deepcopy(data);fault[candidate_pass][cid]['original']=.5
        rejected=C.acceptance(fault,V3);assert rejected['decision']=='DISCARD'
        row=next(r for r in rejected['checks']if r['case_id']==cid and r['candidate_pass']==candidate_pass+1)
        assert row['direct']<=limit and row['original_normalized']>limit and not row['passed']
