"""Fixed Region/stain ABBA controller; all imports are non-scientific/inert."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import runpy
import stat
import statistics
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/regionprops-performance-compare-v3.json'
POOLS=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')


def load(path):return json.loads(Path(path).read_bytes())


def run(argv,log,cwd=None,env=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    log.with_suffix('.command.json').write_text(json.dumps({'argv':list(map(str,argv)),'cwd':None if cwd is None else str(cwd)},indent=2)+'\n')
    started=datetime.now(timezone.utc).isoformat()
    with log.with_suffix('.stdout').open('wb') as out,log.with_suffix('.stderr').open('wb') as err:
        result=subprocess.run(list(map(str,argv)),cwd=cwd,env=env,stdout=out,stderr=err)
    log.with_suffix('.exit.json').write_text(json.dumps({'returncode':result.returncode,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat()})+'\n')
    assert result.returncode==0,(str(log),result.returncode)


def validate_selection(record,listing,role,source,run_id,artifact_id,repository,protocol):
    assert re.fullmatch('[0-9a-f]{40}',source) and re.fullmatch('[1-9][0-9]*',str(run_id))
    assert record['id']==int(run_id) and record['head_sha']==source
    assert record['status']=='completed' and record['conclusion']=='success'
    assert record['head_repository']['full_name']==repository and record['path']=='.github/workflows/ci.yml'
    assert listing['total_count']==len(listing['artifacts'])
    items=[x for x in listing['artifacts'] if x['id']==int(artifact_id)]
    assert len(items)==1;item=items[0]
    assert item['name']=='ci-performance-minimum' and not item['expired']
    assert item['workflow_run']['id']==int(run_id) and item['workflow_run']['head_sha']==source
    assert 0<item['size_in_bytes']<512*1024**2 and re.fullmatch('sha256:[0-9a-f]{64}',item['digest'])
    if role=='baseline':
        b=protocol['baseline'];assert source==b['source'] and str(run_id)==b['run'] and int(artifact_id)==b['artifact']
        assert item['digest']=='sha256:'+b['sha256'] and item['size_in_bytes']==b['bytes']
    else:assert record['event']=='workflow_dispatch'
    return item


def validate_prepare_jobs(jobs):
    # A successful prepare-only measurement job verifies immutable inputs but
    # skips every benchmark command. Retain raw jobs instead of inferring input
    # mode from a successful run or artifact name alone.
    assert jobs['total_count']==len(jobs['jobs'])
    matches=[x for x in jobs['jobs'] if any(s['name']=='Verify source, wheel, runtime, package and SDK before measurements' for s in x['steps'])]
    assert len(matches)==1;job=matches[0];assert job['conclusion']=='success'
    steps=job['steps'];timers=[s for s in steps if s['name'].startswith('Run python benchmarks/')]
    assert timers and all(s['conclusion']=='skipped' for s in timers)
    for name in ('Install exact offline hash-locked runtime','Verify source, wheel, runtime, package and SDK before measurements','Verify measurement inputs remained immutable'):
        found=[s for s in steps if s['name']==name];assert len(found)==1 and found[0]['conclusion']=='success'


def extract_prepared(archive,destination,selection):
    assert archive.stat().st_size==selection['size_in_bytes']
    assert 'sha256:'+hashlib.sha256(archive.read_bytes()).hexdigest()==selection['digest']
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();assert len(infos)==len({i.filename for i in infos})
        assert sum(i.file_size for i in infos)<2*1024**3
        for i in infos:
            p=PurePosixPath(i.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and not stat.S_ISLNK(i.external_attr>>16)
        names=[i for i in infos if i.filename.startswith('prepared/')]
        assert names and any(i.filename=='prepared/ci-artifacts/measurement.json' for i in names)
        for i in names:z.extract(i,destination)


def manifests_equal(a,b):
    assert a['schema']==b['schema']=='ci-offline-measurement-v1'
    assert len(a['wheels'])==len(b['wheels'])==14
    # All 13 dependency wheels and installed scientific packages must be exact.
    assert {k:v for k,v in a['wheels'].items() if v['name']!='scikit-imagexx'}=={k:v for k,v in b['wheels'].items() if v['name']!='scikit-imagexx'}
    for key in ('python','implementation','cache_tag','machine','system','dependency_libraries_sha256'):
        assert a['runtime'][key]==b['runtime'][key],key
    for name in ('numpy','scipy','skimage'):
        assert a['runtime']['packages_sha256'][name]==b['runtime']['packages_sha256'][name]
    assert {k:v for k,v in a['runtime']['versions'].items() if k!='scikit-imagexx'}=={k:v for k,v in b['runtime']['versions'].items() if k!='scikit-imagexx'}
    assert a['runtime']['versions']['numpy']=='1.24.4' and a['runtime']['versions']['scikit-image']=='0.26.0'


def validate_report(report,journal,case_ids):
    rows=report['results'];assert [r['case_id'] for r in rows]==case_ids
    m=report['metadata'];assert m['threads']==1 and m['repeat']==7 and m['target_ms']==100 and m['warmup_calibration_min_ms']==250
    assert set(m['reference_pool_limits'].values())=={'1'} and m['fallback_enabled'] is False
    events=[json.loads(s) for s in Path(journal).read_text().splitlines()]
    assert len([x for x in events if x['event']=='measure_returned'])==len(rows)
    position=0;result={}
    for i,row in enumerate(rows):
        assert row.get('threads',m['threads'])==1 and row['native_accepted'] is True and row['actual_C_returns']
        accepted=[[],[]];counts=[[],[]];warm=[0,0];records=[]
        while position<len(events) and events[position]['event']!='measure_returned':
            before,after=events[position:position+2];position+=2
            assert before['event']=='batch_before' and after['event']=='batch_after'
            for key in ('case_index','case_id','function','round','ordinal','loops'):assert before[key]==after[key]
            assert after['case_index']==i and after['case_id']==row['case_id'] and after['complete'] is True and after['failure'] is None
            assert after['loops']>=1 and after['elapsed_ns']>0
            assert (after['accepted'] is True)==(after['phase']=='sample')
            f=after['function'];assert f in (0,1);records.append(after)
            if after['phase']=='warmup':warm[f]+=after['elapsed_ns']
            if after['accepted']:
                assert after['elapsed_ns']>=100_000_000
                accepted[f].append(after['elapsed_ns']/after['loops']/1e6);counts[f].append(after['loops'])
        returned=events[position];position+=1
        assert returned['event']=='measure_returned' and returned['case_id']==row['case_id'] and returned['case_index']==i
        assert records==row['batch_records'] and accepted==row['samples_ms'] and counts==row['loops'] and warm==row['warmup_elapsed_ns']
        assert all(len(x)==7 for x in accepted) and min(warm)>=250_000_000
        expected_order=[[(f,round_id,ordinal) for ordinal,f in enumerate(order)] for round_id,order in enumerate(row['sample_order'])]
        assert [(x['function'],x['round'],x['ordinal']) for x in records if x['accepted']]==[x for group in expected_order for x in group]
        assert row['sample_order']==[[0,1] if (round_id+i+1)%2==0 else [1,0] for round_id in range(7)]
        for f in (0,1):assert [x['round'] for x in records if x['accepted'] and x['function']==f]==list(range(7))
        med=[statistics.median(x) for x in accepted]
        assert all(math.isfinite(x) and x>0 for x in med)
        assert row['reference_ms']==med[0] and row['cpp_ms']==med[1] and row['speedup']==med[0]/med[1]
        for key in ('samples_ms','loops','reference_ms','cpp_ms','speedup','batch_records'):assert returned['timing'][key]==row[key]
        result[row['case_id']]={'original':med[0],'cpp':med[1]}
    assert position==len(events) and m['raw_samples']==14*len(rows)
    return result


def acceptance(passes,protocol):
    assert len(passes)==4;checks=[]
    g=protocol['fixed_gates']
    for baseline,candidate in protocol['pairs']:
        assert set(passes[baseline])==set(passes[candidate])
        for cid,b in passes[baseline].items():
            c=passes[candidate][cid];direct=c['cpp']/b['cpp'];normalized=(c['cpp']/c['original'])/(b['cpp']/b['original'])
            limit=(g['per_case_limits'][cid] if 'per_case_limits' in g else
                g['region_targets_candidate_over_prior_max'] if cid in g['region_targets'] else
                g['stain_targets_candidate_over_prior_max'] if cid in g['stain_targets'] else g['all20_candidate_over_prior_max'])
            checks.append(dict(case_id=cid,baseline_pass=baseline+1,candidate_pass=candidate+1,direct=direct,original_normalized=normalized,limit=limit,passed=direct<=limit and normalized<=limit))
    return {'decision':'KEEP' if all(c['passed'] for c in checks) else 'DISCARD','checks':checks,'scope':'bounded fixed cohorts only; no whole-library claim'}


def driver(args):
    root=args.source_root.resolve();sys.path[:0]=[str(root/'benchmarks'),str(root/'tools')]
    if args.family=='holdouts':
        script=ROOT/'benchmarks/stain_f32_holdouts.py';argv=[str(script),'--source-root',str(root),'--contract',str(CONTRACT),'--deadline-utc',args.deadline_utc,'--output',str(args.output)]
    else:
        script=root/'benchmarks'/('regionprops_state.py' if args.family=='region' else 'regionprops_scalars.py' if args.family=='scalars' else 'stains.py')
        argv=[str(script),'--threads','1','--repeat','7','--deadline-utc',args.deadline_utc,'--output',str(args.output)]
        if args.family=='half':argv+=['--half-only','--size','512']
    # Stops on the real sampler call after actual original/native validation,
    # before executing its body. No fake timings, output reports or journals.
    class PreflightComplete(Exception):pass
    calls=[]
    def profile(frame,event,value):
        if args.preflight and event=='call' and frame.f_code.co_name=='measure' and frame.f_globals.get('__name__')=='montage_sustained':
            calls.append(True);raise PreflightComplete()
    prior_argv,prior_profile=sys.argv,sys.getprofile();sys.argv=argv
    try:
        if args.preflight:sys.setprofile(profile)
        try:runpy.run_path(str(script),run_name='__main__')
        except PreflightComplete:assert len(calls)==1
    finally:sys.setprofile(prior_profile);sys.argv=prior_argv
    if args.preflight:
        assert len(calls)==1 and not args.output.exists() and not args.output.with_suffix('.batches.jsonl').exists()
        args.output.with_suffix('.preflight.json').write_text(json.dumps({'actual_first_cohort_validated':True,'measure_calls_stopped_before_body':1,'timed_batches':0})+'\n')


def execute(args):
    assert re.fullmatch('[0-9a-f]{40}',args.candidate_source)
    assert re.fullmatch('[1-9][0-9]*',args.candidate_run) and args.candidate_artifact>0
    assert re.fullmatch('[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repository)
    protocol=load(CONTRACT)
    for name,pin in protocol['controller_source_sha256'].items():
        current=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if protocol.get('optimization_version')!=2 and name in ('tools/region_stain_compare.py','.github/workflows/region-stain-same-host.yml'):
            context=load(ROOT/'coverage/contracts/region-stain-controller-v1-context.json')['entries'][name]
            assert hashlib.sha256(context['utf8'].encode()).hexdigest()==context['raw_sha256']==pin
            active=load(ROOT/'coverage/contracts/regionprops-performance-compare-v3.json')
            assert hashlib.sha256(current).hexdigest()==active['controller_source_sha256'][name]
        else:assert hashlib.sha256(current).hexdigest()==pin,name
    out=args.results.resolve();out.mkdir(parents=True,exist_ok=False)
    assert datetime.now(timezone.utc)<datetime.fromisoformat(args.deadline_utc)
    (out/'frozen-inputs.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    def api(endpoint,label):
        raw=subprocess.check_output(['gh','api','repos/'+args.repository+'/actions/'+endpoint]);(out/label).write_bytes(raw);return json.loads(raw)
    sources={'baseline':protocol['baseline']['source'],'candidate':args.candidate_source}
    selections={};manifests={};pythons={}
    env=os.environ.copy();env.update({k:'1' for k in POOLS});env.update(PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('PYTHONPATH',None);env.pop('SKIMAGEXX_DISABLE_SIMD',None)
    for role in ('baseline','candidate'):
        run_id=protocol['baseline']['run'] if role=='baseline' else args.candidate_run
        artifact_id=protocol['baseline']['artifact'] if role=='baseline' else args.candidate_artifact
        record=api('runs/'+str(run_id),role+'-run.json');listing=api('runs/'+str(run_id)+'/artifacts?per_page=100',role+'-artifacts.json')
        selection=validate_selection(record,listing,role,sources[role],run_id,artifact_id,args.repository,protocol)
        if role=='candidate':validate_prepare_jobs(api('runs/'+str(run_id)+'/jobs?per_page=100','candidate-jobs.json'))
        archive=out/(role+'.zip')
        with archive.open('wb') as stream:subprocess.run(['gh','api','repos/'+args.repository+'/actions/artifacts/'+str(artifact_id)+'/zip'],stdout=stream,check=True)
        root=args.snapshots.resolve()/role;assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==sources[role]
        extract_prepared(archive,root/'build',selection)
        prepared=root/'build/prepared';manifest=load(prepared/'ci-artifacts/measurement.json');assert manifest['source']['revision']==sources[role] and len(manifest['wheels'])==14
        venv=out.parent/('.region-stain-'+role+'-venv');assert not venv.exists()
        run([sys.executable,'-m','venv',venv],out/(role+'-venv-create'))
        python=venv/'bin/python';run([python,'-m','pip','install','--no-index','--find-links',prepared/'wheels','--require-hashes','-r',prepared/'ci-artifacts/runtime-lock.txt'],out/(role+'-install'),env=env)
        run([python,root/'tools/ci_measurement_artifacts.py','verify','--wheel-dir',prepared/'wheels','--artifact',prepared/'ci-artifacts/measurement.json'],out/(role+'-verify-before'),cwd=root,env=env)
        selections[role]=selection;manifests[role]=manifest;pythons[role]=python
    manifests_equal(manifests['baseline'],manifests['candidate'])
    def launch(role,family,output,preflight=False):
        cmd=[pythons[role],Path(__file__).resolve(),'driver','--contract',CONTRACT,'--source-root',args.snapshots.resolve()/role,'--family',family,'--output',output,'--deadline-utc',args.deadline_utc]
        if preflight:cmd+=['--preflight']
        assert datetime.now(timezone.utc)<datetime.fromisoformat(args.deadline_utc)
        run(cmd,output.with_suffix('.process'),cwd=args.snapshots.resolve()/role,env=env)
        assert datetime.now(timezone.utc)<datetime.fromisoformat(args.deadline_utc)
    families=protocol.get('families',['region','half','holdouts'])
    for role in ('baseline','candidate'):
        for family in families:launch(role,family,out/(role+'-'+family+'-preflight.json'),True)
    (out/'all-runtimes-preflight-complete.json').write_text(json.dumps({'roles':2,'first_cohorts':2*len(families),'timed_batches':0})+'\n')
    # No network, installation, builds or tests occur below until all four
    # measurement passes terminate. Raw failures survive upload(always).
    passes=[];identity=[]
    ids={family:[x['id'] for x in protocol[{'region':'region_cases','scalars':'scalar_cases','half':'half_cases','holdouts':'holdout_cases'}[family]]]for family in families}
    for index,role in enumerate(protocol['passes'],1):
        combined={}
        for family in families:
            output=out/(str(index)+'-'+role+'-'+family+'.json');launch(role,family,output)
            report=load(output);m=report['metadata'];expected=manifests[role]['runtime']['packages_sha256']['skimagexx']
            core=[value for name,value in expected.items() if name.startswith('_core.') and name.endswith('.so')];assert len(core)==1 and m['core_sha256']==core[0]
            assert m['deadline_utc']==args.deadline_utc
            common={k:m[k] for k in ('cpu_model','logical_cpu_count','avx2_available','python','numpy','scipy','reference','method_sha256')}
            identity.append(common);combined.update(validate_report(report,output.with_suffix('.batches.jsonl'),ids[family]))
        passes.append(combined)
    assert all(x==identity[0] for x in identity)
    for role in ('baseline','candidate'):
        root=args.snapshots.resolve()/role;prepared=root/'build/prepared'
        run([pythons[role],root/'tools/ci_measurement_artifacts.py','verify','--wheel-dir',prepared/'wheels','--artifact',prepared/'ci-artifacts/measurement.json'],out/(role+'-verify-after'),cwd=root,env=env)
    result=acceptance(passes,protocol);(out/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['decision']=='KEEP','fixed scientific gates reject this candidate; retain all losses'


def main(argv=None):
    global CONTRACT
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('execute');e.add_argument('--repository',required=True);e.add_argument('--candidate-source',required=True);e.add_argument('--candidate-run',required=True);e.add_argument('--candidate-artifact',required=True,type=int);e.add_argument('--snapshots',type=Path,required=True);e.add_argument('--results',type=Path,required=True);e.add_argument('--deadline-utc',required=True)
    d=sub.add_parser('driver');d.add_argument('--source-root',type=Path,required=True);d.add_argument('--family',choices=('region','scalars','half','holdouts'),required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--deadline-utc',required=True);d.add_argument('--preflight',action='store_true')
    e.add_argument('--contract',type=Path,default=CONTRACT);d.add_argument('--contract',type=Path,default=CONTRACT)
    args=p.parse_args(argv);CONTRACT=args.contract.resolve()
    deadline=datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset() is not None and deadline.utcoffset().total_seconds()==0,'deadline must be UTC'
    if args.command=='execute':execute(args)
    else:driver(args)


if __name__=='__main__':main()
