"""Offline report-only validator; no measurement, helper, native or scientific imports."""
import argparse,hashlib,json,subprocess
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/join-util-views-report-order-v5.json'
_READ_ONLY=None
_OUTPUT=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    assert _READ_ONLY is not None and _OUTPUT is not None
    assert Path(path).resolve()==_READ_ONLY/'summary.json'
    assert not Path(path).exists(),'Failed campaign summary must remain absent'
    with (_OUTPUT/'summary.json').open('x',encoding='utf-8')as stream:json.dump(value,stream,indent=2);stream.write('\n')

def validate_reports(snapshot,results,manifest,authorities):
    import math,statistics
    snapshot=Path(snapshot);results=Path(results);summaries={}
    context=json.loads((snapshot/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
    for cohort,route in manifest['first_cohort']['routes'].items():
        if cohort=='metrics-contingency':continue
        if route['kind']=='util-executor':specs=route['cases']
        else:specs=json.loads((snapshot/route['contract']).read_bytes())['cases']
        count=len(specs)
        report=json.loads((results/(cohort+'.json')).read_bytes());meta=report['metadata']
        assert {k:meta[k]for k in manifest['benchmark_fingerprint']}==manifest['benchmark_fingerprint']
        assert meta['core_sha256']==manifest['core_sha256']
        assert meta['threads']==1 and meta['repeat']==7 and meta['target_ms']==100
        assert meta['contract_sha256']==sha(snapshot/route['contract'])
        assert len(report['results'])==count
        if route['kind']=='util-executor':
            assert meta['root_binding_sha256']==sha(authorities[cohort]['binding'])
            assert meta['measurement_sha256']==sha(authorities[cohort]['measurement'])
            assert meta['runtime_sha256']==manifest['runtime_sha256']and meta['producer']==manifest['producer']
            assert meta['method_sha256']==context['method']['method_sha256']and meta['retain_all_losses']is True
        else:
            assert meta['raw_samples']==count*14
            assert set(meta['reference_pool_limits'].values())=={'1'}
        assert [row['case_id']for row in report['results']]==[spec['id']for spec in specs]
        journal=[json.loads(line)for line in (results/(cohort+'.batches.jsonl')).read_text().splitlines()]
        assert {event['case_id']for event in journal}=={spec['id']for spec in specs}
        for case_index,(row,spec) in enumerate(zip(report['results'],specs)):
            assert row['native_accepted'] is True
            if route['kind']!='util-executor':assert row['parameters']==spec
            else:
                assert row['api']==route['api']
                assert row['FP'].keys()=={'flags','rounding'}
                assert type(row['FP']['flags'])is int and type(row['FP']['rounding'])is int
            assert list(map(len,row['samples_ms']))==[7,7]
            assert all(math.isfinite(x)and x>0 for arm in row['samples_ms']for x in arm)
            assert list(map(len,row['loops']))==[7,7] and all(type(x)is int and x>0 for arm in row['loops']for x in arm)
            assert row['reference_ms']==statistics.median(row['samples_ms'][0])
            assert row['cpp_ms']==statistics.median(row['samples_ms'][1])
            assert row['speedup']==row['reference_ms']/row['cpp_ms']
            events=[event for event in journal if event['case_id']==row['case_id']]
            batches=row['batch_records']
            assert len(events)==2*len(batches)+1
            assert all(event['case_index']==case_index for event in events)
            timing_keys=('reference_ms','cpp_ms','speedup','samples_ms','loops','sample_labels','warmup_elapsed_ns','sample_order','batch_records')
            assert events[-1]['event']=='measure_returned' and events[-1]['timing']=={key:row[key]for key in timing_keys}
            warm=[0,0];accepted=[[],[]];counts=[[],[]];order=[]
            for before,after,batch in zip(events[0:-1:2],events[1:-1:2],batches):
                assert before['event']=='batch_before' and after==batch and after['event']=='batch_after'
                assert all(before[key]==after[key]for key in ('case_index','case_id','function','round','ordinal','loops'))
                assert after['complete'] is True and after['failure'] is None
                arm=after['function'];elapsed=after['elapsed_ns'];n=after['loops']
                assert arm in (0,1) and type(elapsed)is int and elapsed>0 and type(n)is int and n>0
                if after['phase']=='warmup':
                    assert before['requested_phase']=='warmup' and not after['accepted'] and after['round']==-1
                    warm[arm]+=elapsed
                elif after['phase']=='underlength-calibration':
                    assert before['requested_phase']=='sample' and not after['accepted'] and elapsed<100000000
                else:
                    assert before['requested_phase']=='sample' and after['phase']=='sample' and after['accepted'] and elapsed>=100000000
                    accepted[arm].append(elapsed/n/1e6);counts[arm].append(n)
                    order.append((after['round'],after['ordinal'],arm))
            assert warm==row['warmup_elapsed_ns'] and min(warm)>=250000000
            assert accepted==row['samples_ms'] and counts==row['loops']
            pass_index={'util-executor':0,'dedicated':1}[route['kind']]
            expected_order=[[0,1]if(round_id+case_index+pass_index)%2==0 else[1,0]for round_id in range(7)]
            assert row['sample_order']==expected_order
            assert order==[(round_id,ordinal,arm)for round_id,arms in enumerate(expected_order)for ordinal,arm in enumerate(arms)]
        summaries[cohort]=dict(raw_samples=count*14,
            wins=[row['case_id']for row in report['results']if row['speedup']>1],
            losses=[row['case_id']for row in report['results']if row['speedup']<1],
            ties=[row['case_id']for row in report['results']if row['speedup']==1])
    assert sum(v['raw_samples']for v in summaries.values())==manifest['execution_scope']['canonical_raw_samples']
    write_json(results/'summary.json',dict(families=summaries,scope='Current one-thread public original comparison only; all losses retained; no whole-product qualification or prior-C++ speed claim.'))


def local_authorities(results,manifest):
    """Map only path locations; payload bytes and their existing SHA stay exact."""
    results=Path(results);index=json.loads((results/'cohort-authorities.json').read_bytes())
    assert set(index)==set(manifest['first_cohort']['routes'])
    roles={'binding':'binding.json','go':'root-go.json','measurement':'measurement.json'}
    mapped={}
    for name,values in index.items():
        assert set(values)==set(roles)
        mapped[name]={}
        for role,filename in roles.items():
            remote=PurePosixPath(values[role]);assert remote.is_absolute()
            assert remote.parts[-2:]==('authority-'+name,filename)
            local=results/('authority-'+name)/filename;assert local.is_file()
            mapped[name][role]=str(local.resolve())
    return mapped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','results','output','root-admission'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--root-admission-sha256',required=True)
    args=parser.parse_args()
    # A separate root source decision is mandatory before retained-data replay.
    assert sha(args.root_admission)==args.root_admission_sha256
    admission=json.loads(args.root_admission.read_bytes())
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert admission['passed']is True and admission['source']==source
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    context=json.loads(CONTEXT.read_bytes());manifest_path=ROOT/context['manifest_path']
    assert sha(manifest_path)==context['manifest_raw_sha256'];manifest=json.loads(manifest_path.read_bytes())
    snapshot=args.snapshot.resolve();results=args.results.resolve();output=args.output.resolve()
    assert not output.exists()and output.is_relative_to(results.parents[1])and not output.is_relative_to(results.parent)
    assert not (results/'summary.json').exists(),'Preserve immutable failed results and absent summary'
    assert sha(results.parent/'retention-v1.json')==context['retention_sha256']
    for name,pin in context['source_pins_canonical'].items():
        assert hashlib.sha256((snapshot/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,record in context['raw_files'].items():
        path=results/name;assert path.stat().st_size==record['bytes']and sha(path)==record['sha256'],name
    authorities=local_authorities(results,manifest)
    global _READ_ONLY,_OUTPUT
    _READ_ONLY,_OUTPUT=results,output
    output.mkdir(parents=True,exist_ok=False)
    try:validate_reports(snapshot,results,manifest,authorities)
    except BaseException as error:
        with (output/'failure.json').open('x',encoding='utf-8')as stream:json.dump(dict(status='FAIL_REPORT_ONLY_VALIDATION',type=type(error).__name__,error=str(error),timing_retry=False),stream,indent=2)
        raise
    assert not (results/'summary.json').exists()
    receipt=dict(status='PASS_REPORT_ONLY_RETAINED_OUTPUT_VALIDATION',tool_source=source,root_admission_sha256=args.root_admission_sha256,manifest_sha256=context['manifest_raw_sha256'],retention_sha256=context['retention_sha256'],failed_hosted_campaign_status='failure',summary_sha256=sha(output/'summary.json'),reports=context['reports'],cases=context['cases'],samples=context['samples'],warmup_groups=context['warmup_groups'],authority_relocation='In-memory absolute Linux path to exact local file; original binding/measurement payloads unchanged.',new_original_calls=0,new_native_calls=0,new_helper_calls=0,new_sampler_calls=0,new_timing_runs=0)
    with (output/'validation-receipt.json').open('x',encoding='utf-8')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
