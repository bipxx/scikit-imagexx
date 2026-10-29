"""Pinned whole-public colocalization driver; fixed prospective21-case protocol."""
import argparse, hashlib, importlib.util, inspect, json, os, statistics, sys, traceback
from datetime import datetime, timezone
from pathlib import Path


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_report(report,events,contract,pass_index):
    assert report['status']=='COMPLETE'
    meta=report['metadata'];rows=report['results'];checks=report['checks']
    assert meta['threads']==1 and meta['repeat']==7 and meta['target_ms']==100 and meta['raw_samples']==294 and meta['pass_index']==pass_index
    assert meta['method_sha256']==contract['measurement']['method_sha256'] and meta['case_spec_sha256']==contract['case_spec_sha256']
    assert meta['helpers_sha256']==contract['validation']['helpers_canonical_lf_sha256']
    assert meta['protocol_sha256']=='4f455ef6489dc7207fa2982bdd1c0b7ca3f2412f225a7d0a5f12c1a6ca731885'
    assert [r['id']for r in rows]==[r['id']for r in checks]==[r['id']for r in contract['case_specs']]
    assert all(c['status']=='passed'and c['reference']==c['current']and c['reference_warnings']==c['current_warnings']and c['reference_error']is None and c['current_error']is None for c in checks)
    assert all(c['native_witness']['forward_attempts']==0 and c['native_witness']['c_returns']and c['admitted_stage_returns']>0 for c in checks)
    starts=[e for e in events if e['event']=='batch_before'];ends=[e for e in events if e['event']=='batch_after']
    returned=[e for e in events if e['event']=='measure_returned']
    assert events[0]['event']=='driver_started' and events[-1]==dict(event='driver_complete',cases=21)
    assert len(starts)==len(ends) and len(returned)==21
    assert ends==[e for r in rows for e in r['batch_records']]
    assert [e['case_index']for e in events if e['event']=='case_prechecked']==list(range(21))
    assert [e['case_index']for e in events if e['event']=='case_validated']==list(range(21))
    for before,after in zip(starts,ends):
        assert all(before[k]==after[k]for k in ('case_index','case_id','function','round','ordinal','loops'))
        assert after['complete']is True and after['failure']is None
        assert after['accepted']==(before['requested_phase']=='sample'and after['elapsed_ns']>=100000000)
    for i,row in enumerate(rows):
        assert row['api']==contract['case_specs'][i]['api'] and row['threads']==1
        assert list(map(len,row['samples_ms']))==list(map(len,row['loops']))==[7,7]
        assert row['sample_labels']==['scikit-image','current']
        expected=[[0,1]if(r+i+pass_index)%2==0 else[1,0]for r in range(7)]
        assert row['sample_order']==expected
        for fn in (0,1):
            accepted=[e for e in row['batch_records']if e['accepted']and e['function']==fn]
            assert len(accepted)==7 and [e['round']for e in accepted]==list(range(7))
            assert [e['ordinal']for e in accepted]==[o.index(fn)for o in expected]
            assert all(e['elapsed_ns']>=100000000 and type(e['loops'])is int and e['loops']>0 for e in accepted)
            assert [e['loops']for e in accepted]==row['loops'][fn]
            assert [e['elapsed_ns']/e['loops']/1e6 for e in accepted]==row['samples_ms'][fn]
            warm=sum(e['elapsed_ns']for e in row['batch_records']if e['phase']=='warmup'and e['function']==fn)
            assert warm==row['warmup_elapsed_ns'][fn] and warm>=250000000
        a,b=map(statistics.median,row['samples_ms']);assert a>0 and b>0 and (a,b,a/b)==(row['reference_ms'],row['cpp_ms'],row['speedup'])
        event=returned[i];assert event['case_index']==i and event['case_id']==row['id'] and event['status']=='PENDING_FINAL_HARNESS_VALIDATION'
        assert all(row[k]==v for k,v in event['timing'].items())
    assert datetime.fromisoformat(meta['completed_utc'])<=datetime.fromisoformat(contract['deadline_utc'])


def compare(reports,contract):
    assert len(reports)==4
    indexed=[{r['id']:r for r in report['results']}for report in reports]
    rows=[];target_repeat=[];control_floors=[];original_wins=[]
    gm=statistics.geometric_mean
    for spec in contract['case_specs']:
        case=spec['id'];passes=[data[case]for data in indexed]
        a_cpp=gm([passes[0]['cpp_ms'],passes[3]['cpp_ms']]);b_cpp=gm([passes[1]['cpp_ms'],passes[2]['cpp_ms']])
        a_ratio=gm([passes[0]['speedup'],passes[3]['speedup']]);b_ratio=gm([passes[1]['speedup'],passes[2]['speedup']])
        direct=a_cpp/b_cpp;normalized=b_ratio/a_ratio
        each_direct=[a_cpp/passes[i]['cpp_ms']for i in (1,2)]
        each_normalized=[passes[i]['speedup']/a_ratio for i in (1,2)]
        if case in contract['target_ids']:target_repeat.extend(v>=1.05 for v in each_direct+each_normalized)
        else:control_floors.extend([direct>=.97,normalized>=.97])
        if case in contract['preserved_original_wins']:original_wins.extend(passes[i]['speedup']>1 for i in (1,2))
        rows.append(dict(id=case,role='target'if case in contract['target_ids']else'control',passes=passes,direct=direct,normalized=normalized,candidate_each_direct=each_direct,candidate_each_normalized=each_normalized,cpp_added_ms=b_cpp-a_cpp,candidate_original_losses=[i+1 for i in (1,2)if passes[i]['speedup']<=1],cpp_direct_loss=direct<1,cpp_normalized_loss=normalized<1))
    completed=datetime.now(timezone.utc).isoformat()
    gates=dict(target_repeated_direct_and_normalized=all(target_repeat),balanced_control_floors=all(control_floors),preserved_original_wins=all(original_wins),deadline=datetime.fromisoformat(completed)<=datetime.fromisoformat(contract['deadline_utc']))
    return dict(decision='KEEP'if all(gates.values())else'DISCARD',completed_utc=completed,gates=gates,raw_samples=1176,overall_direct=gm(r['direct']for r in rows),overall_normalized=gm(r['normalized']for r in rows),rows=rows,scope=contract['scope'],note='All source/identity/parity/FP/sample validation must pass before this comparison; no new noise/null gates.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pass-index',type=int,choices=range(1,5),required=True)
    args=parser.parse_args()
    root=args.source_root.resolve();raw=args.protocol.read_bytes();contract=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()=='4f455ef6489dc7207fa2982bdd1c0b7ca3f2412f225a7d0a5f12c1a6ca731885'
    assert not args.output.exists() and not sys.flags.optimize
    for name,pin in contract['validation']['helpers_canonical_lf_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    sys.path.insert(0,str(root/'benchmarks'))
    helpers=load(root/'benchmarks/phase3_public.py','immutable_coloc_helpers')
    sampler_path=Path(__file__).with_name('montage_sustained.py')
    assert hashlib.sha256(sampler_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['measurement']['method_source_driver_sha256']
    sampler=load(sampler_path,'immutable_sustained_sampler')
    assert hashlib.sha256(inspect.getsource(sampler.measure).encode()).hexdigest()==contract['measurement']['method_sha256']
    specs=[s for s in helpers.case_specs()if s['family']=='colocalization']
    assert specs==contract['case_specs'] and len(specs)==21
    assert hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()==contract['case_spec_sha256']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    journal=args.output.with_suffix('.batches.jsonl')
    def emit(event):
        with journal.open('a')as stream:stream.write(json.dumps(event)+'\n');stream.flush()
    report=dict(status='STARTED',metadata=dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),protocol_sha256=hashlib.sha256(raw).hexdigest(),method_sha256=contract['measurement']['method_sha256'],helpers_sha256=contract['validation']['helpers_canonical_lf_sha256'],case_spec_sha256=contract['case_spec_sha256'],pass_index=args.pass_index,threads=1,repeat=7,target_ms=100,raw_samples=294,scope=contract['scope']),checks=[],results=[])
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save();emit(dict(event='driver_started',metadata=report['metadata'],pid=os.getpid()))
    runtime=helpers._runtime();xx=runtime.xx
    previous_workers=xx.get_num_threads();previous_fallback=xx.set_fallback_enabled(False);xx.set_num_threads(1)
    try:
        with runtime.pool_limits(limits=1):
            pools=helpers._pools(runtime);before=helpers.source_fingerprint(root)
            wheel=helpers._verify_wheel(runtime,args.wheel);buffer=runtime.np.getbufsize()
            core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
            report['metadata'].update(source_before=before,core_sha256=core_sha,wheel=wheel,python=sys.version,numpy=runtime.np.__version__,scipy=runtime.scipy.__version__,reference=runtime.skimage.__version__,pools=pools,avx2=xx.has_avx2(),fallback_enabled=False)
            save()
            for index,spec in enumerate(specs):
                helpers._pools(runtime);check=dict(id=spec['id'],api=spec['api'],status='checking');report['checks'].append(check);save()
                helpers._check_case(runtime,spec,check);check['status']='passed';save()
                inputs,options=helpers._fixture(runtime,spec)
                assert helpers._snapshot((inputs,options),runtime.np)==check['input']
                original,native=runtime.functions[spec['api']]
                emit(dict(event='case_prechecked',case_index=index,case_id=spec['id'],input=check['input'],output=check['current']))
                timing=sampler.measure(lambda:original(*inputs,**options),lambda:native(*inputs,**options),7,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=args.pass_index,deadline=contract['deadline_utc'])
                helpers._pools(runtime)
                assert runtime.np.getbufsize()==buffer
                assert helpers._snapshot((inputs,options),runtime.np)==check['input']
                row=dict(id=spec['id'],api=spec['api'],threads=1,**timing)
                report['results'].append(row);save()
                emit(dict(event='case_validated',case_index=index,case_id=spec['id']))
                print(spec['id']+': '+str(timing['speedup']),flush=True)
            assert len(report['checks'])==len(report['results'])==21
            assert helpers.source_fingerprint(root)==before and args.protocol.read_bytes()==raw
            assert helpers._verify_wheel(runtime,args.wheel)==wheel
            assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
            report['metadata'].update(source_after=before,completed_utc=datetime.now(timezone.utc).isoformat())
            report['status']='COMPLETE';save();emit(dict(event='driver_complete',cases=21))
    except BaseException as error:
        report.update(status='FAILED',failure=dict(type=type(error).__name__,text=str(error),traceback=traceback.format_exc()));save();raise
    finally:
        xx.set_num_threads(previous_workers);xx.set_fallback_enabled(previous_fallback)


if __name__=='__main__':main()
