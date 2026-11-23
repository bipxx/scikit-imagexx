"""Explicit sustained sampler and exact frozen Montage validation closure."""
import argparse,hashlib,importlib,importlib.util,inspect,json,math,os,platform,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone


def measure(reference,native,repeat,target_ms,*,journal,case_index,case_id,pass_index,deadline):
    assert repeat==7 and target_ms==100
    target=100_000_000;warm_target=250_000_000;functions=[reference,native]
    events=[];samples=[[],[]];counts=[[],[]];loops=[1,1];warmed=[0,0];orders=[]
    journal.parent.mkdir(parents=True,exist_ok=True)
    def emit(event):
        with journal.open('a')as stream:stream.write(json.dumps(event)+'\n');stream.flush()
    def take(index,n,phase,round_id,ordinal):
        assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline),'sealed deadline expired'
        common=dict(case_index=case_index,case_id=case_id,function=index,round=round_id,ordinal=ordinal,loops=n)
        emit(dict(common,event='batch_before',requested_phase=phase))
        start=time.perf_counter_ns()
        try:
            for _ in range(n):
                result=functions[index]()
                del result
        finally:
            elapsed=time.perf_counter_ns()-start;failure=sys.exc_info()[0]
            accepted=phase=='sample'and elapsed>=target and failure is None
            kind='sample'if accepted else'warmup'if phase=='warmup'else'underlength-calibration'
            event=dict(common,event='batch_after',phase=kind,elapsed_ns=elapsed,complete=failure is None,accepted=accepted,failure=None if failure is None else failure.__name__)
            events.append(event);emit(event)
        assert elapsed>0
        return elapsed
    for index in(0,1):
        elapsed=0
        while warmed[index]<warm_target or elapsed<target:
            elapsed=take(index,loops[index],'warmup',-1,index);warmed[index]+=elapsed
            loops[index]=max(1,math.ceil(loops[index]*target/elapsed*1.15))
    for round_id in range(repeat):
        order=[0,1]if(round_id+case_index+pass_index)%2==0 else[1,0];orders.append(order)
        for ordinal,index in enumerate(order):
            while True:
                n=loops[index];elapsed=take(index,n,'sample',round_id,ordinal)
                loops[index]=max(n+1,math.ceil(n*target/elapsed*1.15))if elapsed<target else max(1,math.ceil(n*target/elapsed*1.15))
                if elapsed>=target:
                    samples[index].append(elapsed/n/1e6);counts[index].append(n);break
    a,b=map(statistics.median,samples)
    result=dict(reference_ms=a,cpp_ms=b,speedup=a/b,samples_ms=samples,loops=counts,sample_labels=['scikit-image','current'],warmup_elapsed_ns=warmed,sample_order=orders,batch_records=events)
    emit(dict(event='measure_returned',case_index=case_index,case_id=case_id,status='PENDING_FINAL_HARNESS_VALIDATION',timing=result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7);parser.add_argument('--output',type=Path,default=Path('build/ci-montage.json'));parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--pass-index',type=int,choices=range(1,9),required=True);args=parser.parse_args()
    ROOT=args.source_root.resolve();CONTRACT=ROOT/'coverage/contracts/montage-benchmark.json'
    frozen_path=ROOT/'benchmarks/montage.py'
    assert hashlib.sha256(frozen_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd'
    loader=importlib.util.spec_from_file_location('unchanged_montage_helpers',frozen_path);helpers=importlib.util.module_from_spec(loader);loader.loader.exec_module(helpers)
    case_specs,fixture,snapshot,oracle_tiles=(getattr(helpers,name)for name in('case_specs','fixture','snapshot','oracle_tiles'))
    sys.path.insert(0,str(ROOT/'benchmarks'))
    protocol=json.loads(args.protocol.read_bytes())
    method_metadata=dict(frozen_harness_sha256='93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd',measurement_protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),method_sha256=hashlib.sha256(inspect.getsource(measure).encode()).hexdigest(),warmup_calibration_min_ms=250,interbatch_sleep_ms=0,pass_index=args.pass_index,method='sustained-adaptive-full-public-v1')
    from provenance import source_fingerprint
    import numpy as np,scipy,skimage,skimagexx as xx
    original=importlib.import_module('skimage.util._montage');implementation=importlib.import_module('skimagexx._montage')
    reference,native=skimage.util.montage,xx.util.montage
    raw_contract=CONTRACT.read_bytes();contract=json.loads(raw_contract)
    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_python_source_sha256']
    limits={key:os.environ[key]for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(1)
    entry=implementation._core.montage_apply;assert inspect.isbuiltin(entry)
    source_before=source_fingerprint(ROOT);core=Path(xx._core.__file__);core_hash=hashlib.sha256(core.read_bytes()).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    reference_code=inspect.unwrap(reference).__code__;native_code=inspect.unwrap(native).__code__
    for index,spec in enumerate(case_specs()):
        image,options=fixture(spec,np,index);before=snapshot(image)
        logical=image if spec['channel_axis']is None else np.moveaxis(image,spec['channel_axis'],-1)
        oracle,shape=oracle_tiles(logical.tobytes(),logical.shape,spec);assert math.prod(shape)<=contract['max_output_elements']
        expected=reference(image,**options);calls=[];accepted=[];prior=sys.getprofile()
        def reject(*a,**kw):raise AssertionError('owned montage forwarded to original whole API')
        def profile(frame,event,value):
            if event=='call'and frame.f_code is reference_code:reject()
            if event=='c_return'and value is entry:calls.append(value.__name__)
            if event=='return'and frame.f_code is native_code:accepted.append(frame.f_locals.get('native_result')is True)
        saved=skimage.util.montage,original.montage;skimage.util.montage=original.montage=reject;sys.setprofile(profile)
        try:actual=native(image,**options)
        finally:sys.setprofile(prior);skimage.util.montage,original.montage=saved
        assert calls==['montage_apply']and accepted==[True]
        a,b=snapshot(actual),snapshot(expected);assert a==b and a['raw']==oracle and a['shape']==shape
        assert not np.shares_memory(actual,image)and not np.shares_memory(expected,image)
        assert snapshot(image)==before and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,100,journal=args.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['case_id'],pass_index=args.pass_index,deadline=protocol['deadline_utc'])
        assert snapshot(image)==before and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(**spec,api='util.montage',backend='native',threads=1,actual_C_returns=calls,native_accepted=True,native_entry_module=entry.__module__,output_metadata={k:v for k,v in a.items()if k!='raw'},output_sha256=hashlib.sha256(a['raw']).hexdigest(),oracle_sha256=hashlib.sha256(oracle).hexdigest(),max_abs_error=0.0,**timing))
        print(spec['case_id']+': '+str(timing['speedup']),flush=True)
    assert len(rows)==9 and all(list(map(len,row['samples_ms']))==[7,7]for row in rows)
    source_after=source_fingerprint(ROOT);assert source_before==source_after and CONTRACT.read_bytes()==raw_contract
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_hash
    report=dict(metadata=dict(**method_metadata,timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**source_after,source_before=source_before,core_path=str(core),core_sha256=core_hash,harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),contract_sha256=hashlib.sha256(raw_contract).hexdigest(),original_source_sha256=contract['original_python_source_sha256'],threads=1,repeat=7,target_ms=100,raw_samples=126,reference_pool_limits=limits,fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,selected_apis=['util.montage'],provider_frontier=contract['provider_frontier'],scope=contract['scope'],timed_boundary=contract['timed_boundary']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Montage: one-thread comparison','',contract['provider_frontier'],'','| Case | Original ms | Current ms | Original/current |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['case_id']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
