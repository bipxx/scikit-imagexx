"""Prospective diagnostic only. Do not execute before root authorizes timing."""
import argparse,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--authorize-diagnostic-timing',action='store_true')
    mode.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    folder=Path(__file__).resolve().parent
    contract=json.loads((folder/'diagnostic-contract.json').read_bytes())
    assert digest(folder/'adapter.cpp')==contract['adapter_sha256']
    assert digest(__file__)==contract['runner_sha256']
    source=args.source_root.resolve()
    assert hashlib.sha256((source/'benchmarks/montage.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['harness_sha256']
    assert digest(source/'coverage/contracts/montage-benchmark.json')==contract['workload_raw_sha256']
    assert all(os.environ[key]=='1'for key in contract['pool_variables'])
    affinity=None
    if hasattr(os,'sched_getaffinity'):
        permitted=sorted(os.sched_getaffinity(0));assert permitted
        os.sched_setaffinity(0,{permitted[0]});affinity=sorted(os.sched_getaffinity(0))
        assert affinity==[permitted[0]]
    import numpy as np,skimagexx as xx,_montage_diagnostic as adapter
    assert digest(xx._core.__file__)==contract['genuine_core_sha256']
    xx.set_num_threads(1);assert xx.get_num_threads()==1
    spec=importlib.util.spec_from_file_location('frozen_montage_fixture',source/'benchmarks/montage.py')
    fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
    specs=fixture_module.case_specs();assert [x['case_id']for x in specs]==contract['case_ids']
    schedule=random.Random(contract['seed']);results=[]
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    for index,case in enumerate(specs):
        image,options=fixture_module.fixture(case,np,index)
        logical=image[...,None]if case['channel_axis']is None else np.moveaxis(image,case['channel_axis'],-1)
        # Exact benchmark inputs are nonnegative small integer words. The sole
        # f64 mean is an exactly rounded integer-rational independent provider.
        logical_bytes=logical.tobytes()
        expected,shape=fixture_module.oracle_tiles(logical_bytes,logical.shape,case)
        if case['fill']=='mean':
            from fractions import Fraction
            import struct
            fill=float(Fraction(sum(int(v[0])for v in struct.iter_unpack('=d',logical_bytes)),logical.size))
        else:fill=case['fill']
        typed_fill=np.atleast_1d(fill).astype(logical.dtype)
        output=np.empty(tuple(shape)+( (1,)if len(shape)==2 else()),dtype=logical.dtype)
        assert output.flags.c_contiguous and output.flags.owndata
        assert not np.shares_memory(output,logical)and not np.shares_memory(output,typed_fill)
        assert logical.dtype==typed_fill.dtype==output.dtype
        before=logical.tobytes(),typed_fill.tobytes()
        prepared=adapter.prepare(str(Path(xx._core.__file__).resolve()),logical,typed_fill,output,*case['grid'],case['padding'])
        identity=adapter.identity(prepared)
        # Both paths use the same held descriptors/pointers and write the same
        # output object. Warmup and all byte comparisons are outside timing.
        for route in (0,1):
            for _ in range(contract['warmup_calls_per_route']):adapter.call(prepared,route,1)
            assert output.tobytes()==expected
        records=[];loops=contract['loops'][index]
        if not args.preflight_only:
            for round_id in range(contract['rounds']):
                pairs=[('wrapper_generic',[('wrapper',0),('generic',1)]),('generic_null',[('generic_A',1),('generic_B',1)])]
                schedule.shuffle(pairs)
                for kind,labels in pairs:
                    order=labels.copy();schedule.shuffle(order)
                    for ordinal,(label,route)in enumerate(order):
                        assert adapter.identity(prepared)==identity
                        start=time.perf_counter_ns();adapter.call(prepared,route,loops);elapsed=time.perf_counter_ns()-start
                        # Includes one identical Python adapter crossing per batch;
                        # no allocation/oracle/public-Python call in timed region.
                        records.append(dict(kind=kind,round=round_id,ordinal=ordinal,label=label,route=route,loops=loops,elapsed_ns=elapsed,ns_per_call=elapsed/loops))
                        assert output.tobytes()==expected
        assert(logical.tobytes(),typed_fill.tobytes())==before
        assert adapter.identity(prepared)==identity and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        results.append(dict(case_id=case['case_id'],identity=identity,input_shape=list(logical.shape),input_strides=list(logical.strides),dtype=logical.dtype.str,fill_sha256=hashlib.sha256(before[1]).hexdigest(),output_shape=list(output.shape),oracle_sha256=hashlib.sha256(expected).hexdigest(),samples=records))
    assert digest(xx._core.__file__)==contract['genuine_core_sha256']
    report=dict(status='PREFLIGHT_ONLY_PASS'if args.preflight_only else'DIAGNOSTIC_COMPLETE',case_count=len(results),oracle_route_checks=2*len(results),warmup_calls=2*len(results)*contract['warmup_calls_per_route'],raw_sample_count=sum(len(x['samples'])for x in results),timing_calls=0 if args.preflight_only else sum(len(x['samples'])for x in results),original_algorithm_calls=0,python=sys.version,scope=contract['scope'],decision='DIAGNOSTIC_ONLY_NO_KEEP_DISCARD',affinity=affinity,unpinned=affinity is None,pid=os.getpid(),contract_sha256=digest(folder/'diagnostic-contract.json'),adapter_binary_sha256=digest(adapter.__file__),core_sha256=digest(xx._core.__file__),results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
