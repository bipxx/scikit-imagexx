"""Prospective exact public label-pair metrics; fixed one-worker sustained sampler."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json'


def source_safe(spec):
    assert spec['api'] in ('metrics.adapted_rand_error','metrics.contingency_table')
    assert type(spec['id']) is str and spec['layout'] in ('C','F')
    shape=spec['shape'];assert type(shape) is list and len(shape)==2
    assert all(type(x)is int and 1<=x<=512 for x in shape)
    count=math.prod(shape);assert 6<=count<=262144
    assert type(spec['labels'])is int and 2<=spec['labels']<=64
    assert count<=2**20 and spec['labels']-1<2**20
    assert spec['dtype']=='int64' and spec['normalize'] in (False,True)
    assert count>=2*spec['labels'] and count-math.ceil(count/spec['labels'])>spec['labels']
    assert spec['ignore']==[0] and spec['alpha']==.5
    # Both sparse axes <=64, COO rows=N, duplicate sort/coalescing at most N;
    # no external sparse table or malformed Cython buffer/helper input.
    # Integer pair sums <=N*(N-1)<2**36, exactly represented in float64.
    return dict(pixels=count,coordinate_bytes=16*count,max_CSR_shape=[spec['labels']]*2,
                max_CSR_logical_slots=spec['labels']**2,source_pair_sum_bound=count*(count-1))


def fixture(spec,np):
    source_safe(spec);n=math.prod(spec['shape']);base=np.arange(n,dtype=np.int64)
    truth=(base%spec['labels']).reshape(spec['shape'])
    test=((base//3+base%7)%spec['labels']).reshape(spec['shape'])
    if spec['layout']=='F':truth=np.asfortranarray(truth);test=np.asfortranarray(test)
    options={'ignore_labels':[0]}
    if spec['api']=='metrics.contingency_table':options['normalize']=spec['normalize']
    else:options['alpha']=spec['alpha']
    return (truth,test),options


def input_state(arguments):
    return [(a.dtype.str,a.dtype.descr,a.dtype.metadata,a.shape,a.strides,bool(a.flags.writeable),
             bool(a.flags.owndata),a.tobytes(order='C'))for a in arguments]


def output_snapshot(value):
    if type(value)is tuple:
        return dict(container='tuple',items=[dict(type=type(v).__name__,dtype=v.dtype.str,words=v.tobytes().hex())for v in value])
    arrays={}
    for name in ('indptr','indices','data'):
        a=getattr(value,name);base=a.base
        arrays[name]=dict(dtype=a.dtype.str,metadata=a.dtype.metadata,shape=a.shape,strides=a.strides,
            owning=bool(a.flags.owndata),writeable=bool(a.flags.writeable),aligned=bool(a.flags.aligned),
            C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),base_type=type(base).__name__,
            base_shape=getattr(base,'shape',None),base_strides=getattr(base,'strides',None),words=a.tobytes(order='C').hex())
    return dict(type=type(value).__module__+'.'+type(value).__name__,format=value.format,shape=value.shape,
        dtype=value.dtype.str,metadata=value.dtype.metadata,has_sorted_indices=value.__dict__.get('_has_sorted_indices'),
        has_canonical_format=value.__dict__.get('_has_canonical_format'),arrays=arrays)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--api',choices=['metrics.adapted_rand_error','metrics.contingency_table'])
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--output',type=Path,default=Path('build/ci-label-pair-metrics.json'))
    args=parser.parse_args();deadline=args.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists()and not journal.exists()
    sys.path.insert(0,str(ROOT/'tools'))
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import label_pair_default_error
    contract_raw=CONTRACT.read_bytes();contract=json.loads(contract_raw)
    assert skimage.__version__=='0.26.0' and np.__version__ in('1.24.4','2.5.4')
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    upstream={n:importlib.import_module('skimage.metrics.'+m)for n,m in(('contingency_table','_contingency_table'),('adapted_rand_error','_adapted_rand_error'))}
    for name,module in upstream.items():assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['upstream_sources']['skimage/metrics/'+module.__name__.rsplit('.',1)[1]+'.py']
    original=importlib.import_module('skimage.metrics');owned=importlib.import_module('skimagexx.metrics')
    wrappers={n:importlib.import_module('skimagexx.'+m)for n,m in(('contingency_table','_contingency_table'),('adapted_rand_error','_label_pair_metrics'))}
    limits={k:os.environ[k]for k in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')};assert set(limits.values())=={'1'}
    assert xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    getter=xx._core.get_num_threads;assert inspect.isbuiltin(getter)
    before=source_fingerprint(ROOT);core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    np_core=importlib.import_module('numpy._core._multiarray_umath'if np.__version__=='2.5.4'else'numpy.core._multiarray_umath')
    sparse_core=importlib.import_module('scipy.sparse._sparsetools')
    runtime_binaries={m.__name__:dict(path=str(m.__file__),sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())for m in(np_core,sparse_core)}
    cpu_model=platform.processor();info=Path('/proc/cpuinfo')
    if info.is_file():cpu_model=next((s.partition(':')[2].strip()for s in info.read_text().splitlines()if s.startswith('model name')),cpu_model)
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    selected=[s for s in contract['cases']if args.api is None or s['api']==args.api]
    for index,spec in enumerate(selected):
        bounds=source_safe(spec);arguments,options=fixture(spec,np);state=input_state(arguments)
        name=spec['api'].split('.')[1];reference=getattr(original,name);native=getattr(owned,name);wrapper=wrappers[name]
        assert reference is getattr(upstream[name],name) and native is getattr(wrapper,name)
        assert str(inspect.signature(reference))==str(inspect.signature(native))
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');expected=reference(*arguments,**options)
        assert not notices and input_state(arguments)==state
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        names=['label_pair_counts_preweighted'] if name=='contingency_table' else ['label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error']
        entries={getattr(xx._core,n):n for n in names};assert all(inspect.isbuiltin(x)for x in entries)
        calls=[];accepted=[];snapshots=[]
        def denied(*a,**k):raise AssertionError('original metric/numerical CSR forwarding forbidden')
        original_codes={getattr(m,n).__code__ for n,m in upstream.items()}
        def profile(frame,event,value):
            assert not(event=='call'and frame.f_code in original_codes),'cached original algorithm called'
            if event=='c_return'and value in entries:calls.append(entries[value])
            if event=='c_return'and value is getter:snapshots.append('get_num_threads')
            probe=wrapper.contingency_table if name=='contingency_table'else wrapper._are_from_i64
            if event=='return'and frame.f_code is probe.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
        # Replace public/private whole algorithms, restore all instrumentation before sampler.
        patches=[(original,n,getattr(original,n))for n in upstream]
        patches +=[(m,n,getattr(m,n))for n,m in upstream.items()]
        if hasattr(upstream['adapted_rand_error'],'contingency_table'):patches.append((upstream['adapted_rand_error'],'contingency_table',upstream['adapted_rand_error'].contingency_table))
        prior_profile=sys.getprofile();fallback=_compat.fallback
        from scipy import sparse
        csr=sparse.csr_array
        def structural(arg,*a,**k):
            assert type(arg)is tuple and len(arg)==3,'coordinate numerical constructor is not owned'
            return csr(arg,*a,**k)
        try:
            for module,key,_ in patches:setattr(module,key,denied)
            _compat.fallback=denied;sparse.csr_array=structural;sys.setprofile(profile)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');actual=native(*arguments,**options)
        finally:
            sys.setprofile(prior_profile);sparse.csr_array=csr;_compat.fallback=fallback
            for module,key,value in patches:setattr(module,key,value)
        expected_calls=['label_pair_counts_preweighted']if name=='contingency_table'else['label_pair_rand_terms','label_pair_rand_divide','label_pair_rand_divide','label_pair_rand_fscore','label_pair_rand_error']
        assert calls==expected_calls and accepted==[True]and snapshots==['get_num_threads']and not notices
        error=label_pair_default_error(spec['api'],actual,expected)
        assert input_state(arguments)==state and xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        timing=measure(lambda:reference(*arguments,**options),lambda:native(*arguments,**options),args.repeat,100,journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert input_state(arguments)==state and xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        rows.append(dict(case_id=spec['id'],api=spec['api'],parameters=spec,bounds=bounds,threads=1,backend='native-partial',actual_C_returns=calls,native_accepted=True,global_worker_snapshots=snapshots,max_abs_error=error,warnings=[],output=output_snapshot(actual),output_kind='whole-CSR-storage'if name=='contingency_table'else'exact-f64-scalar-tuple',**timing))
    assert len(rows)==len(selected) and sum(sum(map(len,r['samples_ms']))for r in rows)==14*len(selected)
    assert CONTRACT.read_bytes()==contract_raw and source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),cpu_model=cpu_model,logical_cpu_count=os.cpu_count(),avx2_available=bool(xx._core.has_avx2()),numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,**before,core_sha256=core_sha,runtime_binaries=runtime_binaries,threads=1,repeat=7,target_ms=100,warmup_calibration_min_ms=250,reference_pool_limits=limits,fallback_enabled=False,deadline_utc=deadline,raw_samples=14*len(selected),selected_api=args.api,scope=contract['scope'],contract_sha256=hashlib.sha256(contract_raw).hexdigest(),method_sha256=contract['method_sha256'],harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
