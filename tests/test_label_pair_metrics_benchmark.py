"""Prospective stdlib-only safety; recording factory is not NumPy materialization."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
CONTRACT=json.loads((ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json').read_bytes())


def driver():
    spec=importlib.util.spec_from_file_location('label_pair_benchmark_probe',ROOT/'benchmarks/label_pair_metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_label_pair_benchmark_sources_method_and_bounds():
    cases=CONTRACT['cases'];assert len(cases)==len({s['id']for s in cases})==8
    assert (CONTRACT['threads'],CONTRACT['repeat'],CONTRACT['target_ms'],CONTRACT['warmup_min_ms'])==(1,7,100,250)
    assert CONTRACT['raw_samples']==112 and CONTRACT['warmup_groups']==16
    assert {s['api']for s in cases}=={'metrics.adapted_rand_error','metrics.contingency_table'}
    bounds=[driver().source_safe(s)for s in cases];assert max(b['pixels']for b in bounds)==262144
    assert max(b['source_pair_sum_bound']for b in bounds)<2**36
    spec=importlib.util.spec_from_file_location('label_pair_benchmark_context',ROOT/'tools/label_pair_benchmark_context.py')
    context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)
    context.validate_source_closure(ROOT,CONTRACT)
    text=(ROOT/'tools/montage_sustained.py').read_text();fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256(''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]).encode()).hexdigest()==CONTRACT['method_sha256']


def test_label_pair_benchmark_import_refuses_science(monkeypatch):
    import builtins
    old=builtins.__import__;attempts=[]
    def guarded(name,*args,**kwargs):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError(name)
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded);assert callable(driver().main)and attempts==[]


@pytest.mark.parametrize('case',CONTRACT['cases'],ids=lambda s:s['id'])
def test_label_pair_benchmark_recording_factory_original_refused(case):
    calls=[]
    class Array:
        def __mod__(self,n):calls.append(('mod',n));return self
        def __floordiv__(self,n):calls.append(('floordiv',n));return self
        def __add__(self,other):calls.append(('add',));return self
        def reshape(self,shape):calls.append(('reshape',shape));return self
    class Recorder:
        int64='int64'
        def arange(self,n,dtype):calls.append(('arange',n,dtype));return Array()
        def asfortranarray(self,a):calls.append(('fortran',));return a
    args,kwargs=driver().fixture(case,Recorder());assert len(args)==2 and kwargs['ignore_labels']==[0]
    assert calls[0]==('arange',case['shape'][0]*case['shape'][1],'int64')
    assert sum(c[0]=='fortran'for c in calls)==(2 if case['layout']=='F' else 0)
    assert kwargs==({'ignore_labels':[0],'normalize':case['normalize']}if case['api']=='metrics.contingency_table'else{'ignore_labels':[0],'alpha':.5})


@pytest.mark.parametrize('fault',('shape','huge','labels','dtype','ignore','alpha','layout'))
def test_label_pair_benchmark_rejects_unaudited_specs(fault):
    spec=copy.deepcopy(CONTRACT['cases'][0])
    if fault=='shape':spec['shape']=[6]
    elif fault=='huge':spec['shape']=[513,512]
    elif fault=='labels':spec['labels']=2**20
    elif fault=='dtype':spec['dtype']='float64'
    elif fault=='ignore':spec['ignore']=[-1]
    elif fault=='alpha':spec['alpha']=float('nan')
    else:spec['layout']='strided'
    with pytest.raises(AssertionError):driver().source_safe(spec)


def test_label_pair_benchmark_restores_instrumentation_before_real_measure():
    source=(ROOT/'benchmarks/label_pair_metrics.py').read_text();ast.parse(source)
    assert source.index('finally:')<source.index('timing=measure(')
    assert 'sys.setprofile(prior_profile);sparse.csr_array=csr;_compat.fallback=fallback'in source
    assert 'lambda:reference(*arguments,**options),lambda:native(*arguments,**options)'in source
    assert 'original metric/numerical CSR forwarding forbidden'in source
    assert 'source_fingerprint(ROOT)==before'in source and 'CONTRACT.read_bytes()==contract_raw'in source
    assert "frame.f_locals.get('native_result')is not None"in source
    assert 'label_pair_default_error(spec['in source


def test_label_pair_active_CLI_keeps_contract_and_uses_predecessor_guard():
    text=(ROOT/'benchmarks/label_pair_metrics.py').read_text()
    assert "CONTRACT=ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json'" in text
    assert 'from label_pair_benchmark_context import validate_source_closure' in text
    assert text.index('validate_source_closure(ROOT,contract)')<text.index('expected=reference(*arguments,**options)')
    assert 'contract_sha256=hashlib.sha256(contract_raw).hexdigest()' in text
    raw=(ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='5b4f3935966aa048eb2d06666a1ea6f548a43d08edd7d43ff68ccd8a211bdfa6'


@pytest.mark.parametrize('fault',('metrics-checker','driver-fixture','resealed-context'))
def test_label_pair_predecessor_context_rejects_tampering(monkeypatch,fault):
    spec=importlib.util.spec_from_file_location('label_pair_benchmark_context_negative',ROOT/'tools/label_pair_benchmark_context.py')
    context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)
    read=Path.read_bytes
    if fault=='metrics-checker':
        target=ROOT/'python/skimagexx/coverage.py';raw=read(target)
        old=b'assert a.dtype==e.dtype and a.tobytes()==e.tobytes()';assert raw.count(old)==1
        raw=raw.replace(old,b'assert a.dtype==e.dtype',1)
    elif fault=='driver-fixture':
        target=ROOT/'benchmarks/label_pair_metrics.py';raw=read(target)
        old=b'truth=(base%spec[\'labels\']).reshape(spec[\'shape\'])';assert raw.count(old)==1
        raw=raw.replace(old,b'truth=(base%2).reshape(spec[\'shape\'])',1)
    else:
        target=ROOT/'coverage/contracts/label-pair-metrics-benchmark-context-v1.json';value=json.loads(read(target))
        value['sources']['python/skimagexx/coverage.py']['predecessor_canonical_lf_sha256']='0'*64
        raw=json.dumps(value).encode()
    monkeypatch.setattr(Path,'read_bytes',lambda path:raw if path==target else read(path))
    with pytest.raises(AssertionError):context.validate_source_closure(ROOT,CONTRACT)
