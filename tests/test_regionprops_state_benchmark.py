"""Small prospective metadata/import checks; no scientific workloads."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).parents[1]


def load_harness():
    spec=importlib.util.spec_from_file_location('prospective_regionprops_benchmark',ROOT/'benchmarks/regionprops_state.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_regionprops_benchmark_prospective_cases_and_bounds():
    module=load_harness();contract=json.loads((ROOT/'coverage/contracts/regionprops-strided-runtime-v1.json').read_bytes());cases=module.case_specs()
    assert len(cases)==12 and len({s['id']for s in cases})==12
    assert {s['operation']for s in cases}=={'constructor','seven-properties'}
    assert {len(s['shape'])for s in cases}=={2,3}and{v['cache']for v in cases}=={True,False}
    assert {'F','reverse','transpose'}<={s['layout']for s in cases}
    assert any(s['pattern']=='sparse'for s in cases)
    for spec in cases:
        bounds=module.case_bounds(spec)
        assert bounds['coords_elements']<=contract['max_coords_elements']
        assert bounds['validation_payload_bytes']<=contract['max_validation_payload_bytes']
    assert contract['threads']==1 and contract['repeat']==7 and contract['target_ms']==100
    assert contract['warmup_calibration_min_ms']==250 and contract['warmup_groups']==24 and contract['raw_samples']==168
    assert contract['method_sha256']==json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())['method_sha256']
    for name,pin in contract['source_closure'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


def test_regionprops_benchmark_import_refuses_scientific_side_effects(monkeypatch):
    original=builtins.__import__
    def checked(name,*args,**kwargs):
        assert name.split('.')[0]not in{'numpy','scipy','skimage','skimagexx'}
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)
    module=load_harness();assert callable(module.main)and len(module.case_specs())==12
    main=next(n for n in ast.parse((ROOT/'benchmarks/regionprops_state.py').read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    before_cases=main.body[:next(i for i,n in enumerate(main.body)if isinstance(n,ast.For)and ast.unparse(n.iter)=='enumerate(case_specs())')]
    published=next(n for n in ast.parse((ROOT/'benchmarks/line_aa.py').read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    def hardware(nodes):
        return [ast.dump(n,include_attributes=False)for n in nodes if
                isinstance(n,ast.Assign)and any(t.id in ('cpu_model','cpu_info','cpu_count')for target in n.targets for t in ast.walk(target)if isinstance(t,ast.Name))
                or isinstance(n,ast.If)and ast.unparse(n.test)=='cpu_info.is_file()']
    assert len(hardware(before_cases))==4 and hardware(before_cases)==hardware(published.body)
    report=next(n for n in main.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='report')
    names={k.arg for n in ast.walk(report)if isinstance(n,ast.Call)for k in n.keywords}
    assert {'cpu_model','logical_cpu_count','avx2_available'}<=names


def test_region_candidate_runtime_contract_preserves_history_and_timed_source():
    historical = json.loads((ROOT/'coverage/contracts/regionprops-state-benchmark.json').read_bytes())
    candidate = json.loads((ROOT/'coverage/contracts/regionprops-strided-runtime-v1.json').read_bytes())
    assert set(candidate) == set(historical)
    assert {key for key in candidate if candidate[key] != historical[key]} <= {'source_closure', 'scope', 'status'}
    assert candidate['cases'] == historical['cases']
    assert set(candidate['source_closure']) == set(historical['source_closure']) | {'python/skimagexx/_regionprops_state_class.py'}
    source = (ROOT/'benchmarks/regionprops_state.py').read_bytes().replace(b'\r\n',b'\n')
    assert source.count(b'regionprops-strided-runtime-v1.json') == 1
    restored = source.replace(b'regionprops-strided-runtime-v1.json',b'regionprops-state-benchmark.json')
    assert hashlib.sha256(restored).hexdigest() == '12c237328a25b462362b19551af70b52dc0868a66bc72c1a4ecfbe5755a39587'
