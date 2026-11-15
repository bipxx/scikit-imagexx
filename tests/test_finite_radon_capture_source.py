"""Source-only readiness contract; no NumPy/helper/target body invocation."""
import ast,hashlib,json
from pathlib import Path
import finite_radon_capture_controller as C
import finite_radon_contract as F
ROOT=Path(__file__).resolve().parents[1]

def test_context_preserves_source_pool_and_zero_budget():
    d=json.loads((ROOT/'coverage/contracts/finite-radon-capture-v1.json').read_bytes())
    assert d['original_budget']==0 and d['frozen_case_source']=='d2109de2a0b7f07e27f0a892585cd7414e36bfbf'
    assert d['case_ids']==[s['id']for s in F.cases()]and len(d['case_ids'])==108
    assert d['api_case_ids']=={'frt2':d['case_ids'][:54],'ifrt2':d['case_ids'][54:]}
    for n,p in d['source_files_raw_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==p,n

def test_actual_helpers_reused_as_source_bytes():
    d=json.loads((ROOT/'coverage/contracts/finite-radon-capture-v1.json').read_bytes());source=Path(C.__file__).read_text();tree=ast.parse(source)
    for name,pin in d['exact_helper_functions'].items():
        node=next(n for n in tree.body if getattr(n,'name',None)==name)
        assert hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()==pin

def test_gates_before_runtime_and_no_per_case_source_hash():
    source=Path(C.__file__).read_text();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assert source.index('q=validate(')<source.index('go=validate_go(')<source.index('import numpy as np')
    loop=next(n for n in ast.walk(main)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='spec'and isinstance(n.iter,ast.Name)and n.iter.id=='specs')
    assert 'source_files_raw_sha256'not in ast.unparse(loop)
    assert 'failed_case_diagnostics'in ast.unparse(loop)and 'failure_writer_error'in ast.unparse(loop)
    assert 'original_budget=0'in source and 'CAPTURE_HAS_FIXED_FULL_POOL_NO_FILTER'in source

def test_refusal_reinstalls_trace_profile_and_preserves_all32():
    source=Path(C.__file__).read_text();tree=ast.parse(source)
    refused=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='refused_call');body=ast.unparse(refused)
    assert 'sys.settrace(trace)'in body and 'sys.setprofile(profiler)'in body
    assert 'sys.settrace(previous_trace)'in body and 'sys.setprofile(previous_profile)'in body
    assert 'bytes(actual) == bytes(saved)'in body
    assert 'finite_radon_observer_v2'in source and "'sticky-rounding-finally-full32'"in source
    assert 'frame.f_code is fn.__code__ for fn in targets.values()'in source
