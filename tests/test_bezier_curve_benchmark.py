"""Pure stdlib benchmark guard/intake; no original/native or NumPy calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[1]
SPEC=importlib.util.spec_from_file_location('bezier_benchmark',ROOT/'benchmarks/bezier_curve.py')
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)
CONTRACT=json.loads(module.CONTRACT.read_bytes())


def test_bezier_benchmark_source_closure_and_method():
    assert CONTRACT['status']=='PROSPECTIVE_FROZEN_UNEXECUTED'
    assert len(CONTRACT['cases'])==len({s['id'] for s in CONTRACT['cases']})==12
    assert CONTRACT['threads']==1 and CONTRACT['repeat']==7
    assert CONTRACT['target_ms']==100 and CONTRACT['warmup_calibration_min_ms']==250
    assert CONTRACT['raw_samples']==168 and CONTRACT['warm_groups']==24
    assert CONTRACT['native_entries']==['bezier_curve_weight','draw_bezier_curve']
    for name,pin in CONTRACT['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert hashlib.sha256(module.REFERENCE.read_bytes()).hexdigest()==CONTRACT['reference_contract_sha256']
    text=(ROOT/'tools/montage_sustained.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='measure')
    source=''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(source.encode()).hexdigest()==CONTRACT['method_sha256']


@pytest.mark.parametrize('spec',CONTRACT['cases'],ids=lambda s:s['id'])
def test_bezier_benchmark_scalar_fixture_intake(spec):
    args,options,bounds=module.fixture(spec)
    assert type(args) is tuple and len(args)==7
    assert args==tuple(spec['args'])
    assert options=={'shape':None if spec['shape'] is None else tuple(spec['shape'])}
    assert bounds['coordinate_pairs_upper_bound']<=50000
    assert bounds['payload_bytes_upper_bound']==16*bounds['coordinate_pairs_upper_bound']


def test_bezier_benchmark_real_sampler_and_outside_validation():
    tree=ast.parse((ROOT/'benchmarks/bezier_curve.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure']
    assert len(calls)==1
    call=calls[0];assert all(isinstance(n,ast.Lambda) for n in call.args[:2])
    assert call.args[2].attr=='repeat' and call.args[3].value==100
    assert {k.arg for k in call.keywords}=={'journal','case_index','case_id','pass_index','deadline'}
    checked=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_output']
    assert len(checked)==1 and checked[0].lineno<call.lineno
    restoration=[n for n in ast.walk(main) if isinstance(n,ast.Try) and n.finalbody]
    assert any(n.lineno<call.lineno and max(x.end_lineno for x in n.finalbody)<call.lineno for n in restoration)


def test_bezier_benchmark_algorithm_imports_only_inside_main():
    tree=ast.parse((ROOT/'benchmarks/bezier_curve.py').read_text())
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            assert not any(n.split('.')[0] in ('numpy','scipy','skimage','skimagexx') for n in names)
