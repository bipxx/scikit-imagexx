"""Prospective source/import metadata only; no geometry or timing calls."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_circle_aa_benchmark_frozen_cases_source_and_bounds():
    raw = (ROOT/'coverage/contracts/circle-perimeter-aa.json').read_bytes()
    reference = json.loads(raw)
    contract = json.loads((ROOT/'coverage/contracts/circle-perimeter-aa-benchmark.json').read_bytes())
    assert contract['reference_contract_sha256'] == hashlib.sha256(raw).hexdigest()
    assert contract['cases'] == reference['benchmark_cases'] and len(contract['cases']) == 8
    assert len({s['id'] for s in contract['cases']}) == 8
    assert contract['threads'] == 1 and contract['repeat'] == 7
    assert contract['raw_samples'] == 8*2*7 and contract['warmup_groups'] == 16
    assert contract['target_ms'] == 100 and contract['warmup_min_ms'] == 250
    assert contract['max_output_elements'] == reference['benchmark_max_output_elements'] == 1000000
    assert contract['max_radius'] == 4096 and contract['max_ulp'] == 0
    assert set(contract['reference_pool_limits'].values()) == {'1'} and len(contract['reference_pool_limits']) == 4
    method = json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())
    assert contract['method_sha256'] == method['method_sha256']
    for name, pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin, name
    for spec in contract['cases']:
        assert spec['api'] == 'draw.circle_perimeter_aa' and spec['kind'] == 'ordinary'
        assert spec['args'] == spec['safe_args'] and not spec['policy']
        row, column, radius = spec['args']
        assert all(type(v) is int for v in (row, column, radius))
        assert abs(radius) <= 4096 and radius*radius <= (1 << 63)-1
        assert all(-(1 << 63)+abs(radius) <= v <= (1 << 63)-1-abs(radius) for v in (row, column))
        assert 3*(8+16*max(0, radius-1)) <= contract['max_output_elements']
        if spec['kwargs']:
            assert set(spec['kwargs']) == {'shape'} and len(spec['kwargs']['shape']) == 2
            assert all(type(v) is int and v > 0 for v in spec['kwargs']['shape'])
    assert max(3*(8+16*max(0, s['args'][2]-1)) for s in contract['cases']) > reference['max_output_elements']


def test_circle_aa_benchmark_import_has_no_scientific_side_effects(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **options):
        assert name.split('.')[0] not in {'numpy', 'scipy', 'skimage', 'skimagexx'}
        return original(name, *args, **options)

    monkeypatch.setattr(builtins, '__import__', checked)
    spec = importlib.util.spec_from_file_location('circle_aa_benchmark_import_probe', ROOT/'benchmarks/circle_perimeter_aa.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_circle_aa_hardware_capture_reuses_current_line_aa_template():
    reference = ast.parse((ROOT/'benchmarks/line_aa.py').read_text())
    candidate = ast.parse((ROOT/'benchmarks/circle_perimeter_aa.py').read_text())
    blocks = []
    for tree in (reference, candidate):
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        index = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'cpu_model' for t in node.targets))
        blocks.append([ast.dump(node) for node in main.body[index:index+4]])
    assert blocks[0] == blocks[1]
    source = (ROOT/'benchmarks/circle_perimeter_aa.py').read_text()
    assert "cpu_model=cpu_model, logical_cpu_count=cpu_count, avx2_available=avx2_available" in source
    assert source.index('cpu_model = platform.processor()') < source.index("enumerate(contract['cases'])")


def test_circle_aa_empty_output_identity_guards_without_scientific_calls():
    tree = ast.parse((ROOT/'benchmarks/circle_perimeter_aa.py').read_text())
    guards = [node.test for node in ast.walk(tree) if isinstance(node, ast.Assert)
              and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
              and node.test.func.id == 'all' and isinstance(node.test.args[0], ast.GeneratorExp)
              and any(isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                      and part.func.attr == 'shares_memory' for part in ast.walk(node.test))]
    assert len(guards) == 2
    # NumPy reports no shared elements for empty arrays, including the same
    # empty object. Exercise the actual assertion expressions with inert values.
    inert_numpy = SimpleNamespace(shares_memory=lambda first, second:
                                  bool(first.size and second.size and first is second))
    for size in (0, 3):
        array, distinct = SimpleNamespace(size=size), SimpleNamespace(size=size)
        assert array is array and inert_numpy.shares_memory(array, array) == bool(size)
        for guard in guards:
            code = compile(ast.Expression(guard), '<benchmark-owner-assertion>', 'eval')
            scope = dict(np=inert_numpy, array=array, expected=[array], actual=[array, array], i=0)
            assert not eval(code, scope)
            scope.update(expected=[distinct], actual=[array, distinct])
            assert eval(code, scope)
