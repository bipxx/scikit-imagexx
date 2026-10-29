"""Inert workload, checker and timing-boundary checks; no scientific calls."""
import ast
import builtins
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'benchmarks/draw_coordinates.py'


def load():
    spec = importlib.util.spec_from_file_location('draw_benchmark_metadata', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_and_frozen_workloads_do_not_import_scientific_packages(monkeypatch):
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in ('numpy', 'scipy', 'skimage', 'skimagexx')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    module = load()
    reference = json.loads(module.CONTRACT.read_bytes())
    contract = json.loads(module.EXECUTION.read_bytes())
    assert module.case_specs() == reference['benchmark_cases'] == contract['cases']
    assert [s['id'] for s in contract['cases']] == [
        'line/tiny', 'line/long', 'rectangle/tiny', 'rectangle/2d', 'rectangle/3d',
        'disk/tiny', 'disk/dense', 'disk/clipped', 'ellipse/tiny', 'ellipse/axes',
        'ellipse/rotated', 'ellipse/clipped']
    assert hashlib.sha256(module.CONTRACT.read_bytes()).hexdigest() == contract['reference_contract_sha256']
    assert (contract['threads'], contract['repeat'], contract['target_ms'],
            contract['warmup_ms_per_function'], contract['raw_samples']) == (1, 7, 100, 250, 168)
    assert contract['method_sha256'] == 'c05b519d7470f5ac8b87d13d971df64f404e1d3a283120cfbee0d340767a9fb0'
    assert len(reference['original_observations']['1.24.4']) == 88
    assert len(reference['original_observations']['2.5.4']) == 88
    assert 'not uint8/float32/float64 pixel-image kernels' in contract['dtype_scope']


@pytest.mark.parametrize('args,expected', [
    ([0, 0, 7, 3], ([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 2, 3, 3])),
    ([0, 0, 2, 1], ([0, 1, 2], [0, 1, 1])),
    ([2, 1, 0, 0], ([2, 1, 0], [1, 0, 0])),
    ([3, -4, 3, -4], ([3], [-4]))])
def test_line_oracle_ties_endpoint_order_and_degenerate_without_numpy(args, expected):
    stub = SimpleNamespace(intp='intp', asarray=lambda values, dtype: values)
    assert load().oracle(dict(api='draw.line', args=args, kwargs={}), stub) == list(expected)


class Array:
    def __init__(self, raw, shape=(2,), strides=(8,), base=None):
        self.raw, self.shape, self.strides, self.base = raw, shape, strides, base
        self.size = math.prod(shape)
        self.dtype = SimpleNamespace(str='<i8', metadata=None)
        self.flags = SimpleNamespace(owndata=base is None, writeable=True, aligned=True,
                                     c_contiguous=True, f_contiguous=True)
    def tobytes(self, order='C'):
        assert order == 'C'
        return self.raw


def test_exact_checker_rejects_word_layout_container_and_alias_changes():
    module = load()
    dtype = SimpleNamespace(str='<i8', metadata=None)
    np = SimpleNamespace(ndarray=Array, intp='intp', dtype=lambda _: dtype,
                         shares_memory=lambda a, b: a is b)
    make = lambda: (Array(bytes(range(16))), Array(bytes(range(16, 32))))
    actual, expected, oracle = make(), make(), make()
    assert module.checked_outputs(actual, expected, oracle, np, 4)['container'] == 'tuple'
    with pytest.raises(AssertionError):
        module.checked_outputs(list(actual), expected, oracle, np, 4)
    with pytest.raises(AssertionError):
        module.checked_outputs(actual, expected, oracle, np, 3)
    with pytest.raises(AssertionError):
        module.checked_outputs(actual, actual, oracle, np, 4)
    for field, value in (('raw', bytes(16)), ('strides', (16,)), ('base', SimpleNamespace(shape=(2, 2), strides=(16, 8)))):
        previous = getattr(expected[0], field)
        setattr(expected[0], field, value)
        with pytest.raises(AssertionError):
            module.checked_outputs(actual, expected, oracle, np, 4)
        setattr(expected[0], field, previous)


def test_precheck_and_restoration_precede_unchanged_sustained_sampler():
    tree = ast.parse(SOURCE.read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    timer = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'measure')
    check = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'checked_outputs')
    cleanup = next(n for n in ast.walk(main) if isinstance(n, ast.Try))
    assert cleanup.end_lineno < check.lineno < timer.lineno
    assert [ast.unparse(v) for v in timer.args] == [
        'lambda: reference(*call_args, **options)', 'lambda: native(*call_args, **options)', 'args.repeat', '100']
    text = ast.unparse(main)
    assert "event == 'c_return' and value is entry" in text
    assert "calls == [entry_name] and accepted == [True]" in text
    assert "compiled._line = denied" in text and "frame.f_code in original_codes" in text
    assert "from montage_sustained import measure" in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr.startswith('perf_counter') for n in ast.walk(tree))
