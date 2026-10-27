"""Metadata-only checks for the fixed Euler benchmark; no numerical imports."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'benchmarks/euler_number.py'
spec = importlib.util.spec_from_file_location('euler_benchmark', PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_fixed_geometry_dtype_keys_and_sample_count():
    cases = harness.case_specs()
    assert len(cases) == len({r['case_id'] for r in cases}) == 12
    shapes = {(256, 256), (1024, 1024), (32, 32, 32), (64, 64, 64)}
    assert {tuple(r['shape']) for r in cases} == shapes
    for shape in shapes:
        rows = [r for r in cases if tuple(r['shape']) == shape]
        assert {r['dtype'] for r in rows} == {'uint8', 'float32', 'float64'}
        assert all(r['rank'] == len(shape) for r in rows)
    assert len(cases)*2*7 == 168


def test_preparation_is_seeded_binary_same_pixels_per_dtype():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    values = {node.targets[0].id: ast.unparse(node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert values['gray'] == "np.random.default_rng(2026).integers(0, 256, spec['shape'], dtype=np.uint8)"
    assert values['image'] == "(gray > 127).astype(spec['dtype'])"


def test_timer_calls_public_api_directly_without_default_preparation():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'measure']
    call, = calls
    assert [ast.unparse(n) for n in call.args] == [
        'lambda: skimage.measure.euler_number(image)', 'lambda: xx.measure.euler_number(image)', 'args.repeat', '15']
    assert 'default_call' not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_exact_shared_checker_and_scalar_type_precede_measurement():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    checker, = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'euler_default_error']
    timer, = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'measure']
    assert [ast.unparse(n) for n in checker.args] == ['image', 'actual', 'expected']
    assert checker.lineno < timer.lineno
    assertions = {ast.unparse(n.test): n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)}
    assert assertions['type(actual) is type(expected)'] < timer.lineno
    assert assertions['actual == expected'] < timer.lineno
    assert 'error == 0.0 and xx.get_num_threads() == 1' in assertions


def test_generic_benchmark_has_only_euler_specific_exact_branch():
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_text(encoding='utf-8'))
    checker, = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'euler_default_error']
    assert [ast.unparse(n) for n in checker.args] == ['call_args[0]', 'actual', 'expected']
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "api == 'measure.euler_number'"]
    assert len(branches) == 1 and any(n is checker for n in ast.walk(branches[0]))
    scopes = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'euler_scope' for t in n.targets)]
    scope, = scopes
    assert isinstance(scope, ast.Constant) and all(s in scope.value for s in ('NumPy', 'padding', 'exact integer', '2**53'))
