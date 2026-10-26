"""Metadata-only checks for the fixed RGB benchmark; no image execution."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'benchmarks/rgb2yuv_storage.py'
spec = importlib.util.spec_from_file_location('rgb_storage', PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_closed_case_keys_and_actual_storage_targets():
    for byteorder, swapped in [('little', '>'), ('big', '<')]:
        cases = harness.case_specs(byteorder)
        assert len(cases) == len({c['case_id'] for c in cases}) == 12
        assert {c['size'] for c in cases} == {256, 1024}
        for size in (256, 1024):
            rows = [c for c in cases if c['size'] == size]
            assert {c['storage_dtype'] for c in rows} == {'u1', '=f4', '=f8', '=f2', swapped+'f4', swapped+'f8'}
            assert sum(c['group'] == 'target' for c in rows) == 3
            assert all(c['shape'] == [size, size, 3] for c in rows)


def test_timer_contains_public_calls_only():
    tree = ast.parse(PATH.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'measure']
    assert len(calls) == 1
    call, = calls
    assert [ast.unparse(n) for n in call.args] == [
        'lambda: skimage.color.rgb2yuv(image)', 'lambda: xx.color.rgb2yuv(image)', 'args.repeat', '15']


def test_existing_color_roundoff_allowance_is_unchanged():
    tree = ast.parse(PATH.read_text())
    expressions = {ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AugAssign))}
    original = ast.parse((ROOT / 'tests/test_color_transforms.py').read_text())
    prior = {ast.unparse(n.value) for n in ast.walk(original) if isinstance(n, (ast.Assign, ast.AugAssign))}
    allowance = '4 * np.finfo(np.float32).eps * np.max(np.sum(np.abs(matrix), axis=1)) * magnitude'
    assert allowance in expressions & prior
    assert '2e-06 if actual.dtype == np.float32 else 2e-13' in expressions & prior
