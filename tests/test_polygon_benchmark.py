"""Metadata-only guards for the fixed polygon comparison; no image calls."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/polygon.py'


def test_fixed_work_and_complexity_cases_import_without_science():
    spec = importlib.util.spec_from_file_location('polygon_benchmark_metadata', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.case_specs()
    expected = {(kind, size, vertices)
                for kind, sizes in (('points', (4096, 65536)), ('grid', (256, 1024)))
                for size in sizes for vertices in (4, 32, 128)}
    assert len(rows) == len({r['case_id'] for r in rows}) == 12
    assert {(r['kind'], r['size'], r['vertices']) for r in rows} == expected
    assert {r['api'] for r in rows} == {'measure.points_in_poly', 'measure.grid_points_in_poly'}


def test_exact_raw_checker_precedes_unchanged_paired_timer():
    tree = ast.parse(SOURCE.read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    checker = next(n for n in calls if n.func.id == 'polygon_default_error')
    timer = next(n for n in calls if n.func.id == 'measure')
    assert checker.lineno < timer.lineno
    assert [ast.unparse(a) for a in checker.args] == ['None', 'actual', 'expected']
    assert [ast.unparse(a) for a in timer.args[2:]] == ['args.repeat', '17']
    assert 'default_call' not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert 'reference(*call_args)' in ast.unparse(main) and 'native(*call_args)' in ast.unparse(main)


def test_both_public_modes_and_private_guards_precede_freeze():
    text = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
    producer = text.split('  numpy-minimum:', 1)[1].split('  measure-minimum:', 1)[0]
    for target in ('polygon-normal.xml', 'polygon-scalar.xml', 'polygon-intake.xml'):
        assert producer.count(target) == 1
        assert producer.index(target) < producer.index('ci_measurement_artifacts.py freeze')


@pytest.mark.parametrize('api', ('measure.points_in_poly', 'measure.grid_points_in_poly'))
def test_generic_benchmark_uses_exact_output_policy(api):
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "api in ('measure.points_in_poly', 'measure.grid_points_in_poly')")
    calls = []
    a, e, image = object(), object(), object()
    def checker(*args):
        calls.append(args)
        return 0.
    namespace = dict(api=api, polygon_default_error=checker, actual=a, expected=e, call_args=(image,))
    exec(compile(ast.Module(branch.body, type_ignores=[]), '<polygon checker route>', 'exec'), namespace)
    assert calls == [(image, a, e)] and namespace['error'] == 0.
