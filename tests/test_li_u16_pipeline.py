"""Exact bounded Li parity through the installed public package.

The child authenticates/loads the frozen original observer before importing the
candidate. This keeps original-only recording assumptions intact even when the
parent pytest session has already imported skimagexx. No private adapter is used.
"""
import collections
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys


def audit_installed_product():
    observer_path = Path(__file__).with_name('test_li_u16_bounded.py')
    spec = importlib.util.spec_from_file_location('li_u16_public_original', observer_path)
    observer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = observer
    spec.loader.exec_module(observer)
    cases = observer.cases()
    expected = {case['id']: observer.observe(case) for case in cases}

    import skimagexx as xx
    from skimagexx import _core
    assert inspect.isbuiltin(_core.li_prepare)
    assert _core.li_prepare.__module__ == 'skimagexx._core'
    assert xx.filters.threshold_li.__module__ == 'skimagexx.filters'
    assert inspect.signature(xx.filters.threshold_li) == inspect.signature(observer.ORIGINAL)
    assert _core.get_num_threads() == xx.get_num_threads() == 1
    forbidden = {observer.ORIGINAL.__code__, observer.BASE.SOURCE.histogram.__code__}
    previous_fallback = xx.set_fallback_enabled(False)
    totals = collections.Counter()
    try:
        for workers in (1, 4):
            xx.set_num_threads(workers)
            assert _core.get_num_threads() == xx.get_num_threads() == workers
            for case in cases:
                completions = []
                attempts = []

                def guard(frame, event, arg):
                    if event == 'call' and frame.f_code in forbidden:
                        attempts.append(frame.f_code.co_name)
                        raise AssertionError('original numerical forwarding during strict Li call')
                    if event == 'c_return':
                        owner = getattr(arg, '__self__', None)
                        if (owner is not None and type(owner).__module__ == 'skimagexx._core'
                                and getattr(arg, '__name__', None) == 'run'):
                            completions.append('run')

                previous_profile = sys.getprofile()
                sys.setprofile(guard)
                try:
                    actual = observer.observe(case, xx.filters.threshold_li)
                finally:
                    sys.setprofile(previous_profile)
                assert not attempts, (case['id'], attempts)
                route = case['proposed_route']
                assert len(completions) == int(route == 'solver'), (case['id'], completions)
                if route == 'provider':
                    assert actual['error'] is not None
                    assert actual['error']['type'] == 'NotImplementedError', case['id']
                    xx.set_fallback_enabled(True)
                    try:
                        forwarded = observer.observe(case, xx.filters.threshold_li)
                    finally:
                        xx.set_fallback_enabled(False)
                    diagnostics = [w for w in forwarded['warnings'] if w['category'] == 'FallbackWarning']
                    assert diagnostics == [dict(category='FallbackWarning', message=
                        'skimage.filters.threshold_li executed by scikit-image fallback')]
                    actual = dict(forwarded, warnings=[w for w in forwarded['warnings']
                                                       if w['category'] != 'FallbackWarning'])
                assert actual == expected[case['id']], (workers, case['id'], actual, expected[case['id']])
                totals[route] += 1
                assert _core.get_num_threads() == workers
    finally:
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)
    assert totals == dict(solver=42, early=10, **{'native-error': 4, 'provider': 30})
    print(json.dumps(dict(profile=observer.AUTH['profile'], observations=sum(totals.values()),
                          actual_CPP_solver_completions=totals['solver'], routes=dict(totals),
                          private_adapter=False, operation_timings=0), sort_keys=True))


def test_installed_li_u16_exact_original_parity():
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                             '--audit-installed-product'], capture_output=True, text=True,
                            encoding='utf-8', timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['observations'] == 86
    assert report['actual_CPP_solver_completions'] == 42
    assert report['private_adapter'] is False


if __name__ == '__main__':
    assert sys.argv[1:] == ['--audit-installed-product']
    audit_installed_product()
