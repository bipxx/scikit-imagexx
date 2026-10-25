"""Installed public NMI gate; exact private assertions, ordinary sole core.

The compatibility tests explicitly enable fallback and count its warning. The
separate strict214 audit disables fallback, denies original numerical calls,
and records actual owned compiled completions at both global worker limits.
No private module, synthetic package, or replacement numerical source is used.
"""
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'nmi_public_frozen_assertions', ROOT / 'tests/test_normalized_mutual_information.py')
FROZEN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FROZEN)


@pytest.fixture(scope='module')
def installed():
    assert not any(name.startswith('SKIMAGEXX_NMI_') for name in os.environ)
    assert '_nmi_core' not in sys.modules
    assert '_nmi_test_package' not in sys.modules
    import skimagexx as xx
    from skimagexx.coverage import native_functions, parity_tolerance
    wrapper = importlib.import_module('skimagexx._normalized_mutual_information')
    metrics = importlib.import_module('skimagexx.metrics')
    compat = importlib.import_module('skimagexx._compat')
    core = xx._core
    assert core.__name__ == 'skimagexx._core' and wrapper._core is core
    package = Path(xx.__file__).resolve().parent
    for module in (core, wrapper, metrics, compat):
        assert Path(module.__file__).resolve().parent == package
    assert package != (ROOT / 'python/skimagexx').resolve()
    assert xx.get_num_threads is core.get_num_threads
    assert xx.set_num_threads is core.set_num_threads
    assert core.get_num_threads() == 1 and core.has_openmp is True
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD') == '1':
        assert not core.has_avx2()
    for name in ('nmi_prepare', 'nmi_statistics', 'nmi_validate_inputs'):
        entry = getattr(core, name)
        assert inspect.isbuiltin(entry) and entry.__module__ == core.__name__
        assert entry.__name__ == name
    function = metrics.normalized_mutual_information
    assert function is wrapper.normalized_mutual_information
    assert function.__module__ == 'skimagexx._normalized_mutual_information'
    assert str(inspect.signature(function)) == '(image0, image1, *, bins=100)'
    assert metrics.__all__.count('normalized_mutual_information') == 1
    assert tuple(native_functions()).count('metrics.normalized_mutual_information') == 1
    assert len(FROZEN.CASES) == 214 and len({c['id'] for c in FROZEN.CASES}) == 214
    assert FROZEN.POLICY['rtol'] == 2e-12 and FROZEN.POLICY['atol'] == 2e-13
    for dtype in ('uint8', 'float32', 'float64'):
        assert parity_tolerance('metrics.normalized_mutual_information', dtype) == (2e-12, 2e-13)
    previous = xx.set_fallback_enabled(False)
    assert xx.set_fallback_enabled(False) is False
    try:
        yield core, wrapper, compat, metrics
    finally:
        core.set_num_threads(1)
        xx.set_fallback_enabled(previous)


@pytest.fixture
def native(installed):
    """Compatibility lane: provider calls are explicit and individually checked."""
    core, wrapper, compat, _ = installed
    assert core.get_num_threads() == 1
    previous = compat.set_fallback_enabled(True)
    try:
        yield core, wrapper, compat
    finally:
        core.set_num_threads(1)
        compat.set_fallback_enabled(previous)


# Exact function objects and parameter marks. Metadata-only unsafe controls
# remain metadata-only; their inputs never reach an image or original oracle.
for _name, _value in vars(FROZEN).items():
    if _name.startswith('test_') and _name != 'test_histogram_membership_edges_and_entropy':
        globals()[_name] = _value


@pytest.mark.parametrize('case', FROZEN.CASES, ids=lambda case: case['id'])
def test_histogram_membership_edges_and_explicit_frontiers(native, case):
    """No public skips: each old detail frontier has concrete policy checks."""
    _, wrapper, compat = native
    exceptional = bool(case['policy']) or case['id'] == 'rank-mismatch'
    prepared = None if exceptional else wrapper._prepare(case['image0'], case['image1'], case['bins'])
    if not exceptional and prepared is not None:
        # Exact frozen membership/count/edge/width/entropy assertions.
        FROZEN.test_histogram_membership_edges_and_entropy(native, case)
        return
    # The private histogram-detail test skipped this scope. Keep its API row
    # fully exercised with the unchanged complete compatibility assertion,
    # then explicitly verify the disabled-fallback behavior as well.
    FROZEN.test_original_contract_and_explicit_native_frontier(native, case)
    previous = compat.set_fallback_enabled(False)
    try:
        actual = FROZEN.observe(wrapper.normalized_mutual_information, case)
    finally:
        compat.set_fallback_enabled(previous)
    assert not [w for w in actual['warnings'] if w['category'] == 'FallbackWarning']
    if case['native'] is False or (not exceptional and prepared is None):
        assert actual.get('exception', {}).get('type') == 'builtins.NotImplementedError'
        assert actual['exception']['message'] == (
            'skimage.metrics.normalized_mutual_information: this call is outside native coverage; fallback is disabled')
    else:
        from skimage.metrics import normalized_mutual_information as original
        FROZEN.compare_observations(FROZEN.observe(original, case), actual)


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('case', FROZEN.CASES, ids=lambda case: case['id'])
def test_strict_public_route_and_actual_compiled_completions(installed, case, workers):
    import numpy as np
    import skimage.metrics
    import skimage.metrics.simple_metrics as original_metrics
    import scipy.stats
    core, _, compat, metrics = installed
    original = original_metrics.normalized_mutual_information
    expected = FROZEN.observe(original, case)
    snapshots = [image.tobytes() if isinstance(image, (np.ndarray, np.generic)) else None
                 for image in (case['image0'], case['image1'])]
    forbidden = {function.__code__ for function in (
        original, scipy.stats.entropy, original_metrics.entropy)
        if hasattr(function, '__code__')}
    attempts = []
    completions = []

    def guard(frame, event, arg):
        if event == 'call' and frame.f_code in forbidden:
            attempts.append(frame.f_code.co_name)
            raise AssertionError('Original numerical forwarding during strict public NMI')
        if event == 'c_return' and any(arg is getattr(core, name)
                                      for name in ('nmi_prepare', 'nmi_statistics')):
            completions.append(arg.__name__)

    previous_fallback = compat.set_fallback_enabled(False)
    previous_profile = sys.getprofile()
    try:
        core.set_num_threads(workers)
        assert core.get_num_threads() == workers
        sys.setprofile(guard)
        actual = FROZEN.observe(metrics.normalized_mutual_information, case)
    finally:
        sys.setprofile(previous_profile)
        core.set_num_threads(1)
        compat.set_fallback_enabled(previous_fallback)
    assert not attempts, (case['id'], attempts)
    assert not [w for w in actual['warnings'] if w['category'] == 'FallbackWarning']
    refused = actual.get('exception', {}).get('type') == 'builtins.NotImplementedError'
    if case['native'] is False:
        assert refused, (case['id'], actual)
    elif case['native'] is True:
        assert not refused, (case['id'], actual)
    if refused:
        assert actual['exception']['message'] == (
            'skimage.metrics.normalized_mutual_information: this call is outside native coverage; fallback is disabled')
    else:
        FROZEN.compare_observations(expected, actual)
        assert completions == ([] if case['id'] == 'rank-mismatch'
                               else ['nmi_prepare', 'nmi_statistics']), (case['id'], completions)
    for snapshot, image in zip(snapshots, (case['image0'], case['image1'])):
        if snapshot is not None:
            assert image.tobytes() == snapshot


def test_original_denial_guard_positive_control(installed):
    from skimage.metrics import normalized_mutual_information as original
    attempts = []
    previous = sys.getprofile()

    def reject(frame, event, arg):
        if event == 'call' and frame.f_code is original.__code__:
            attempts.append(True)
            raise RuntimeError('controlled original NMI denial')

    try:
        sys.setprofile(reject)
        with pytest.raises(RuntimeError, match='controlled original NMI denial'):
            original(None, None)
    finally:
        sys.setprofile(previous)
    assert attempts == [True]
