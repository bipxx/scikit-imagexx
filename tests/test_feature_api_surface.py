"""Fast surface checks for all 477 APIs, not numerical or native parity.

Resolve names under the default fallback-enabled availability policy without
calling functions, constructing classes, loading assets, or inspecting members.
Passing also includes upstream-backed APIs and grants no C++ coverage credit.
"""
import importlib
import inspect
import json
from pathlib import Path
import re
import warnings

import pytest


INVENTORY = json.loads(
    (Path(__file__).resolve().parents[1] / 'coverage/api.json').read_text(encoding='utf-8')
)
ENTRIES = INVENTORY['entries']
# These pinned extension/exception classes expose no inspectable constructor.
# A new failure elsewhere must fail the test, never become a skipped check.
OPAQUE_SIGNATURES = frozenset({
    'feature.Cascade', 'graph.MCP', 'graph.MCP_Connect', 'graph.MCP_Flexible',
    'graph.MCP_Geometric', 'util.FailedEstimationAccessError',
    'util.PendingSkimage2Change',
})


@pytest.fixture(scope='module')
def surface_packages():
    upstream = importlib.import_module('skimage')
    assert INVENTORY['reference_version'] == upstream.__version__ == '0.26.0'
    assert len(ENTRIES) == len({entry['api'] for entry in ENTRIES}) == 477
    assert {entry['kind'] for entry in ENTRIES} == {'callable', 'class'}
    assert {entry['api'] for entry in ENTRIES if entry['signature'] is None} == OPAQUE_SIGNATURES
    candidate = importlib.import_module('skimagexx')
    previous = candidate.set_fallback_enabled(True)
    try:
        yield upstream, candidate
    finally:
        candidate.set_fallback_enabled(previous)


def _signature(value, api, package):
    try:
        return inspect.signature(value)
    except (TypeError, ValueError) as error:
        if api not in OPAQUE_SIGNATURES:
            pytest.fail(f'{package}.{api}: signature unexpectedly unavailable: {error}')
        assert inspect.isclass(value), f'{package}.{api}: opaque entry is no longer a class'
        return None


def _signature_text(signature):
    # Match the inventory's two display-only normalizations. No parameter,
    # default, annotation, or callable-kind mismatch is otherwise ignored.
    text = re.sub(r' at 0x[0-9a-fA-F]+', '', str(signature))
    import numpy as np
    if any(parameter.default is np.max for parameter in signature.parameters.values()):
        text = text.replace('<function amax>', '<function max>')
    return text


@pytest.mark.parametrize('entry', ENTRIES, ids=lambda entry: entry['api'])
def test_feature_api_surface(entry, surface_packages):
    upstream, candidate = surface_packages
    api = entry['api']
    module, name = api.rsplit('.', 1)
    assert name and not name.startswith('_'), f'{api}: inventory name is not public'
    original = getattr(importlib.import_module(f'{upstream.__name__}.{module}'), name)
    with warnings.catch_warnings():
        # Resolving an upstream-backed class can warn without invoking it.
        warnings.simplefilter('ignore', candidate.FallbackWarning)
        target = getattr(importlib.import_module(f'{candidate.__name__}.{module}'), name)
    assert callable(original), f'skimage.{api}: public entry is not callable'
    assert callable(target), f'skimagexx.{api}: public entry is not callable'
    expected_class = entry['kind'] == 'class'
    assert inspect.isclass(original) == expected_class, f'skimage.{api}: category drift'
    assert inspect.isclass(target) == expected_class, f'skimagexx.{api}: category differs'
    original_signature = _signature(original, api, 'skimage')
    candidate_signature = _signature(target, api, 'skimagexx')
    if entry['signature'] is None:
        assert original_signature is None and candidate_signature is None, (
            f'{api}: opaque-signature inventory changed; inspect the newly available signature'
        )
    else:
        assert _signature_text(original_signature) == entry['signature'], (
            f'skimage.{api}: signature differs from the pinned inventory'
        )
        assert _signature_text(candidate_signature) == _signature_text(original_signature), (
            f'skimagexx.{api}: signature differs from skimage 0.26.0'
        )
