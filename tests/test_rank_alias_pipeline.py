"""Private native alias parity; no public admission or forwarding claim."""
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load('rank_alias_contract_provider', Path(__file__).with_name('test_rank_alias_contracts.py'))
if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    native = load('skimagexx._rank_alias_candidate', os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    native._core = importlib.import_module(os.environ['SKIMAGEXX_RANK_BACKEND'])
else:
    native = load('skimagexx._rank_alias_candidate', Path(__file__).parents[1] / 'python/skimagexx/_rank.py')
# Private adapter only: retain the production normalization/error path and
# original objects, bypassing the historical Python compatibility admission.
native._r2_alias_admission = lambda *args: (False, None)


@pytest.fixture(autouse=True, params=[1, 4])
def workers(request):
    previous = native._core.get_num_threads()
    native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(previous)


@pytest.fixture(scope='module')
def frozen():
    return json.loads(contracts.MATRIX.read_text())


@pytest.mark.parametrize('case', contracts.CASES, ids=[row[0] for row in contracts.CASES])
def test_native_matches_every_frozen_owner_and_observation(case, frozen):
    key, name, spec = case
    expected = frozen['observations'][frozen['snapshots'][contracts.profile()][key]]
    actual = contracts.observe(getattr(native, name), spec)
    assert actual == expected, key


def test_alias_capability_is_separate_from_algorithm_support():
    assert native._core.rank_alias_interface == 'rank-alias-v1'
    assert native._core.rank_histogram_capabilities == 8191
    assert native._core.rank_alias_capabilities == sum(1 << i for i in range(5, 9))


@pytest.mark.parametrize('operation', range(5, 9))
@pytest.mark.parametrize('unsafe', ['wide-float', 'wide-partial-byte', 'integral-entropy'])
def test_undefined_alias_domains_reject_before_any_write(operation, unsafe):
    if unsafe == 'integral-entropy' and operation != 8:
        return
    shape = (3, 5)
    owner = np.full(256, 0x5a, np.uint8)
    image = np.ndarray(shape, np.uint16 if unsafe != 'integral-entropy' else np.uint8, buffer=owner)
    image[:] = np.arange(15).reshape(shape)
    output = np.ndarray(shape + (1,), np.float64 if unsafe == 'wide-float' else np.uint8, buffer=owner)
    initial = owner.copy()
    message = {'wide-float': 'uint16 image/float', 'wide-partial-byte': 'full histogram', 'integral-entropy': 'integral entropy'}[unsafe]
    if operation == 8 and unsafe == 'wide-partial-byte':
        message = 'integral entropy'
    with pytest.raises(ValueError, match=message):
        native._core.rank_run(image, np.ones((3, 3), np.uint8), None, output, [0, 0], 256, operation)
    assert owner.tobytes() == initial.tobytes()


def test_ordinary_native_calls_do_not_use_upstream(monkeypatch):
    from skimage.filters import rank
    def forbidden(*args, **kwargs):
        raise AssertionError('upstream rank forwarding')
    monkeypatch.setattr(native, 'fallback', forbidden)
    for name in contracts.NAMES:
        monkeypatch.setattr(rank, name, forbidden)
        _, _, spec = next(row for row in contracts.CASES if row[1] == name and '/population-drift/2d' in row[0])
        args, kwargs, _ = contracts.make_case(spec)
        getattr(native, name)(*args, **kwargs)
