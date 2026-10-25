"""Actual private H/reconstruction composition against immutable source records."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT/'coverage/contracts/h-extrema-integer.json'
POLICY = json.loads(POLICY_PATH.read_bytes())
for _name, _sha in POLICY['source_sha256'].items():
    assert hashlib.sha256((ROOT/_name).read_bytes()).hexdigest() == _sha, _name


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initial_native_spec(spec):
    # Prospective input partition only. No expected/candidate output is consulted.
    if spec['dtype'] not in ('u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8'):
        return False
    if 'family' in spec:
        return spec['family'] in ('integer-saturation', 'full-integer-span', 'wide-integer-rounding', 'integer-cutoff')
    return spec.get('layout') != 'tagged' and spec.get('height', 'one') in ('one', 'zero', 'two', 'out-of-range')


REFERENCE = load('h_source_reference_auth', ROOT/'tests/h_extrema_reference.py')
_profile = os.environ['SKIMAGEXX_H_EXTREMA_PROFILE']
PLAN, SOURCE_AUTHENTICATION = REFERENCE.authenticate(_profile)
LEGACY = load('h_private_legacy_observer', ROOT/'tests/test_h_extrema_contracts.py')
BOUNDARIES = load('h_private_boundary_observer', ROOT/'tests/test_h_extrema_boundaries.py')
import numpy as np

assert LEGACY.R.profile() == _profile
MATRIX = json.loads((ROOT/'coverage/contracts/h-extrema.json').read_bytes())
assert hashlib.sha256((ROOT/'coverage/contracts/h-extrema.json').read_bytes()).hexdigest() == POLICY['reference_matrix_sha256']
SPECS = LEGACY.cases() + BOUNDARIES.cases()
assert [spec['id'] for spec in SPECS] == POLICY['all_case_ids']
assert [spec['id'] for spec in SPECS if initial_native_spec(spec) and spec['id'] not in PLAN['legacy_unsafe_ids']] == POLICY['initial_native_ids']
MODULE_PATH = Path(os.environ['SKIMAGEXX_H_EXTREMA_CORE'])
MODULE_SHA = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
assert MODULE_SHA == os.environ['SKIMAGEXX_H_EXTREMA_CORE_SHA256']
CORE = load('_h_extrema_core', MODULE_PATH)
assert CORE.get_num_threads() == 1
assert CORE.h_extrema_interface == 'h-extrema-integer-v1' and CORE.h_extrema_full_API is False
assert CORE.reconstruction_interface == 'reconstruction-v1' and CORE.reconstruction_capabilities == 3
for _name in ('h_extrema_integer_range', 'h_extrema_integer_seed', 'h_extrema_integer_mask', 'reconstruction_run'):
    _entry = getattr(CORE, _name)
    assert inspect.isbuiltin(_entry) and _entry.__module__ == CORE.__name__
if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):
    assert not CORE.has_avx2()
PACKAGE = types.ModuleType('_h_extrema_owned')
PACKAGE.__path__ = [str(ROOT/'python/skimagexx')]
PACKAGE._core = CORE
sys.modules[PACKAGE.__name__] = PACKAGE
sys.modules[PACKAGE.__name__+'._core'] = CORE
OWNED = load(PACKAGE.__name__+'._h_extrema', ROOT/'python/skimagexx/_h_extrema.py')
assert OWNED._core is CORE and OWNED.reconstruction.__globals__['_core'] is CORE
FORBIDDEN = {function.__code__ for function in LEGACY.ORIGINAL.values()}
FORBIDDEN.add(LEGACY.ORIGINAL_REC.__code__)
FORBIDDEN.add(LEGACY.R.SOURCE.rank_order.__code__)
ROWS = []
GUARDS = []
JOURNAL_PATH = Path(os.environ['SKIMAGEXX_H_EXTREMA_REPORT']).with_suffix('.jsonl')
JOURNAL = JOURNAL_PATH.open('x', encoding='utf-8', newline='\n')


@pytest.fixture(scope='session', autouse=True)
def retained_report():
    yield
    JOURNAL.close()
    CORE.set_num_threads(1)
    report = dict(schema='h-extrema-actual-private-pipeline-v1', profile=_profile,
                  source_authentication=SOURCE_AUTHENTICATION, module=dict(path=str(MODULE_PATH), sha256=MODULE_SHA),
                  source_sha256=POLICY['source_sha256'], policy_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
                  has_avx2=CORE.has_avx2(), disable_simd=os.environ.get('SKIMAGEXX_DISABLE_SIMD'),
                  default_workers=1, final_workers=CORE.get_num_threads(),
                  observations=ROWS, guards=GUARDS, full_API_complete=False,
                  original_algorithm_calls=0, operation_timings=0, public_acceptance=False)
    with Path(os.environ['SKIMAGEXX_H_EXTREMA_REPORT']).open('x', encoding='utf-8', newline='\n') as output:
        json.dump(report, output, indent=2, allow_nan=False); output.write('\n')


def retain(row):
    ROWS.append(row)
    JOURNAL.write(json.dumps(row, allow_nan=False)+'\n'); JOURNAL.flush()


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('spec', SPECS, ids=lambda spec: spec['id'])
def test_exact_private_pipeline(spec, workers):
    name = spec['id']
    if workers == 4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF'):
        with pytest.raises(ValueError, match='no OpenMP support'):
            CORE.set_num_threads(4)
        retain(dict(id=name, workers=workers, status='portable-worker4-explicit-rejection'))
        pytest.skip('portable private foundation explicitly rejects worker4')
    if name in PLAN['legacy_unsafe_ids']:
        retain(dict(id=name, workers=workers, status='source-unsafe-blocked', actual_calls=0))
        pytest.skip('unsafe original intake is not an upstream exception contract')
    CORE.set_num_threads(workers)
    assert CORE.get_num_threads() == workers
    CORE.h_extrema_counts(True); CORE.reconstruction_counts(True)
    expected = (MATRIX['additive_profiles'] if 'family' in spec else MATRIX['profiles'])[_profile]['observations'][name]
    attempts = []
    def guard(frame, event, arg):
        if event == 'call' and frame.f_code in FORBIDDEN:
            attempts.append(frame.f_code.co_name)
            raise AssertionError('upstream H-extrema/reconstruction/rank-order forwarding forbidden')
    previous = sys.getprofile()
    sys.setprofile(guard)
    try:
        function = getattr(OWNED, spec['function'])
        actual = BOUNDARIES.observe(spec, LEGACY, function) if 'family' in spec else LEGACY.observe(spec, function)
    finally:
        sys.setprofile(previous)
    stats, reconstruction = CORE.h_extrema_counts(), CORE.reconstruction_counts()
    selected = initial_native_spec(spec)
    exact = LEGACY.public_observation(actual) == LEGACY.public_observation(expected)
    row = dict(id=name, workers=workers, prospective_native=selected, actual=actual,
               expected_reference='additive_profiles' if 'family' in spec else 'profiles',
               expected_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
               H_counts=stats, reconstruction_counts=reconstruction, forwarding_attempts=attempts,
               public_observation_exact=exact, status='native' if selected else 'mandatory-followup')
    retain(row)
    assert not attempts and actual['original_numeric_completed'] == 0
    if selected:
        assert exact, name
        assert reconstruction['run']['completed'] == expected['original_numeric_completed'], name
        assert stats['mask']['completed'] == expected['original_numeric_completed'], name
    else:
        assert actual['error']['type'] == 'NotImplementedError'
        assert all(value['completed'] == 0 for value in stats.values())
        assert reconstruction['run']['completed'] == 0


def test_safe_metadata_only_intake_guards():
    # This actual exported metadata function never dereferences image pixels.
    # Stale/escaping storage is only passed to that metadata-only function.
    validate = CORE.h_extrema_validate_metadata
    for attribute in ('base', 'dtype', 'shape', 'strides', 'flags'):
        def forbidden(self):
            raise AssertionError('Python ndarray metadata attribute called')
        subtype = type('MetadataTrap_'+attribute, (np.ndarray,), {attribute: property(forbidden)})
        image = np.arange(8, dtype=np.uint16).view(subtype)
        assert validate(image) == 8
        GUARDS.append(dict(id='direct-C-slot/'+attribute, passed=True, pixel_reads=0))
    root = np.arange(16, dtype=np.uint16)
    view = root[8:]
    root.resize((8,), refcheck=False)
    with pytest.raises(ValueError, match='outside owning allocation'):
        validate(view)
    GUARDS.append(dict(id='preexisting-resized-owner/metadata-only', passed=True, pixel_reads=0))
    image = np.arange(8, dtype=np.uint16)
    indirect = np.lib.stride_tricks.as_strided(image, shape=(4,), strides=(2,))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        validate(indirect)
    GUARDS.append(dict(id='in-bounds-unknown-owner', passed=True, pixel_reads=0))
    for name, value in (('bytearray', bytearray(8)), ('bytes', bytes(8))):
        image = np.frombuffer(value, dtype=np.uint16)
        with pytest.raises(ValueError, match='ndarray-backed storage'):
            validate(image)
        GUARDS.append(dict(id='foreign-owner/'+name, passed=True, pixel_reads=0))
    calls = []
    class ScalarProbe:
        def __index__(self): calls.append('index'); raise AssertionError('scalar conversion callback')
        def __float__(self): calls.append('float'); raise AssertionError('scalar conversion callback')
    image = np.arange(8, dtype=np.uint16)
    with pytest.raises(TypeError, match='exact integer'):
        CORE.h_extrema_integer_seed(image, ScalarProbe(), 0)
    with pytest.raises(TypeError, match='exact integer'):
        CORE.h_extrema_integer_seed(image, 1, ScalarProbe())
    with pytest.raises(TypeError, match='exact float'):
        CORE.h_extrema_integer_mask(image, np.arange(8, dtype=np.float64), ScalarProbe(), 0)
    assert not calls
    GUARDS.append(dict(id='scalar-protocols-rejected-before-native-read', passed=True, pixel_reads=0))
