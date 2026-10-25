"""Exact private H floating pipeline; immutable 816+70 original observations.

This gate has no original algorithm execution and makes no public/SDK claim.
The independently frozen reference directory may be separate from this owned
production worktree. Source and module hashes are mandatory before imports.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import pytest

if not os.environ.get('SKIMAGEXX_H_EXTREMA_CORE'):
    pytest.skip('private H-extrema core explicitly required; no native/public claim', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ.get('SKIMAGEXX_H_EXTREMA_FLOAT_REFERENCE', ROOT))
PROFILE = os.environ['SKIMAGEXX_H_EXTREMA_PROFILE']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The six owned files are frozen together before any candidate build/call.
# The external manifest's hash is supplied by the reviewed private composition.
SOURCE_MANIFEST_PATH = Path(os.environ['SKIMAGEXX_H_EXTREMA_SOURCE_MANIFEST'])
SOURCE_MANIFEST_SHA = digest(SOURCE_MANIFEST_PATH)
assert SOURCE_MANIFEST_SHA == os.environ['SKIMAGEXX_H_EXTREMA_SOURCE_MANIFEST_SHA256']
SOURCE_MANIFEST = json.loads(SOURCE_MANIFEST_PATH.read_bytes())
OWNED_PATHS = {
    'include/skimagexx/h_extrema.hpp', 'src/h_extrema.cpp', 'src/bind_h_extrema.cpp',
    'python/skimagexx/_h_extrema.py', 'tests/test_h_extrema_floating_native.cpp',
    'tests/test_h_extrema_floating_pipeline.py',
}
assert set(SOURCE_MANIFEST['source_sha256']) == OWNED_PATHS
for _name, _sha in SOURCE_MANIFEST['source_sha256'].items():
    assert digest(ROOT / _name) == _sha, _name
assert digest(ROOT / 'python/skimagexx/_reconstruction.py') == 'c190d7e8c879da82facdd4bbfd9d751535eed4a3f08870b7088d14b4372d4fd0'

# These are exact, original-only 8fb reference bytes, not candidate expectations
# generated from a new implementation. A future metadata-only successor needs
# its own explicit integration authentication; this private gate stays frozen.
REFERENCE_PINS = {
    'tests/test_h_extrema_floating_cases.py': 'f127526cd66c5043edd66d82518db3abb2a8956d3b29f1931057566ec090de1e',
    'tests/h_extrema_floating_reference.py': '389e5cffe1b399c0deb3c8dce8424a414b9b2329e16907e6afa86415fab0dfb0',
    'coverage/contracts/h-extrema-floating-plan.json': '67d6b11cbe3d78215aa637f167dd1cbf3a7543f568d44e9cda93dcea1cf901a0',
    'coverage/contracts/h-extrema-floating.json': '566ed99dfcfe70ea9d24deb1e8237389de9ed838a32d3909571c29a8d9a2a8df',
}
for _name, _sha in REFERENCE_PINS.items():
    assert digest(REFERENCE_ROOT / _name) == _sha, _name
REFERENCE = load('h_floating_private_reference', REFERENCE_ROOT / 'tests/h_extrema_floating_reference.py')
FLOAT_PLAN, SOURCE_AUTHENTICATION = REFERENCE.authenticate(PROFILE)
LEGACY = load('h_floating_private_legacy', REFERENCE_ROOT / 'tests/test_h_extrema_contracts.py')
BOUNDARIES = load('h_floating_private_boundaries', REFERENCE_ROOT / 'tests/test_h_extrema_boundaries.py')
FLOATING = load('h_floating_private_addendum', REFERENCE_ROOT / 'tests/test_h_extrema_floating_cases.py')
import numpy as np

assert LEGACY.R.profile() == PROFILE
POLICY_PATH = ROOT / 'coverage/contracts/h-extrema-integer.json'
POLICY = json.loads(POLICY_PATH.read_bytes())
assert digest(ROOT / 'coverage/contracts/h-extrema.json') == POLICY['reference_matrix_sha256']
MATRIX = json.loads((ROOT / 'coverage/contracts/h-extrema.json').read_bytes())
FLOAT_MATRIX = json.loads((REFERENCE_ROOT / 'coverage/contracts/h-extrema-floating.json').read_bytes())
OLD_SPECS = LEGACY.cases() + BOUNDARIES.cases()
FLOAT_SPECS = FLOATING.cases()
SPECS = OLD_SPECS + FLOAT_SPECS
assert len(SPECS) == len({spec['id'] for spec in SPECS}) == 886
assert [spec['id'] for spec in OLD_SPECS] == POLICY['all_case_ids']
INTEGER_IDS = set(POLICY['initial_native_ids'])
PENDING_IDS = set(POLICY['mandatory_followup_ids'])
UNSAFE_IDS = set(POLICY['source_unsafe_blocked_ids'])
assert (len(INTEGER_IDS), len(PENDING_IDS), len(UNSAFE_IDS)) == (424, 378, 14)
assert not (INTEGER_IDS & PENDING_IDS or INTEGER_IDS & UNSAFE_IDS or PENDING_IDS & UNSAFE_IDS)
FLOAT_IDS = {spec['id'] for spec in FLOAT_SPECS}


def expected_row(spec):
    if spec['id'] in FLOAT_IDS:
        return FLOAT_MATRIX['profiles'][PROFILE]['observations'][spec['id']]
    section = 'additive_profiles' if 'family' in spec else 'profiles'
    return MATRIX[section][PROFILE]['observations'][spec['id']]


def final_provider_case(spec):
    # Prospective scalar/subclass frontier. No candidate result chooses it.
    return (spec.get('layout') == 'tagged' or spec.get('height') == 'probe'
            or spec.get('family') in ('height-protocol', 'nonrepresentable-height', 'wide-unsigned-comparison'))


def expects_real_mask(spec):
    row = expected_row(spec)
    return (spec['id'] not in INTEGER_IDS and spec['id'] not in UNSAFE_IDS
            and row['error'] is None and row['original_numeric_completed'] == 1
            and not final_provider_case(spec))


def stage_contract(rows, integer_route):
    # The proved integer seed has always used a C-order scratch allocation.
    # Its source contract is dtype/shape/exact logical bytes, not the original
    # ufunc's private F-order scratch layout. Retain every raw field in the
    # journal; compare all fields for the newly literal floating preparation.
    if not integer_route:
        return rows
    projected = []
    for row in rows:
        item = dict(row)
        item['seed'] = {key: row['seed'][key] for key in ('type', 'dtype', 'metadata', 'shape', 'raw')}
        projected.append(item)
    return projected


# Fixed source-derived targets are checked before any candidate call. Errors
# after a completed flood are deliberately not counted as completed H masks.
assert sum(expects_real_mask(s) for s in OLD_SPECS) == (284 if PROFILE == 'minimum' else 280)
assert sum(expects_real_mask(s) for s in FLOAT_SPECS) == (44 if PROFILE == 'minimum' else 40)

MODULE_PATH = Path(os.environ['SKIMAGEXX_H_EXTREMA_CORE'])
MODULE_SHA = digest(MODULE_PATH)
assert MODULE_SHA == os.environ['SKIMAGEXX_H_EXTREMA_CORE_SHA256']
CORE = load('_h_extrema_core', MODULE_PATH)
assert CORE.get_num_threads() == 1
assert CORE.h_extrema_interface == 'h-extrema-integer-v1'
assert CORE.h_extrema_real_interface == 'h-extrema-real-v1' and CORE.h_extrema_full_API is False
assert CORE.reconstruction_interface == 'reconstruction-v1' and CORE.reconstruction_capabilities == 3
ENTRY_NAMES = ('h_extrema_integer_range', 'h_extrema_integer_seed', 'h_extrema_integer_mask',
               'h_extrema_real_mask', 'reconstruction_run')
RETURNS = {}
for _name in ENTRY_NAMES:
    _entry = getattr(CORE, _name)
    assert inspect.isbuiltin(_entry) and _entry.__name__ == _name and _entry.__module__ == CORE.__name__
    RETURNS[_entry] = _name
if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):
    assert not CORE.has_avx2()
PACKAGE = types.ModuleType('_h_extrema_floating_owned')
PACKAGE.__path__ = [str(ROOT / 'python/skimagexx')]
PACKAGE._core = CORE
sys.modules[PACKAGE.__name__] = PACKAGE
sys.modules[PACKAGE.__name__ + '._core'] = CORE
OWNED = load(PACKAGE.__name__ + '._h_extrema', ROOT / 'python/skimagexx/_h_extrema.py')
assert OWNED._core is CORE and OWNED.reconstruction.__globals__['_core'] is CORE
FORBIDDEN = {function.__code__ for function in LEGACY.ORIGINAL.values()}
FORBIDDEN.update((LEGACY.ORIGINAL_REC.__code__, LEGACY.R.SOURCE.rank_order.__code__,
                  LEGACY.SOURCE._add_constant_clip.__code__, LEGACY.SOURCE._subtract_constant_clip.__code__))
ROWS, GUARDS = [], []
REPORT_PATH = Path(os.environ['SKIMAGEXX_H_EXTREMA_REPORT'])
JOURNAL = REPORT_PATH.with_suffix('.jsonl').open('x', encoding='utf-8', newline='\n')


@pytest.fixture(scope='session', autouse=True)
def retained_report():
    yield
    JOURNAL.close()
    CORE.set_num_threads(1)
    report = dict(schema='h-extrema-floating-actual-private-pipeline-v1', profile=PROFILE,
                  source_authentication=SOURCE_AUTHENTICATION,
                  source_manifest=dict(path=str(SOURCE_MANIFEST_PATH), sha256=SOURCE_MANIFEST_SHA),
                  source_sha256=SOURCE_MANIFEST['source_sha256'], reference_sha256=REFERENCE_PINS,
                  module=dict(path=str(MODULE_PATH), sha256=MODULE_SHA),
                  old_policy_sha256=digest(POLICY_PATH), old_integer_ids=424, mandatory_followup_ids=378,
                  unsafe_blocked_ids=14, additive_original_ids=70,
                  has_avx2=CORE.has_avx2(), disable_simd=os.environ.get('SKIMAGEXX_DISABLE_SIMD'),
                  default_workers=1, final_workers=CORE.get_num_threads(), observations=ROWS, guards=GUARDS,
                  full_API_complete=False, public_acceptance=False, operation_timings=0)
    with REPORT_PATH.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def retain(row):
    ROWS.append(row)
    JOURNAL.write(json.dumps(row, allow_nan=False) + '\n')
    JOURNAL.flush()


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('spec', SPECS, ids=lambda spec: spec['id'])
def test_exact_owned_pipeline(spec, workers):
    name = spec['id']
    if workers == 4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF'):
        with pytest.raises(ValueError, match='no OpenMP support'):
            CORE.set_num_threads(4)
        retain(dict(id=name, workers=workers, status='portable-worker4-explicit-rejection', actual_calls=0))
        pytest.skip('portable private foundation explicitly rejects worker4')
    if name in UNSAFE_IDS:
        retain(dict(id=name, workers=workers, status='source-unsafe-blocked', actual_calls=0))
        pytest.skip('unsafe original intake is not an upstream exception contract')
    CORE.set_num_threads(workers)
    assert CORE.get_num_threads() == workers
    CORE.h_extrema_counts(True)
    CORE.h_extrema_real_counts(True)
    CORE.reconstruction_counts(True)
    expected = expected_row(spec)
    attempts, stages = [], []
    c_returns = {entry: 0 for entry in ENTRY_NAMES}
    providers = dict(preparation=0, clipping=0, final_mask=0)
    saved_reconstruction = OWNED.reconstruction

    def reconstruction(seed, mask, *args, **kwargs):
        row = dict(seed=LEGACY.value(seed), mask=LEGACY.value(mask), args=LEGACY.value(args),
                   kwargs=LEGACY.value(kwargs), completed=False)
        stages.append(row)
        result = saved_reconstruction(seed, mask, *args, **kwargs)
        row['completed'] = True
        row['result'] = LEGACY.value(result)
        return result

    def guard(frame, event, arg):
        if event == 'call':
            if frame.f_code in FORBIDDEN:
                attempts.append(frame.f_code.co_name)
                raise AssertionError('upstream H-extrema/reconstruction/rank-order forwarding forbidden')
            for key, function in (('preparation', OWNED._real), ('clipping', OWNED._clipped_seed),
                                  ('final_mask', OWNED._provider_mask)):
                if frame.f_code is function.__code__:
                    providers[key] += 1
        elif event == 'c_return' and arg in RETURNS:
            c_returns[RETURNS[arg]] += 1

    previous = sys.getprofile()
    OWNED.reconstruction = reconstruction
    sys.setprofile(guard)
    try:
        function = getattr(OWNED, spec['function'])
        if name in FLOAT_IDS:
            actual = FLOATING.observe(spec, LEGACY, function)
        elif 'family' in spec:
            actual = BOUNDARIES.observe(spec, LEGACY, function)
        else:
            actual = LEGACY.observe(spec, function)
    finally:
        sys.setprofile(previous)
        OWNED.reconstruction = saved_reconstruction
    integer, real, rec = CORE.h_extrema_counts(), CORE.h_extrema_real_counts(), CORE.reconstruction_counts()
    exact = LEGACY.public_observation(actual) == LEGACY.public_observation(expected)
    stages_exact = stage_contract(stages, name in INTEGER_IDS) == stage_contract(expected['reconstruction'], name in INTEGER_IDS)
    row = dict(id=name, workers=workers, actual=actual, actual_owned_reconstruction=stages,
               expected_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
               integer_counts=integer, real_mask_counts=real, reconstruction_counts=rec,
               actual_c_returns=c_returns, provider_calls=providers, forwarding_attempts=attempts,
               public_observation_exact=exact, intermediate_stages_exact=stages_exact,
               original_integer_route=name in INTEGER_IDS, prospective_real_mask=expects_real_mask(spec))
    retain(row)  # Retain failures before asserting; never filter adverse rows.
    assert not attempts and actual['original_numeric_completed'] == 0
    assert exact, name
    assert stages_exact, name
    assert rec['run']['completed'] == c_returns['reconstruction_run'] == expected['original_numeric_completed'], name
    if name in INTEGER_IDS:
        assert providers == dict(preparation=0, clipping=0, final_mask=0)
        assert real == dict(attempted=0, completed=0)
        assert integer['mask']['completed'] == c_returns['h_extrema_integer_mask'] == expected['original_numeric_completed']
    else:
        assert providers['preparation'] == 1
        assert integer['range'] == dict(attempted=0, completed=0) and integer['mask'] == dict(attempted=0, completed=0)
        assert real['completed'] == int(expects_real_mask(spec)), name
        assert real['completed'] <= c_returns['h_extrema_real_mask'] <= real['attempted']
        expected_provider = expected['error'] is None and expected['original_numeric_completed'] == 1 and final_provider_case(spec)
        assert providers['final_mask'] == int(expected_provider), name


def test_safe_storage_guards_and_unchanged_integer_intake():
    for attribute in ('base', 'dtype', 'shape', 'strides', 'flags'):
        def forbidden(self):
            raise AssertionError('overridable ndarray metadata was read')
        subtype = type('MetadataTrap_' + attribute, (np.ndarray,), {attribute: property(forbidden)})
        array = np.arange(8, dtype=np.float32).view(subtype)
        guard = CORE._HExtremaStorage(array)
        assert guard.check() == 8
        GUARDS.append(dict(id='real-direct-C-slot/' + attribute, passed=True, pixel_reads=0))
    for dtype, shape, count in (('f4', (0, 3), 0), ('?', (3,), 3), ('c8', (2,), 2), ('f2', (), 1)):
        assert CORE._HExtremaStorage(np.empty(shape, dtype=dtype)).check() == count
        GUARDS.append(dict(id='generic-storage/' + dtype + '/' + str(shape), passed=True, pixel_reads=0))
    with pytest.raises(TypeError, match='integer stage'):
        CORE.h_extrema_validate_metadata(np.empty(3, dtype=np.float32))
    GUARDS.append(dict(id='integer-v1-still-refuses-float32', passed=True, pixel_reads=0))
    root = np.arange(16, dtype=np.float64)
    view = root[8:]
    guard = CORE._HExtremaStorage(view)
    root.resize((8,), refcheck=False)
    # Only the metadata checker sees the stale view; no numerical entry does.
    with pytest.raises(ValueError, match='owner metadata changed'):
        guard.check()
    with pytest.raises(ValueError, match='outside owning allocation'):
        CORE._HExtremaStorage(view)
    GUARDS.append(dict(id='resized-root-and-current-extent/metadata-only', passed=True, pixel_reads=0))
    for value in (bytearray(16), bytes(16)):
        with pytest.raises(ValueError, match='ndarray-backed storage'):
            CORE._HExtremaStorage(np.frombuffer(value, dtype=np.float64))
    array = np.arange(8, dtype=np.float64)
    indirect = np.lib.stride_tricks.as_strided(array, shape=(4,), strides=(8,))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        CORE._HExtremaStorage(indirect)
    GUARDS.append(dict(id='foreign-and-in-bounds-indirect-owners', passed=True, pixel_reads=0))


def test_height_admission_does_not_invoke_protocols():
    calls = []
    class Height:
        def __float__(self): calls.append('float'); raise AssertionError('height conversion probe')
        def __index__(self): calls.append('index'); raise AssertionError('height index probe')
        def __array__(self, *args, **kwargs): calls.append('array'); raise AssertionError('height array probe')
    image = np.zeros(3, dtype=np.float32)
    reconstructed = np.zeros(3, dtype=np.float32)
    CORE.h_extrema_real_counts(True)
    for height in (Height(), 4.0 + 2.0**-23, np.uint64(2**53 + 1)):
        assert CORE.h_extrema_real_mask(image, reconstructed, height, 0) is None
    assert not calls and CORE.h_extrema_real_counts() == dict(attempted=3, completed=0)
    GUARDS.append(dict(id='height-protocol-and-inexact-threshold-refusal', passed=True, completed_masks=0))


def test_array_height_storage_precedes_first_range_read(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('ptp')
        raise AssertionError('range read preceded height storage validation')
    image = np.zeros(3, dtype=np.float32)
    height = np.frombuffer(bytes(8), dtype=np.float64)
    monkeypatch.setattr(OWNED.np, 'ptp', forbidden)
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        OWNED.h_maxima(image, height)
    assert not calls
    GUARDS.append(dict(id='array-height-owner-before-first-provider-read', passed=True, pixel_reads=0))
