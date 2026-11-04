"""Reference-only finite half-input scope; every controller ID is dry-run first."""
import base64
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import zlib
import pytest
if __package__:
    from .combine_stains_half_contract import API, MAX_INPUT_ELEMENTS, MAX_OUTPUT_ELEMENTS, NUMERICAL_POLICY, UNASSESSED, cases, fixture, source_safe
    from .combine_stains_half_observer import observe, encode, scientific_profile, SOURCE_SHA, UTILS_SHA, DTYPE_SHA, SIGNATURE
else:
    from combine_stains_half_contract import API, MAX_INPUT_ELEMENTS, MAX_OUTPUT_ELEMENTS, NUMERICAL_POLICY, UNASSESSED, cases, fixture, source_safe
    from combine_stains_half_observer import observe, encode, scientific_profile, SOURCE_SHA, UTILS_SHA, DTYPE_SHA, SIGNATURE
ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/combine-stains-half.json'
CASES = cases()


@contextmanager
def without_color_activity():
    prefixes = ('skimage.color', 'skimagexx.color', 'skimagexx._core')
    def forbidden(name):
        return any(name == p or name.startswith(p + '.') for p in prefixes)
    modules = {name: module for name, module in sys.modules.items() if forbidden(name)}
    previous, original_import, original_dynamic = sys.getprofile(), builtins.__import__, importlib.import_module
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert not forbidden(name) and not (name in ('skimage', 'skimagexx') and 'color' in (fromlist or ()))
        return original_import(name, globals, locals, fromlist, level)
    def guarded_dynamic(name, package=None):
        assert not forbidden(name)
        return original_dynamic(name, package)
    def profile(frame, event, value):
        module = frame.f_globals.get('__name__', '') if event == 'call' else (getattr(value, '__module__', '') or '') if event == 'c_call' else ''
        assert not forbidden(module)
    builtins.__import__, importlib.import_module = guarded_import, guarded_dynamic
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        builtins.__import__, importlib.import_module = original_import, original_dynamic
    assert {name: module for name, module in sys.modules.items() if forbidden(name)} == modules


def test_combine_stains_half_prospective_source_safety():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['api'] == API and matrix['signature'] == SIGNATURE
    assert matrix['cases'] == CASES and len(CASES) == len({s['id'] for s in CASES}) == 12
    assert matrix['numerical_policy'] == NUMERICAL_POLICY
    assert matrix['original_source_sha256'] == dict(colorconv=SOURCE_SHA, utils=UTILS_SHA, dtype=DTYPE_SHA)
    assert matrix['unassessed'] == list(UNASSESSED)
    assert matrix['native_implementation'] == 'pending'
    for spec in CASES:
        source_safe(spec)
    # These3 complete files record historical capture context. Their old bytes
    # and original pins stay immutable; live producer dependencies remain strict.
    context_raw = (ROOT / 'coverage/contracts/combine-stains-half-source-context-v1.json').read_bytes()
    assert hashlib.sha256(context_raw).hexdigest() == 'bdd55b52ecc3eb1fafb502b2762fd233c1853fb233e02d27a85e745d2c32e6f5'
    context = json.loads(context_raw)
    assert context['schema'] == 'combine-stains-half-source-context-v1'
    assert context['reference_commit'] == '22c7d6addfeb36e82df4bda0814e2ca687349603'
    assert context['reference_contract_sha256'] == hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    historical = {'python/skimagexx/coverage.py', 'tests/test_stains.py',
                  'tests/test_combine_stains_half_reference.py'}
    live = {'tests/combine_stains_half_contract.py', 'tests/combine_stains_half_observer.py',
            'tests/draw_coordinates_observer.py'}
    assert set(context['entries']) == historical
    assert set(context['live_source_paths']) == live
    assert set(matrix['source_freeze']) == historical | live
    for name, entry in context['entries'].items():
        assert entry['role'] == 'historical_capture_context' and entry['Git_revision'] == context['reference_commit']
        archived = entry['text'].encode('utf-8')
        assert len(archived) == entry['bytes']
        assert hashlib.sha256(archived).hexdigest() == entry['raw_sha256']
        assert hashlib.sha256(archived.replace(b'\r\n', b'\n')).hexdigest() == entry['source_freeze_canonical_lf_sha256'] == matrix['source_freeze'][name]
    for name in live:
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == matrix['source_freeze'][name]


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
def test_combine_stains_half_controller_intake_originals_refused(spec):
    np = scientific_profile()
    calls = []
    def refused(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('combine-half-original-refused-before-capture')
    with without_color_activity():
        args, options, buffers = fixture(spec, np)
        assert args[0].dtype.type is np.float16 and args[0].dtype.isnative
        assert args[0].shape[spec['channel_axis']] == 3
        assert args[0].size <= MAX_INPUT_ELEMENTS
        json.dumps(encode((args, options, buffers), np), allow_nan=False)
        row = observe(refused, spec, np)
    assert len(calls) == 1 and row['result'] is None
    assert row['error'] == ['builtins', 'RuntimeError', 'combine-half-original-refused-before-capture']
    assert row['input_before'] == row['input_after'] and all(row['checks'].values())
    assert not row['events'] and not row['warnings']


@pytest.mark.parametrize('profile', ('1.24.4', '2.5.4'))
def test_combine_stains_half_saved_original_rows(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective only; original capture pending')
    rows = matrix['original_observations'][profile]
    assert len(rows) == 12 and [r['id'] for r in rows] == [s['id'] for s in CASES]
    for spec, row in zip(CASES, rows):
        assert row['api'] == API and row['error'] is None
        assert row['input_before'] == row['input_after'] and all(row['checks'].values())
        assert not row['warnings'] and not row['events']
        result = row['result']
        assert result['type'] == 'ndarray' and result['dtype'] in ('<f4', '<f8')
        assert math.prod(result['shape']) == source_safe(spec)['output_elements'] <= MAX_OUTPUT_ELEMENTS
        assert len(row['output_input_aliases']) == 2
        assert all(not alias['same_object'] and not alias['shares_memory'] for alias in row['output_input_aliases'])
        pending = [row]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if 'raw_zlib_base64' in value:
                    raw = zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest() == value['raw_sha256']
                    assert len(raw) == math.prod(value['shape']) * int(value['dtype'][2:])
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
