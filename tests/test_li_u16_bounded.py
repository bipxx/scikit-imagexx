"""Original-only prefreeze for a bounded uint16 extension of existing Li.

No native implementation is imported or exercised by reference recording.
The existing312 warn-policy corpus and40 default-policy cases stay immutable.
"""
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/li-u16-bounded.json'
BASE_TEST = ROOT / 'tests/test_iterative_threshold_contracts.py'
BASE_MATRIX = ROOT / 'coverage/contracts/iterative-thresholds.json'
BASE_TEST_SHA = '63e7449e2458c3725bf8ea2b95ad84db94bd1a89bad03225b6a365750b577dd9'
BASE_MATRIX_SHA = '34ea03c49663324c8169069884391fe3bacc311476a78dbe8999e6dbfbe3e95e'
DEFAULT_POLICY = dict(divide='warn', over='warn', under='ignore', invalid='warn')


def authenticate_before_import():
    assert hashlib.sha256(BASE_TEST.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == BASE_TEST_SHA
    assert hashlib.sha256(BASE_MATRIX.read_bytes()).hexdigest() == BASE_MATRIX_SHA
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    profile = 'windows' if os.name == 'nt' else 'minimum' if versions['numpy'] == '1.24.4' else 'modern'
    assert versions == dict(numpy='1.24.4' if profile == 'minimum' else '2.5.4',
                            scipy='1.15.3' if profile == 'minimum' else '1.18.1',
                            **{'scikit-image': '0.26.0'})
    frozen = json.loads(BASE_MATRIX.read_bytes())['profiles'][profile]
    expected, = [r for r in frozen['provenance'] if r['path'].replace('\\', '/').endswith('/filters/thresholding.py')]
    source = Path(importlib.metadata.distribution('scikit-image').locate_file('skimage/filters/thresholding.py'))
    raw = source.read_bytes()
    assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    return dict(profile=profile, versions=versions, executable=sys.executable, python=sys.version,
                source=dict(path=str(source), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))


AUTH = authenticate_before_import()
import numpy as np

_spec = importlib.util.spec_from_file_location('li_u16_frozen_base', BASE_TEST)
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)
ORIGINAL = BASE.ORIGINAL['threshold_li']


def cases():
    rows = []
    for layout in ('C', 'F', 'step', 'reverse', 'endian', 'readonly'):
        rows.append(dict(id='default/default-u2-' + layout, base_id='threshold_li/default-u2-' + layout,
                         layout=layout, proposed_route='solver'))
    rows.append(dict(id='default/storage-u2-unaligned', base_id='threshold_li/storage-u2-unaligned',
                     layout='unaligned', proposed_route='solver'))
    for name, route in (
        ('upper-edge', 'solver'), ('shifted', 'solver'), ('empty', 'early'),
        ('scalar-array', 'early'), ('constant-zero', 'early'), ('constant-255', 'early'),
        ('constant-outside', 'provider'), ('numpy-scalar', 'provider'), ('metadata', 'solver'),
        ('tolerance-zero', 'solver'), ('tolerance-small', 'solver'), ('tolerance-infinite', 'solver'),
        ('guess-float', 'solver'), ('guess-float32', 'solver'), ('guess-uint16', 'solver'),
        ('guess-no-update', 'solver'), ('shifted-guess-no-update', 'solver'),
        ('guess-invalid-low', 'native-error'), ('guess-invalid-high', 'native-error'),
        ('underflow-warn', 'provider'), ('underflow-raise', 'provider'),
        ('all-ignore', 'solver'), ('other-errors-raise', 'solver'), ('underflow-call', 'provider'),
        ('callback-record', 'provider'), ('callback-abort2', 'provider'), ('callback-invalid', 'provider'),
        ('callable-guess', 'provider'), ('mutating-guess', 'provider'),
        ('tolerance-protocol', 'provider'), ('negative-tolerance-abort', 'provider'),
        ('outside-256', 'provider'), ('outside-65535', 'provider'), ('tagged', 'provider'),
        ('constant-invalid-keywords', 'early'), ('worker-8193', 'solver')):
        rows.append(dict(id='boundary/' + name, kind=name, proposed_route=route))
    assert len(rows) == 43 and len({r['id'] for r in rows}) == 43
    return rows


def fixture(spec, events):
    if 'base_id' in spec:
        old, = [r for r in BASE.cases() if r['id'] == spec['base_id']]
        return (*BASE.fixture(old, events), dict(DEFAULT_POLICY))
    kind = spec['kind']
    image = np.array([0, 0, 1, 2, 3, 4, 4, 5, 7, 9, 11, 12], dtype=np.uint16).reshape(3, 4)
    options = {}
    policy = dict(DEFAULT_POLICY)
    if kind == 'upper-edge': image = np.array([0, 1, 2, 254, 255], dtype=np.uint16)
    elif kind in ('shifted', 'shifted-guess-no-update'): image = image + np.uint16(10)
    elif kind == 'empty': image = np.empty((0, 3), dtype=np.uint16)
    elif kind == 'scalar-array': image = np.array(3, dtype=np.uint16)
    elif kind == 'constant-zero': image = np.zeros((3, 4), dtype=np.uint16)
    elif kind == 'constant-255': image = np.full((3, 4), 255, dtype=np.uint16)
    elif kind == 'constant-outside': image = np.full((3, 4), 65535, dtype=np.uint16)
    elif kind == 'numpy-scalar': image = np.uint16(3)
    elif kind == 'metadata': image = image.astype(np.dtype('u2', metadata={'li-u16': 'frozen'}))
    elif kind == 'outside-256': image = np.array([0, 1, 7, 255, 256], dtype=np.uint16)
    elif kind == 'outside-65535': image = np.array([0, 1, 7, 255, 65535], dtype=np.uint16)
    elif kind == 'tagged': image = BASE.layout(image, 'tagged')
    elif kind == 'worker-8193': image = ((np.arange(8193, dtype=np.uint32) * 37) % 256).astype(np.uint16)
    if kind == 'tolerance-zero': options['tolerance'] = 0.
    elif kind == 'tolerance-small': options['tolerance'] = .125
    elif kind == 'tolerance-infinite': options['tolerance'] = np.inf
    elif kind == 'guess-float': options['initial_guess'] = 1.25
    elif kind == 'guess-float32': options['initial_guess'] = np.float32(1.25)
    elif kind == 'guess-uint16': options['initial_guess'] = np.uint16(2)
    elif kind in ('guess-no-update', 'shifted-guess-no-update'):
        options.update(initial_guess=11.25 if kind.startswith('shifted') else 1.25, tolerance=np.inf)
    elif kind == 'guess-invalid-low': options['initial_guess'] = 0.
    elif kind == 'guess-invalid-high': options['initial_guess'] = 12.
    if kind == 'underflow-warn': policy['under'] = 'warn'
    elif kind == 'underflow-raise': policy['under'] = 'raise'
    elif kind == 'underflow-call': policy['under'] = 'call'
    elif kind == 'all-ignore': policy = dict.fromkeys(DEFAULT_POLICY, 'ignore')
    elif kind == 'other-errors-raise': policy.update(divide='raise', over='raise', invalid='raise')
    calls = [0]
    def callback(value):
        calls[0] += 1
        events.append(['iter_callback', BASE.value(value)])
        if kind in ('callback-abort2', 'negative-tolerance-abort') and calls[0] == 2:
            raise RuntimeError('li-u16-callback-stop')
        if calls[0] >= 32: raise RuntimeError('li-u16-callback-bound')
    if kind in ('callback-record', 'callback-abort2'): options['iter_callback'] = callback
    elif kind == 'callback-invalid': options['iter_callback'] = 7
    elif kind == 'negative-tolerance-abort': options.update(tolerance=-.5, iter_callback=callback)
    def initial(array):
        events.append(['initial_guess', BASE.value(array), bool(np.shares_memory(array, image))])
        if kind == 'mutating-guess':
            array.flat[0] = array.max()
            events.append(['initial_mutation', BASE.value(array)])
        return np.mean(array)
    if kind in ('callable-guess', 'mutating-guess'): options['initial_guess'] = initial
    class Tolerance(float):
        def __bool__(self): events.append(['tolerance_bool']); return True
        def __rmul__(self, other): events.append(['tolerance_rmul', other]); return float(self) * other
    if kind == 'tolerance-protocol': options['tolerance'] = Tolerance(.5)
    if kind == 'constant-invalid-keywords':
        image = np.full((2, 3), 3, dtype=np.uint16)
        options.update(initial_guess=[1], tolerance=Tolerance(.5), iter_callback=7)
    return image, options, policy


def observe(spec, function=ORIGINAL):
    events = []
    image, options, policy = fixture(spec, events)
    before, options_before = BASE.value(image), BASE.value(options)
    old_policy, old_handler = np.geterr(), np.geterrcall()
    handler_events = []
    result, error = None, None
    try:
        if policy['under'] == 'call': np.seterrcall(lambda text, flag: handler_events.append([text, flag]))
        handler = np.geterrcall()
        with warnings.catch_warnings(record=True) as caught, np.errstate(**policy):
            warnings.simplefilter('always')
            entered = np.geterr()
            try: result = function(image, **options)
            except Exception as exc: error = dict(type=type(exc).__name__, message=str(exc))
            after_policy = np.geterr()
            handler_unchanged = np.geterrcall() is handler
        seen = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
    finally:
        np.seterrcall(old_handler)
        assert np.geterr() == old_policy and np.geterrcall() is old_handler
    return dict(input_before=before, input_after=BASE.value(image), options_before=options_before,
        options_after=BASE.value(options), result=BASE.value(result), error=error, warnings=seen,
        events=events, policy_entered=entered, policy_after_call=after_policy,
        errcall_unchanged=handler_unchanged, errcall_events=handler_events,
        input_result_alias=bool(isinstance(result,np.ndarray) and isinstance(image,np.ndarray) and np.shares_memory(result,image)),
        NumPy_buffer_size=np.getbufsize())


def record_reference(destination):
    contract_raw = CONTRACT.read_bytes()
    contract = json.loads(contract_raw)
    own_raw = Path(__file__).read_bytes()
    assert hashlib.sha256(own_raw.replace(b'\r\n', b'\n')).hexdigest() == contract['observer_canonical_LF_sha256']
    assert cases() == contract['cases']
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    rows = [dict(id=spec['id'], observation=observe(spec)) for spec in cases()]
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    body = dict(schema='li-u16-bounded-original-profile-v1', source_authentication=AUTH,
        contract_raw_sha256=hashlib.sha256(contract_raw).hexdigest(),
        observer_raw_sha256=hashlib.sha256(own_raw).hexdigest(),
        observer_canonical_LF_sha256=contract['observer_canonical_LF_sha256'],
        function_source_sha256=hashlib.sha256(inspect.getsource(ORIGINAL).encode()).hexdigest(),
        candidate_calls=0, operation_timings=0, cases=rows)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(AUTH['profile'], len(rows), 'original-only Li uint16 observations; candidate calls0; timings0')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-reference', required=True)
    record_reference(parser.parse_args().record_reference)
