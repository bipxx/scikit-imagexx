"""Two provider conformance references, captured once per pytest process.

These records compare literal provider behavior on the current runtime. They
never replace the immutable original-first gold or grant native ownership.
"""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

RUNTIME_REFERENCE_IDS = ('tolerance-provider-positive', 'tolerance-mutates-coordinates')
ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'tests/approximate_polygon_contract.py': '6f79e5c91f761bd67b5499abc6842460ee132f347edce337c3168ea0252f821f',
    'tests/test_approximate_polygon_public.py': 'e08bfdfe6cdef0a0ad24a5b2ed3660fcaf02ea9d2fa678b429eb8221a2d58c4e',
    'coverage/contracts/approximate-polygon.json': '101bb58c8ceae38d0a5dd2d66a8cfa628d069e41e2ef4d30740bb5e5c59ee758'}
ORIGINAL_SHA = '26a27d89545d158f6b542857297995639b251ed3af0428c8c86a56c84928aabe'
_STARTED = False
_CACHE = None
_PROFILE = None


def _validate_sources():
    for name, pin in PINS.items():
        raw = (ROOT / name).read_bytes()
        if name.endswith('.py'): raw = raw.replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() != pin:
            raise RuntimeError('RDP runtime reference source pin mismatch: ' + name)


def _prepare(np):
    _validate_sources()
    if np.__version__ not in ('1.24.4', '2.5.4'):
        raise RuntimeError('RDP runtime reference has an unassessed NumPy profile')
    original = importlib.import_module('skimage.measure._polygon')
    raw = Path(original.__file__).read_bytes().replace(b'\r\n', b'\n')
    reference = original.approximate_polygon
    if (hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA
            or str(inspect.signature(reference)) != '(coords, tolerance)'
            or importlib.metadata.version('scikit-image') != '0.26.0'):
        raise RuntimeError('RDP runtime reference original source/signature mismatch')
    if __package__:
        from .approximate_polygon_contract import cases
        from .test_approximate_polygon_public import observe
    else:
        from approximate_polygon_contract import cases
        from test_approximate_polygon_public import observe
    specs = [next(spec for spec in cases() if spec['id'] == name) for name in RUNTIME_REFERENCE_IDS]
    core = (np._core if hasattr(np, '_core') else np.core)._multiarray_umath
    provenance = dict(scope='Current-runtime provider conformance only; immutable gold with 120 rows remains unchanged.',
        python=sys.version, numpy=np.__version__, scipy=importlib.metadata.version('scipy'),
        original_source=str(original.__file__), original_source_sha256=ORIGINAL_SHA,
        signature='(coords, tolerance)', source_pins=PINS,
        numpy_core=str(core.__file__), numpy_core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        FP_policy=np.geterr(), FP_handler_type=type(np.geterrcall()).__name__, FP_buffer_elements=np.getbufsize())
    return specs, reference, observe, provenance


def runtime_reference(case_id, np):
    """Return an independent full-row value; a partial capture is never replayed."""
    global _STARTED, _CACHE, _PROFILE
    if case_id not in RUNTIME_REFERENCE_IDS:
        raise ValueError('RDP runtime reference is restricted to the two protocol IDs')
    if _CACHE is not None:
        if np.__version__ != _PROFILE:
            raise RuntimeError('RDP runtime reference process profile changed')
        return json.loads(_CACHE[case_id])
    if _STARTED:
        raise RuntimeError('RDP runtime reference already started and did not complete; no replay')
    _STARTED = True
    specs, reference, observe, provenance = _prepare(np)
    mode = 'scalar' if os.environ.get('SKIMAGEXX_DISABLE_SIMD') else 'normal'
    folder = ROOT / 'build/ci-artifacts'; folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('approximate-polygon-runtime-reference-' + np.__version__ + '-' + mode + '-' + str(os.getpid()) + '.jsonl')
    rows = {}
    with path.open('x', encoding='utf-8') as journal:
        for spec in specs:
            journal.write(json.dumps(dict(stage='before', id=spec['id'], provenance=provenance), allow_nan=False) + '\n')
            journal.flush()
            row = observe(reference, spec, np)
            journal.write(json.dumps(dict(stage='after', id=spec['id'], observation=row), allow_nan=False) + '\n')
            journal.flush()
            rows[spec['id']] = json.dumps(row, allow_nan=False).encode('utf-8')
    _CACHE, _PROFILE = rows, np.__version__
    return json.loads(_CACHE[case_id])
