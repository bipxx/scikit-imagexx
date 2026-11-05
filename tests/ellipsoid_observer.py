"""Capture controller; importing this file performs no image computation."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings

if __package__:
    from .ellipsoid_contract import API, fixture
    from .draw_coordinates_observer import encode, scientific_profile
else:
    from ellipsoid_contract import API, fixture
    from draw_coordinates_observer import encode, scientific_profile

SOURCE_SHA = '78c9a91b754d8a0ca59e8eccd75fa9458dc12a529230f542858d4ee6ae6c57ca'
SIGNATURE = '(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False)'


def reference_runtime():
    np = scientific_profile()
    module = importlib.import_module('skimage.draw.draw3d')
    source = Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    assert str(inspect.signature(module.ellipsoid)) == SIGNATURE
    grid_module = importlib.import_module(type(np.mgrid).__module__)
    return np, module.ellipsoid, dict(
        python=sys.version, numpy=np.__version__, skimage='0.26.0',
        source_canonical_lf_sha256=SOURCE_SHA, source_path=module.__file__,
        numpy_grid_source_path=grid_module.__file__,
        numpy_grid_source_sha256=hashlib.sha256(Path(grid_module.__file__).read_bytes()).hexdigest(),
        calls_per_spec=1, native_calls=0, timings=False)


def observe(call, spec, np):
    args, options, bound = fixture(spec, np)
    before = encode((args, options), np)
    old_policy, old_handler, old_buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    events, result, error = [], None, None
    def handler(message, flags):
        events.append(dict(message=message, flags=flags))
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(call=handler):
            policy = np.geterr().copy()
            try:
                result = call(*args, **options)
            except Exception as exc:
                error = [type(exc).__module__, type(exc).__name__, str(exc)]
            stable = np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == old_buffer
    if error is None:
        assert isinstance(result, np.ndarray) and result.ndim == 3
        assert result.size <= bound['max_output_elements']
        assert result.dtype.kind in 'bf'
    row = dict(id=spec['id'], api=API, result=encode(result, np), error=error,
               inputs_before=before, inputs_after=encode((args, options), np), events=events,
               warnings=[[type(w.message).__module__, type(w.message).__name__, str(w.message)] for w in notices],
               checks=dict(inputs_unchanged=before == encode((args, options), np),
                           policy_unchanged=stable,
                           state_restored=np.geterr() == old_policy and np.geterrcall() is old_handler and np.getbufsize() == old_buffer))
    assert all(row['checks'].values())
    json.dumps(row, allow_nan=False)
    return row
