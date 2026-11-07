"""One supplied call, source-frozen factory; no resolver or capture authority."""
import json
import warnings
from crop_observer import encode as legacy_encode
from flood_fill_exception_codec import encode_exception_args
from flood_fill_contract import factory


def encode(value, np):
    if isinstance(value, complex):
        return dict(python_complex=[legacy_encode(value.real, np), legacy_encode(value.imag, np)])
    if isinstance(value, (tuple, list)):
        return dict(container=type(value).__name__, items=[encode(v, np) for v in value])
    if isinstance(value, dict):
        return {k: encode(v, np) for k, v in value.items()}
    return legacy_encode(value, np)


def observe(call, spec, np):
    args, options = factory(spec)
    before = encode((args, options), np)
    arrays = [v for v in args + list(options.values()) if isinstance(v, np.ndarray)]
    pointers = [a.ctypes.data for a in arrays]
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result = None
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = call(*args, **options)
        except Exception as exc:
            error = dict(module=type(exc).__module__, name=type(exc).__name__,
                         qualname=type(exc).__qualname__, message=str(exc),
                         args=encode_exception_args(exc.args, np))
    aliases = []
    if isinstance(result, np.ndarray):
        aliases = [dict(identity=result is a, base_is_input=result.base is a,
                        shares_memory=bool(np.shares_memory(result, a))) for a in arrays]
    row = dict(id=spec['id'], api=spec['api'], before=before,
               after=encode((args, options), np), result=None if error else encode(result, np),
               error=error, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
               aliases=aliases, pointers_stable=[p == a.ctypes.data for p, a in zip(pointers, arrays)],
               numpy_before=dict(policy=policy, handler_type=type(handler).__name__, buffer=buffer),
               numpy_after=dict(policy=np.geterr(), handler_type=type(np.geterrcall()).__name__, buffer=np.getbufsize()),
               supplied_call_attempts=1)
    json.dumps(row, allow_nan=False)
    return row
