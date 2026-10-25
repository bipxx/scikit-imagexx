"""Prospective original-only observer; no scientific imports at module import.

Requires explicit authenticated NumPy and original morphology modules. This
file provides no native admission and performs no calls unless observe runs.
"""
import base64
import hashlib
import sys
import warnings
import zlib


def snapshot(value, np):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('object pointer words are outside this initial packet')
        if value.size > 32768 or value.nbytes > 262144:
            raise ValueError('reference snapshot budget exceeded')
        raw = value.tobytes(order='C')
        chain, current, seen = [], value, set()
        while isinstance(current, np.ndarray):
            if id(current) in seen: raise ValueError('cyclic owner chain')
            seen.add(id(current))
            if current.nbytes > 262144:
                raise ValueError('owner snapshot budget exceeded')
            owner_raw = current.tobytes(order='C')
            chain.append(dict(dtype=current.dtype.str, shape=list(current.shape),
                              strides=list(current.strides), own=bool(current.flags.owndata),
                              writeable=bool(current.flags.writeable),
                              raw_size=len(owner_raw), sha256=hashlib.sha256(owner_raw).hexdigest(),
                              raw_zlib_base64=base64.b64encode(zlib.compress(owner_raw)).decode('ascii')))
            current = current.base
        if current is not None: raise TypeError('uncertified non-ndarray owner')
        return dict(kind='array', dtype=value.dtype.str, shape=list(value.shape),
                    strides=list(value.strides), c=bool(value.flags.c_contiguous),
                    f=bool(value.flags.f_contiguous), aligned=bool(value.flags.aligned),
                    writeable=bool(value.flags.writeable), own=bool(value.flags.owndata),
                    raw_size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode('ascii'),
                    owners=chain)
    if isinstance(value, np.generic):
        return dict(kind='numpy-scalar', type=type(value).__name__, dtype=value.dtype.str,
                    raw=value.tobytes().hex())
    if isinstance(value, np.dtype): return dict(kind='dtype', str=value.str)
    if isinstance(value, (tuple, list)):
        return dict(kind=type(value).__name__, items=[snapshot(x, np) for x in value])
    if isinstance(value, dict): return {k: snapshot(v, np) for k, v in value.items()}
    if value is None or isinstance(value, (bool, str, int, float)): return value
    if isinstance(value, complex): return dict(kind='complex', real=value.real, imag=value.imag)
    raise TypeError('uncertified observer value: ' + type(value).__name__)


def arrays(value, np):
    if isinstance(value, np.ndarray): return [value]
    if isinstance(value, (tuple, list)):
        return [a for x in value for a in arrays(x, np)]
    if isinstance(value, dict): return [a for x in value.values() for a in arrays(x, np)]
    return []


def entry_profiler(code):
    """Count same-code recursion without retaining a frame, argument or array."""
    counts = dict(outer_call=0, outer_return=0, nested_call=0, nested_return=0,
                  total_call=0, total_return=0)
    depth = 0
    def profile(frame, event, arg):
        nonlocal depth
        if frame.f_code is not code: return
        if event == 'call':
            counts['total_call'] += 1
            counts['outer_call' if depth == 0 else 'nested_call'] += 1
            depth += 1
        elif event == 'return':
            depth -= 1
            counts['total_return'] += 1
            counts['outer_return' if depth == 0 else 'nested_return'] += 1
    return counts, profile


def observe(spec, factory, np, original, journal):
    """One small original call; flush actual row before checking invariants.

    Authentication/RECORD/source/versions must be performed by a future runner
    before importing fixtures. This is deliberately not an executable runner.
    """
    if any(k == 'skimagexx' or k.startswith('skimagexx.') for k in sys.modules):
        raise RuntimeError('original-only observer refuses a candidate import')
    function = getattr(original, spec['api'])
    if not function.__module__.startswith('skimage.morphology'):
        raise RuntimeError('wrong original entry')
    if sys.getprofile() is not None: raise RuntimeError('profiler already installed')
    state = (dict(np.geterr()), np.geterrcall(), np.getbufsize())
    counts, profile = entry_profiler(function.__code__)
    try:
        args, kwargs = factory.fixture(spec, np)
        before = snapshot((args, kwargs), np)
    except Exception as error:
        journal(dict(id=spec['id'], phase='fixture-observer-error',
                     error=[type(error).__module__, type(error).__name__, str(error)]))
        raise
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sys.setprofile(profile)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            error = [type(exc).__module__, type(exc).__name__, str(exc)]
        finally:
            sys.setprofile(None)
    try:
        returned_arrays, inputs = arrays(result, np), arrays((args, kwargs), np)
        row = dict(id=spec['id'], phase='actual-original', before=before,
                   after=snapshot((args, kwargs), np), result=snapshot(result, np),
                   error=error, entries=counts,
                   warnings=[[x.category.__module__, x.category.__name__, str(x.message)]
                             for x in caught],
                   aliases=[[r is a, bool(np.shares_memory(r, a))]
                            for r in returned_arrays for a in inputs],
                   output_aliases=[[r is a, bool(np.shares_memory(r, a))]
                                   for r in returned_arrays for a in returned_arrays],
                   state_before=dict(policy=state[0], handler_none=state[1] is None,
                                     buffer=state[2]),
                   restored=(state[0] == np.geterr() and state[1] is np.geterrcall()
                             and state[2] == np.getbufsize()),
                   entry_identity_preserved=getattr(original, spec['api']) is function)
    except Exception as exc:
        journal(dict(id=spec['id'], phase='postcall-observer-error', error=[
            type(exc).__module__, type(exc).__name__, str(exc)], entries=counts))
        raise
    journal(row)
    assert row['restored'] and row['entry_identity_preserved']
    # Three source branches recurse once into their own dense generator. These
    # are genuine entries, not independent cases or repeated capture attempts.
    api, mode = spec['api'], spec['kwargs'].get('decomposition')
    nested = int((api == 'ellipse' and mode == 'crosses') or
                 (api == 'octahedron' and mode == 'sequence') or
                 (api == 'octagon' and mode == 'sequence' and
                  spec['args'][0] <= 2 and spec['args'][1] <= 2))
    assert counts == dict(outer_call=1, outer_return=1, nested_call=nested,
                          nested_return=nested, total_call=1+nested, total_return=1+nested)
    return row
