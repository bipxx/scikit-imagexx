"""Deterministic NMI corpus shared by parity tests and future isolated timing.

Construction and oracle calls belong outside any measured interval. The timed
subset contains only ordinary finite calls; this module performs no timing.
"""


def cases():
    import numpy as np

    result = []
    def add(name, a, b, bins=100, native=True, policy=None, trace=None, timed=False):
        result.append(dict(id=name, image0=a, image1=b, bins=bins,
                           native=native, policy=policy, trace=trace, timed=timed))

    dtypes = ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
              'uint64', 'int64', 'float16', 'float32', 'float64')
    for dtype in dtypes:
        x = (np.arange(63).reshape(7, 9) % 17).astype(dtype)
        y = ((np.arange(63).reshape(7, 9) * 7 + 3) % 19).astype(dtype)
        if x.dtype.kind == 'f':
            x = x / np.array(8, dtype=dtype)
            y = y / np.array(8, dtype=dtype)
        for layout in ('c', 'f', 'transpose', 'reverse', 'step', 'broadcast', 'endian', 'unaligned', 'readonly'):
            a, b = x.copy(), y.copy()
            if layout == 'f': a, b = np.asfortranarray(a), np.asfortranarray(b)
            elif layout == 'transpose': a, b = a.T, b.T
            elif layout == 'reverse': a, b = a[::-1, ::-1], b[:, ::-1]
            elif layout == 'step': a, b = a[::2, ::2], b[::2, ::2]
            elif layout == 'broadcast': a, b = np.broadcast_to(a[0], a.shape), np.broadcast_to(b[:, :1], b.shape)
            elif layout == 'endian': a, b = a.astype(a.dtype.newbyteorder('>')), b.astype(b.dtype.newbyteorder('>'))
            elif layout == 'unaligned':
                a = np.ndarray(a.shape, dtype=a.dtype, buffer=bytearray(a.nbytes + 1), offset=1)
                b = np.ndarray(b.shape, dtype=b.dtype, buffer=bytearray(b.nbytes + 1), offset=1)
                a[...] = x; b[...] = y
            elif layout == 'readonly': a.flags.writeable = False; b.flags.writeable = False
            add(dtype + '-' + layout, a, b, timed=layout in ('c', 'reverse'))
        for bins in (1, 2, 3, (7, 11)):
            add(dtype + '-bins-' + str(bins), x, y, bins)

    for left, right in (('uint8', 'float16'), ('int16', 'float32'), ('uint32', 'float32'), ('uint64', 'int64'), ('float32', 'float64')):
        add('mixed-' + left + '-' + right, np.arange(35, dtype=left).reshape(5, 7),
            (np.arange(35, dtype=right) % 11).reshape(5, 7), (13, 9))
    for shape0, shape1 in (((2, 7), (5, 3)), ((0, 3), (4, 5)), ((2, 1, 5), (1, 3, 2)), ((7,), (11,))):
        a = np.arange(np.prod(shape0), dtype=np.int16).reshape(shape0)
        b = np.arange(np.prod(shape1), dtype=np.int16).reshape(shape1)
        add('padding-' + str(shape0) + '-' + str(shape1), a, b, 7)
    for shape in ((), (1,), (2, 3), (2, 3, 4), (1, 2, 1, 3)):
        add('constant-' + str(shape), np.full(shape, 2.5), np.full(shape, -3.0))
    a = np.arange(32, dtype=np.float64)
    add('one-constant', np.zeros_like(a), a)
    add('perfect-correlation', a, a.copy(), 32)
    add('perfect-anticorrelation', a, a[::-1], 32)
    add('independent-grid', np.tile(np.arange(8), 8), np.repeat(np.arange(8), 8), 8)
    edge = np.linspace(-1., 1., 9)
    values = np.concatenate((edge, np.nextafter(edge, -np.inf), np.nextafter(edge, np.inf)))
    add('edge-neighbors', values, values[::-1], 8)
    add('signed-zeros', np.array([-0., +0., 1., -1.]), np.array([+0., -0., -1., 1.]), 2)
    add('numpy-scalars', np.float32(2.), np.int16(3))
    rare = np.zeros(2048); rare[-1] = 1
    add('ill-conditioned-rare-event', rare, rare[::-1], 2, native=False)
    for dtype in ('float16', 'float32', 'float64'):
        a = np.array([1, 1], dtype=dtype)
        a[1] = np.nextafter(a[1], np.array(2, dtype=dtype))
        add('narrow-' + dtype, a, a[::-1], 100, native=None)
        info = np.finfo(dtype)
        add('extreme-' + dtype, np.array([-info.max, info.max], dtype=dtype), a, native=False)
        add('subnormal-' + dtype, np.array([0, np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))], dtype=dtype), a, native=False)
    for kind, values in (('nan', [0., np.nan]), ('positive-inf', [0., np.inf]), ('negative-inf', [-np.inf, 0.])):
        add(kind, np.array(values), np.array([0., 1.]), native=False)
    for dtype in ('uint64', 'int64'):
        add('integer-inexact-' + dtype, np.array([2**53 + 1, 2**53 + 3], dtype=dtype), np.array([0, 1], dtype=dtype), native=False)
    add('empty', np.empty((0, 3)), np.empty((0, 5)), native=False)
    add('rank-mismatch', np.zeros((2, 3)), np.zeros(6), native=True)
    add('list-input', [[1, 2]], np.array([[1, 2]]), native=False)
    add('complex', np.array([1 + 2j, 2 + 1j]), np.array([1., 2.]), native=False)
    add('object', np.array([1., 2.], object), np.array([2., 1.]), native=False)
    add('masked', np.ma.array([1., 2., 3.], mask=[0, 1, 0]), np.array([3., 2., 1.]), native=False)
    for name, bins in [('zero', 0), ('negative', -1), ('float', 2.), ('boolean', True),
                       ('numpy-int', np.int64(7)), ('wrong-length', (2, 3, 4)),
                       ('edges', [np.array([0., 1., 3.]), np.array([0., 2., 3.])]),
                       ('descending', [[3., 0.], [0., 3.]]), ('duplicate-edges', [[0., 0., 3.], [0., 3.]])]:
        add('custom-bins-' + name, np.array([0., 1., 2., 3.]), np.array([3., 2., 1., 0.]), bins, native=False)
    trace = []
    class Index:
        def __index__(self): trace.append('__index__'); return 7
        def __lt__(self, value): trace.append('__lt__'); return False
    add('index-protocol', np.arange(7.), np.arange(7.), Index(), native=False, trace=trace)
    trace2 = []
    class Array(np.ndarray):
        def __array_function__(self, func, types, args, kwargs):
            trace2.append(func.__name__)
            return super().__array_function__(func, types, args, kwargs)
    add('array-protocol', np.arange(7.).view(Array), np.arange(7.), native=False, trace=trace2)
    for policy in ('ignore', 'raise', 'call', 'log'):
        add('zero-entropy-policy-' + policy, np.zeros(4), np.ones(4), policy=policy,
            native=policy in ('ignore', 'raise'))
    large = (np.arange(512 * 513, dtype=np.uint32).reshape(512, 513) % 257).astype(np.float64)
    add('worker-sized', large, large[:, ::-1], (31, 29), timed=True)
    return result


def benchmark_cases():
    """One-worker cases include adverse/provider routes; report every route.

    These cases must stay in benchmark coverage even while their native route
    is pending. No timer or execution is provided by this factory.
    """
    adverse = {'ill-conditioned-rare-event', 'narrow-float16', 'narrow-float32',
               'extreme-float16', 'extreme-float32', 'integer-inexact-uint64',
               'constant-(2, 3)', 'one-constant', 'custom-bins-edges'}
    return [case for case in cases() if case['timed'] or case['id'] in adverse]
