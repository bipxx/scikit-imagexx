"""Prospective block-SUM contract, frozen before original/candidate outputs.

Ordinary bool/integer/f16/f32/f64 default sums require owned block arithmetic.
Integer promotion/wrap and all metadata are exact. For each finite floating
block, S=sum(abs(typed padded inputs)), m=block volume and u=accumulator eps/2:
the difference of two ordered sums is bounded by 2*gamma_m*S, gamma_m=m*u/(1-m*u).
Half uses a float32 accumulator and additionally accounts for two final half
roundings: eps16*(abs(reference)+gamma_m*S)/(1-eps16/2). No relative floor.
This is a prospective forward-error bound, not a fitted cross-backend tolerance.
Nonfinite masks, source warnings/errors, zero signs, integer words and state
remain exact. Custom functions/kwargs and exceptional floating sums are explicit
primitive providers and earn no owned reduction credit.
"""
import math

API = 'measure.block_reduce'
STAGE = 'block_reduce_sum'
UNASSESSED = ('forged/stale storage', 'arbitrary array protocols', 'object arithmetic',
              'extended dtypes', 'unbounded geometry', 'custom concurrent mutation')


def cases():
    rows = []
    def add(id, dtype='float64', shape=(6, 8), block=2, **options):
        row = dict(id=id, dtype=dtype, shape=shape, block=block,
                   layout='C', pattern='ordinary', route='native',
                   outcome='return', policy={})
        row.update(options); rows.append(row)
    for dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                  'int64', 'uint64', 'float16', 'float32', 'float64'):
        add('dtype-' + dtype, dtype)
    for name, shape, block in [('scalar', (), 2), ('line', (7,), 3),
            ('nondivisible', (5, 7), (2, 3)), ('volume', (3, 5, 7), (1, 2, 3)),
            ('rank4', (2, 3, 4, 5), (1, 2, 2, 3)), ('identity', (3, 5), 1),
            ('large-block', (2, 3), (4, 5)), ('numpy-int-block', (6, 8), 'np-int')]:
        add('geometry-' + name, shape=shape, block=block)
    for layout in ('F', 'step', 'reverse', 'endian', 'readonly', 'unaligned', 'metadata', 'broadcast'):
        add('layout-' + layout, dtype='float32', layout=layout)
    add('padding-integer-fraction', dtype='int16', shape=(3, 5), cval=1.75)
    add('padding-unsigned-negative', dtype='uint8', shape=(3, 5), cval=-1, outcome='source')
    add('padding-float32', dtype='float32', shape=(3, 5), cval='np-f32')
    add('padding-nan', shape=(3, 5), cval='nan', route='provider')
    add('padding-complex-real', shape=(3, 5), cval='complex', route='provider', outcome='source')
    for n in (7, 8, 9, 127, 128, 129):
        add('association-' + str(n), dtype='float32', shape=(2*n,), block=n, pattern='cancellation')
    add('wrap-int64', dtype='int64', pattern='wide-signed')
    add('wrap-uint64', dtype='uint64', pattern='wide-unsigned')
    add('bool-noncanonical', dtype='bool', pattern='bool-bytes')
    add('empty-2d', shape=(0, 4))
    add('empty-3d', shape=(2, 0, 3), block=(1, 2, 3))
    add('negative-zero', pattern='negative-zero')
    add('nonfinite', pattern='nonfinite', route='provider')
    add('overflow-warn', dtype='float32', pattern='overflow', route='provider')
    add('overflow-raise', dtype='float32', pattern='overflow', route='provider',
        outcome='error', policy={'over': 'raise'})
    for kwargs in ('dtype-f64', 'dtype-i8', 'initial', 'keepdims', 'where', 'out', 'axis', 'unknown'):
        add('kwargs-' + kwargs, dtype='float32', kwargs=kwargs, route='provider',
            outcome='error' if kwargs in ('axis', 'unknown') else 'return')
    for func in ('mean', 'max', 'min', 'median', 'custom', 'raise'):
        add('func-' + func, func=func, route='provider', outcome='error' if func == 'raise' else 'return')
    add('callback-mutates-view', func='mutate', route='provider')
    add('callback-mutates-padding', shape=(5, 7), func='mutate', route='provider')
    add('complex-sum', dtype='complex64', route='provider')
    for name, block in [('length', (2,)), ('zero', 0), ('negative', -2),
                        ('fractional', 2.5), ('float-integral', 2.0), ('list', [2, 2])]:
        add('invalid-' + name, block=block, route='error', outcome='error')
    add('image-list', layout='list', route='error', outcome='error')
    assert len(rows) == 72 and len({s['id'] for s in rows}) == 72
    return rows


def fixture(spec, np):
    shape = spec['shape']; n = math.prod(shape); dtype = np.dtype(spec['dtype'])
    values = np.arange(n, dtype=np.int64) % 19
    if dtype.kind in 'fc': values = (values - 9) / 7
    elif dtype.kind == 'i': values = values - 9
    image = np.array(values, dtype=dtype).reshape(shape).copy()
    p = spec['pattern']
    if p == 'cancellation':
        image.fill(1); image.flat[0] = 2**24; image.flat[-1] = -2**24
    elif p == 'wide-signed': image.flat[:] = np.resize(np.array([2**63-1, 2**63-1, -2**63, -3], dtype=dtype), n)
    elif p == 'wide-unsigned': image.flat[:] = np.resize(np.array([2**64-1, 2**64-1, 3, 7], dtype=dtype), n)
    elif p == 'bool-bytes': image.view(np.uint8).flat[:] = np.resize(np.array([0, 2, 128, 255], dtype=np.uint8), n)
    elif p == 'negative-zero': image.fill(-0.)
    elif p == 'nonfinite': image.flat[:4] = [np.nan, np.inf, -np.inf, -0.]
    elif p == 'overflow': image.fill(np.finfo(np.float32).max)
    if dtype.kind == 'c': image += np.complex64(0.25j)
    layout = spec['layout']
    if layout == 'F': image = np.asfortranarray(image)
    elif layout == 'step':
        backing = np.zeros(tuple(s*2 for s in image.shape), dtype=image.dtype)
        view = backing[tuple(slice(None, None, 2) for _ in image.shape)]; view[...] = image; image = view
    elif layout == 'reverse': image = image[::-1, ::-1]
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'unaligned':
        backing = np.empty(image.nbytes+1, dtype=np.uint8)
        view = np.ndarray(image.shape, dtype=image.dtype, buffer=backing, offset=1); view[...] = image; image = view
    elif layout == 'metadata': image = image.astype(np.dtype(image.dtype, metadata={'units': 'block-values'}))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], image.shape)
    elif layout == 'list': image = image.tolist()
    block = np.int64(2) if spec['block'] == 'np-int' else spec['block']
    options = dict(block_size=block); events = []
    if 'cval' in spec:
        v = spec['cval']; options['cval'] = {'np-f32': np.float32(1/3), 'nan': np.nan, 'complex': 1+2j}.get(v, v) if isinstance(v, str) else v
    kw = spec.get('kwargs')
    if kw:
        options['func_kwargs'] = {'dtype-f64': dict(dtype=np.float64), 'dtype-i8': dict(dtype=np.int8),
            'initial': dict(initial=3.5), 'keepdims': dict(keepdims=True), 'where': dict(where=False),
            'out': dict(out=np.full((3, 4), -91., dtype=np.float64)), 'axis': dict(axis=0), 'unknown': dict(unexpected=1)}[kw]
    kind = spec.get('func')
    if kind in ('mean', 'max', 'min', 'median'): options['func'] = getattr(np, kind)
    elif kind:
        def callback(blocked, axis, gain=1, note=None):
            events.append(dict(shape=list(blocked.shape), strides=list(blocked.strides), dtype=blocked.dtype.str,
                               writeable=bool(blocked.flags.writeable), axis=list(axis), gain=gain, note=note))
            if kind == 'raise': raise RuntimeError('intentional block callback failure')
            if kind == 'mutate': blocked.flat[0] = 17
            return np.sum(blocked, axis=axis) * gain
        options['func'] = callback
        if kind == 'custom': options['func_kwargs'] = dict(gain=2, note='literal callback kwargs')
    return image, options, events


def error_limits(spec, blocked, expected, np):
    if spec['route'] != 'native' or expected.dtype.kind not in 'f': return np.zeros(expected.shape, dtype=np.float64)
    rank = len(spec['shape']); axes = tuple(range(rank, blocked.ndim))
    m = math.prod(blocked.shape[rank:]); component = np.float32 if expected.dtype.itemsize <= 4 else np.float64
    u = float(np.finfo(component).eps)/2; assert m*u < 1
    gamma = m*u/(1-m*u)
    S = np.sum(np.abs(np.asarray(blocked, dtype=np.float64)), axis=axes)
    bound = 2*gamma*S
    if expected.dtype.itemsize == 2:
        eps = float(np.finfo(np.float16).eps)
        bound += eps*(np.abs(np.asarray(expected, dtype=np.float64))+gamma*S)/(1-eps/2)
    finite = np.isfinite(expected)
    assert np.all(np.isfinite(bound[finite])) and np.all(bound[finite] >= 0)
    return bound
