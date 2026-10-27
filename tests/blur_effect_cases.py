"""Prospective blur-effect cases: exact source stages, no numerical allowance.

The initial owned domain is bounded positive rank1..3 ordinary real storage
after conversion. Nonfinite/subnormal/extreme and rank4 fixtures explicitly
retain primitive stages. Only the three declared provider-NaN fixtures admit
NaN word differences; every raw observation is retained. All other words,
including finite values, infinities and signed zeros, remain exact.
"""
API = 'measure.blur_effect'
STAGES = ('uniform_filter1d', 'correlate_sparse')
UNASSESSED = (
    'zero spatial axes through original C filtering',
    'forged storage and enormous shape/filter allocations',
    'custom image and h_size protocols beyond the explicit input errors',
    'nonfinite callback argument mutation and alternate floating rounding modes',
)


def cases():
    rows = []
    def add(name, **options):
        row = dict(id=name, shape=(9, 13), dtype='float64', layout='C',
                   pattern='texture', h_size=11, channel_axis=None,
                   reducer='default', route='native', outcome='return',
                   nan_mask=False, policy={})
        row.update(options)
        rows.append(row)
    for dtype in ('float32', 'float64'):
        add('rank1-'+dtype, shape=(19,), dtype=dtype)
    for dtype in ('uint8', 'float32', 'float64', 'int16', 'bool'):
        add('rank2-'+dtype, dtype=dtype)
    for dtype in ('float32', 'float64'):
        add('rank3-'+dtype, shape=(5, 7, 9), dtype=dtype)
    add('rank4-provider', shape=(4, 4, 4, 4), route='provider')
    add('scalar-none', shape=(), reducer='none', route='intake')
    add('scalar-default-error', shape=(), route='intake', outcome='error')
    for layout in ('F', 'step', 'reverse', 'endian', 'unaligned', 'readonly', 'metadata', 'broadcast'):
        add('layout-'+layout, dtype='float32', layout=layout)
    for dtype in ('uint8', 'float16', 'float32'):
        add('rgb-last-'+dtype, shape=(9, 13, 3), dtype=dtype, channel_axis=-1)
    add('rgb-first', shape=(3, 9, 13), dtype='float32', channel_axis=0)
    add('rgb-middle', shape=(9, 3, 13), channel_axis=1)
    for window in (1, 2, 31):
        add('window-'+str(window), h_size=window)
    add('window-numpy-int', h_size='numpy-int')
    for pattern in ('zeros', 'constant', 'ramp', 'impulse', 'signed'):
        add('pattern-'+pattern, dtype='float32', pattern=pattern)
    for shape in ((2,), (1, 7), (3, 5), (4, 4)):
        add('tiny-'+'x'.join(map(str, shape)), shape=shape)
    add('tiny-invalid-raise', shape=(3, 5), policy={'invalid': 'raise'}, outcome='error')
    for reducer in ('none', 'mean', 'mutate', 'warn', 'raise', 'typeerror'):
        add('reducer-'+reducer, reducer=reducer,
            outcome='error' if reducer in ('raise', 'typeerror') else 'return')
    for pattern in ('nan', 'inf', 'subnormal', 'extreme'):
        add('provider-'+pattern, pattern=pattern, route='provider',
            nan_mask=pattern in ('nan', 'inf', 'extreme'))
    add('half-gray-error', dtype='float16', route='intake', outcome='error')
    add('complex-error', dtype='complex64', route='intake', outcome='error')
    add('list-error', layout='list', route='intake', outcome='error')
    add('channel-axis-error', channel_axis=3, route='intake', outcome='error')
    add('channel-type-error', channel_axis='bad', route='intake', outcome='error')
    add('channel-size-error', shape=(9, 13, 4), channel_axis=-1, route='intake', outcome='error')
    for name, window in (('zero', 0), ('negative', -2), ('float', 2.0), ('string', 'bad')):
        add('window-'+name+'-error', h_size=window, route='intake', outcome='error')
    add('parallel-float32', shape=(513, 513), dtype='float32')
    assert len(rows) == 60 and len({row['id'] for row in rows}) == 60
    return rows


def fixture(spec, np):
    shape = tuple(spec['shape'])
    count = 1
    for size in shape:
        count *= size
    raw = ((np.arange(count, dtype=np.int64) * 37 + 11) % 251).reshape(shape)
    dtype = np.dtype(spec['dtype'])
    if dtype.kind == 'b':
        image = (raw % 3 == 0)
    elif dtype.kind in 'ui':
        image = raw.astype(dtype)
    else:
        image = (raw / 256).astype(dtype)
    pattern = spec['pattern']
    if pattern == 'zeros': image[...] = 0
    elif pattern == 'constant': image[...] = .5
    elif pattern == 'ramp':
        image[...] = np.arange(shape[-1], dtype=image.dtype) / 16
    elif pattern == 'impulse':
        image[...] = 0
        image[tuple(size//2 for size in shape)] = 1
    elif pattern == 'signed': image -= image.dtype.type(.5)
    elif pattern == 'nan': image[2, 3] = np.nan
    elif pattern == 'inf': image[2, 3] = np.inf
    elif pattern == 'subnormal':
        image[...] = np.array([1], dtype=np.uint64).view(np.float64)[0]
    elif pattern == 'extreme': image[...] = np.finfo(np.float64).max / 4

    layout = spec['layout']
    if layout == 'F': image = np.asfortranarray(image)
    elif layout == 'step':
        storage = np.empty(tuple(size*2 for size in shape), dtype=image.dtype)
        image_view = storage[tuple(slice(None, None, 2) for _ in shape)]
        image_view[...] = image
        image = image_view
    elif layout == 'reverse': image = image[::-1, ::-1]
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'unaligned':
        storage = bytearray(image.nbytes + 1)
        view = np.ndarray(shape, dtype=image.dtype, buffer=storage, offset=1)
        view[...] = image
        image = view
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'metadata': image = image.view(np.dtype(image.dtype, metadata={'unit': 'linear'}))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], shape)
    elif layout == 'list': image = image.tolist()

    events = []
    options = dict(h_size=np.int32(5) if spec['h_size'] == 'numpy-int' else spec['h_size'],
                   channel_axis=spec['channel_axis'])
    reducer = spec['reducer']
    if reducer == 'none': options['reduce_func'] = None
    elif reducer == 'mean': options['reduce_func'] = np.mean
    elif reducer != 'default':
        def reduce(values):
            events.append(['call', type(values).__name__, [
                [type(value).__module__, type(value).__name__, value.dtype.str,
                 value.tobytes().hex()] for value in values]])
            assert all(np.isfinite(value) for value in values)
            if reducer == 'mutate':
                values[0] = np.float64(.25)
                values.append(np.float32(.75))
                events.append(['mutated', len(values)])
                return values
            if reducer == 'warn':
                __import__('warnings').warn('blur reducer warning', RuntimeWarning)
                return values[-1]
            if reducer == 'raise': raise RuntimeError('blur reducer failure')
            raise TypeError('blur reducer body TypeError; no retry')
        options['reduce_func'] = reduce
    return image, options, events
