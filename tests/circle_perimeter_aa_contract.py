"""Bounded source-safe circle AA fixtures; no draw calls during construction."""
if __package__:
    from .draw_coordinates_contract import ScalarProvider
else:
    from draw_coordinates_contract import ScalarProvider
API = 'draw.circle_perimeter_aa'
MAX_OUTPUT_ELEMENTS = 4096
INTP_MIN, INTP_MAX = -(1 << 63), (1 << 63) - 1
UNASSESSED = (
    'All empty/short shape carriers: unchecked shape[0]/shape[1]; never call originals',
    'Radius squares, negation, recurrence or origin translation outside signed intptr range',
    'Arbitrary shape indexing/scalar protocols, object/complex/nonfinite shapes and conversions',
    'Unbounded allocation, concurrent mutation, forged storage and allocation failure',
    'Floating environment outside ordinary default rounding',
    'No native, frontend, SIMD, worker or performance completion credit from reference observations')


def cases():
    result = []
    def add(name, args, kind='ordinary', kwargs=None, safe=None, holdout=False):
        result.append(dict(id='circle-aa/' + name, api=API, kind=kind, args=list(args),
                           safe_args=list(args) if safe is None else safe,
                           kwargs={} if kwargs is None else kwargs, policy={}, holdout=holdout))
    for radius in (-3, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 17, 31):
        add('radius-' + str(radius), (0, 0, radius), holdout=radius in (13, 17))
    for name, args in (
        ('positive-origin', (5, 7, 4)), ('negative-origin', (-4, -7, 5)),
        ('translated', (2**30, -2**30, 7)),
        ('near-intp-limits', (INTP_MAX - 64, INTP_MIN + 64, 7))):
        add(name, args, holdout=name == 'near-intp-limits')
    for name, args, shape, kind in (
        ('clip', (4, 5, 5), [8, 9], 'shape-tuple'),
        ('clip-away', (-20, -20, 3), [4, 6], 'shape-tuple'),
        ('zero-shape', (0, 0, 3), [0, 5], 'shape-tuple'),
        ('negative-shape', (0, 0, 3), [-1, 6], 'shape-tuple'),
        ('extra-shape', (3, 4, 4), [6, 7, 8], 'shape-tuple'),
        ('list-shape', (3, 4, 4), [6, 7], 'shape-list'),
        ('array-shape', (4, 5, 5), [8, 9], 'shape-intp'),
        ('reverse-shape', (4, 5, 5), [8, 9], 'shape-reverse'),
        ('float-shape', (3, 4, 4), [6.5, 7.5], 'shape-tuple'),
        ('uint64-shape', (4, 5, 5), [8, 9], 'shape-uint64')):
        add(name, args, kind, {'shape': shape}, holdout=name == 'uint64-shape')
    for dtype in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
        add('np-' + dtype, (1, 2, 5), 'np-' + dtype)
    for kind, args, safe in (
        ('bool', (False, True, True), [0, 1, 1]),
        ('scalar-array', (1, 2, 5), [1, 2, 5]),
        ('provider-first', (1, 2, 5), [1, 2, 5]),
        ('providers-all', (1, 2, 5), [1, 2, 5]),
        ('float-first', (1.0, 2, 5), [1, 2, 5]),
        ('np-float-radius', (1, 2, 5.5), [1, 2, 5])):
        add(kind, args, kind, safe=safe)
    for kind, args, kwargs in (
        ('overflow-first', [2**100, 2, 5], {}),
        ('overflow-last', [1, 2, 2**100], {}),
        ('none-first', [None, 2, 5], {}),
        ('missing', [1, 2], {}),
        ('unexpected-keyword', [1, 2, 5], {'method': 'bresenham'})):
        add(kind, args, kind, kwargs, safe=[])
    add('fp-raise', (0, 0, 5))
    result[-1]['policy'] = dict(invalid='raise', divide='raise', over='raise')
    return result


def source_safe(spec, maximum=MAX_OUTPUT_ELEMENTS):
    values = spec['safe_args']
    if not values:
        assert spec['kind'] in ('overflow-first', 'overflow-last', 'none-first', 'missing', 'unexpected-keyword')
        return dict(preentry_rejection=True)
    assert len(values) == 3 and all(type(v) is int for v in values)
    origin_r, origin_c, radius = values
    assert abs(radius) <= 4096 and radius * radius <= INTP_MAX
    extent = max(abs(radius), 1)
    assert all(INTP_MIN + extent <= v <= INTP_MAX - extent for v in (origin_r, origin_c))
    shape = spec['kwargs'].get('shape')
    if shape is not None:
        assert type(shape) is list and len(shape) >= 2
        assert all(type(v) in (int, float) and abs(v) <= 8192 for v in shape)
    bound = 8 + 16 * max(radius - 1, 0)
    assert 3 * bound <= maximum
    return dict(preentry_rejection=False, squared_radius=radius * radius,
                output_entries_upper_bound=bound)


def fixture(spec, np):
    source_safe(spec)
    args, options, events, buffers = list(spec['args']), spec['kwargs'].copy(), [], {}
    kind = spec['kind']
    if kind.startswith('np-') and kind != 'np-float-radius':
        args = [np.dtype(kind[3:]).type(v) for v in args]
    elif kind == 'np-float-radius':
        args[2] = np.float64(args[2])
    elif kind == 'scalar-array':
        args[0] = np.array(args[0], dtype=np.int64)
        buffers['r'] = args[0]
    elif kind == 'provider-first':
        args[0] = ScalarProvider(args[0], events, 'r')
    elif kind == 'providers-all':
        args = [ScalarProvider(v, events, name) for v, name in zip(args, ('r', 'c', 'radius'))]
    if 'shape' in options:
        shape = options['shape']
        if kind == 'shape-tuple':
            options['shape'] = tuple(shape)
        elif kind == 'shape-list':
            options['shape'] = list(shape)
        else:
            dtype = np.uint64 if kind == 'shape-uint64' else np.intp
            options['shape'] = np.array(shape[::-1] if kind == 'shape-reverse' else shape, dtype=dtype)
            if kind == 'shape-reverse':
                options['shape'] = options['shape'][::-1]
            buffers['shape'] = options['shape']
    return tuple(args), options, events, buffers


def benchmark_cases():
    return [dict(id='circle-aa/' + name, api=API, kind='ordinary', args=list(args),
                 safe_args=list(args), kwargs=kwargs, policy={}, holdout=False)
            for name, args, kwargs in (
                ('tiny', (0, 0, 2), {}), ('zero', (0, 0, 0), {}),
                ('cache', (0, 0, 32), {}), ('long', (0, 0, 4096), {}),
                ('clipped', (128, 128, 256), {'shape': [256, 256]}),
                ('clipped-away', (-8192, -8192, 128), {'shape': [64, 64]}),
                ('translated', (2**30, -2**30, 128), {}), ('negative', (0, 0, -3), {}))]
