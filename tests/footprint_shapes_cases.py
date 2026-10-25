"""Prospective footprint fixtures. Importing this module needs only stdlib.

No reference observations or candidate behavior are encoded here. The caller
supplies an authenticated NumPy module only after source/safety admission.
"""
import copy

APIS = ('ellipse', 'octagon', 'octahedron', 'star', 'footprint_from_sequence')
DTYPES = ('bool', 'uint8', 'int16', 'float32', 'float64', 'complex64', 'S2', 'U2')


def cases():
    rows = []
    def add(api, tag, args, **kwargs):
        rows.append(dict(id=api + '/' + tag, api=api, args=args, kwargs=kwargs,
                         partition='reference', policy='default'))
    for w, h in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 3), (3, 1),
                 (2, 4), (4, 2), (3, 3), (4, 5), (7, 2), (8, 8)):
        for mode in (None, 'crosses'):
            add('ellipse', f'{w}-{h}-{mode}', [w, h], decomposition=mode)
    for m, n in ((0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0),
                 (1, 1), (2, 2), (3, 1), (1, 3), (4, 3), (7, 4)):
        for mode in (None, 'sequence'):
            add('octagon', f'{m}-{n}-{mode}', [m, n], decomposition=mode)
    for radius in range(9):
        for mode in (None, 'sequence'):
            add('octahedron', f'{radius}-{mode}', [radius], decomposition=mode)
    for radius in (0.5, 1.5, 2.25, 3.75):
        add('octahedron', 'fractional-' + str(radius), [radius])
    for a in range(8):
        add('star', str(a), [a])
    for dtype in DTYPES:
        for api, args, modes in (('ellipse', [3, 2], (None, 'crosses')),
                                 ('octagon', [3, 2], (None, 'sequence')),
                                 ('octahedron', [3], (None, 'sequence')),
                                 ('star', [3], (None,))):
            for mode in modes:
                kw = {'dtype': dtype}
                if api != 'star': kw['decomposition'] = mode
                add(api, 'dtype-' + dtype + '-' + str(mode), args, **kw)
    for api, args in (('ellipse', [2, 2]), ('octagon', [2, 2]), ('octahedron', [2])):
        for mode in ('unknown', '', 7):
            add(api, 'invalid-decomposition-' + repr(mode), args, decomposition=mode)
    # Each error is a Python/NumPy allocation/index/conversion path; no forged
    # descriptors, custom allocation callbacks, huge sizes or Cython inputs.
    for api, arguments in (
        ('ellipse', ([-1, 2], [2, -1], [1.5, 2], [None, 2])),
        ('octagon', ([0, 0], [-1, 0], [1.5, 1], [None, 1])),
        ('octahedron', ([-1], [None], ['x'], [1+2j])),
        ('star', ([-1], [None], [1.5], ['x']))):
        for j, args in enumerate(arguments):
            # Complex is represented explicitly for JSON portability.
            if api == 'octahedron' and j == 3: args = [{'complex': [1, 2]}]
            add(api, 'intake-' + str(j), args)
    for pattern in ('cross', 'box', 'asymmetric', 'even', 'line', 'cube',
                    'mixed', 'first-zero', 'later-zero', 'repeat-two',
                    'bool-repeat', 'singleton'):
        for layout in ('C', 'F', 'step', 'reverse', 'readonly'):
            add('footprint_from_sequence', pattern + '-' + layout, [],
                pattern=pattern, layout=layout)
    for bad in ('array', 'empty', 'bad-item', 'float-repeat', 'not-sequence',
                'missing-repeat', 'wrong-rank'):
        add('footprint_from_sequence', 'invalid-' + bad, [], invalid=bad)
    return copy.deepcopy(rows)


def future_cases():
    """Unexecuted whole-call workload candidates; never reference credit."""
    rows = []
    for size in (16, 64, 128, 256):
        for api, args in (('ellipse', [size-1, size-1]),
                          ('octagon', [size, size//2]),
                          ('octahedron', [min(size//8, 24)]),
                          ('star', [size//2])):
            rows.append(dict(id=f'{api}/future-{size}', api=api, args=args,
                             kwargs={}, partition='future-unexecuted'))
        rows.append(dict(id=f'footprint_from_sequence/future-{size}',
                         api='footprint_from_sequence', args=[],
                         kwargs={'pattern': 'box', 'layout': 'C',
                                 'future_repetitions': size//2},
                         partition='future-unexecuted'))
    return rows


def fixture(spec, np):
    args = [complex(*x['complex']) if isinstance(x, dict) else x for x in spec['args']]
    kw = dict(spec['kwargs'])
    if spec['api'] != 'footprint_from_sequence':
        if 'dtype' in kw: kw['dtype'] = np.dtype(kw['dtype'])
        return tuple(args), kw
    bad = kw.get('invalid')
    if bad:
        fp = np.ones((3, 3), dtype=np.uint8)
        value = {'array': fp, 'empty': (), 'bad-item': (7,),
                 'float-repeat': ((fp, 1.5),), 'not-sequence': None,
                 'missing-repeat': ((fp,),),
                 'wrong-rank': ((fp, 1), (np.ones((3,), dtype=np.uint8), 1))}[bad]
        return (value,), {}
    pattern, layout = kw['pattern'], kw['layout']
    cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    data = {'cross': cross, 'box': [[1]*3]*3,
            'asymmetric': [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
            'even': [[1, 0], [1, 1]], 'line': [[1, 1, 1]],
            'cube': [[[1]*3]*3]*3, 'singleton': [[1]]}.get(pattern, cross)
    def arrange(data):
        a = np.array(data, dtype=np.uint8, order='F' if layout == 'F' else 'C')
        if layout == 'step':
            b = np.zeros(tuple(2*s for s in a.shape), dtype=a.dtype)
            view = b[tuple(slice(None, None, 2) for _ in a.shape)]
            view[...] = a; a = view
        if layout == 'reverse': a = a[tuple(slice(None, None, -1) for _ in a.shape)]
        if layout == 'readonly': a.flags.writeable = False
        return a
    fp = arrange(data)
    count = kw.get('future_repetitions', 2 if pattern == 'repeat-two' else 1)
    if pattern == 'first-zero': count = 0
    if pattern == 'bool-repeat': count = True
    sequence = [(fp, count)]
    if pattern in ('mixed', 'later-zero'):
        sequence.append((arrange([[1, 1, 1]]), 0 if pattern == 'later-zero' else 2))
    return (tuple(sequence),), {}
