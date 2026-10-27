"""Prospective Crofton contracts; no candidate observations informed this policy.

Threshold/stencil histogram counts are exact. The 16-term binary64 weighted
sum uses the source-ordered coefficient values and the bound 2*gamma_32*S,
where u=2**-53, gamma_32=32*u/(1-32*u), and S=sum(abs(c[i])*h[i]).
The factor two covers reference and candidate multiply/add rounding trees.
S and the budget are evaluated with exact rational arithmetic, then rounded
up once to binary64. Zero, dtype, nonfinite masks, warnings and state are
exact. A different floating reduction tree receives no bit-identity claim
against the original; native worker1/4 results must be bit-identical.
"""
from fractions import Fraction
import math
import warnings

API = 'measure.perimeter_crofton'
STAGES = ('crofton_counts', 'crofton_perimeter')
UNASSESSED = ('forged/stale storage', 'extended real dtypes', 'noncanonical bool bytes',
              'arbitrary ndarray subclass protocols', 'arbitrary object comparisons')


class DirectionProbe:
    def __init__(self, mode, events):
        self.mode, self.events = mode, events

    def __repr__(self):
        return 'DirectionProbe(' + repr(self.mode) + ')'

    def __eq__(self, other):
        self.events.append(('directions.eq', other))
        if self.mode == 'raise':
            raise RuntimeError('crofton direction probe')
        if self.mode == 'warn':
            warnings.warn('crofton direction probe', UserWarning)
        return self.mode != 'warn'


def cases():
    rows = []

    def add(id, pattern='labels', dtype='uint8', layout='C', directions=4,
            route='native', outcome='return', policy=None):
        rows.append(dict(id=id, pattern=pattern, dtype=dtype, layout=layout,
                         directions=directions, route=route, outcome=outcome, policy=policy or {}))

    for pattern in ('empty00', 'empty05', 'empty40', 'single0', 'single1',
                    'row', 'column', 'ring', 'diagonal', 'large'):
        add('geometry-' + pattern, pattern)
    for dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                  'int64', 'uint64', 'float16', 'float32', 'float64'):
        add('dtype-' + dtype, dtype=dtype)
    for layout in ('F', 'step', 'reverse', 'endian', 'readonly', 'unaligned', 'metadata', 'broadcast'):
        add('layout-' + layout, pattern='even', dtype='float64', layout=layout)
    for name, value in (('two', 2), ('numpy-int', 'np-int2'), ('numpy-float', 'np-float2'),
                        ('three', 3), ('zero', 0), ('none', None), ('text', '2'),
                        ('array-scalar', 'array-scalar'), ('array-many', 'array-many'),
                        ('eq-true', 'probe-true'), ('eq-warn', 'probe-warn'), ('eq-raise', 'probe-raise')):
        add('directions-' + name, directions=value,
            outcome='error' if name in ('array-many', 'eq-raise') else 'return')
    add('nonfinite', pattern='nonfinite', dtype='float64')
    add('nonfinite-raise', pattern='nonfinite', dtype='float64', policy={'invalid': 'raise'})
    add('complex-provider', pattern='complex', dtype='complex128', route='provider', outcome='source')
    add('object-provider', pattern='object', dtype='object', route='provider')
    for pattern in ('scalar', 'rank1', 'rank3', 'list'):
        add('intake-' + pattern, pattern, route='intake', outcome='error')
    assert len(rows) == 50 and len({row['id'] for row in rows}) == 50
    return rows


def fixture(spec, np):
    pattern, dtype = spec['pattern'], spec['dtype']
    shapes = {'empty00': (0, 0), 'empty05': (0, 5), 'empty40': (4, 0),
              'single0': (1, 1), 'single1': (1, 1), 'row': (1, 7), 'column': (6, 1),
              'ring': (7, 9), 'diagonal': (6, 6), 'large': (513, 513), 'even': (6, 8)}
    shape = shapes.get(pattern, (5, 7))
    if pattern in ('scalar', 'rank1', 'rank3'):
        image = np.ones({'scalar': (), 'rank1': (7,), 'rank3': (2, 3, 4)}[pattern], dtype=dtype)
    elif pattern == 'list':
        image = [[0, 1], [1, 0]]
    elif pattern == 'nonfinite':
        image = np.array([[0., -0., math.nan, math.inf], [-math.inf, -2., 3., math.nan]], dtype=dtype)
    elif pattern == 'complex':
        image = np.array([[0j, 1j, -1j], [1-2j, -1+2j, 2+3j]], dtype=dtype)
    elif pattern == 'object':
        image = np.array([[0, -2, 3], [5, 0, -1]], dtype=object)
    elif pattern in ('single1', 'row', 'column'):
        image = np.ones(shape, dtype=dtype)
    elif pattern.startswith('empty') or pattern == 'single0':
        image = np.zeros(shape, dtype=dtype)
    elif pattern == 'ring':
        image = np.ones(shape, dtype=dtype); image[2:-2, 2:-2] = 0
    elif pattern == 'diagonal':
        image = np.eye(6, dtype=dtype)
    else:
        values = (0, 2**63, 2**63+1, 2**64-1) if dtype == 'uint64' else (
            (-2**63, -1, 0, 2**63-1) if dtype == 'int64' else (
            (0, 1, 2, 3) if dtype.startswith('uint') or dtype == 'bool' else (-2, 0, 1, 3)))
        image = np.array([[values[(row*5+col*3) % len(values)] for col in range(shape[1])]
                          for row in range(shape[0])], dtype=dtype)
    layout = spec['layout']
    if layout == 'F': image = np.asfortranarray(image)
    elif layout == 'step':
        owner = np.zeros(tuple(2*n for n in image.shape), dtype=image.dtype)
        owner[::2, ::2] = image; image = owner[::2, ::2]
    elif layout == 'reverse': image = image[::-1, ::-1]
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'unaligned':
        owner = np.zeros(image.nbytes+1, dtype=np.uint8)
        view = np.ndarray(image.shape, dtype=image.dtype, buffer=owner, offset=1)
        view[...] = image; image = view
    elif layout == 'metadata': image = image.astype(np.dtype(image.dtype, metadata={'crofton': 'source'}))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1].copy(), image.shape)
    events = []; directions = spec['directions']
    if directions == 'np-int2': directions = np.int64(2)
    elif directions == 'np-float2': directions = np.float32(2)
    elif directions == 'array-scalar': directions = np.array(2)
    elif directions == 'array-many': directions = np.array([2, 4])
    elif isinstance(directions, str) and directions.startswith('probe-'):
        directions = DirectionProbe(directions[6:], events)
    return image, {'directions': directions}, events


def error_limit(histogram, coefficients):
    assert len(histogram) == len(coefficients) == 16
    scale = sum((abs(Fraction.from_float(float(c))) * int(n)
                 for c, n in zip(coefficients, histogram)), Fraction(0))
    if scale == 0: return 0.
    allowance = float(2 * Fraction(32, 2**53-32) * scale)
    assert math.isfinite(allowance) and allowance >= 0
    return math.nextafter(allowance, math.inf)
