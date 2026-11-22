"""Exact stored Hough peak inputs and endpoint recipes."""

from hashlib import sha256


ENDPOINTS = ("retained-cross-default", "sparse-96x180", "dense-noise-128x180", "tied-plateaus-112x180-cap")


def sha(raw):
    return sha256(raw).hexdigest()


def snapshot(array, exact=True):
    raw = array.tobytes(order='C')
    result = dict(type=type(array).__name__, dtype=array.dtype.str, metadata=array.dtype.metadata is not None,
                  size=array.size, shape=list(array.shape), strides=list(array.strides),
                  c_order=bool(array.flags.c_contiguous), f_order=bool(array.flags.f_contiguous),
                  aligned=bool(array.flags.aligned), owns=bool(array.flags.owndata), writable=bool(array.flags.writeable),
                  base=type(array.base).__name__, sha256=sha(raw))
    if exact:
        result['data_hex'] = raw.hex()
    return result


def restore(np, record):
    shape = tuple(record['shape'])
    vals = np.empty(shape, dtype=np.dtype(record['dtype']))
    vals[...] = np.frombuffer(bytes.fromhex(record['data_hex']), dtype=vals.dtype).reshape(shape)
    if not record['owns']:
        vals = vals.view()
    if not record['writable']:
        vals.flags.writeable = False
    observed = snapshot(vals)
    assert {key: observed[key] for key in record} == record
    return vals


def inputs(np, rows, cols, sites=(), dtype='uint64'):
    h = np.zeros((rows, cols), dtype=dtype)
    for row, col, value in sites:
        h[row, col] = value
    return h, np.linspace(-np.pi / 2, np.pi / 2, cols, endpoint=False), np.arange(rows, dtype='float64') - rows // 2


def cases(np, previous):
    def old(name):
        saved = next(row['result'] for row in previous['rows'] if row['id'] == name)
        return tuple(restore(np, saved[field]) for field in ('accumulator', 'angles', 'distances'))
    separated = ((2, 2, 10), (6, 7, 6), (10, 12, 5), (13, 16, 4))
    threshold = inputs(np, 17, 20, separated)
    wrapped = inputs(np, 11, 12, ((3, 0, 12), (8, 11, 10), (2, 11, 9), (7, 10, 8)))
    plateau = inputs(np, 13, 15, (*((r, c, 12) for r in (2, 3) for c in (2, 3)),
                                   *((r, c, 10) for r in (7, 8) for c in (7, 8))))
    hollow = inputs(np, 11, 13, ((r, c, 14) for r in (3, 4, 5) for c in (4, 5, 6) if (r, c) != (4, 5)))
    tied = inputs(np, 24, 28, ((r, c, 20) for r in (2, 7, 12, 17) for c in (2, 7, 12, 17, 22)))
    controls = dict(min_distance=1, min_angle=1)
    yield 'retained-diagonal', old('diagonal-default'), dict(min_distance=2, min_angle=5), 'reused', 'diagonal-default'
    yield 'retained-blank', old('no-votes-default'), {}, 'reused', 'no-votes-default'
    yield 'retained-rounding-neighbors', old('signed-half-and-neighbors'), controls, 'reused', 'signed-half-and-neighbors'
    yield 'threshold-default-strict', threshold, controls, 'bounded', None
    yield 'threshold-explicit-strict', threshold, dict(**controls, threshold=5.0), 'bounded', None
    yield 'threshold-zero', threshold, dict(**controls, threshold=0), 'bounded', None
    yield 'threshold-nan', threshold, dict(**controls, threshold=float("nan")), 'compatibility', None
    yield 'reflected-periodic-boundary', wrapped, controls, 'bounded', None
    yield 'zero-distance-row-exclusion', inputs(np, 11, 12, ((0, 0, 12), (0, 11, 11))), controls, 'bounded', None
    yield 'centroid-nearest-even', plateau, controls, 'bounded', None
    yield 'centroid-in-hollow-plateau', hollow, controls, 'bounded', None
    yield 'diagonally-connected-peak-members', inputs(np, 10, 12, ((2, 3, 9), (3, 4, 9))), controls, 'bounded', None
    yield 'ties-priority-unlimited', tied, controls, 'bounded', None
    yield 'ties-priority-limited', tied, dict(**controls, num_peaks=5), 'bounded-provider', None
    yield 'cap-zero-dtype', tied, dict(**controls, num_peaks=0), 'bounded', None
    yield 'radius-angle-clamped', inputs(np, 9, 3, ((4, 1, 8),)), dict(min_distance=0, min_angle=99), 'bounded', None
    rs, cs, ds = threshold
    yield 'reversed-non-c-accumulator', (rs[:, ::-1], cs[::-1], ds), controls, 'compatibility', None
    yield 'mixed-coordinate-dtypes', (rs.astype('int32'), cs.astype('float32'), ds.astype('int32')), controls, 'compatibility', None
    yield 'empty-space', inputs(np, 0, 4), {}, 'error', None
    yield 'invalid-negative-min-distance', threshold, dict(min_distance=-1, min_angle=1), 'error', None
    yield 'invalid-active-fractional-cap', tied, dict(**controls, num_peaks=2.5), 'error', None
    yield ENDPOINTS[0], old('cross-default'), {}, 'benchmark', 'cross-default'
    a, ang, dst = inputs(np, 96, 180)
    y, x = np.indices(a.shape)
    a[(13 * x + 17 * y + x * y) % 127 == 0] = ((x + 3 * y) % 59 + 1)[(13 * x + 17 * y + x * y) % 127 == 0]
    yield ENDPOINTS[1], (a, ang, dst), {}, 'benchmark', None
    a, ang, dst = inputs(np, 128, 180)
    y, x = np.indices(a.shape)
    a[...] = (17 * x + 23 * y + 7 * x * y) % 251
    yield ENDPOINTS[2], (a, ang, dst), dict(min_distance=4, min_angle=5), 'benchmark', None
    tiles = ((r + dr, c + dc, 24) for r in range(4, 110, 9) for c in range(4, 178, 11)
             for dr in (0, 1) for dc in (0, 1))
    yield ENDPOINTS[3], inputs(np, 112, 180, tiles), dict(min_distance=1, min_angle=2, num_peaks=11), 'benchmark', None
