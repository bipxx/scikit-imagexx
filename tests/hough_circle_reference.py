"""Exact retained original Hough circle case recipes."""
from hashlib import sha256


CIRCLE_FIXED = ('blank-three-radii', 'sparse-normalized', 'dense-unit-counts', 'full-output-sparse')


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


def circle_cases(np):
    corner = np.zeros((4, 6), dtype=bool)
    corner[0, 0] = corner[0, 5] = corner[3, 3] = True
    field = np.array([[0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                      [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0]], dtype=np.uint8)
    yield 'scalar-zero-normalized', (corner, 0), {}, 'bounded'
    yield 'scalar-zero-unit-counts', (corner, 0), {'normalize': False}, 'bounded'
    yield 'clipped-border', (corner, [1, 2]), {}, 'bounded'
    yield 'full-border', (corner, [1, 2]), {'full_output': True}, 'bounded'
    yield 'full-zero-pad', (corner, [0, 0]), {'full_output': True}, 'bounded'
    yield 'radius-order-duplicates', (field, np.array([3, 0, 1, 3], dtype=np.int16)), {}, 'bounded'
    yield 'duplicate-normalized-rounding', (np.ones((9, 10), dtype=bool), (3, 4, 5)), {}, 'bounded'
    yield 'nonbinary-u8-unit-votes', (field * np.uint8(127), (0, 1, 2)), {'normalize': False}, 'bounded'
    yield 'rectangular-nonempty-all-zero', (np.zeros((6, 7), dtype=bool), [2, 1]), {}, 'bounded'
    yield 'float-radii-truncate', (field, [2.9, 0.2, 1.2]), {}, 'compatibility'
    yield 'negative-radius-clipped', (field, [-2, 0, 2]), {}, 'compatibility'
    yield 'strided-and-signed-input', ((field.astype(np.int16) * -5)[:, ::-1], [1, 2]), {}, 'compatibility'
    yield 'empty-image', (np.empty((0, 7), dtype=bool), [2]), {}, 'compatibility'
    yield 'empty-radii-clipped', (field, []), {}, 'compatibility'
    yield 'empty-radii-full', (field, []), {'full_output': True}, 'error'
    yield 'one-dimensional-image', (np.array([1, 0, 1], dtype=bool), 2), {}, 'error'
    yield 'two-dimensional-radii', (field, np.array([[1, 2]], dtype=np.intp)), {}, 'error'
    yield CIRCLE_FIXED[0], (np.zeros((96, 128), dtype=bool), [2, 4, 7]), {}, 'benchmark'
    y, x = np.indices((112, 128))
    yield CIRCLE_FIXED[1], (((x * y + 17 * x + 13 * y) % 137) == 0, [3, 7, 11]), {}, 'benchmark'
    y, x = np.indices((72, 80))
    yield CIRCLE_FIXED[2], ((((5 * x + 3 * y) % 7) != 0).astype(np.uint8), (3, 7)), {'normalize': False}, 'benchmark'
    y, x = np.indices((80, 96))
    yield CIRCLE_FIXED[3], (((7 * x + 11 * y + x * y) % 103) == 2, (2, 7, 12)), {'full_output': True}, 'benchmark'
