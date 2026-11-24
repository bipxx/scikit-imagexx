"""Deterministic safe inputs for exact FAST response checks."""

RING = (
    (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3),
)
TIMED = ('u8-small-default', 'u8-gridded-n9', 'f32-textured-n12', 'f64-textured-n9')


def textured(np, rows, cols, dtype):
    yy, xx = np.indices((rows, cols), dtype=np.int64)
    x = (yy * 103 + xx * 267 + (yy * xx * 17) + (yy // 13 + xx // 17) * 73) % 1021
    if dtype == 'uint8':
        return (x % 256).astype(dtype)
    return (x / 1023 - 0.125).astype(dtype)


def ring(np, dtype, center, values):
    image = np.full((7, 7), center, dtype=dtype)
    for (dy, dx), value in zip(RING, values):
        image[3 + dy, 3 + dx] = value
    return image


def cases(np):
    yield 'u8-small-default', textured(np, 48, 64, 'uint8'), {}, True
    yield 'u8-gridded-n9', textured(np, 512, 512, 'uint8'), {'n': 9}, True
    yield 'f32-textured-n12', textured(np, 192, 256, 'float32'), {'threshold': 0.14}, True
    yield 'f64-textured-n9', textured(np, 288, 352, 'float64'), {'n': 9, 'threshold': 0.12}, True
    arc = [0.82 + i / 301 for i in range(16)]
    arc[5:9] = [0.32, 0.44, 0.18, 0.32]
    yield 'f32-wrap-arc-score-order', ring(np, 'float32', 0.32, arc), {}, True
    yield 'f64-dark-eight', ring(np, 'float64', 0.51, [0.12] * 8 + [0.51] * 8), {'n': 8}, True
    yield 'f64-split-rejected', ring(np, 'float64', 0.5, ([0.9] * 4 + [0.5]) * 3 + [0.5]), {'n': 5}, True
    equal = ring(np, 'float32', 0.5, [0.625] * 16)
    yield 'strict-equality', equal, {'n': 1, 'threshold': 0.125}, True
    pushed = np.nextafter(np.float32(0.625), np.float32(1))
    yield 'f32-nextafter-bright', ring(np, 'float32', 0.5, [pushed] * 16), {'n': 16, 'threshold': 0.125}, True
    yield 'bright-at-n1', ring(np, 'float64', 0.5, [0.875] + [0.5] * 15), {'n': 1}, True
    yield 'zero-threshold', ring(np, 'float32', 0.5, [0.5625] * 16), {'n': 16, 'threshold': 0.}, True
    yield 'width-six', textured(np, 11, 6, 'uint8'), {}, True
    yield 'height-six', textured(np, 6, 11, 'float64'), {}, True
    yield 'u8-offset-owner', textured(np, 12, 15, 'uint8')[1:-1], {}, True
    yield 'f32-offset-owner', textured(np, 14, 13, 'float32')[1:-1], {'n': 5}, True
    u8_readonly = textured(np, 8, 8, 'uint8')
    u8_readonly.flags.writeable = False
    yield 'u8-readonly', u8_readonly, {'n': 7}, True
    f32_readonly = textured(np, 8, 8, 'float32')
    f32_readonly.flags.writeable = False
    yield 'f32-readonly-error', f32_readonly, {}, False
    yield 'squeezed-3d', textured(np, 10, 11, 'uint8')[None], {'n': 4}, False
    yield 'noncontiguous', textured(np, 13, 13, 'float32')[:, ::-1], {'n': 8}, False
    yield 'ordinary-int16', textured(np, 10, 10, 'uint8').astype('int16'), {}, False
    yield 'ordinary-f16', textured(np, 10, 10, 'float32').astype('float16'), {'n': 4}, False
    yield 'n-zero', textured(np, 10, 10, 'uint8'), {'n': 0}, False
    yield 'n-too-large', textured(np, 10, 10, 'uint8'), {'n': 17}, False
    yield 'negative-threshold', textured(np, 10, 10, 'uint8'), {'n': 9, 'threshold': -0.15}, False
    yield 'rank-one-error', np.zeros(10, dtype='uint8'), {}, False


def snapshot(value):
    import hashlib
    return {
        'shape': list(value.shape), 'dtype': value.dtype.str,
        'strides': list(value.strides), 'owns': bool(value.flags.owndata),
        'writeable': bool(value.flags.writeable), 'type': type(value).__name__,
        'base_type': type(value.base).__name__, 'nonzero': int((value != 0).sum()),
        'digest': hashlib.sha256(value.tobytes()).hexdigest(),
    }
