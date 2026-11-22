"""Exact fixed and structural find_contours inputs, independent of the implementation."""

TIMED = ('f64-ridges-default', 'f64-masked-grid-high', 'f32-islands-high', 'u8-tiles-low')


def cases(np):
    y, x = np.indices((128, 192), dtype=np.int64)
    ridges = (np.abs(x % 37 - 17) + 5 * np.abs(y % 23 - 11)).astype('float64') / 257
    yield 'f64-ridges-default', ridges, {}, True
    y, x = np.indices((224, 256), dtype=np.int64)
    grid = ((17 * x + 13 * y + (y // 5) * (x // 11)) % 251).astype('float64') / 257
    mask = ((x - 106)**2 + (y - 104)**2 > 31**2) & ((x + 5 * y) % 19 != 0)
    yield 'f64-masked-grid-high', grid, dict(level=.47, fully_connected='high', mask=mask), True
    y, x = np.indices((128, 176), dtype=np.int64)
    islands = (((x % 23 - 10)**2 + (y % 19 - 9)**2) / 211).astype('float32')
    yield 'f32-islands-high', islands, dict(level=.28, positive_orientation='high'), True
    y, x = np.indices((320, 352), dtype=np.int64)
    tiles = (((y // 7) * 119 + (x // 9) * 89) % 256).astype('uint8')
    yield 'u8-tiles-low', tiles, dict(level=120.), True

    for case in range(16):
        tile = np.array([(case >> bit) & 1 for bit in range(4)], dtype='float64').reshape(2, 2)
        for join in ('low','high'):
            for orientation in ('low','high'):
                yield f'cell-{case}-{join}-{orientation}', tile, dict(level=.5, fully_connected=join,
                    positive_orientation=orientation), True

    islands = np.zeros((9, 13), dtype='float64')
    islands[2:4, 3:6] = .92
    islands[5:8, 8:12] = .81
    yield 'closed-islands-in-scan-order', islands, dict(level=.37), True
    yield 'orientation-reverses-each-array', islands, dict(level=.37, positive_orientation='high'), True
    cuts = np.ones_like(islands, dtype='bool')
    cuts[:, 4] = False
    yield 'mask-cuts-into-open-contours', islands, dict(level=.37, mask=cuts), True
    cuts.flags.writeable = False
    yield 'readonly-mask', islands, dict(level=.37, mask=cuts), True
    edge = np.arange(27, dtype='float64').reshape(3, 9)
    edge.flags.writeable = False
    yield 'readonly-decimal-edges', edge, dict(level=12.31), True
    plateau = np.array([[2., 0., 1., 3.], [0., 1., 1., 0.], [3., 1., 0., 4.], [0., 3., 1., 0.]])
    yield 'plateau-exact-level-splits', plateau, dict(level=1.), True
    reversed_image = plateau[::-1]
    yield 'negative-stride-fallback', reversed_image, dict(level=1.), False
    omitted = islands.copy()
    omitted[3, 4] = np.nan
    yield 'nan-cell-skip-fallback', omitted, dict(level=.37), False
    yield 'all-nan-warning-fallback', np.full((2, 3), np.nan), {}, False
    yield 'rank-three-error-fallback', np.zeros((2, 2, 3)), {}, False
    yield 'narrow-image-error-fallback', np.zeros((1, 3)), {}, False
    yield 'mask-shape-before-type-error', islands, dict(level=.37, mask=np.zeros((2, 2))), False
    yield 'mask-type-error-fallback', islands, dict(level=.37, mask=np.zeros(islands.shape,'uint8')), False
    yield 'invalid-orientation-fallback', islands, dict(positive_orientation='middle'), False
    yield 'both-invalid-options-fallback', islands, dict(fully_connected='middle', positive_orientation='middle'), False
    yield 'f32-default-fallback', islands.astype('float32'), {}, False
    yield 'uint8-default-overflow-warning', np.array([[250,200],[250,200]],'uint8'), {}, False


def input_snapshot(image):
    from hashlib import sha256
    return {'type':type(image).__name__,'shape':list(image.shape),'dtype':image.dtype.str,
            'strides':list(image.strides),'writeable':bool(image.flags.writeable),
            'digest':sha256(image.tobytes()).hexdigest()}


def output_snapshot(value, raw):
    from hashlib import sha256
    members = []
    for item in value:
        packed = item.tobytes()
        raw.update(packed)
        members.append({'type':type(item).__name__,'shape':list(item.shape),'dtype':item.dtype.str,
            'strides':list(item.strides),'writeable':bool(item.flags.writeable),'owns':bool(item.flags.owndata),
            'base_type':type(item.base).__name__, 'base_owns':bool(item.base.flags.owndata) if item.base is not None else None,
            'closed':bool((item[0] == item[-1]).all()),'digest':sha256(packed).hexdigest(),
            'first':[float(x).hex()for x in item[0]],'last':[float(x).hex()for x in item[-1]]})
    return {'type':type(value).__name__,'count':len(value),'contours':members}
