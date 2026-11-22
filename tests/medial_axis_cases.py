"""Exact frozen medial-axis case recipes and observation shape."""

from hashlib import sha256

FIXED = ({'id': 'symmetric-thick', 'shape': [129, 129], 'recipe': 'solid', 'seed': 206, 'return_distance': False},
 {'id': 'perforated-distance', 'shape': [160, 192], 'recipe': 'holes', 'seed': 19, 'return_distance': True},
 {'id': 'masked-components-distance',
  'shape': [128, 192],
  'recipe': 'masked',
  'seed': 95,
  'return_distance': True},
 {'id': 'seeded-corridors', 'shape': [96, 160], 'recipe': 'corridors', 'seed': 43, 'return_distance': False})


def digest(raw):
    return sha256(raw).hexdigest()


def fixture(spec, np):
    height, width = spec['shape']
    row, col = np.ogrid[:height, :width]
    shape = spec['recipe']
    mask = None
    if shape == 'solid':
        image = (row - 64)**2 + (col - 64)**2 <= 50**2
    elif shape == 'holes':
        image = (row > 6) & (row < 154) & (col > 9) & (col < 182)
        image &= ((row - 45)**2 + (col - 67)**2 > 24**2) & ((row - 108)**2 + (col - 126)**2 > 27**2)
    elif shape == 'masked':
        image = ((row - 43)**2 + (col - 60)**2 <= 37**2) | ((row - 80)**2 + (col - 131)**2 <= 37**2)
        mask = ((col + row * 2) % 31 > 4) & (row > 11) & (col < 181)
    else:
        assert shape == 'corridors'
        image = (row % 19 < 7) | (col % 27 < 8)
    return image, mask


def cases(plan, np):
    for spec in plan['fixed']:
        image, mask = fixture(spec, np)
        yield spec['id'], image, mask, spec['return_distance'], spec['seed'], str(spec['seed'])
    tied = np.zeros((11, 15), dtype=bool)
    tied[2:8, 3:11] = True
    for seed in (0, 1):
        yield f'tie-even-rectangle-seed-{seed}', tied, None, True, seed, str(seed)
    bank = np.random.Generator(np.random.PCG64(501))
    for name, mask in [('generator-first', None), ('generator-next-masked', np.indices(tied.shape)[1] < 8)]:
        yield name, tied, mask, True, bank, 'shared Generator(PCG64(501))'
    generator = np.random.PCG64(84)
    yield 'explicit-bit-generator', tied, None, False, generator, 'PCG64(84)'
    for name, image in [('one-pixel-zero', np.zeros((1, 1), bool)),
                        ('one-pixel-one', np.ones((1, 1), bool)),
                        ('row-strip', np.array([[0, 1, 1, 1, 0, 1, 0]], bool)),
                        ('column-strip', np.array([[1, 1, 1, 0, 1]], bool).T),
                        ('empty', np.empty((0, 5), bool)),
                        ('all-zero', np.zeros((5, 7), bool)),
                        ('all-foreground', np.ones((4, 6), bool))]:
        yield name, image, None, True, 32, '32'
    signed = np.array([[0, 0, 0, 0, 0], [0, -3, 256, -257, 0], [0, 0, -5, 0, 0]], np.int16)
    yield 'signed-numeric', signed, None, True, 12, '12'
    unusual = np.array([[0, -0., 0, 0], [0, np.nan, -np.inf, 0], [0, np.inf, -0.5, 0]], np.float64)
    yield 'nonfinite-numeric', unusual, None, True, 12, '12'
    readonly = np.asfortranarray(tied)
    readonly.flags.writeable = False
    yield 'readonly-fortran', readonly, None, True, 12, '12'
    yield 'strided-negative', tied[::-1, ::-2], None, True, 12, '12'
    tagged = tied.view(np.dtype(bool, metadata={'contract': 'medial-axis'}))
    yield 'tagged-bool', tagged, None, True, 12, '12'
    for name, image, mask, seed in [
        ('mask-shape-error', tied, np.ones((2, 3), bool), 12),
        ('mask-type-error', tied, np.ones(tied.shape, np.float64), 12),
        ('rank-one-error', np.zeros(5, bool), None, 12),
        ('rank-three-small-error', np.ones((2, 2, 2), bool), None, 12),
        ('negative-seed-error', tied, None, -1),
        ('legacy-randomstate-error', tied, None, np.random.RandomState(5)),
    ]:
        yield name, image, mask, False, seed, ('RandomState(5)' if name == 'legacy-randomstate-error' else str(seed))


def array_snapshot(array, small=False):
    raw = array.tobytes()
    report = dict(type=type(array).__module__ + '.' + type(array).__name__, shape=list(array.shape),
                  dtype=array.dtype.str, dtype_metadata=dict(array.dtype.metadata) if array.dtype.metadata else None,
                  strides=list(array.strides), c=bool(array.flags.c_contiguous), f=bool(array.flags.f_contiguous),
                  owns=bool(array.flags.owndata), writeable=bool(array.flags.writeable),
                  base=type(array.base).__name__, sha256=digest(raw))
    if small and array.size <= 180:
        report['raw_hex'] = raw.hex()
    return report
