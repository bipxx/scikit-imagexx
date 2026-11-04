"""Prospective finite half-input stain fixtures; no color algorithm calls."""
API = 'color.combine_stains'
MAX_INPUT_ELEMENTS = 24
MAX_OUTPUT_ELEMENTS = 64
STANDARD = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]
UNASSESSED = (
    'Other stain APIs remain unchanged; no native/public completion credit from reference',
    'Nonfinite or extreme half values, overflow/underflow warnings and hardware floating environment',
    'Non-native endian, image/matrix subclasses, object/complex/protocol providers',
    'Rank1/batched/broadcast matrices and malformed channel axes/shapes',
    'Concurrent mutation, forged storage, allocation failure or unbounded allocation')
NUMERICAL_POLICY = dict(prepared_input='float32', rtol=2e-6, atol=2e-6,
    matrix_scale='multiply atol by max(1,max(sum(abs(matrix),axis=0))); empty matrix scale1',
    provenance='Unchanged float32 branch of coverage.parity_tolerance plus tests/test_stains.py::compare',
    structural='Exact observed dtype/metadata/shape/layout/container/errors/warnings/input-state and alias rules; raw original words retained, not universally bitwise candidate arithmetic',
    no_relaxation='Policy frozen before original or candidate observations; no float16-labelled weaker budget')


def cases():
    rows = [
        ('standard-f32', 'c', 'standard', 'float32', -1),
        ('standard-f64', 'c', 'standard', 'float64', -1),
        ('custom-k0', 'c', 'k0', 'float32', -1),
        ('custom-k1', 'c', 'k1', 'float32', -1),
        ('custom-k7', 'c', 'k7', 'float32', -1),
        ('channel-zero', 'planar', 'standard', 'float64', 0),
        ('reverse', 'reverse', 'standard', 'float64', -1),
        ('fortran', 'f', 'standard', 'float32', -1),
        ('readonly-unaligned', 'readonly-unaligned', 'standard', 'float64', -1),
        ('empty', 'empty', 'standard', 'float32', -1),
        ('pixel', 'pixel', 'standard', 'float64', -1),
        ('dtype-metadata', 'metadata', 'standard', 'float32', -1)]
    return [dict(id='combine-half/' + name, api=API, layout=layout, matrix=matrix,
                 matrix_dtype=dtype, channel_axis=axis,
                 policy=dict(all='raise') if name == 'reverse' else {},
                 holdout=name in ('custom-k7', 'reverse', 'pixel'))
            for name, layout, matrix, dtype, axis in rows]


def source_safe(spec):
    assert spec['api'] == API and spec['layout'] in ('c', 'planar', 'reverse', 'f', 'readonly-unaligned', 'empty', 'pixel', 'metadata')
    assert spec['matrix'] in ('standard', 'k0', 'k1', 'k7')
    assert spec['matrix_dtype'] in ('float32', 'float64')
    assert spec['channel_axis'] == (0 if spec['layout'] == 'planar' else -1)
    width = dict(standard=3, k0=0, k1=1, k7=7)[spec['matrix']]
    pixels = 0 if spec['layout'] == 'empty' else 1 if spec['layout'] == 'pixel' else 4 if spec['layout'] == 'reverse' else 8
    assert pixels * width <= MAX_OUTPUT_ELEMENTS
    # All image magnitudes<=0.75 and coefficients<=1. With adjust<14, a
    # three-term dot-product exponent has magnitude<=31.5 (safe in float32).
    return dict(input_elements_upper_bound=MAX_INPUT_ELEMENTS,
                output_elements=pixels * width, exponent_absolute_upper_bound=31.5,
                pointers='Only owned NumPy allocations or valid views within them; no forged spans')


def fixture(spec, np):
    source_safe(spec)
    values = [-0.25, -0.0, 0.0, 2.0**-24, 0.125, 0.25, 0.5, 0.75]
    image = np.array(values * 3, dtype=np.float16).reshape(2, 4, 3).copy()
    layout = spec['layout']
    if layout == 'planar':
        image = np.ascontiguousarray(np.moveaxis(image, -1, 0))
    elif layout == 'reverse':
        image = image[::-1, ::2, ::-1]
    elif layout == 'f':
        image = np.asfortranarray(image)
    elif layout == 'readonly-unaligned':
        unaligned = np.ndarray(image.shape, dtype=np.float16,
                               buffer=bytearray(image.nbytes + 1), offset=1)
        unaligned[:] = image
        unaligned.flags.writeable = False
        image = unaligned
    elif layout == 'empty':
        image = np.empty((0, 3), dtype=np.float16)
    elif layout == 'pixel':
        image = image[0, 0].copy()
    elif layout == 'metadata':
        image = image.view(np.dtype(np.float16, metadata={'half-stain': 'observe-conversion'}))
    if spec['matrix'] == 'standard':
        matrix = np.array(STANDARD, dtype=spec['matrix_dtype'])
    elif spec['matrix'] == 'k0':
        matrix = np.empty((3, 0), dtype=spec['matrix_dtype'])
    elif spec['matrix'] == 'k1':
        matrix = np.array([[0.125], [-0.2], [0.0625]], dtype=spec['matrix_dtype'])
    else:
        matrix = np.array([[((i + 2*j) % 7 - 3) / 16 for j in range(7)]
                           for i in range(3)], dtype=spec['matrix_dtype'])
    assert type(image) is np.ndarray and image.dtype.type is np.float16 and image.dtype.isnative
    assert image.size <= MAX_INPUT_ELEMENTS and np.isfinite(image).all() and (np.abs(image) <= 0.75).all()
    assert type(matrix) is np.ndarray and matrix.shape[0] == 3 and matrix.size <= 21
    assert np.isfinite(matrix).all() and (np.abs(matrix) <= 1).all()
    options = dict(channel_axis=spec['channel_axis'])
    return (image, matrix), options, dict(image=image, matrix=matrix)
