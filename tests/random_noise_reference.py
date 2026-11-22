"""Exact profile-specific stock outputs and PCG64 states for random_noise."""

from hashlib import sha256
import json
from pathlib import Path
import warnings


PATH = Path(__file__).resolve().parents[1] / 'coverage/contracts/random-noise-stock-v1.json'
PIN = '9f45cebaa26c4f493b2bbeae58533bcc2e2eac0e3b4a0bde97ad892917dac496'
MODERN_PATH = PATH.with_name('random-noise-stock-modern-v1.json')
MODERN_PIN = '19c4d5c26b1d73726ee28da865c4dfe7264fe66ab9b1f2070028722f7efac19b'
FIXED = ('gaussian-u8-default-384x512', 'speckle-f64-signed-no-clip-448x576',
         'sp-u16-interior-384x640', 'pepper-f64-signed-512x512')


def reference_profile(platform, numpy_version):
    profiles = {
        ('win32', '2.5.4'): 'windows',
        ('linux', '1.24.4'): 'linux-minimum',
        ('linux', '2.5.4'): 'linux-modern',
    }
    try:
        return profiles[platform, numpy_version]
    except KeyError:
        raise ValueError(f'No saved random reference for {platform} with NumPy {numpy_version}') from None


def reference(profile):
    raw = PATH.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(raw).hexdigest() == PIN
    profiles = json.loads(raw)
    if profile != 'linux-modern':
        return profiles[profile]
    raw = MODERN_PATH.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(raw).hexdigest() == MODERN_PIN
    modern = json.loads(raw)
    assert modern['saved_profile'] == profile
    rows = profiles[modern['base_profile']]
    overrides = modern['overrides']
    assert len(rows) == modern['case_count'] and overrides.keys() <= {row['id'] for row in rows}
    return [dict(row, **overrides.get(row['id'], {})) for row in rows]


def build_image(np, data):
    values = (-1.0, -0.875, -0.25, -0.0, 0.0, 0.0625, 0.25, 0.5,
              0.625, 0.75, 0.875, 0.9375, 0.984375, 0.9921875, 1.0)
    if 'shape' in data:
        height, width = data['shape']
        r, c = np.ogrid[:height, :width]
        word = 37*r + 13*c + 7*r*c
        if data['dtype'] == 'float64':
            return (word % 2047 - 1023).astype(np.float64) / 1024
        dtype = np.dtype(data['dtype'])
        return (word & np.iinfo(dtype).max).astype(dtype)
    name = data['image']
    if name == 'u8-ramp':
        return (np.arange(15, dtype=np.uint8) * 17).reshape(3, 5).copy()
    if name == 'positive-i16':
        return np.arange(15, dtype=np.int16).reshape(3, 5).copy()
    if name == 'f32-fortran':
        return np.array(np.array(values).reshape(3, 5), dtype=np.float32, order='F')
    if name == 'u8-empty':
        return np.empty((0, 5), dtype=np.uint8)
    if name == 'u8-three-unique':
        return np.array([[0, 60, 200], [200, 60, 0]], dtype=np.uint8)
    if name == 'f64-negative-one':
        return np.full((3, 5), -1.0)
    if name == 'f64-zero-edges':
        return np.array([[-0.0, 0.0, 0.25], [0.5, 0.75, 1.0]], dtype=np.float64)
    assert name in ('f64-edges', 'f64-nan')
    image = np.array(values, dtype=np.float64).reshape(3, 5).copy()
    if name == 'f64-nan':
        image[0, 3] = np.nan
    return image


def make_call(np, row):
    recipe = row['recipe']
    image = build_image(np, recipe)
    kwargs = dict(recipe.get('kwargs', {}), clip=recipe['clip'])
    if 'local_vars' in recipe:
        shapes = {'same': lambda: np.full(image.shape, .01), 'row': lambda: np.full((1, image.shape[1]), .01),
                  'scalar': lambda: np.float64(.01), 'nonpositive': lambda: np.zeros(image.shape),
                  'mismatch': lambda: np.full((2, 2), .01)}
        kwargs['local_vars'] = shapes[recipe['local_vars']]()
    if 'scalar_var' in recipe:
        kwargs['var'] = (-.25 if recipe['scalar_var'] == 'python-negative' else np.float64(-.25))
    owner = recipe.get('rng', 'int')
    rng = recipe['seed'] if owner == 'int' else np.random.default_rng(recipe['seed'])
    if owner == 'advanced-pcg64':
        rng.integers(0, 17, size=3, dtype=np.uint32)
    assert (None if owner == 'int' else rng.bit_generator.state) == row['rng_before']
    if not recipe.get('omit_mode'):
        kwargs['mode'] = recipe['mode']
    return (image,), dict(kwargs, rng=rng)


def typename(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def snapshot(np, array):
    chain = []
    current = array
    while isinstance(current, np.ndarray):
        chain.append(dict(shape=list(current.shape), strides=list(current.strides), dtype=current.dtype.str,
                          type=typename(current), c_contiguous=bool(current.flags.c_contiguous),
                          f_contiguous=bool(current.flags.f_contiguous), aligned=bool(current.flags.aligned),
                          owns=bool(current.flags.owndata), writable=bool(current.flags.writeable)))
        current = current.base
    data = array.tobytes(order='C')
    return dict(array_chain=chain, terminal_base=typename(current), logical_sha256=sha256(data).hexdigest()), data


def observe(np, call, row, retained=None):
    args, kwargs = make_call(np, row)
    image, = args
    before, _ = snapshot(np, image)
    assert before == row['input']
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        if row['native']:
            output = call(*args, **kwargs)
        else:
            try:
                call(*args, **kwargs)
            except NotImplementedError as error:
                assert 'fallback is disabled' in str(error)
            else:
                raise AssertionError('expected strict compatibility refusal')
    rng = kwargs['rng']
    state = None if type(rng) is int else rng.bit_generator.state
    assert snapshot(np, image)[0] == before
    notes = [dict(type=typename(item.message), message=str(item.message)) for item in emitted]
    if not row['native']:
        assert state == row['rng_before'] and not notes
        return None, dict(strict_refusal=True, rng_after=state)
    assert notes == row['warnings'] and state == row['rng_after']
    assert type(output) is np.ndarray and not np.shares_memory(image, output)
    details, bits = snapshot(np, output)
    assert details == row['output']
    if 'data_hex' in row:
        assert bits.hex() == row['data_hex']
    if retained is not None:
        assert bits == retained
    return output, dict(output_sha256=details['logical_sha256'], warnings=notes, rng_after=state)
