"""Frozen scikit-image output words and storage for local mean resize."""

from hashlib import sha256
import json
from pathlib import Path
import warnings


PATH = Path(__file__).resolve().parents[1] / 'coverage/contracts/resize-local-mean-stock-v1.json'
PIN = '1090477dbd0443bb9c7535865c2186922a8725649de90191310684bfb0679c03'
FIXED = ('u8-small-normalized', 'u8-mid-normalized', 'u16-aniso-raw', 'u16-largest-normalized')
COUNTEREXAMPLES = ('uint8-scaled-8x4', 'uint8-scaled-8x8', 'uint16-scaled-8x4', 'uint16-scaled-8x8')


def reference():
    raw = PATH.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(raw).hexdigest() == PIN
    return json.loads(raw)


def digest(array):
    return sha256(array.tobytes()).hexdigest()


def image_for(np, case):
    height, width = case['shape']
    row, col = np.ogrid[:height, :width]
    dtype = np.dtype(case['dtype'])
    limit = np.iinfo(dtype).max
    image = ((257 * row + 17 * col + 13 * row * col) & limit).astype(dtype)
    factor_y, factor_x = height // case['output'][0], width // case['output'][1]
    pattern = case.get('pattern')
    if pattern == 'all-zero':
        image.fill(0)
    elif pattern == 'all-max':
        image.fill(limit)
    elif pattern == 'impulses':
        image.fill(0)
        image[0, 0] = image[-1, -1] = image[factor_y, factor_x - 1] = limit
    elif pattern in ('near-max', 'near-max-readonly'):
        image = limit - (image & 3)
        if pattern == 'near-max-readonly':
            image.flags.writeable = False
    elif pattern == 'view':
        image = image.view()
    return image


def array_details(array, np):
    current = array
    chain = []
    while isinstance(current, np.ndarray):
        chain.append(dict(shape=list(current.shape), strides=list(current.strides), dtype=current.dtype.str,
                          type=type(current).__module__ + '.' + type(current).__qualname__,
                          c=bool(current.flags.c_contiguous), f=bool(current.flags.f_contiguous),
                          own=bool(current.flags.owndata), writeable=bool(current.flags.writeable),
                          aligned=bool(current.flags.aligned)))
        current = current.base
    return dict(array_chain=chain, terminal_base_type=type(current).__name__, raw_sha256=digest(array))


def observe(np, call, image, row, compat=False):
    case = row['case']
    before = digest(image)
    assert before == row['input_sha256']
    if 'input_writeable' in row:
        assert bool(image.flags.writeable) == row['input_writeable'] and bool(image.flags.owndata) == row['input_owns']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call(image, tuple(case['output']), preserve_range=case['preserve_range'])
    metadata = array_details(result, np)
    assert metadata == row['stock_output']
    assert digest(image) == before and bool(np.shares_memory(image, result)) == row['stock_input_share']
    messages = [{'type': w.category.__name__, 'message': str(w.message)} for w in caught]
    expected = row['warnings']
    if compat:
        expected = [{'type': 'FallbackWarning', 'message': 'skimage.transform.resize_local_mean executed by scikit-image fallback'},
                    *expected]
    assert messages == expected
    return result, metadata
