"""Exact stock digests, topology, and Generator state for random_shapes."""

from hashlib import sha256
import json
from pathlib import Path
import warnings


PATH = Path(__file__).resolve().parents[1] / 'coverage/contracts/random-shapes-stock-v1.json'
PIN = '6b6cada86b1d03afe5b9217827882f47671e80cc6f3771e748a7028056dc278c'
MODERN_PATH = PATH.with_name('random-shapes-stock-modern-v1.json')
MODERN_PIN = '56928923effa619b10da15b82d4543c7289ebb42887838d7fa9b51411adef348'
FIXED = ('fixed-rectangle-overlap-gray', 'fixed-circle-disjoint-rgb',
         'fixed-ellipse-planar-four-channel', 'fixed-mixed-retry-rgb')


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


def typename(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def make_call(np, row):
    args = (tuple(row['image_shape']), row['max_shapes'])
    kwargs = dict(row['options'])
    if 'intensity_range' in kwargs:
        kwargs['intensity_range'] = tuple(tuple(v) if type(v) is list else v for v in kwargs['intensity_range'])
    owner = row['rng_carrier']
    rng = row['seed'] if owner == 'int' else np.random.default_rng(row['seed'])
    if owner == 'advanced-pcg64':
        rng.integers(0, 17, size=3, dtype=np.uint32)
    assert (None if owner == 'int' else rng.bit_generator.state) == row['rng_before']
    kwargs['rng'] = rng
    return args, kwargs


def array_snapshot(array, np):
    chain = []
    cursor = array
    while isinstance(cursor, np.ndarray):
        chain.append(dict(shape=list(cursor.shape), strides=list(cursor.strides), dtype=cursor.dtype.str,
                          type=typename(cursor), c=bool(cursor.flags.c_contiguous), f=bool(cursor.flags.f_contiguous),
                          owns=bool(cursor.flags.owndata), aligned=bool(cursor.flags.aligned),
                          writeable=bool(cursor.flags.writeable)))
        cursor = cursor.base
    return dict(array_chain=chain, terminal_base=typename(cursor), logical_sha256=sha256(array.tobytes()).hexdigest())


def label_snapshot(labels):
    assert type(labels) is list
    types = set()
    out = []
    for label in labels:
        assert type(label) is tuple and len(label) == 2 and type(label[0]) is str
        name, box = label
        assert type(box) is tuple and len(box) == 2
        assert all(type(pair) is tuple and len(pair) == 2 for pair in box)
        types.update(typename(v) for pair in box for v in pair)
        out.append(dict(category=name, limits=[[int(v) for v in pair] for pair in box],
                        box_integer_types=[[typename(v) for v in pair] for pair in box]))
    array = dict(container_type=typename(labels), items=out, coord_types=sorted(types))
    return sha256(json.dumps(array, separators=(',', ':')).encode()).hexdigest()


def observe(np, call, row, retained=None):
    args, options = make_call(np, row)
    output = None
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        try:
            record = call(*args, **options)
        except (ValueError, TypeError, KeyError) as error:
            actual = dict(error=dict(type=typename(error), message=str(error)))
            assert actual['error'] == row['error']
        else:
            assert type(record) is tuple and len(record) == 2 and typename(record) == row['container_type']
            output, labels = record
            assert array_snapshot(output, np) == row['image'] and type(output) is np.ndarray
            assert label_snapshot(labels) == row['labels_sha256'] and len(labels) == row['label_count']
            if retained is not None:
                assert output.tobytes().hex() == retained['image']['data_hex']
            actual = dict(output_sha256=row['image']['logical_sha256'], labels_sha256=row['labels_sha256'],
                          accepted=len(labels))
    warning_records = [dict(type=typename(item.message), message=str(item.message)) for item in emitted]
    assert warning_records == row['warnings']
    rng = options['rng']
    state = None if row['rng_carrier'] == 'int' else rng.bit_generator.state
    assert state == row['rng_after']
    return output, dict(actual, warnings=warning_records, rng_after=state)
