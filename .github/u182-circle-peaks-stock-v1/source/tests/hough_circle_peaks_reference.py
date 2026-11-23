"""Frozen stock observations and prepared workloads for circular peaks."""

from hashlib import sha256
import json
from pathlib import Path
import warnings


PATH = Path(__file__).resolve().parents[1] / 'coverage/contracts/hough-circle-peaks-stock-v1.json'
PIN = 'cbe53ceb7e81926337637ef40324f7725fe3de002dbe96e8818a7785e804d2d9'
FIXED = ('peak-default-real', 'peak-normalized-ranking', 'peak-cross-radius-pruning', 'peak-capped-equal')
CIRCLE_IDS = ('full-output-sparse', 'dense-unit-counts', 'sparse-normalized')


def reference():
    data = PATH.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(data).hexdigest() == PIN
    return json.loads(data)


def snapshot(array, exact=False):
    raw = array.tobytes(order='C')
    result = dict(type=type(array).__name__, dtype=array.dtype.str, metadata=array.dtype.metadata is not None,
                  size=array.size, shape=list(array.shape), strides=list(array.strides),
                  c_order=bool(array.flags.c_contiguous), f_order=bool(array.flags.f_contiguous),
                  aligned=bool(array.flags.aligned), owns=bool(array.flags.owndata), writable=bool(array.flags.writeable),
                  base=type(array.base).__name__, sha256=sha256(raw).hexdigest())
    if exact:
        result['data_hex'] = raw.hex()
    return result


def restore(np, row):
    array = np.empty(row['shape'], dtype=row['dtype'])
    array[:] = np.frombuffer(bytes.fromhex(row['data_hex']), dtype=array.dtype).reshape(array.shape)
    assert snapshot(array, True) == row
    return array


def circle_fields(np, hough_circle):
    """Prepare only the three retained Hough volume recipes whose bytes are not embedded."""
    y, x = np.indices((80, 96))
    yield 'full-output-sparse', hough_circle((7 * x + 11 * y + x * y) % 103 == 2, (2, 7, 12), full_output=True)
    y, x = np.indices((72, 80))
    yield 'dense-unit-counts', hough_circle(((5 * x + 3 * y) % 7 != 0).astype(np.uint8), (3, 7), normalize=False)
    y, x = np.indices((112, 128))
    yield 'sparse-normalized', hough_circle((x * y + 17 * x + 13 * y) % 137 == 0, [3, 7, 11])


def cases(np, prepared, profile):
    source = reference()['prepared_circle_records']
    assert set(prepared) == set(CIRCLE_IDS)
    assert all(snapshot(prepared[key]) == source[key] for key in CIRCLE_IDS)
    small = {key: restore(np, source[key]) for key in ('full-border', 'scalar-zero-normalized')}
    votes = np.zeros((3, 18, 22), dtype=np.float64)
    votes[0, 4, 5] = 10
    votes[0, 7, 8] = 6
    votes[0, 12, 13] = 5
    votes[1, 5, 6] = 18
    votes[1, 12, 13] = 7
    votes[2, 6, 7] = 14
    votes[2, 12, 14] = 3
    radii = np.array([2, 5, 8], dtype=np.intp)
    ordinary_ids = ('raw-default', 'normalized-sorts-raw-output', 'default-centers-retained',
                    'inclusive-cross-suppression', 'normalized-cross-suppression', 'threshold-strict-and-plane-cap',
                    'total-cap-default-view', 'total-cap-zero', 'plane-cap-zero', 'fractional-active-cap')
    inputs = {key: (votes, radii) for key in ordinary_ids}
    inputs.update({
        'blank': (np.zeros_like(votes), radii),
        'from-full-border': (small['full-border'], np.array([1, 2], dtype=np.intp)),
        'zero-radius-normalize': (small['scalar-zero-normalized'], np.array([0], dtype=np.intp)),
        'reordered-radius-dtype': (votes[[2, 0, 1]], np.array([8, 2, 5], dtype=np.int16)),
        'short-radii-zip': (votes, radii[:2]),
        FIXED[0]: (prepared[CIRCLE_IDS[0]], np.array([2, 7, 12], dtype=np.intp)),
        FIXED[1]: (prepared[CIRCLE_IDS[1]], np.array([3, 7], dtype=np.intp)),
        FIXED[2]: (prepared[CIRCLE_IDS[2]], np.array([3, 7, 11], dtype=np.intp)),
    })
    equal = np.zeros((2, 42, 56), dtype=np.float64)
    equal[:, 2:-1:4, 2:-1:4] = 19
    inputs[FIXED[3]] = (equal, np.array([3, 7], dtype=np.intp))
    for row in reference()['profiles'][profile]:
        if row['id'] in inputs:
            args = inputs[row['id']]
        else:
            args = tuple(restore(np, a) for a in row['inputs'])
        assert [snapshot(a, 'data_hex' in expected) for a, expected in zip(args, row['inputs'], strict=True)] == row['inputs']
        yield args, row


def observe(call, np, args, row, fallback_warning=False):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        try:
            output = call(*args, **row['options'])
        except (ValueError, RuntimeError, IndexError, TypeError) as error:
            assert [type(error).__module__, type(error).__name__, str(error)] == row['error']
            observed = {'error': row['error']}
        else:
            assert type(output) is tuple and len(output) == 4
            assert [snapshot(a, 'data_hex' in saved) for a, saved in zip(output, row['result'], strict=True)] == row['result']
            assert [[bool(np.shares_memory(a, b)) for b in args] for a in output] == row['result_aliases_inputs']
            observed = {'output_sha256': [data['sha256'] for data in row['result']]}
    notices = [[type(w.message).__module__, w.category.__name__, str(w.message)] for w in emitted]
    expected = row['warnings']
    if fallback_warning:
        expected = [['skimagexx._compat', 'FallbackWarning',
                     'skimage.transform.hough_circle_peaks executed by scikit-image fallback'], *expected]
    assert notices == expected
    assert [snapshot(a, 'data_hex' in saved) for a, saved in zip(args, row['inputs'], strict=True)] == row['inputs']
    return observed
