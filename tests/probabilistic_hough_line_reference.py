"""Frozen original Hough-line input and output observations."""

from hashlib import sha256
import json
from pathlib import Path
import warnings


FIXED = ('default-cross-diagonal', 'gapped-three-angles', 'sparse-two-diagonals', 'dense-default-bands')
REFERENCE = Path(__file__).resolve().parents[1] / 'coverage/contracts/probabilistic-hough-line-reference-v1.jsonl'
REFERENCE_SHA256 = '63f336bc2ac60043d8374f471251f01fcb879244520fc08bbf079d5ddb0890cc'


def frozen_rows():
    data = REFERENCE.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(data).hexdigest() == REFERENCE_SHA256
    return tuple(json.loads(row) for row in data.splitlines())


def snapshot(value, content=False):
    data = value.tobytes()
    row = dict(type=type(value).__name__, dtype=value.dtype.str, metadata=value.dtype.metadata is not None,
               size=value.size, shape=list(value.shape), strides=list(value.strides),
               c_order=value.flags.c_contiguous, f_order=value.flags.f_contiguous,
               aligned=value.flags.aligned, owns=value.flags.owndata, writable=value.flags.writeable,
               base=type(value.base).__name__, sha256=sha256(data).hexdigest())
    if content:
        row['data_hex'] = data.hex()
    return row



def cases(np):
    cross = np.zeros((19, 23), dtype=bool)
    cross[8, 2:21] = True
    cross[1:18, 11] = True
    theta = np.array([0., np.pi / 2])
    options = dict(threshold=1, line_length=7, line_gap=0)
    yield 'first-theta-wins-tie', cross, theta, 17, options, 'bounded'
    yield 'reordered-theta-wins-tie', cross, theta[::-1].copy(), 17, options, 'bounded'
    yield 'same-seed-passed-generator', cross, theta, np.random.default_rng(17), options, 'provider'
    advanced = np.random.default_rng(17)
    advanced.integers(0, 31, size=5)
    yield 'advanced-passed-generator', cross, theta, advanced, options, 'provider'
    yield 'strided-image-and-theta', (cross.astype(np.int16) * -7)[:, ::-1], theta[::-1], 17, options, 'compatibility'
    gapped = np.zeros((13, 33), dtype=np.uint8)
    gapped[4, 2:14] = 4
    gapped[4, 16:28] = 231
    horizontal = np.array([np.pi / 2])
    yield 'gap-exactly-two', gapped, horizontal, 99, dict(threshold=3, line_length=8, line_gap=2), 'bounded'
    yield 'gap-two-exceeds-one', gapped, horizontal, 99, dict(threshold=3, line_length=8, line_gap=1), 'bounded'
    diagonal = np.zeros((15, 19), dtype=bool)
    dial = np.arange(6 + 1)
    diagonal[3 + dial, 4 + dial] = True
    axis = np.array([-np.pi / 4])
    yield 'component-length-inclusive', diagonal, axis, 5, dict(threshold=1, line_length=6, line_gap=0), 'bounded'
    yield 'component-length-rejects-euclidean', diagonal, axis, 5, dict(threshold=1, line_length=7, line_gap=0), 'bounded'
    yield 'empty-theta-positive-threshold', cross, np.empty(0, dtype=np.float64), np.random.default_rng(23), options, 'control'
    yield 'all-background-no-generator-draw', np.zeros((13, 19), dtype=bool), None, np.random.default_rng(81), {}, 'control'
    one = np.zeros((12, 13), dtype=bool)
    one[3, 8] = True
    yield 'single-foreground-no-generator-draw', one, None, np.random.default_rng(81), {}, 'control'

    mask = np.zeros((80, 96), dtype=bool)
    span = np.arange(8, 71)
    mask[19, 2:92] = True
    mask[3:77, 74] = True
    mask[span, span + 4] = True
    yield FIXED[0], mask, None, 42, {}, 'benchmark'

    mask = np.zeros((64, 96), dtype=np.uint8)
    for y in (13, 31):
        mask[y, 5:31] = 1
        mask[y, 34:59] = 2
        mask[y, 62:86] = 3
    mask[7:59, 77] = 255
    yield FIXED[1], mask, np.array([np.pi / 2, 0., -np.pi / 4]), 87, dict(threshold=5, line_length=20, line_gap=4), 'benchmark'

    yy, xx = np.indices((96, 112))
    mask = (23 * xx + 17 * yy + xx * yy) % 241 == 5
    span = np.arange(11, 84)
    mask[span, span + 8] = True
    mask[span, 102 - span] = True
    angled = np.concatenate((np.linspace(-1.4, 1.4, 97), np.array([-np.pi / 4, np.pi / 4])))
    yield FIXED[2], mask, angled, 103, dict(threshold=6, line_length=25, line_gap=3), 'benchmark'

    mask = np.zeros((72, 88), dtype=bool)
    for row in range(6, 65, 7):
        mask[row:row + 2, 3:84] = True
    for col in range(8, 82, 11):
        mask[3:69, col:col + 2] = True
    yield FIXED[3], mask, None, 137, dict(threshold=9, line_length=24, line_gap=2), 'benchmark'

    cross = np.zeros((19, 23), dtype=bool)
    cross[8, 2:21] = True
    cross[1:18, 16] = True
    yield 'vertical-first', cross, theta, 17, options, 'bounded'
    yield 'horizontal-first', cross, theta[::-1].copy(), 17, options, 'bounded'


def observe(call, np, case, saved):
    case_id, image, theta, rng, options, scope = case
    assert case_id == saved['id'] and options == saved['options'] and scope == saved['scope']
    before = snapshot(image, scope != 'benchmark')
    angles = None if theta is None else snapshot(theta, True)
    assert before == saved['image'] and angles == saved['theta']
    if type(rng) is np.random.Generator:
        assert rng.bit_generator.state == saved['rng']['before']
    else:
        assert rng == saved['rng']['seed']
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        result = call(image, theta=theta, rng=rng, **options)
    assert type(result) is list and len(result) == saved['result_len']
    assert all(type(line) is tuple and len(line) == 2 and
               all(type(point) is tuple and len(point) == 2 and all(type(i) is int for i in point) for point in line)
               for line in result)
    tuples = [part for line in result for part in (line, *line)]
    assert len({id(value) for value in tuples}) == len(tuples) == saved['tuple_objects']
    encoded = json.dumps(result, separators=(',', ':')).encode()
    assert sha256(encoded).hexdigest() == saved['result_sha256'] and json.loads(encoded) == saved['result']
    assert [[type(w.message).__module__, w.category.__name__, str(w.message)] for w in emitted] == saved['warnings']
    assert snapshot(image, scope != 'benchmark') == before
    assert (None if theta is None else snapshot(theta, True)) == angles
    if type(rng) is np.random.Generator:
        assert rng.bit_generator.state == saved['rng']['actual_after']
    return result
