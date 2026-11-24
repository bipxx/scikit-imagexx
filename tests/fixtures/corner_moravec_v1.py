"""Frozen byte inputs and private response metadata for Moravec."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((_ROOT / 'coverage/contracts/corner-moravec-v1.json').read_bytes())
FIXED = tuple(case['id'] for case in CONTRACT['fixed_one_worker_cases'])
ADMITTED = tuple(CONTRACT['observations'])


def pattern(shape):
    height, width = shape
    rows, columns = np.ogrid[:height, :width]
    return ((17 * rows + 29 * columns + 5 * rows * columns) & 255).astype(np.uint8)


def make_input(case):
    row = CONTRACT['observations'][case]
    shape = tuple(row['shape'])
    if case == 'impulse-example':
        image = np.zeros(shape, np.uint8)
        image[3, 3] = 255
    elif case == 'all-equal':
        image = np.full(shape, 151, np.uint8)
    elif case == 'axis-ramp':
        image = np.tile(np.arange(shape[1], dtype=np.uint8), (shape[0], 1))
    else:
        image = pattern(shape)
    if case == 'readonly':
        image.flags.writeable = False
    options = {} if row['default_call'] else {'window_size': row['window_size']}
    return image, options


def assert_frozen_response(case, image, actual, *, stock=False):
    expected = CONTRACT['observations'][case]
    assert sha256(image.tobytes(order='C')).hexdigest() == expected['input_sha256']
    assert type(actual) is np.ndarray and actual.shape == image.shape
    assert actual.dtype == np.dtype('float64')
    assert actual.strides == (8 * image.shape[1], 8)
    assert actual.flags.c_contiguous and not actual.flags.f_contiguous
    assert actual.flags.writeable and actual.flags.aligned and not actual.flags.owndata
    assert not np.shares_memory(actual, image)
    assert sha256(actual.tobytes(order='C')).hexdigest() == expected['response_sha256']

    view = actual.base
    assert type(view) is memoryview and view.format == 'd' and view.itemsize == 8
    assert view.ndim == 2 and view.shape == actual.shape and view.strides == actual.strides
    assert view.suboffsets == () and view.nbytes == actual.nbytes
    assert view.c_contiguous and view.contiguous and not view.f_contiguous and not view.readonly
    exporter = view.obj
    if stock:
        type_name = type(exporter).__module__ + '.' + type(exporter).__qualname__
        assert type_name == CONTRACT['layout']['stock_exporter']
        owner = exporter.base
    else:
        assert type(exporter) is np.ndarray
        owner = exporter
    assert type(owner) is np.ndarray and owner.flags.owndata and owner.base is None
    assert owner.flags.writeable and owner.flags.c_contiguous
    assert owner.shape == actual.shape and owner.strides == actual.strides
    assert np.shares_memory(actual, owner)
    return owner
