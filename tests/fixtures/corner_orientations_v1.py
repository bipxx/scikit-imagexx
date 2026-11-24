"""Frozen deterministic inputs and response assertions for corner_orientations."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((_ROOT / 'coverage/contracts/corner-orientations-v1.json').read_bytes())
FIXED = tuple(spec['id'] for spec in CONTRACT['fixed_cases'])


def make_case(case_id):
    spec = next(row for row in CONTRACT['fixed_cases'] if row['id'] == case_id)
    h, w = spec['shape']
    rows, cols = np.ogrid[:h, :w]
    image = ((17 * rows + 29 * cols + 5 * rows * cols) % 256).astype(np.uint8)
    extent = spec['mask']
    radius = extent // 2
    order = np.arange(spec['corners'], dtype=np.intp)
    corners = np.empty((len(order), 2), dtype=np.intp)
    corners[:, 0] = radius + (17 * order + 7) % (h - 2 * radius)
    corners[:, 1] = radius + (31 * order + 13) % (w - 2 * radius)
    if spec['sparse']:
        y, x = np.ogrid[:extent, :extent]
        mask = (((2 * y + x) % 4) != 1).astype(np.uint8)
    else:
        mask = np.ones((extent, extent), dtype=np.uint8)
    return image, corners, mask, spec


def expected_sha256(spec):
    if 'sha256' in spec:
        return spec['sha256']
    return spec['sha256_windows'] if sys.platform == 'win32' else spec['sha256_linux']


def assert_memoryview_response(actual, length, dtype_str='<f8', *, stock=False):
    itemsize = np.dtype(dtype_str).itemsize
    assert type(actual) is np.ndarray
    assert actual.shape == (length,)
    assert actual.dtype.str == dtype_str
    assert actual.strides == (itemsize,)
    assert actual.flags.c_contiguous and actual.flags.f_contiguous
    assert actual.flags.writeable and actual.flags.aligned and not actual.flags.owndata
    view = actual.base
    assert type(view) is memoryview and view.ndim == 1
    assert view.shape == actual.shape and view.strides == actual.strides
    assert view.suboffsets == () and view.nbytes == actual.nbytes and not view.readonly
    exporter = view.obj
    if stock:
        type_name = type(exporter).__module__ + '.' + type(exporter).__qualname__
        assert type_name == CONTRACT['layout']['stock_exporter']
        owner = exporter.base
    else:
        assert type(exporter) is np.ndarray
        owner = exporter
    assert type(owner) is np.ndarray and owner.flags.owndata and owner.base is None
    assert owner.shape == actual.shape and (length == 0 or np.shares_memory(actual, owner))
    return owner


def sha(array):
    return sha256(array.tobytes(order='C')).hexdigest()
