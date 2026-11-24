"""Input recipes and case-specific storage checks from the frozen table reference."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((_ROOT / 'coverage/contracts/regionprops-table-v1.json').read_bytes())
FIXED_CASES = tuple(case['id'] for case in CONTRACT['fixed_one_worker_cases'])
CASE_IDS = tuple(CONTRACT['observations'])
KEYS = ('label', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3')


def make_input(case):
    if case == 'all-background':
        return np.zeros((32, 32), np.int32)
    if case == 'dense-one-label':
        return np.ones((256, 256), np.int32)
    if case == 'tiled-4096':
        row, col = np.ogrid[:512, :512]
        return np.where((row % 8 < 2) & (col % 8 < 2),
                        (row // 8) * 64 + (col // 8) + 1, 0).astype(np.int32)
    if case == 'fragmented':
        words = np.arange(1024 * 1024, dtype=np.int32).reshape(1024, 1024)
        return np.where(words % 8 == 0, 0, 1 + words % 4096)
    if case == 'one-pixel':
        return np.array([[4096]], np.int32)
    if case == 'high-only-gapped':
        return np.array([[0, 4096, 0], [0, 17, 0]], np.int32)
    if case in ('gapped-disconnected', 'read-only-view'):
        image = np.array([[7, 0, 0, 3, 0, 7], [0, 0, 0, 0, 0, 0],
                          [11, 0, 3, 0, 7, 0], [0, 11, 0, 0, 0, 0]], np.int32)
        if case == 'read-only-view':
            image = image.view()
            image.flags.writeable = False
        return image
    raise ValueError(case)


def word_hash(array):
    return sha256(array.tobytes(order='C')).hexdigest()


def assert_frozen_table(case, image, table):
    witness = CONTRACT['observations'][case]
    assert word_hash(image) == witness['input_sha256']
    assert type(table) is dict and tuple(table) == KEYS
    count = witness['label_rows']
    columns = tuple(table.values())
    for name, column in table.items():
        assert type(column) is np.ndarray
        assert column.shape == (count,) and column.dtype == np.dtype(np.intp)
        assert column.flags.writeable and column.flags.aligned
        assert not np.shares_memory(column, image)
        assert word_hash(column) == witness['columns_sha256'][name]

    labels, *boxes = columns
    stride = labels.itemsize
    assert labels.strides == (stride,) and labels.flags.owndata == bool(count)
    label_base = labels if count else labels.base
    assert type(label_base) is np.ndarray and label_base.flags.owndata and label_base.base is None
    assert label_base.shape == (max(count, 1),) and label_base.flags.writeable
    assert word_hash(label_base) == witness['label_base_sha256']
    base = boxes[0].base
    assert type(base) is np.ndarray and base.shape == (max(count, 1), 4)
    assert base.flags.owndata and base.flags.c_contiguous and base.flags.writeable and base.base is None
    assert word_hash(base) == witness['bbox_base_sha256']
    for index, box in enumerate(boxes):
        assert box.base is base and not box.flags.owndata and box.strides == (4 * stride,)
        assert box.ctypes.data - base.ctypes.data == index * stride
        assert box.flags.c_contiguous == box.flags.f_contiguous == (count <= 1)
    for index, column in enumerate(columns):
        assert all(not np.shares_memory(column, other) for other in columns[:index])
    return label_base, base
