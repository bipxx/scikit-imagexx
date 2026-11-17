"""Meaningful run-copy domains; execution requires an admitted native core/F1.

The existing59 assignment cases retain dtype/shape/error coverage. These add
padded long runs, opaque exceptional float words and an initially-full live
positive-stride alias that must keep the ordered mismatch fallback.
"""
import pytest
from test_flood_fill_public import target, genuine_global_worker_and_caller
from test_flood_assignment_protocol import assign, guarded


def padded_mask(np, shape, layout):
    storage = np.ones(tuple(n + 2 for n in shape), dtype=bool, order=layout)
    return storage, storage[tuple(slice(1, n + 1) for n in shape)]


def word_values(np, count, dtype):
    values = np.empty(count, dtype=dtype)
    payload = (bytes(range(256)) * ((values.nbytes + 255) // 256))[:values.nbytes]
    values.view('u1')[:] = np.frombuffer(payload, dtype='u1')
    if count >= 4:
        # sNaN, qNaN, negative zero, and a finite word: assignment is byte-only.
        values.view('u4')[:4] = (0x7f800001, 0x7fc01234, 0x80000000, 0x3f800001)
    return values


def expected_bytes(output, mask, values):
    result = bytearray(output.tobytes())
    rhs, size, selected = values.tobytes(), output.itemsize, 0
    for index, flag in enumerate(mask.flat):
        if flag:
            result[index * size:(index + 1) * size] = rhs[selected * size:(selected + 1) * size]
            selected += 1
    return bytes(result)


@pytest.mark.parametrize('shape', ((3, 7), (2, 3, 5)))
@pytest.mark.parametrize('layout', ('C', 'F'))
@pytest.mark.parametrize('partial', (False, True))
def test_padded_full_and_partial_long_runs_keep_raw_words_and_bounds(target, assign,
                                                                  shape, layout, partial):
    np, _ = target
    dtype = np.dtype('f4', metadata={'kind': 'flood-run-words'})
    store, output = guarded(np, dtype, 'C', shape)
    mask_store, mask = padded_mask(np, shape, layout)
    if partial:
        mask.flat[::5] = False
    values = word_values(np, int(np.count_nonzero(mask)), dtype)
    expected = expected_bytes(output, mask, values)
    outside, mask_before, rhs_before = store.tobytes(), mask_store.tobytes(), values.tobytes()
    pointer, metadata = output.ctypes.data, output.dtype.metadata
    assert assign(output, mask, values) is None
    assert output.tobytes() == expected
    assert output.ctypes.data == pointer and output.dtype.metadata == metadata
    assert store[:32].tobytes() == outside[:32] and store[-32:].tobytes() == outside[-32:]
    assert mask_store.tobytes() == mask_before and values.tobytes() == rhs_before


def test_initially_full_positive_alias_keeps_live_reads_and_partial_error(target, assign):
    np, _ = target
    raw = np.ones(5, dtype=bool)
    output, mask = raw[1:], raw[:-1]
    values = np.array([False, True, False, True], dtype=bool)
    with pytest.raises(ValueError, match='value count mismatch'):
        assign(output, mask, values)
    assert raw.tobytes() == bytes([1, 0, 1, 1, 0])


@pytest.mark.parametrize('layout', ('C', 'F'))
def test_partial_mask_with_full_RHS_keeps_mismatch_writes(target, assign, layout):
    np, _ = target
    dtype = np.dtype('f4')
    store, output = guarded(np, dtype, 'C', (3, 7))
    mask_store, mask = padded_mask(np, output.shape, layout)
    mask.flat[::5] = False
    values = word_values(np, output.size, dtype)
    expected = expected_bytes(output, mask, values)
    before, mask_before, rhs_before = store.tobytes(), mask_store.tobytes(), values.tobytes()
    with pytest.raises(ValueError, match='value count mismatch'):
        assign(output, mask, values)
    assert output.tobytes() == expected
    assert store[:32].tobytes() == before[:32] and store[-32:].tobytes() == before[-32:]
    assert mask_store.tobytes() == mask_before and values.tobytes() == rhs_before


def test_nonunit_mask_strides_keep_ordered_fallback(target, assign):
    np, _ = target
    _, output = guarded(np, np.dtype('f4'), 'C', (3, 7))
    backing = np.ones((6, 14), dtype=bool)
    mask = backing[::2, ::2]
    mask.flat[::4] = False
    values = word_values(np, int(np.count_nonzero(mask)), output.dtype)
    expected = expected_bytes(output, mask, values)
    before = backing.tobytes()
    assert assign(output, mask, values) is None
    assert output.tobytes() == expected and backing.tobytes() == before


def test_contiguous_noncanonical_bool_bytes_select_nonzero_runs(target, assign):
    np, _ = target
    _, output = guarded(np, np.dtype('f4'), 'C', (2, 5))
    raw = np.array([2, 128, 0, 255, 3, 0, 4, 129, 255, 5], dtype='u1')
    mask = raw.view(bool).reshape(output.shape)
    values = word_values(np, 8, output.dtype)
    expected, before = expected_bytes(output, mask, values), raw.tobytes()
    assert assign(output, mask, values) is None
    assert output.tobytes() == expected and raw.tobytes() == before
