"""Owned MARK copy/scatter cases beyond the retained small public golds."""
import importlib

import numpy as np
import pytest


@pytest.fixture(scope='module')
def core():
    return importlib.import_module('skimagexx')._core


@pytest.mark.parametrize('shape', ((5, 7), (3, 5, 3), (2, 3, 4, 3)))
@pytest.mark.parametrize('dtype', ('uint8', '>u2', 'float16', 'float32', 'float64'))
@pytest.mark.parametrize('layout', ('C', 'F', 'reverse', 'step', 'unaligned'))
def test_prepare_preserves_logical_pixels_and_owns_storage(core, shape, dtype, layout):
    values = (np.arange(np.prod(shape), dtype=np.uint16) % 251).reshape(shape)
    image = values.astype(dtype)
    if layout == 'F':
        image = np.asfortranarray(image)
    elif layout == 'reverse':
        image = image[(slice(None, None, -1),) * image.ndim]
    elif layout == 'step':
        image = np.repeat(image, 2, axis=0)[::2]
    elif layout == 'unaligned':
        buffer = np.empty(image.nbytes + 1, dtype=np.uint8)
        unaligned = np.ndarray(shape, dtype=image.dtype, buffer=buffer, offset=1)
        unaligned[...] = image
        image = unaligned
    before = image.tobytes()
    expected_dtype = np.dtype('float32' if image.dtype.kind == 'f' and image.itemsize <= 4 else 'float64')
    expected = image.astype(expected_dtype)
    if image.dtype.kind in 'ui':
        expected *= np.float64(1.0 / np.iinfo(image.dtype).max)
    if image.ndim == 2:
        expected = np.repeat(expected[..., None], 3, axis=-1)
    out = core.mark_boundaries_prepare(image, False, False, 1)
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert out.tobytes() == expected.tobytes()
    assert out.flags.owndata and not np.shares_memory(out, image)
    assert image.tobytes() == before


@pytest.mark.parametrize('shape', ((1 << 20, 0), (0, 1 << 20)))
@pytest.mark.parametrize('dtype', ('uint16', 'float32'))
def test_empty_gray_F_metadata_never_requires_pixel_storage(core, shape, dtype):
    owner = np.empty(0, dtype=dtype)
    # The existing ndarray-backed policy admits empty descriptors. Deliberate
    # F-like strides exercise non-C layout metadata without allocating pixels.
    image = np.ndarray(shape, dtype=owner.dtype, buffer=owner,
                       strides=(owner.itemsize, (max(shape[0], 1) + 1) * owner.itemsize))
    out = core.mark_boundaries_prepare(image, False, False, 1)
    assert out.shape == shape + (3,) and out.size == out.nbytes == 0
    assert out.dtype == np.dtype('float32' if dtype == 'float32' else 'float64')
    # NumPy canonicalizes strides when allocating an owning empty result.
    assert out.strides == np.empty(out.shape, dtype=out.dtype).strides
    assert out.flags.owndata and image.size == owner.size == 0


@pytest.mark.parametrize('dtype,words', (
    ('float32', (0x00000000, 0x80000000, 0x00000001, 0x7f800000, 0x7fc12345, 0x7f812345)),
    ('float64', (0x0000000000000000, 0x8000000000000000, 0x0000000000000001,
                 0x7ff0000000000000, 0x7ff8123456789abc, 0x7ff0123456789abc)),
))
@pytest.mark.parametrize('reverse', (False, True))
def test_prepare_copies_float_storage_words_without_arithmetic(core, dtype, words, reverse):
    raw = np.array(words, dtype='uint32' if dtype == 'float32' else 'uint64').reshape(2, 3)
    image = raw.view(dtype)
    if reverse:
        image = image[:, ::-1]
        raw = raw[:, ::-1]
    out = core.mark_boundaries_prepare(image, False, False, 1)
    expected = np.repeat(raw[..., None], 3, axis=-1)
    assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('order', ('C', 'F', 'gray-F'))
@pytest.mark.parametrize('palette_kind', ('scalar', 'channel', 'selected'))
def test_paint_preserves_selected_palette_order_and_normalizes_bool_storage(core, dtype, order, palette_kind):
    selection = np.array([[0, 2, 255, 0, 1], [255, 0, 1, 2, 0], [0, 0, 255, 1, 0]], dtype=np.uint8).view(np.bool_)
    selected = int(np.count_nonzero(selection))
    if order == 'gray-F':
        gray = np.asfortranarray(np.zeros(selection.shape, dtype=dtype))
        image = core.mark_boundaries_prepare(gray, False, False, 1)
        assert image.strides == (3 * image.itemsize, 3 * selection.shape[0] * image.itemsize, image.itemsize)
    else:
        image = np.empty(selection.shape + (3,), dtype=dtype, order=order)
    image.fill(-4)
    expected = image.copy(order='K')
    colors = (np.array(2, dtype=dtype) if palette_kind == 'scalar' else
              np.array([1, -2, 3], dtype=dtype) if palette_kind == 'channel' else
              np.arange(selected * 3, dtype=dtype).reshape(selected, 3))
    expected[selection] = colors
    assert core.mark_boundaries_count(selection) == selected
    assert core.mark_boundaries_paint(image, selection, colors) is True
    assert image.tobytes() == expected.tobytes()


@pytest.mark.parametrize('order', ('C', 'F'))
def test_invalid_selected_palette_does_not_partially_paint(core, order):
    selection = np.ones((3, 5), dtype=np.bool_)
    image = np.empty((3, 5, 3), dtype=np.float64, order=order)
    image.fill(-4)
    before = image.tobytes()
    with pytest.raises(ValueError, match='broadcast'):
        core.mark_boundaries_paint(image, selection, np.zeros((14, 3), dtype=np.float64))
    assert image.tobytes() == before


@pytest.mark.parametrize('shape', ((1, 1), (1, 7), (7, 1), (5, 7)))
def test_outline_square_matches_full_neighborhood_at_every_edge(core, shape):
    mask = np.zeros(shape, dtype=np.bool_)
    mask[0, 0] = True
    mask[-1, -1] = True
    if min(shape) > 2:
        mask[2, 3] = True
    padded = np.pad(mask, 1, mode='edge')
    expected = np.array([[np.any(padded[r:r + 3, c:c + 3]) for c in range(shape[1])]
                         for r in range(shape[0])], dtype=np.bool_)
    actual = core.mark_boundaries_outline(mask)
    assert actual.tobytes() == expected.tobytes() and actual.flags.owndata
