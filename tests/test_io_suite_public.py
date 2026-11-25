import inspect
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from PIL import Image
from skimage import io as stock
from skimage.io.manage_plugins import plugin_store
from tifffile import imwrite

from skimagexx import FallbackWarning, set_fallback_enabled
from skimagexx import _io_suite as native


@pytest.fixture
def file():
    paths = []

    def make(suffix, contents=b''):
        with NamedTemporaryFile(
            dir=Path(__file__).with_name('fixtures'), prefix='io_suite_', suffix=suffix, delete=False,
        ) as stream:
            paths.append(Path(stream.name))
            stream.write(contents)
        return paths[-1]

    yield make
    for path in paths:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize('name', stock.__all__)
def test_public_signatures(name):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(stock, name))


@pytest.mark.parametrize('name', ['ImageCollection', 'MultiImage'])
def test_class_members(name):
    actual, expected = getattr(native, name), getattr(stock, name)
    assert issubclass(actual, expected)
    for member in dir(expected):
        left = inspect.getattr_static(actual, member)
        right = inspect.getattr_static(expected, member)
        if isinstance(right, property):
            assert isinstance(left, property)
            left, right = left.fget, right.fget
        elif isinstance(right, (classmethod, staticmethod)):
            assert type(left) is type(right)
            left, right = left.__func__, right.__func__
        if callable(right):
            try:
                signature = inspect.signature(right)
            except (ValueError, TypeError):
                continue
            assert inspect.signature(left) == signature, member
    assert issubclass(native.MultiImage, native.ImageCollection)


@pytest.mark.parametrize('dtype', ['B', 'H', 'l', 'q', 'f', 'd', 'D', '?'])
@pytest.mark.parametrize('strided', [False, True])
def test_native_concatenation(dtype, strided):
    image = np.arange(120).astype(dtype).reshape(8, 15)
    images = [image[1::2, ::-2], image[::2, ::-2]] if strided else [image, image.copy()]
    expected = stock.concatenate_images(images)
    previous = set_fallback_enabled(False)
    try:
        result = native.concatenate_images(images)
    finally:
        set_fallback_enabled(previous)
    np.testing.assert_array_equal(result, expected)
    assert (result.strides, result.dtype, result.flags.owndata) == (expected.strides, expected.dtype, True)


@pytest.mark.parametrize('kind', ['metadata', 'foreign-byte-order', 'promotion', 'object', 'fortran', 'generator'])
def test_concatenation_fallback(kind):
    inputs = {
        'metadata': [np.ones(3, dtype=np.dtype('f4', metadata={'units': 'pixels'}))] * 2,
        'foreign-byte-order': [np.arange(8, dtype='>i2')] * 2,
        'promotion': [np.arange(8, dtype='f8'), np.arange(8, dtype='u4')],
        'object': [np.array([{'x': 1}])] * 2,
        'fortran': [np.asfortranarray(np.ones((3, 4)))] * 2,
        'generator': [np.zeros((2, 3)), np.ones((2, 3))],
    }[kind]
    expected = stock.concatenate_images(inputs)
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning):
            actual = native.concatenate_images(iter(inputs) if kind == 'generator' else inputs)
    finally:
        set_fallback_enabled(previous)
    np.testing.assert_array_equal(actual, expected)
    assert (actual.dtype, actual.dtype.metadata, actual.strides) == (
        expected.dtype, expected.dtype.metadata, expected.strides,
    )


@pytest.mark.parametrize('images', [[], [np.zeros(3), np.zeros(5)]])
def test_concatenation_error(images):
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning), pytest.raises(ValueError, match='Image dimensions must agree'):
            native.concatenate_images(images)
    finally:
        set_fallback_enabled(previous)


def test_concatenation_scalar_arrays():
    images = [np.array(1), np.array(2)]
    np.testing.assert_array_equal(native.concatenate_images(images), stock.concatenate_images(images))


@pytest.mark.parametrize('rgb', [False, True])
def test_netpbm_read_and_write(file, rgb, monkeypatch):
    shape = (5, 7, 3) if rgb else (5, 7)
    pixels = np.arange(np.prod(shape), dtype='u1').reshape(shape)
    path = file('.ppm' if rgb else '.pgm')
    stock.imsave(path, pixels, check_contrast=False)
    expected_bytes = path.read_bytes()
    expected_array = stock.imread(path)

    def unexpected_provider(*args, **kwargs):
        pytest.fail('canonical NetPBM call reached the Python codec')

    monkeypatch.setattr(native, '_imread', unexpected_provider)
    monkeypatch.setattr(native, '_imsave', unexpected_provider)
    for fname in (path, str(path)):
        assert native.imsave(fname, pixels, check_contrast=False) is None
        assert path.read_bytes() == expected_bytes
        result = native.imread(fname)
        assert result.flags.writeable and result.flags.owndata
        np.testing.assert_array_equal(result, expected_array)


def test_codec_options_warnings_paths_and_metadata(file):
    png = file('.png')
    pixels = np.ones((5, 7), dtype=bool)
    for fn in (stock.imsave, native.imsave):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fn(png, pixels, plugin='imageio')
        messages = [(type(w.message), str(w.message)) for w in caught]
        if fn is stock.imsave:
            expected_messages = messages
        else:
            assert messages == expected_messages
    contents = png.read_bytes()
    np.testing.assert_array_equal(native.imread(BytesIO(contents)), stock.imread(BytesIO(contents)))
    tiff = file('.tiff')
    pixels = np.arange(120, dtype='u2').reshape(3, 5, 8)
    imwrite(tiff, pixels, photometric='minisblack', metadata={'axes': 'ZYX'})
    np.testing.assert_array_equal(native.imread(tiff), stock.imread(tiff))
    with pytest.raises(FileNotFoundError) as original:
        stock.imread(str(png) + '.ppm')
    with pytest.raises(type(original.value)) as actual:
        native.imread(str(png) + '.ppm')
    assert str(original.value) == str(actual.value)


def test_noncanonical_netpbm_and_plugin_selection(file, monkeypatch):
    path = file('.pgm', b'P5\n# comment\n2 1\n15\n\x00\x0f')
    np.testing.assert_array_equal(native.imread(path), stock.imread(path))
    custom_result = np.ones((3, 3))
    monkeypatch.setitem(plugin_store, 'imread', [('test_plugin', lambda fname: custom_result)])
    assert native.imread(path) is custom_result
    monkeypatch.setitem(plugin_store, 'imread_collection', [('test_plugin', lambda *args: 'plugin collection')])
    assert native.imread_collection(path) == 'plugin collection'


def test_netpbm_decode_boundaries(file, monkeypatch):
    path = file('.pgm', b'P5\n2 1\n65535\n\x00\x01\x10\x00')
    np.testing.assert_array_equal(native.imread(path), stock.imread(path))
    path = file('.pgm', b'P5\n3 2\n255\n012345')
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 2)
    with pytest.raises(Image.DecompressionBombError):
        native.imread(path)


@pytest.mark.parametrize('conserve_memory', [True, False])
def test_collections(conserve_memory):
    def loader(index, multiplier=1):
        return np.full((3, 4), index * multiplier)

    def observe(cls):
        collection = cls([2, 1, 0], conserve_memory, loader, multiplier=2)
        assert collection.conserve_memory is conserve_memory
        assert collection.files == [2, 1, 0]
        item = collection[-1]
        assert item is collection[2]
        collection.reload()
        assert item is not collection[2]
        with pytest.raises(IndexError, match='There are only 3 images'):
            collection[3]
        with pytest.raises(TypeError, match='slicing must be'):
            collection[1.5]
        sliced = collection[::-2]
        assert isinstance(sliced, cls)
        return sliced.concatenate()

    np.testing.assert_array_equal(observe(native.ImageCollection), observe(stock.ImageCollection))
    factory = native.imread_collection_wrapper(loader)
    assert inspect.signature(factory) == inspect.signature(stock.imread_collection_wrapper(loader))
    assert type(factory([2, 1, 0])) is native.ImageCollection


def test_collection_loader_changes_shape():
    def collection(cls):
        buffer = np.arange(8)

        def loader(index):
            buffer.shape = (4, 2) if index == 1 else (2, 4)
            return buffer

        return cls([0, 1], load_func=loader)

    for cls in (native.ImageCollection, stock.ImageCollection):
        with pytest.raises(ValueError, match='Image dimensions must agree'):
            collection(cls).concatenate()


def test_collection_file_loading_and_multiimage(file):
    tiff = file('.tif')
    pixels = np.arange(60, dtype='u2').reshape(3, 4, 5)
    imwrite(tiff, pixels, photometric='minisblack')
    for cls in ('MultiImage', 'ImageCollection'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            actual = getattr(native, cls)(str(tiff))
            expected = getattr(stock, cls)(str(tiff))
            assert actual.files == expected.files
            assert len(actual) == len(expected)
            if cls == 'MultiImage':
                assert actual.filename == expected.filename
            np.testing.assert_array_equal(actual.concatenate(), expected.concatenate())
    for function in (native.imread_collection, stock.imread_collection):
        collection = function(str(tiff), conserve_memory=False)
        assert not collection.conserve_memory
        np.testing.assert_array_equal(collection[0], pixels)


@pytest.mark.parametrize('kind, contents', [
    ('sift', b'2 2\r\n-0 .25 +1e1 -2.5 5 6\r\n-2 4 6 8 10 12\r\n'),
    ('sift', b'0 128\n'),
    ('surf', b'3\n1\n-1 .125 2 3 4 5 6 7\n'),
    ('surf', b'1\n1\n1 2 3 4 5 6\n'),
])
def test_native_feature_files(file, kind, contents):
    path = file('.key', contents)
    old, new = (getattr(module, f'load_{kind}') for module in (stock, native))
    previous = set_fallback_enabled(False)
    try:
        actual, expected = new(str(path)), old(str(path))
    finally:
        set_fallback_enabled(previous)
    assert actual.dtype.descr == expected.dtype.descr
    assert actual.tobytes() == expected.tobytes()
    previous = set_fallback_enabled(True)
    try:
        with open(path) as stream, pytest.warns(FallbackWarning):
            np.testing.assert_array_equal(new(stream), expected)
    finally:
        set_fallback_enabled(previous)


def test_feature_file_bad_count(file):
    path = file('.key', b'2 0\n1 2 3 4\n')
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning), pytest.raises(OSError, match='Invalid SIFT feature file'):
            native.load_sift(str(path))
    finally:
        set_fallback_enabled(previous)


def test_stack_cross_namespace_and_errors():
    original = stock.image_stack[:]
    stock.image_stack.clear()
    try:
        image = np.zeros((2, 3))
        native.push(img=image)
        assert stock.pop() is image
        stock.push(image)
        assert native.pop() is image
        with pytest.raises(IndexError, match='pop from empty list'):
            native.pop()
        with pytest.raises(ValueError, match='Can only push ndarrays'):
            native.push([])
        for name in ('pop', 'push'):
            with pytest.raises(TypeError) as expected:
                getattr(stock, name)(unknown=True)
            with pytest.raises(TypeError) as actual:
                getattr(native, name)(unknown=True)
            assert str(actual.value) == str(expected.value)
    finally:
        stock.image_stack[:] = original
