"""Native array and simple-file paths with scikit-image's codec and collection protocols."""

from functools import lru_cache
from pathlib import Path, WindowsPath, PosixPath
from os import stat
from os.path import abspath, basename
from stat import S_ISREG
from struct import Struct

import numpy as np
from PIL import Image
from skimage import io as _stock
from skimage._shared.utils import DEPRECATED, deprecate_parameter, warn
from skimage.exposure import is_low_contrast
from skimage.io._plugins import imageio_plugin
from skimage.io.manage_plugins import plugin_store

from . import _compat, _core

__all__ = [
    'ImageCollection', 'MultiImage', 'concatenate_images', 'imread', 'imread_collection',
    'imread_collection_wrapper', 'imsave', 'load_sift', 'load_surf', 'pop', 'push',
]

_imread = _stock.imread
_imsave = _stock.imsave
_imread_collection = _stock.imread_collection
_imageio_read = imageio_plugin.imread
_imageio_write = imageio_plugin.imsave
_imageio_collection = imageio_plugin.imread_collection
_max_file_bytes = 64 * 1024 * 1024
_png_header = Struct('>16sII')
_png_prefix = b'\x89PNG\r\n\x1a\n\0\0\0\rIHDR'
_uint8 = np.dtype(np.uint8)

image_stack = _stock.image_stack
pop, push = _core.io_stack_api(image_stack, np.ndarray, _stock.push, _stock.pop)


def concatenate_images(ic):
    if type(ic) in (list, tuple):
        result = _core.io_concatenate(ic, np.ndarray, False)
        if result is not None:
            return result
    return _compat.fallback('io', 'concatenate_images', ic)


class ImageCollection(_stock.ImageCollection):
    def concatenate(self):
        if type(self) in (ImageCollection, MultiImage) and self.conserve_memory is False:
            frames = self.data
            if (
                type(frames) is np.ndarray and frames.dtype == object
                and frames.shape == (len(self),) and '_check_imgnum' not in self.__dict__
            ):
                # The inherited indexer only reads a populated, non-conserving cache.
                result = _core.io_concatenate(tuple(frames), np.ndarray, False)
                if result is not None:
                    return result
        # Index each frame before fetching the next so a loader can reuse and
        # reshape its ndarray without changing the earlier views.
        expanded = [image[None, ...] for image in self]
        result = _core.io_concatenate(expanded, np.ndarray, True)
        if result is not None:
            return result
        try:
            return np.concatenate(expanded)
        except ValueError:
            raise ValueError('Image dimensions must agree.')


class MultiImage(_stock.MultiImage, ImageCollection):
    pass


def _local_path(fname):
    if type(fname) is str:
        if '://' not in fname and not fname.startswith(('<', '~', 'imageio:', 'file:\\')):
            return fname
        return ''
    if type(fname) in (Path, WindowsPath, PosixPath):
        return str(fname.resolve())
    return ''


def _active(kind, function):
    plugins = plugin_store[kind]
    return plugins and plugins[0][1] is function


def _zip_member(lower_path):
    return '.zip/' in lower_path or '.zip\\' in lower_path


def _read_bytes(path):
    try:
        info = stat(path)
        if not S_ISREG(info.st_mode) or not 0 < info.st_size <= _max_file_bytes:
            return b''
        with open(path, 'rb') as stream:
            data = stream.read(info.st_size + 1)
            return data if len(data) == info.st_size else b''
    except (OSError, ValueError):
        # The codec or descriptor reader owns path errors and their messages.
        return b''


def _read_png(path):
    try:
        info = stat(path)
        if not S_ISREG(info.st_mode):
            return None
        # Buffer small compressed rasters across Pillow's header seeks and reads.
        buffer_size = info.st_size if 8192 < info.st_size <= 256 * 1024 else -1
        with open(path, 'rb', buffering=buffer_size) as stream:
            header = stream.read(_png_header.size)
            if len(header) != _png_header.size:
                return None
            signature, width, height = _png_header.unpack(header)
            limit = Image.MAX_IMAGE_PIXELS
            if signature != _png_prefix or (limit is not None and (
                type(limit) is not int or width * height > limit
            )):
                return None  # Let the provider own diagnostics.
            with Image.open(stream, formats=('PNG',)) as image:
                if (image.mode not in ('1', 'L', 'RGB', 'RGBA', 'I;16')
                        or image.custom_mimetype == 'image/apng'):
                    return None
                # Pillow exports the pixels through __array_interface__, which
                # loads the raster and any trailing metadata.
                pixels = np.array(image)
                if 'exif' not in image.info:
                    return pixels
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        pass
    return None


def imread(fname, as_gray=False, plugin=DEPRECATED, **plugin_args):
    if plugin is DEPRECATED and not plugin_args and as_gray is False and _active('imread', _imageio_read):
        path = _local_path(fname)
        if path:
            fname = path
        lower_path = path.lower()
        suffix = lower_path[-4:]
        if suffix == '.png' and not _zip_member(lower_path):
            result = _read_png(abspath(path))
            if result is not None:
                return result
        elif suffix in ('.pgm', '.ppm', '.pnm') and not _zip_member(lower_path):
            data = _read_bytes(path)
            if data:
                limit = Image.MAX_IMAGE_PIXELS
                if limit is None or (type(limit) is int and limit >= 0):
                    maximum = _max_file_bytes if limit is None else min(limit, _max_file_bytes)
                    image = _core.io_decode_netpbm(data, maximum)
                    if image is not None:
                        return image
    return _imread(fname, as_gray, plugin=plugin, **plugin_args)


# Use the upstream decorator frame for the warning's source and registry.
@deprecate_parameter('plugin', start_version='0.25', stop_version='0.27')
def _warn_low_contrast(path, *, plugin=DEPRECATED):
    warn(f'{path} is a low contrast image')


def imsave(fname, arr, plugin=DEPRECATED, *, check_contrast=True, **plugin_args):
    if (
        plugin is DEPRECATED and not plugin_args and type(check_contrast) is bool
        and type(arr) is np.ndarray and arr.dtype == _uint8
        and _active('imsave', _imageio_write)
    ):
        path = _local_path(fname)
        if path:
            fname = path
        lower_path = path.lower()
        suffix = lower_path[-4:]
        if suffix in ('.png', '.pgm', '.ppm') and not _zip_member(lower_path):
            if suffix == '.png' and basename(lower_path) != '.png' and arr.size and (
                arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4))
            ):
                if check_contrast:
                    low_contrast = _core.io_gray_png_low_contrast(arr) if arr.ndim == 2 else None
                    if low_contrast is None:
                        low_contrast = is_low_contrast(arr)
                    if low_contrast:
                        _warn_low_contrast(path)
                        # A warning handler may change the active writer.
                        if not _active('imsave', _imageio_write):
                            return _imsave(fname, arr, plugin=plugin, check_contrast=False)
                    check_contrast = False
                try:
                    image = Image.fromarray(arr)
                except (TypeError, ValueError):
                    pass  # Let the provider report unsupported arrays.
                else:
                    with image:
                        try:
                            stream = open(abspath(path), 'wb')
                        except (OSError, ValueError):
                            pass  # Retain the provider's path diagnostics.
                        else:
                            with stream:
                                image.save(stream, format='PNG')
                            return None
            elif suffix in ('.pgm', '.ppm') and not check_contrast:
                header = _core.io_netpbm_header(arr, suffix == '.ppm')
                if header is not None:
                    try:
                        stream = open(path, 'wb')
                    except (OSError, ValueError):
                        pass
                    else:
                        with stream:
                            stream.write(header)
                            stream.write(arr)
                        return None
    return _imsave(fname, arr, plugin=plugin, check_contrast=check_contrast, **plugin_args)


def imread_collection_wrapper(imread):
    def imread_collection(load_pattern, conserve_memory=True):
        return ImageCollection(load_pattern, conserve_memory=conserve_memory, load_func=imread)

    return imread_collection


def imread_collection(load_pattern, conserve_memory=True, plugin=DEPRECATED, **plugin_args):
    if plugin is DEPRECATED and not plugin_args and _active('imread_collection', _imageio_collection):
        return ImageCollection(load_pattern, conserve_memory=conserve_memory, load_func=_imageio_read)
    return _imread_collection(load_pattern, conserve_memory, plugin=plugin, **plugin_args)


@lru_cache(maxsize=16)
def _feature_dtype(surf, width):
    if surf:
        fields = [('column', float), ('row', float), ('second_moment', (float, 3)), ('sign', float)]
    else:
        fields = [('row', float), ('column', float), ('scale', float), ('orientation', float)]
    return np.dtype([*fields, ('data', (float, width))])


def _load_features(f, surf):
    if type(f) is str:
        data = _read_bytes(f)
        if data:
            result = _core.io_read_features(data, surf)
            if result is not None:
                values, width = result
                return values.view(_feature_dtype(surf, width))
    return _compat.fallback('io', 'load_surf' if surf else 'load_sift', f)


def load_sift(f):
    return _load_features(f, False)


def load_surf(f):
    return _load_features(f, True)


# Preserve the stock documentation without changing the leaf's call signatures.
for _name in __all__:
    if _name not in ('pop', 'push'):
        globals()[_name].__doc__ = getattr(_stock, _name).__doc__
