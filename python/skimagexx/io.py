"""Native array and image I/O paths with scikit-image compatibility."""
from ._compat import module_attribute
_IO_EXPORTS = (
    'ImageCollection', 'MultiImage', 'concatenate_images', 'imread',
    'imread_collection', 'imread_collection_wrapper', 'imsave',
    'load_sift', 'load_surf', 'pop', 'push',
)

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = list(_IO_EXPORTS)


def __getattr__(name):
    if name in _IO_EXPORTS:
        from . import _io_suite
        value = getattr(_io_suite, name)
        globals()[name] = value
        return value
    return module_attribute('io', name)
