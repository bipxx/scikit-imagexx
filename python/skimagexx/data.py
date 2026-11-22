"""Local sample data and explicit upstream compatibility."""
from ._compat import module_attribute
from ._binary_blobs import binary_blobs
from ._data_chelsea import cat, chelsea
from ._data_clock_horse import clock, horse
from ._data_textures import brick, grass, gravel, text

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('data', name)
