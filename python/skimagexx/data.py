"""Local sample data and packaged dataset loaders."""
from ._compat import module_attribute
from ._binary_blobs import binary_blobs
from ._data_camera_coffee import camera, coffee
from ._data_chelsea import cat, chelsea
from ._data_checkerboard import checkerboard
from ._data_clock_horse import clock, horse
from ._data_file_hash import file_hash
from ._data_loaders_a import (
    astronaut, brain, cell, cells3d, coins, colorwheel, download_all,
    eagle, hubble_deep_field, human_mitosis, immunohistochemistry,
    kidney, lbp_frontal_face_cascade_filename, lfw_subset,
)
from ._data_loaders_b import (
    lily, logo, microaneurysms, moon, nickel_solidification, page,
    palisades_of_vogt, protein_transport, retina, rocket, skin,
    stereo_motorcycle, vortex,
)
from ._data_shepp_logan_phantom import shepp_logan_phantom
from ._data_textures import brick, grass, gravel, text

__all__ = [
    'astronaut', 'binary_blobs', 'brain', 'brick', 'camera', 'cat', 'cell',
    'cells3d', 'checkerboard', 'chelsea', 'clock', 'coffee', 'coins',
    'colorwheel', 'download_all', 'eagle', 'file_hash', 'grass', 'gravel',
    'horse', 'hubble_deep_field', 'human_mitosis', 'immunohistochemistry',
    'kidney', 'lbp_frontal_face_cascade_filename', 'lfw_subset', 'lily',
    'logo', 'microaneurysms', 'moon', 'nickel_solidification', 'page',
    'palisades_of_vogt', 'protein_transport', 'retina', 'rocket',
    'shepp_logan_phantom', 'skin', 'stereo_motorcycle', 'text', 'vortex',
]

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('data', name)
