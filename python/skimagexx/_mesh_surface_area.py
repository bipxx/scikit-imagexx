"""Owned finite mesh arithmetic; literal gathering and primitive frontiers."""
import numpy as np
from . import _core

_PROFILES = {'1.24.4': 1, '2.5.4': 2}


def mesh_surface_area(verts, faces):
    # Full fancy indexing is the FIRST source operation, including unused
    # fourth indices. Do not repeat it on a native numerical refusal.
    actual_verts = verts[faces]
    profile = _PROFILES.get(np.__version__, 0)
    if (profile and type(actual_verts)is np.ndarray and actual_verts.ndim==3
            and actual_verts.shape[1]>=3 and actual_verts.shape[2]==3
            and actual_verts.dtype.char in ('f', 'd')):
        prepared = np.ascontiguousarray(actual_verts, dtype=actual_verts.dtype.newbyteorder('='))
        total = _core.mesh_surface_area(prepared, profile, np.getbufsize(), _core.get_num_threads())
        if total is not None:
            return total / 2.0
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    del actual_verts
    return ((np.cross(a, b) ** 2).sum(axis=1) ** 0.5).sum() / 2.0
