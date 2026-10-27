"""Native labels, partial colocalization and explicit compatibility APIs."""
from ._components import label
from ._colocalization import intersection_coeff, manders_coloc_coeff, manders_overlap_coeff
from ._compat import module_attribute

__all__ = ['label', 'intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('measure', name)


from ._moments import centroid, inertia_tensor, moments, moments_central

__all__ += ['moments', 'moments_central', 'centroid', 'inertia_tensor']

from ._moments_derived import moments_hu, moments_normalized

__all__ += ['moments_normalized', 'moments_hu']


from ._moments import inertia_tensor_eigvals

__all__ += ['inertia_tensor_eigvals']


from ._moments import moments_coords, moments_coords_central

__all__ += ['moments_coords', 'moments_coords_central']


from ._shannon_entropy import shannon_entropy

__all__ += ['shannon_entropy']


from ._block_reduce import block_reduce

__all__ += ['block_reduce']


from ._perimeter_crofton import perimeter_crofton

__all__ += ['perimeter_crofton']


from ._perimeter import perimeter

__all__ += ['perimeter']


from ._euler_number import euler_number

__all__ += ['euler_number']
