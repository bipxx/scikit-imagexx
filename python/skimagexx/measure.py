"""Native labels, partial colocalization and explicit compatibility APIs."""
from ._components import label
from ._colocalization import intersection_coeff, manders_coloc_coeff, manders_overlap_coeff
from ._compat import module_attribute

__all__ = ['label', 'intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('measure', name)


from ._moments import centroid, moments, moments_central

__all__ += ['moments', 'moments_central', 'centroid']

from ._moments_derived import moments_hu, moments_normalized

__all__ += ['moments_normalized', 'moments_hu']
