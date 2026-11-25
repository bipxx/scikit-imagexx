"""Native image registration algorithms with compatibility dispatch."""
from ._compat import module_attribute
from ._phase_cross_correlation import phase_cross_correlation
from ._registration_flows import optical_flow_ilk, optical_flow_tvl1

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = ['optical_flow_ilk', 'optical_flow_tvl1', 'phase_cross_correlation']


def __getattr__(name):
    return module_attribute('registration', name)
