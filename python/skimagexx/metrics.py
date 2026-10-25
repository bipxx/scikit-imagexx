"""Native error metrics and explicit compatibility for remaining metrics."""
from . import _core
from ._metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio
from ._normalized_mutual_information import normalized_mutual_information
from ._compat import module_attribute

__all__ = ['mean_squared_error', 'normalized_root_mse', 'peak_signal_noise_ratio',
           'normalized_mutual_information']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('metrics', name)
