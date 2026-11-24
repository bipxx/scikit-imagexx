"""Pure-compute 400 x 400 float64 Shepp-Logan phantom sample."""

from . import _core


def shepp_logan_phantom():
    """Shepp Logan Phantom."""
    return _core.data_shepp_logan_phantom()


__all__ = ['shepp_logan_phantom']
