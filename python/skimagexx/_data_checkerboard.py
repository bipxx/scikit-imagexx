"""Pure-compute 200 x 200 uint8 checkerboard sample."""

from . import _core


def checkerboard():
    """Checkerboard image."""
    return _core.data_checkerboard()


__all__ = ['checkerboard']
