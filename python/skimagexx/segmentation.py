"""Native component consumers with explicit compatibility fallback."""
from functools import wraps
import numpy as np
from ._compat import module_attribute,fallback
from . import _component_consumers as _consumers

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []
__all__ = ['clear_border','relabel_sequential']
from ._level_set import checkerboard_level_set, disk_level_set
__all__ += ['checkerboard_level_set','disk_level_set']


@wraps(_consumers.clear_border)
def clear_border(*args,**kwargs):
    source=args[0] if args else kwargs.get('labels')
    mask=args[3] if len(args)>3 else kwargs.get('mask')
    arrays=[source,kwargs.get('out'),mask]
    if source is not None and any(value is not None and type(value) is not np.ndarray for value in arrays):
        return fallback('segmentation','clear_border',*args,**kwargs)
    return _consumers.clear_border(*args,**kwargs)


@wraps(_consumers.relabel_sequential)
def relabel_sequential(*args,**kwargs):
    source=args[0] if args else kwargs.get('label_field')
    if source is not None and type(source) is not np.ndarray:
        return fallback('segmentation','relabel_sequential',*args,**kwargs)
    return _consumers.relabel_sequential(*args,**kwargs)


def __getattr__(name):
    return module_attribute('segmentation', name)
