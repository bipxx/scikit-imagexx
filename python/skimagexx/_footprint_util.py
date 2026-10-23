"""View-preserving footprint reflection and native fixed-record padding."""
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from . import _core


def _sequence(footprint):
    if hasattr(footprint,'__array_interface__'):return False
    if isinstance(footprint,Sequence):
        if not all(isinstance(item,Sequence) and len(item)==2
                   and hasattr(item[0],'__array_interface__')
                   and isinstance(item[1],Integral) for item in footprint):
            raise ValueError('All elements of footprint sequence must be a 2-tuple where '
                             'the first element of the tuple is an ndarray and the second '
                             'is an integer indicating the number of iterations.')
    else:raise ValueError('footprint must be either an ndarray or Sequence')
    return True


def mirror_footprint(footprint):
    if _sequence(footprint):return tuple((mirror_footprint(item),count) for item,count in footprint)
    return _core.footprint_mirror(np.asarray(footprint))


def pad_footprint(footprint,*,pad_end=True):
    if _sequence(footprint):return tuple((pad_footprint(item,pad_end=pad_end),count) for item,count in footprint)
    image=np.asarray(footprint)
    # Evaluate truthiness once per even axis, in source order: arbitrary
    # objects may change their answer or raise on a later __bool__ call.
    padding=[((0,1) if pad_end else (1,0)) if size%2==0 else (0,0) for size in image.shape]
    return _core.footprint_pad(image,padding)


__all__=['mirror_footprint','pad_footprint']
