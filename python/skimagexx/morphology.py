"""Native flat morphology; unsupported calls use the explicit fallback policy."""
from collections.abc import Sequence
from functools import wraps
from numbers import Integral, Real
import warnings
import numpy as np
from . import _core
from ._compat import fallback, module_attribute
from ._footprint_decomposition import footprint_rectangle, diamond, disk, ball
from ._footprint_util import mirror_footprint, pad_footprint
from ._footprint_shapes import ellipse, octagon, octahedron, star, footprint_from_sequence
from ._components import label
from . import _component_consumers as _consumers
from ._reconstruction import reconstruction
from ._h_extrema import h_maxima, h_minima
from ._max_tree import (max_tree, max_tree_local_maxima, area_opening, area_closing,
                        diameter_opening, diameter_closing)

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = ['erosion','dilation','opening','closing','white_tophat','black_tophat',
           'binary_erosion','binary_dilation','binary_opening','binary_closing',
           'footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint','label',
           'remove_small_objects','remove_small_holes','max_tree','max_tree_local_maxima',
           'area_opening','area_closing','diameter_opening','diameter_closing','reconstruction','h_maxima','h_minima',
           'ellipse','octagon','octahedron','star','footprint_from_sequence']
_DTYPES = tuple(np.dtype(t) for t in ('bool','uint8','int8','uint16','int16','uint32',
                                    'int32','uint64','int64','float32','float64'))
_MODES = ('reflect','constant','nearest','mirror','wrap','max','min','ignore')
_SEQUENCE_ERROR = ('All elements of footprint sequence must be a 2-tuple where the first '
                   'element of the tuple is an ndarray and the second is an integer '
                   'indicating the number of iterations.')


def __getattr__(name):
    return module_attribute('morphology',name)


@wraps(_consumers.remove_small_objects)
def remove_small_objects(*args,**kwargs):
    source=args[0] if args else kwargs.get('ar')
    out=kwargs.get('out')
    if source is not None and (type(source) is not np.ndarray or
                              out is not None and type(out) is not np.ndarray):
        return fallback('morphology','remove_small_objects',*args,**kwargs)
    return _consumers.remove_small_objects(*args,**kwargs)


@wraps(_consumers.remove_small_holes)
def remove_small_holes(*args,**kwargs):
    source=args[0] if args else kwargs.get('ar')
    out=kwargs.get('out')
    if source is not None and (type(source) is not np.ndarray or
                              out is not None and type(out) is not np.ndarray):
        return fallback('morphology','remove_small_holes',*args,**kwargs)
    return _consumers.remove_small_holes(*args,**kwargs)


def _plans(footprint,ndim,binary,first_dilation=False):
    if footprint is None:
        coordinates = np.zeros((2*ndim+1,ndim),dtype=np.int64)
        for d in range(ndim):
            coordinates[2*d+1,d]=-1
            coordinates[2*d+2,d]=1
        coordinates=coordinates[np.lexsort(coordinates[:,::-1].T)]
        return [(coordinates,1,ndim==1)]
    if hasattr(footprint,'__array_interface__'):
        sequence=((footprint,1),)
    elif isinstance(footprint,Sequence):
        if not all(isinstance(item,Sequence) and len(item)==2 and
                   hasattr(item[0],'__array_interface__') and isinstance(item[1],Integral)
                   for item in footprint):
            raise ValueError(_SEQUENCE_ERROR)
        sequence=footprint
    else:
        raise TypeError('footprint must be either an ndarray or Sequence')
    if not len(sequence):
        raise IndexError('tuple index out of range')
    plans=[]
    for fp,repeats in sequence:
        fp=np.asarray(fp,dtype=bool)
        if fp.ndim!=ndim:
            if first_dilation and fp.ndim<ndim: raise IndexError('tuple index out of range')
            raise RuntimeError('structure and input must have same dimensionality' if binary else
                               'sequence argument must have length equal to input rank')
        if not all(fp.shape):
            raise ValueError('All-zero footprint is not supported.')
        coordinates=np.argwhere(fp)
        centers=np.array(fp.shape)//2
        if not binary:
            centers-=np.array(fp.shape)%2==0
        coordinates=np.ascontiguousarray(coordinates-centers,dtype=np.int64)
        separable=bool(np.all(fp)) and all(s%2 for s in fp.shape)
        plans.append((coordinates,int(repeats),separable))
    return plans


def _call(name,image,footprint,out,mode,cval=0.0):
    op=__all__.index(name)
    binary=op>=6
    args=(image,)
    options={'footprint':footprint,'out':out,'mode':mode}
    if not binary:
        options['cval']=cval
    eligible=(type(image) is np.ndarray and image.ndim>0 and image.dtype in _DTYPES
              and (out is None or (type(out) is np.ndarray and out.dtype in _DTYPES
                   and out.shape==image.shape and out.flags.writeable
                   and (binary or out.dtype==image.dtype))) and isinstance(cval,Real))
    if not eligible:
        return fallback('morphology',name,*args,**options)
    if binary:
        extra = {'binary_erosion':' Note the pixel shift by 1 for even-sized footprints (see docstring notes).',
                 'binary_dilation':' Note the lack of mirroring for non-symmetric footprints (see docstring notes).'}
        warnings.warn(f'`{name}` is deprecated since version 0.26 and will be removed in version 0.28. '
                      f'Use `skimage.morphology.{name[7:]}` instead.'+extra.get(name,''),FutureWarning,stacklevel=3)
    if mode not in (_MODES[5:] if binary else _MODES):
        raise ValueError(f'unsupported mode, got {mode!r}')
    plans=_plans(footprint,image.ndim,binary,name in ('dilation','closing','black_tophat','binary_dilation','binary_closing'))
    result,unsupported,invalid,overflow=_core.morphology(image,plans,op,_MODES.index(mode),float(cval),out)
    if unsupported:
        return fallback('morphology',name,*args,**options)
    if invalid:
        np.subtract(np.array([np.inf]),np.array([np.inf]))
    if overflow:
        limit=np.finfo(image.dtype).max
        np.subtract(np.array([limit],dtype=image.dtype),np.array([-limit],dtype=image.dtype))
    return result


def erosion(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('erosion',image,footprint,out,mode,cval)


def dilation(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('dilation',image,footprint,out,mode,cval)


def opening(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('opening',image,footprint,out,mode,cval)


def closing(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('closing',image,footprint,out,mode,cval)


def white_tophat(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('white_tophat',image,footprint,out,mode,cval)


def black_tophat(image,footprint=None,out=None,*,mode='reflect',cval=0.0):
    return _call('black_tophat',image,footprint,out,mode,cval)


def binary_erosion(image,footprint=None,out=None,*,mode='ignore'):
    return _call('binary_erosion',image,footprint,out,mode)


def binary_dilation(image,footprint=None,out=None,*,mode='ignore'):
    return _call('binary_dilation',image,footprint,out,mode)


def binary_opening(image,footprint=None,out=None,*,mode='ignore'):
    return _call('binary_opening',image,footprint,out,mode)


def binary_closing(image,footprint=None,out=None,*,mode='ignore'):
    return _call('binary_closing',image,footprint,out,mode)
