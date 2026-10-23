"""Connected-component metadata around the version1 native label pipeline.

The initial native scope covers boolean and real 8/16/32/64-bit array types.
Metadata follows scikit-image0.26.0 (BSD-3-Clause; see
third_party/scikit-image-LICENSE.txt).
"""
import numpy as np
from . import _core


def label(label_image, background=None, return_num=False, connectivity=None):
    """Label connected equal-valued regions in logical C traversal order."""
    boolean=label_image.dtype==bool
    swaps=[]
    if boolean:
        # Preserve upstream comparison/ambiguity precedence before connectivity.
        background=1 if background==1 else 0
        prepared=None
    else:
        ones=[axis for axis,n in enumerate(label_image.shape) if n==1][::-1]
        bigs=[axis for axis,n in enumerate(label_image.shape) if n>1]
        for one,big in zip(ones,bigs):
            if one<big:break
            swaps.append((one,big))
        arranged=label_image
        for one,big in swaps:arranged=label_image.swapaxes(one,big)
        prepared,invalid=_core.components_prepare_index(arranged)
        if invalid:
            # A tiny scalar probe reproduces the active NumPy error policy;
            # native preparation already converted every image pixel.
            np.asarray([np.nan]).astype(np.intp)
        if label_image.ndim<1 or label_image.ndim>3:
            raise NotImplementedError('Only for images of dimension 1-3 are supported, '
                                      f'got a {label_image.ndim}D one')
        background=0 if background is None else _core.components_background(background)
    ndim=label_image.ndim
    if connectivity is None:connectivity=ndim
    if not 1<=connectivity<=ndim:
        raise ValueError(f'Connectivity for {ndim}D image should be in [1, ..., {ndim}]. Got {connectivity}.')
    effective=(sum(bool(step<=connectivity) for step in range(1,ndim+1)) if boolean
               else _core.components_background(connectivity))
    if boolean:prepared=_core.components_prepare_bool(label_image)
    dtype=np.dtype(np.intp if not boolean or label_image.size>=2**31-2 else np.int32)
    if not boolean and label_image.dtype==np.dtype(np.intp):dtype=label_image.dtype
    output,count=_core.components_label(prepared,background,effective,boolean,dtype)
    if swaps:
        base=output
        for one,big in swaps[::-1]:output=base.swapaxes(one,big)
    return (output,count) if return_num else output
