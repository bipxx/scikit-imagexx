"""Join segmentations using native relabel, encode, unique and remap stages.

Scalar addition and promotion metadata use the live NumPy scalar authority.
Sorts are ordered; independent stages use the existing optional global budget.
ArrayMap is the project's owned implementation, not upstream class identity.
"""
import numpy as np
from . import _core
from ._compat import fallback
from ._component_consumers import ArrayMap, relabel_sequential


def join_segmentations(s1, s2, return_mapping: bool = False):
    """Return the join of two label fields, optionally with two shared-key maps."""
    if s1.shape != s2.shape:
        raise ValueError('Cannot join segmentations of different shape. '+f's1.shape: {s1.shape}, s2.shape: {s2.shape}')
    if type(s1) not in (np.ndarray,np.memmap) or type(s2) not in (np.ndarray,np.memmap):
        return fallback('segmentation','join_segmentations',s1,s2,return_mapping=return_mapping)
    if any(a.dtype.kind not in 'biuf' or a.dtype.itemsize not in (1,2,4,8) for a in (s1,s2)):
        return fallback('segmentation','join_segmentations',s1,s2,return_mapping=return_mapping)
    s1_relabeled,_,backward_map1=relabel_sequential(s1)
    s2_relabeled,_,backward_map2=relabel_sequential(s2)
    # ORIGINAL s2 maximum, not its rank maximum. Scalar overflow policy is live.
    factor=_core.consumer_max(s2)+np.uint8(1)
    if int(factor)<=0:
        return fallback('segmentation','join_segmentations',s1,s2,return_mapping=return_mapping)
    product_type=np.result_type(factor,s1_relabeled)
    sum_type=np.result_type(product_type,s2_relabeled.dtype)
    j_initial=_core.join_encode(s1_relabeled,s2_relabeled,int(factor),product_type,sum_type)
    j,_,map_j_to_j_initial=relabel_sequential(j_initial)
    if not return_mapping:return j
    # The third relabel already owns these sorted encoded labels.
    labels_j=map_j_to_j_initial.out_values
    division_type=np.result_type(labels_j,factor)
    if division_type.kind not in 'iu':
        raise TypeError('The dtype of an array to be remapped should be integer.')
    labels_s1_relabeled,labels_s2_relabeled=_core.join_divmod(labels_j,int(factor),division_type)
    keys=map_j_to_j_initial.in_values
    map1=ArrayMap(keys,backward_map1[labels_s1_relabeled])
    map2=ArrayMap(keys,backward_map2[labels_s2_relabeled])
    return j,map1,map2
