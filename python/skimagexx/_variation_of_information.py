"""Owned finite VI entropy stages; public shared registration remains pending.

Existing owned contingency aggregation is reused. Its NumPy membership/weight
normalization and SciPy CSR representation are explicit computation providers;
they are not counted as native VI arithmetic. All entropy/log/reduction work is
C++. Supplied plain CSR buffers are read without sorting/mutating their caches.
"""
import numpy as np
import scipy.sparse as sparse
from . import _core
from ._contingency_table import contingency_table


def _aligned(value,dtype):
    _core.label_pair_check_storage(value)
    out=np.ascontiguousarray(value,dtype=dtype)
    return out if out.flags.aligned else out.copy(order='C')


def variation_of_information(image0=None,image1=None,*,table=None,ignore_labels=()):
    # Original shape checking is unconditional, even for a supplied table.
    if image0.shape!=image1.shape:
        raise ValueError('Input images must have the same dimensions.')
    if table is None:
        if type(image0)is not np.ndarray or type(image1)is not np.ndarray:
            raise NotImplementedError('VI arbitrary image protocols remain unassessed')
        table=contingency_table(image0,image1,ignore_labels=ignore_labels,normalize=True)
    if type(table)not in(sparse.csr_array,sparse.csr_matrix):
        raise NotImplementedError('VI requires assessed plain CSR table representation')
    if table.dtype.kind!='f'or table.dtype.itemsize!=8 or not table.dtype.isnative:
        raise NotImplementedError('VI non-native or non-f64 supplied-table arithmetic remains unassessed')
    # Normalize physical transport once, after source prefix callbacks. No
    # sparse sums, products, logs, sorting or whole original metric forwarding.
    data=_aligned(table.data,np.float64)
    indices=_aligned(table.indices,np.int64)
    indptr=_aligned(table.indptr,np.int64)
    rows,columns=table.shape
    result=_core.variation_of_information_csr(data,indices,indptr,rows,columns,_core.get_num_threads())
    if result is None:
        raise NotImplementedError('VI typed arithmetic/storage/FENV domain refused; no algorithm fallback')
    return result
