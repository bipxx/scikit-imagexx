"""Source-ordered contingency prefix, owned weighted counts, structural CSR.

NumPy conversion/membership/normalization and SciPy class/index/storage packing
are declared providers. Unsupported typed suffixes use the already prepared
coordinate/weight objects; the original metric and its prefix are never replayed.
Backing-tail payload words outside the logical CSR arrays remain unassessed.
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse._sputils import get_index_dtype
from . import _core


def _aligned_i64(value):
    result = np.ascontiguousarray(value, dtype=np.int64)
    return result if result.flags.aligned else result.copy(order='C')


def _structural_csr(native_result, truth, test):
    shape, indptr, indices, data = native_result
    # COO's sparse-array index selection does not inspect values to downcast
    # int64 carriers. Passing SDK int64 arrays directly would change int16
    # source carriers to int64 public indices, so select before construction.
    index_dtype = get_index_dtype((truth, test), maxval=max(*shape, data.size), check_contents=False)
    indices = indices.astype(index_dtype, copy=False)
    indptr = indptr.astype(index_dtype, copy=False)
    cont = sparse.csr_array((data, indices, indptr), shape=shape, copy=False)
    # Original coordinate construction sum_duplicates establishes these flags;
    # the structural three-array path needs the same proven invariants.
    cont.has_sorted_indices = True
    cont.has_canonical_format = True
    return cont


def contingency_table(im_true, im_test, *, ignore_labels=None, normalize=False, sparse_type='matrix'):
    """Bounded native preweighted aggregation with source-ordered providers.

    Native labels are stable ordinary integer arrays with at most2**20 entries
    and values in[0,2**20-1]. The normalization callback executes before native
    coordinate intake. CSR errors and unsupported suffixes remain named SciPy
    providers, without forwarding the whole original contingency algorithm.
    """
    if ignore_labels is None:
        ignore_labels = []
    # Storage checks precede possible reshape copies for ordinary ndarrays.
    if type(im_test) is np.ndarray:
        _core.label_pair_check_storage(im_test)
    im_test_r = im_test.reshape(-1)
    if type(im_true) is np.ndarray:
        _core.label_pair_check_storage(im_true)
    im_true_r = im_true.reshape(-1)
    data = np.isin(im_true_r, ignore_labels, invert=True).astype(float)
    if normalize:
        data /= np.count_nonzero(data)
    # These are the CURRENT coordinates after callbacks, not a pre-callback
    # cast, and the already-normalized data is not recomputed on any refusal.
    native_result = None
    ordinary = (type(im_true_r) is np.ndarray and type(im_test_r) is np.ndarray
                and im_true_r.dtype.kind in 'iu' and im_test_r.dtype.kind in 'iu'
                and im_true_r.dtype.metadata is None and im_test_r.dtype.metadata is None
                and im_true_r.size == im_test_r.size == data.size
                and 0 < data.size <= 2**20)
    if ordinary:
        # A callback can expose new geometry/storage; this is a new trust
        # boundary before the native suffix releases the GIL.
        _core.label_pair_check_storage(im_true_r)
        _core.label_pair_check_storage(im_test_r)
        truth, test = _aligned_i64(im_true_r), _aligned_i64(im_test_r)
        weights = np.ascontiguousarray(data, dtype=np.float64)
        if not weights.flags.aligned:
            weights = weights.copy(order='C')
        native_result = _core.label_pair_counts_preweighted(truth, test, weights, _core.get_num_threads())
    if native_result is None:
        # Literal suffix only: empty/negative/error and unassessed numerical
        # domains do not regain native credit by using this coordinate factory.
        cont = sparse.csr_array((data, (im_true_r, im_test_r)))
    else:
        cont = _structural_csr(native_result, im_true_r, im_test_r)
    if sparse_type == 'matrix':
        cont = sparse.csr_matrix(cont)
    elif sparse_type != 'array':
        raise ValueError(f"`sparse_type` must be 'array' or 'matrix', got {sparse_type}")
    return cont
