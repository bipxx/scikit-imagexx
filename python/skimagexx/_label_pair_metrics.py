"""Private typed count stage; no public metric or original forwarding.

The caller has already performed source-ordered conversion to stable native
contiguous int64 buffers. SciPy class/index packing, source error delivery,
normalization callback phases and Rand/VI reductions/logs remain explicit next
stages. This internal result is not a public contingency table.
"""
from . import _core


def _counts_from_i64(truth,test,ignored,normalize):
    native_result=_core.label_pair_counts(truth,test,ignored,normalize,_core.get_num_threads())
    return native_result
