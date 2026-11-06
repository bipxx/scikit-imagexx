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


def _rand_divide_result(native_result):
    """Deliver this stage before starting the next, using current NumPy policy."""
    import numpy as np
    import warnings
    if native_result is None:
        raise NotImplementedError('adapted Rand scalar stage requires nearest/gradual finite typed domain')
    value,status,bits=native_result
    policy,handler=np.geterr(),np.geterrcall()
    if any(v not in ('ignore','warn','raise','call') for v in policy.values()):
        raise NotImplementedError('adapted Rand print/log error-policy providers remain unassessed')
    flags=((status&1)<<3)|((status&2)>>1)|((status&4)>>1)|((status&8)>>1)
    for bit,key,label in ((1,'divide','divide by zero'),(2,'over','overflow'),(4,'under','underflow'),(8,'invalid','invalid value')):
        if not flags&bit:continue
        action=policy[key]
        if action=='warn':warnings.warn(label+' encountered in scalar divide',RuntimeWarning,stacklevel=2)
        elif action=='raise':raise FloatingPointError(label+' encountered in scalar divide')
        elif action=='call':
            if handler is None:raise NameError(f'python callback specified for {label} (in  scalar divide) but no function found.')
            handler(label,flags)
    return value,bits


def _are_from_i64(truth,test,ignored,alpha=0.5):
    """Private stable-input ARE path, without public conversion/CSR claims.

    Supplied tables, arbitrary protocols and normalization callbacks are not
    routed through this typed table=None helper. No original metric is called.
    """
    import math
    native_result=_core.label_pair_rand_terms(truth,test,ignored,_core.get_num_threads())
    if native_result is None:
        raise NotImplementedError('adapted Rand requires the bounded stable int64 input domain')
    # Source performs contingency construction before alpha validation.
    if type(alpha) not in (int,float):
        raise NotImplementedError('adapted Rand alpha protocol/nonfinite domain remains unassessed')
    if alpha<0.0 or alpha>1.0:raise ValueError('alpha must be between 0 and 1')
    if not math.isfinite(alpha):
        raise NotImplementedError('adapted Rand alpha protocol/nonfinite domain remains unassessed')
    joint,row,column=native_result
    precision,_=_rand_divide_result(_core.label_pair_rand_divide(joint,row))
    recall,_=_rand_divide_result(_core.label_pair_rand_divide(joint,column))
    _,bits=_rand_divide_result(_core.label_pair_rand_fscore(joint,row,column,alpha))
    result=_core.label_pair_rand_error(bits)
    if result is None:
        raise NotImplementedError('adapted Rand subtraction requires the unchanged nearest/gradual environment')
    are,status,_=result
    assert status==0
    return are,precision,recall


def adapted_rand_error(image_true=None,image_test=None,*,table=None,ignore_labels=(0,),alpha=0.5):
    """Owned finite table=None ARE partition; shared public routing is pending.

    Exact plain integer ndarrays with stable storage, at most2**20 samples and
    labels in[0,2**20-1] are admitted. NumPy reshape/integer conversion and scalar
    policy delivery are explicit structural providers. Supplied tables, array
    subclasses, arbitrary ignore/alpha protocols and print/log/FENV-changing
    callbacks remain unassessed; no whole original algorithm is forwarded.
    """
    import numpy as np
    if image_test is not None and image_true is not None:
        if type(image_true)is not np.ndarray or type(image_test)is not np.ndarray:
            raise NotImplementedError('adapted Rand requires assessed plain ndarray carriers')
        if image_true.shape!=image_test.shape:
            raise ValueError('Input images must have the same dimensions.')
    if table is not None:
        raise NotImplementedError('adapted Rand supplied-table mutation/CSR providers remain unassessed')
    if type(image_true)is not np.ndarray or type(image_test)is not np.ndarray:
        raise NotImplementedError('adapted Rand requires assessed plain ndarray carriers')
    if not 0<image_true.size<=2**20:
        raise NotImplementedError('adapted Rand empty/large-array boundary remains unassessed')
    _core.label_pair_check_storage(image_test)
    test=image_test.reshape(-1)
    _core.label_pair_check_storage(image_true)
    truth=image_true.reshape(-1)
    if any(x.dtype.kind not in 'iu'or x.itemsize not in (1,2,4,8)or x.dtype.metadata is not None for x in (truth,test)):
        raise NotImplementedError('adapted Rand noninteger/metadata dtype providers remain unassessed')
    if ignore_labels is None:ignore_labels=()
    if type(ignore_labels)not in (tuple,list)or len(ignore_labels)>64 or any(type(x)is not int or not -2**63<=x<2**63 for x in ignore_labels):
        raise NotImplementedError('adapted Rand ignore-label protocol remains unassessed')
    # Conversion preserves each admitted integer word's value; larger/negative
    # coordinates are refused by the native intake, without allocating by label.
    truth=np.ascontiguousarray(truth,dtype=np.int64)
    test=np.ascontiguousarray(test,dtype=np.int64)
    ignored=np.array(ignore_labels,dtype=np.int64)
    return _are_from_i64(truth,test,ignored,alpha)
