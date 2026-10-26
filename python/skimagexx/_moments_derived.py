"""Derived moments: source scalar providers plus owned arithmetic stages.

Assessed 4x4 real defaults use the bounded owned batch stage. Other inputs
keep NumPy powers, first division, promotion and output assignment. The final division is native for assessed scalar dtypes and
ignore/warn/raise/call policies; print/log and other dtypes are primitive NumPy
providers, never upstream whole-function forwarding. No speed claim is made.
"""
import itertools
import warnings
import numpy as np
from . import _core

_FLAGS=((1,"divide","divide by zero"),(2,"over","overflow"),
        (4,"under","underflow"),(8,"invalid","invalid value"))


def _divide(first,denominator):
    dtype=np.result_type(first,denominator)
    policy,handler=np.geterr(),np.geterrcall()
    if (dtype.char not in "efdFD" or np.__version__ not in ("1.24.4","2.5.4")
            or any(p not in ("ignore","warn","raise","call")for p in policy.values())):
        return first/denominator
    value,flags=_core.moments_normalize(first,denominator,dtype.char)
    # Same combined status and error order as the existing colocalization
    # dispatcher. This operation's policy snapshot survives error callbacks.
    for bit,key,label in _FLAGS:
        if not flags&bit:continue
        action=policy[key]
        if action=="warn":warnings.warn(label+" encountered in scalar divide",RuntimeWarning,stacklevel=3)
        elif action=="raise":raise FloatingPointError(label+" encountered in scalar divide")
        elif action=="call":
            if handler is None:raise NameError(f"python callback specified for {label} (in  scalar divide) but no function found.")
            handler(label,flags)
    return dtype.type(value)


def moments_normalized(mu,order=3,spacing=None):
    storage_checked=False
    if (type(mu) is np.ndarray and type(order) is int and order==3 and spacing is None
            and np.__version__ in ("1.24.4","2.5.4") and mu.shape==(4,4)
            and mu.dtype.char in ("f","d") and mu.dtype.isnative
            and mu.dtype.metadata is None and mu.flags.c_contiguous and mu.flags.aligned):
        batch=_core.moments_normalize(mu,np.__version__=="2.5.4" and mu.dtype.itemsize==4)
        storage_checked=True
        if batch is not None:return batch
        # None is a metadata probe, not an accepted numerical completion.
    if not storage_checked and isinstance(mu,np.ndarray):_core.fft_check(mu)
    if np.any(np.array(mu.shape)<=order):
        raise ValueError("Shape of image moments must be >= `order`")
    if spacing is None:spacing=np.ones(mu.ndim)
    nu=np.zeros_like(mu)
    mu0=mu.ravel()[0]
    scale=min(spacing)
    for powers in itertools.product(range(order+1),repeat=mu.ndim):
        if sum(powers)<2:
            nu[powers]=np.nan
        else:
            # Keep first divide BEFORE mass power: warnings/raises are
            # observable, and NP1/NP2 scalar promotion differs here.
            first=mu[powers]/scale**sum(powers)
            denominator=mu0**(sum(powers)/nu.ndim+1)
            nu[powers]=_divide(first,denominator)
    return nu


def moments_hu(nu):
    if isinstance(nu,np.ndarray):_core.fft_check(nu)
    dtype=np.float32 if nu.dtype=='float32' else np.float64
    values=nu.astype(dtype,copy=False)
    result=_core.moments_hu(values)
    # The original returns np.asarray of its typed memoryview, not the owning
    # ndarray directly. Preserve the public base/ownership contract.
    return np.asarray(memoryview(result))
