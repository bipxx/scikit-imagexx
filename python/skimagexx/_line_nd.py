"""Owned line-ND sampling with explicit NumPy conversion and FP-policy tails.

Equal native-endian f16/f32/f64/i8/u8/i64/u64 vectors and the captured pair
of float64 scalars enter the kernel. Other dtype/rank/flag domains are gaps. Tiny
operands are copied after both array conversions: this supports the assessed
read-only, unaligned and external-buffer cases without changing shared storage
policy, and preserves the source's snapshots across floating-point callbacks.
"""
import warnings
import numpy as np
from . import _core

# Actual scalar operation, sampled once; NumPy scalar remainder promotion differs
# between the retained runtimes. Later private promotion-state changes are outside
# the assessed boundary, without a mutable policy/cache framework.
_SCALAR_MODULO_NATIVE = (np.float32(0) % 1).dtype == np.dtype(np.float32)


def line_nd(start, stop, *, endpoint=False, integer=True):
    if type(endpoint) is not bool or type(integer) is not bool:
        raise NotImplementedError('draw.line_nd: nonbuiltin boolean flag protocols are outside the assessed partition')
    start = np.asarray(start)
    stop = np.asarray(stop)
    # Conversion/ownership providers only; geometry and rounding are C++ work.
    first = np.array(start, copy=True, order='C', subok=False)
    last = np.array(stop, copy=True, order='C', subok=False)
    native_result = _core.draw_line_nd(first, last, endpoint, integer, _core.get_num_threads(), _SCALAR_MODULO_NATIVE)
    if native_result is None:
        # Preserve the original pre-sampling NumPy error classes/messages for
        # empty, mismatched, string, boolean and nonfinite operands. A successful
        # reduction here does not make an unsupported geometry native.
        int(np.ceil(np.max(np.abs(stop - start))))
        raise NotImplementedError('draw.line_nd: dtype/rank/geometry is outside the assessed native partition')
    base, divide_underflow, multiply_underflow = native_result
    for occurred, operation in ((divide_underflow, 'divide'), (multiply_underflow, 'multiply')):
        if not occurred:
            continue
        mode, handler = np.geterr()['under'], np.geterrcall()
        message = f'underflow encountered in {operation}'
        if mode == 'warn':
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        elif mode == 'raise':
            raise FloatingPointError(message)
        elif mode == 'call':
            if handler is None:
                raise NameError(f'python callback specified for underflow (in  {operation}) but no function found.')
            handler('underflow', 4)
        elif mode != 'ignore':
            raise NotImplementedError('draw.line_nd: underflow print/log policy is outside the assessed partition')
    if integer and start.ndim == 0:
        # Source checks len(start) only after sampling, including scalar errors.
        len(start)
    if start.ndim == stop.ndim == 0 or integer:
        return tuple(base)
    return tuple(base.T)
