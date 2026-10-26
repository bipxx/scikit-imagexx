"""Owned ordinary unique counts and entropy with explicit type providers."""
import math
import numpy as np
from . import _core


def shannon_entropy(image, base=2):
    """Calculate Shannon entropy, retaining source base-validation ordering.

    Standard real image counts and entropy arithmetic are owned C++ stages.
    Unsupported unique types use NumPy unique; scalar boxing and logarithmic
    base conversion retain the source NumPy/Python operations.
    """
    array = np.asanyarray(image)
    workers = _core.get_num_threads()
    counts = _core.shannon_counts(array, workers) if type(array) is np.ndarray else None
    if counts is None:
        _, counts = np.unique(image, return_counts=True)
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    S = np.float64(_core.shannon_entropy_counts(counts, workers))
    if base is not None:
        S /= math.log(base)
    return S
