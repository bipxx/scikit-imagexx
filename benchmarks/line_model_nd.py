"""Complete public calls proposed by the frozen LineModelND reference."""

from functools import partial
import warnings

import numpy as np


CASE_IDS = ("constructor-copy-2d", "fit-two-3d", "fit-svd-128x3", "legacy-estimate-two-3d",
            "residual-large-3d", "predict-large-3d", "predict-x-large-2d", "predict-y-large-2d")


def _points(count):
    i = np.arange(count)
    t = i / 4 - 16
    return np.column_stack((t, 2*t+(i%3-1)/8, -t+(i%7-3)/16))


def public_call_pairs(stock, candidate):
    origin, direction = np.array([1.,2.]), np.array([3.,4.])
    two = np.array([[1.,2.,3.],[4.,6.,8.]])
    svd = _points(128)
    large = _points(8192)
    t = np.arange(8192) / 8 - 128
    models3 = [cls([1.,2.,3.], [2.,-1.,0.5]) for cls in (stock, candidate)]
    models2 = [cls(origin, direction) for cls in (stock, candidate)]
    updates = [cls([1.,2.,3.], [2.,-1.,0.5]) for cls in (stock, candidate)]
    calls = (
        (partial(stock, origin, direction), partial(candidate, origin, direction)),
        (partial(stock.from_estimate, two), partial(candidate.from_estimate, two)),
        (partial(stock.from_estimate, svd), partial(candidate.from_estimate, svd)),
        tuple(partial(model.estimate, two) for model in updates),
        tuple(partial(model.residuals, large) for model in models3),
        tuple(partial(model.predict, t) for model in models3),
        tuple(partial(model.predict_x, t) for model in models2),
        tuple(partial(model.predict_y, t) for model in models2),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`estimate` is deprecated", category=FutureWarning)
        for name, (baseline, owned) in zip(CASE_IDS, calls, strict=True):
            yield name, baseline, owned
