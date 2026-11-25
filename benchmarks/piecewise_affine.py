"""One-thread public piecewise-affine fit and application cases.

Run from the repository root with the candidate package installed::

    python -c "from benchmarks.run import measure; from benchmarks.piecewise_affine import public_call_pairs; from skimage import transform as s; import skimagexx as x; x.set_num_threads(1); x.set_fallback_enabled(False); [print(name, measure(a, b, 7, 10)) for name, a, b in public_call_pairs(s.PiecewiseAffineTransform, x.transform.PiecewiseAffineTransform)]"
"""

from functools import partial

import numpy as np


def public_call_pairs(original, candidate):
    rng = np.random.default_rng(207)
    for count in (8, 64):
        src = rng.uniform(-1, 1, size=(count, 2))
        dst = src + 0.12 * np.sin(src[:, ::-1] * 3)
        stock_fit = partial(original.from_estimate, src, dst)
        native_fit = partial(candidate.from_estimate, src, dst)
        before, after = stock_fit(), native_fit()
        for model, reference in ((after, before), (after.inverse, before.inverse)):
            assert len(model.affines) == len(reference.affines)
            for a, b in zip(model.affines, reference.affines):
                np.testing.assert_allclose(a.params, b.params, atol=1e-10, rtol=1e-10)
        yield f'fit-{count}', stock_fit, native_fit
        for rows in (32, 16384):
            coords = rng.uniform(-1.2, 1.2, size=(rows, 2))
            stock_apply, native_apply = partial(before, coords), partial(after, coords)
            np.testing.assert_allclose(native_apply(), stock_apply(), atol=1e-10, rtol=1e-10)
            yield f'apply-{count}-{rows}', stock_apply, native_apply
