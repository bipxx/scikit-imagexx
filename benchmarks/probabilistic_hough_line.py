"""Frozen prepared-mask one-worker comparisons for the progressive Hough line search."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public prepared mask and theta including RNG, trig, native search and result allocation')
FIXED_CASES = ('default-cross-diagonal', 'gapped-three-angles', 'sparse-two-diagonals', 'dense-default-bands')


def make_cases():
    import numpy as np

    recipes = runpy.run_path(str(ROOT / 'tests/probabilistic_hough_line_reference.py'))
    endpoints = tuple(case for case in recipes['cases'](np) if case[-1] == 'benchmark')
    assert tuple(row[0] for row in endpoints) == FIXED_CASES
    for case, image, angles, seed, options, _ in endpoints:
        yield case, (image,), dict(options, theta=angles, rng=seed)
