"""Frozen prepared-image serial ellipse Hough public comparisons."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public prepared-mask call including output allocation')
FIXED_CASES = ('documented-axis-ellipse', 'asymmetric-binary-step', 'two-small-rings', 'clipped-sheared-ring')


def make_cases():
    import numpy as np

    recipes = runpy.run_path(str(ROOT / 'tests/hough_ellipse_reference.py'))
    rows = tuple(row for row in recipes['frozen_rows']() if row['scope'] == 'benchmark')
    assert tuple(row['id'] for row in rows) == FIXED_CASES
    for row in rows:
        yield row['id'], (recipes['restore'](np, row),), row['options']
