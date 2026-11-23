"""Frozen prepared-image, one-worker circular Hough public comparisons."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public prepared-image call with normal radius handling and output allocation')


def make_cases():
    import numpy as np

    recipes = runpy.run_path(str(ROOT / 'tests/hough_circle_reference.py'))
    for case, args, options, scope in recipes['circle_cases'](np):
        if scope == 'benchmark':
            yield case, args, options
