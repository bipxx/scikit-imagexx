"""Frozen prepared-volume one-worker comparisons for circular peaks."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public prepared accumulator volume including ranking, pruning and output allocation')


def make_cases(np, prepared, profile):
    recipes = runpy.run_path(str(ROOT / 'tests/hough_circle_peaks_reference.py'))
    rows = [(args, row) for args, row in recipes['cases'](np, prepared, profile) if row['scope'] == 'benchmark']
    assert tuple(row['id'] for _, row in rows) == recipes['FIXED']
    for args, row in rows:
        yield row['id'], args, row['options']
