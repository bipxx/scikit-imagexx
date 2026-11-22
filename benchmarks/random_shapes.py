"""Four frozen end-to-end seeded random shape generation recipes."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='whole public call including seed, RNG, warnings, raster, and output allocation')


def make_cases(np, profile):
    freeze = runpy.run_path(str(ROOT / 'tests/random_shapes_reference.py'))
    rows = [row for row in freeze['reference'](profile) if row['scope'] == 'benchmark']
    assert tuple(row['id'] for row in rows) == freeze['FIXED']
    for row in rows:
        args, kwargs = freeze['make_call'](np, row)
        yield row['id'], args, kwargs
