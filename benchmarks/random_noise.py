"""Fixed seeded public random_noise workloads, including allocation."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='prepared image, full public conversion, provider draws, native transform, clip and allocation')


def make_cases(np, profile):
    refs = runpy.run_path(str(ROOT / 'tests/random_noise_reference.py'))
    rows = {row['id']: row for row in refs['reference'](profile)}
    for case in refs['FIXED']:
        row = rows[case]
        args, kwargs = refs['make_call'](np, row)
        yield case, args, kwargs, row['native']
