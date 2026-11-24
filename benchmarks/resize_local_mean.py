"""Original four workloads with explicit coverage in the bounded local mean route."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public prepared image including range conversion and output allocation')


def make_cases(np, profile):
    frozen = runpy.run_path(str(ROOT / 'tests/resize_local_mean_reference.py'))
    rows = frozen['reference']()[profile]['observations']
    for name in frozen['FIXED']:
        row = rows[name]
        case = row['case']
        yield name, (frozen['image_for'](np, case), tuple(case['output'])), {
            'preserve_range': case['preserve_range']}, row['native']
