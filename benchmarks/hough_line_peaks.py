"""Fixed one-worker public Hough peak comparisons on prepared accumulators."""

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              call='public call with prepared Hough accumulator, includes output allocation and disposal')


def make_cases():
    import numpy as np

    recipe = runpy.run_path(str(ROOT / 'tests/hough_line_peaks_reference.py'))
    frozen = json.loads((ROOT / 'coverage/contracts/hough-line-peaks-reference-v1.json').read_bytes())
    for name, args, options, scope, _ in recipe['cases'](np, {'rows': frozen['parent_hough_rows']}):
        if scope == 'benchmark':
            yield name, args, options
