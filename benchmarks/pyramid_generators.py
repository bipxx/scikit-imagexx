"""Fixed fully consumed public pyramid generators."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
METHOD = dict(workers=1, samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              includes='iterator construction, conversion, all Gaussian/resize stages, yields and exhaustion')


def fixed_cases(np):
    recipes = runpy.run_path(str(ROOT / 'tests/pyramid_generator_cases.py'))
    for row in recipes['fixed_recipes']():
        yield row['id'], row['api'], recipes['make_image'](np, row), row['options']


def public_call(function, image, options):
    return tuple(function(image, **options))
