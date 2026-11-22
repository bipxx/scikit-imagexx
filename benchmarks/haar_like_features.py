"""Fixed allocation-inclusive public-call fixtures for the two Haar APIs."""

import runpy
from pathlib import Path

_CASES = Path(__file__).resolve().parents[1] / "tests/haar_feature_cases.py"


def fixed_cases():
    recipe = runpy.run_path(str(_CASES))
    by_id = {spec["id"]: spec for spec in recipe["cases"]()}
    return [by_id[name] for name in recipe["TIMED"]], recipe


def public_call(function, args, kwargs):
    return function(*args, **kwargs)
