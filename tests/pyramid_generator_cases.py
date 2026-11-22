"""Generators and image recipes shared by stock comparisons and timing."""
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SPATIAL = runpy.run_path(str(ROOT / 'tests/pyramid_spatial_cases.py'))
CONTRACT = ROOT / 'coverage/contracts/pyramid-generators-stock-v1.json'
PIN = 'aaa8402c2d2eee661a1c4fdbdf98b4065b16217e9cb0f73b9ee5c6ec204e3a3e'
INPUTS = {
    'max-zero-float-identity': ((5, 7), 'float32'),
    'max-zero-unsigned-preserve': ((5, 7), 'uint8'),
    'default-channel-only-remainder': ((1, 3, 27), 'float32'),
    'nearest-index-modulo': ((5, 7, 3), 'float64'),
    'near-one-stagnation': ((2, 3), 'float64'),
    'one-by-one-explicit-max': ((1, 1), 'float32'),
    'yield-mutation': ((5, 7), 'float32'),
    'pre-first-mutation': ((5, 7), 'uint8'),
    'generator-send-close': ((5, 7), 'float32'),
    'generator-throw': ((5, 7), 'float64'),
    'invalid-factor': ((5, 7), 'uint8'),
    'negative-sigma': ((5, 7), 'float32'),
    'invalid-order': ((5, 7), 'float32'),
    'noninteger-max': ((2, 3), 'float64'),
    'negative-nonsentinel-max': ((2, 3), 'float32'),
    'nan-warning-timing': ((4, 5), 'float64'),
    'half-fortran': ((5, 7), 'float16'),
}


def contract():
    data = CONTRACT.read_bytes().replace(b'\r\n', b'\n')
    assert sha256(data).hexdigest() == PIN
    return json.loads(data)


def fixed_recipes():
    return contract()['fixed']


def make_image(np, recipe):
    if 'input_recipe' in recipe:
        parent = next(item for item in SPATIAL['FIXED'] if item['id'] == recipe['input_recipe'])
        return SPATIAL['fixture'](parent['shape'], parent['dtype'], np)
    case = recipe['id']
    shape, dtype = INPUTS[case]
    if case == 'nan-warning-timing':
        return np.full(shape, np.nan, dtype=dtype)
    image = SPATIAL['fixture'](shape, dtype, np)
    return np.asfortranarray(image) if case == 'half-fortran' else image


def rows():
    data = contract()
    observations = {row['case']: row for row in data['observations']}
    for recipe in data['fixed']:
        yield recipe, observations[recipe['id']]
    for name in ('gaussian', 'laplacian'):
        for recipe in data['controls']:
            yield recipe, observations[name + '-' + recipe['id']]
