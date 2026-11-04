"""Source-audited public mutation sequences, no original replay in tests."""
import base64
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib
import pytest
if __package__:
    from .regionprops_public_mutation_contract import API,PROPERTIES,UNASSESSED,cases,fixture,source_safe
    from .regionprops_public_mutation_observer import observe,scientific_profile,SOURCE_SHA,UTILS_SHA,SIGNATURE
else:
    from regionprops_public_mutation_contract import API,PROPERTIES,UNASSESSED,cases,fixture,source_safe
    from regionprops_public_mutation_observer import observe,scientific_profile,SOURCE_SHA,UTILS_SHA,SIGNATURE
ROOT = Path(__file__).parents[1]
MATRIX = ROOT/'coverage/contracts/regionprops-public-mutation.json'
CASES = cases()


@contextmanager
def without_region_activity():
    prefixes = ('skimage.measure','skimagexx.measure','skimagexx._regionprops_state','skimagexx._core')
    def forbidden(name):
        return any(name == p or name.startswith(p+'.') for p in prefixes)
    before = {n:m for n,m in sys.modules.items() if forbidden(n)}
    previous,real_import,real_dynamic = sys.getprofile(),builtins.__import__,importlib.import_module
    def checked(name,globals=None,locals=None,fromlist=(),level=0):
        assert not forbidden(name) and not (name in ('skimage','skimagexx') and 'measure' in (fromlist or ()))
        return real_import(name,globals,locals,fromlist,level)
    def dynamic(name,package=None):
        assert not forbidden(name)
        return real_dynamic(name,package)
    def profile(frame,event,value):
        module = frame.f_globals.get('__name__','') if event == 'call' else (getattr(value,'__module__','') or '') if event == 'c_call' else ''
        assert not forbidden(module)
    builtins.__import__,importlib.import_module = checked,dynamic
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        builtins.__import__,importlib.import_module = real_import,real_dynamic
    assert {n:m for n,m in sys.modules.items() if forbidden(n)} == before


def test_regionprops_public_mutation_prospective_source_safety():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['api'] == API and matrix['signature'] == SIGNATURE
    assert matrix['cases'] == CASES and len(CASES) == len({s['id'] for s in CASES}) == 12
    assert matrix['properties'] == list(PROPERTIES) and matrix['unassessed'] == list(UNASSESSED)
    assert matrix['original_source_sha256'] == dict(regionprops=SOURCE_SHA,regionprops_utils=UTILS_SHA)
    for spec in CASES:
        source_safe(spec)
    for name,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_regionprops_public_mutation_controller_intake_originals_refused(spec):
    np = scientific_profile()
    refused_calls,synthetic_calls = [],[]
    def refused(*args,**options):
        refused_calls.append(1)
        raise RuntimeError('region-public-mutation-original-refused')
    def synthetic(image,**options):
        synthetic_calls.append(1)
        ndim = image.ndim
        slices = tuple(slice(a,b) for a,b in spec['original_slice'])
        region = SimpleNamespace(label=1,slice=slices,bbox=tuple([p[0] for p in spec['original_slice']]+[p[1] for p in spec['original_slice']]),
            image=np.zeros((2,)*ndim,dtype=bool),coords=np.empty((0,ndim),dtype=np.intp),num_pixels=np.int64(0),area=np.float64(0))
        return [region,object()]
    with without_region_activity():
        fixture(spec,np)
        rejected = observe(refused,spec,np)
        exercised = observe(synthetic,spec,np)
    assert refused_calls == synthetic_calls == [1]
    assert rejected['constructor_error'] == ['builtins','RuntimeError','region-public-mutation-original-refused']
    assert not rejected['steps'] and all(rejected['checks'].values())
    assert len(exercised['steps']) == len(spec['sequence'])
    assert all(step['error'] is None and not step['warnings'] for step in exercised['steps'])
    assert all(exercised['checks'].values())


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_regionprops_public_mutation_saved_original_rows(profile):
    matrix = json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective only; original calls pending')
    rows = matrix['original_observations'][profile]
    assert len(rows) == 12 and [r['id'] for r in rows] == [s['id'] for s in CASES]
    for spec,row in zip(CASES,rows):
        assert row['returned_count'] == 2 and row['constructor_error'] is None and not row['constructor_warnings']
        assert row['region_type'] == ['skimage.measure._regionprops','RegionProperties']
        assert all(row['checks'].values()) and row['input_before'] == row['input_after']
        assert len(row['steps']) == len(spec['sequence'])
        assert all(step['error'] is None and not step['warnings'] for step in row['steps'])
        image_steps = [i for i,s in enumerate(row['steps']) if s['action'] == 'get' and s['argument'] == 'image']
        assert len(image_steps) in (2,3)
        pair = next(p for p in row['output_pair_aliases'] if p['first'] == image_steps[-2] and p['second'] == image_steps[-1])
        assert pair['same_object'] is spec['cache']
        assert all(not p['same_object'] and not p['shares_memory'] for p in row['output_input_aliases'])
        pending = [row]
        while pending:
            value = pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64' in value:
                    raw = zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest() == value['raw_sha256']
                    assert len(raw) == math.prod(value['shape'])*int(value['dtype'][2:])
                pending.extend(value.values())
            elif isinstance(value,list):
                pending.extend(value)
