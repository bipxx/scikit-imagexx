"""Check the fixed ellipsoid-stats source and fixture scope without science."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'coverage/contracts/ellipsoid-stats-v1.json').read_bytes())


def test_fixed_original_scope_and_timing_cases():
    fixture = json.loads((ROOT / CONTRACT['reference']['fixture']).read_bytes())
    assert fixture['api'] == 'draw.ellipsoid_stats'
    assert set(fixture['profiles']) == {'minimum','modern'}
    for profile in fixture['profiles'].values():
        rows = profile['rows']
        assert profile['skimage'] == '0.26.0'
        assert profile['original_source_sha256'] == CONTRACT['reference']['original_source_sha256']
        assert len(rows) == len({row['id'] for row in rows}) == profile['original_calls'] == 73
        assert [row['id'] for row in rows if row['timed']] == CONTRACT['timing']['case_ids']
        assert sum(row['id'].startswith('triaxial-permutation') for row in rows) == 6
        assert all(len(row['args']) == 3 and ('result' in row) != ('error' in row) for row in rows)


def test_dtype_changes_and_extreme_warning_boundaries_are_explicit():
    profiles = json.loads((ROOT / CONTRACT['reference']['fixture']).read_bytes())['profiles']
    def indexed(profile):
        return {row['id']:row for row in profiles[profile]['rows']}
    minimum, modern = indexed('minimum'), indexed('modern')
    assert [item['kind'] for item in minimum['np.float16-standard']['result']] == ['numpy.float64']*2
    assert [item['kind'] for item in modern['np.float16-standard']['result']] == ['numpy.float16','numpy.float32']
    assert modern['np-float16-overflow-square']['warnings']
    assert minimum['np.longdouble-standard']['error']['kind']=='TypeError'
    assert modern['extreme-underflow']['error']['kind']=='ZeroDivisionError'


def test_owned_integral_and_explicit_provider_boundaries():
    for path, sha in CONTRACT['source_canonical_sha256'].items():
        raw = (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(raw).hexdigest()==sha, path
    frontend = ast.parse((ROOT/'python/skimagexx/_ellipsoid_stats.py').read_bytes())
    public = next(node for node in frontend.body if isinstance(node, ast.FunctionDef) and node.name=='ellipsoid_stats')
    assert [arg.arg for arg in public.args.args]==['a','b','c']
    calls = [ast.unparse(node.func) for node in ast.walk(public) if isinstance(node, ast.Call)]
    assert calls.count('_core._ellipsoid_stats_builtin')==1 and calls.count('fallback')==1
    assert CONTRACT['timing']['actual_authority'] is None
