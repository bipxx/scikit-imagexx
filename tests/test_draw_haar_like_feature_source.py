"""Source-only Haar drawing checks with no scientific imports."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = 'coverage/contracts/draw-haar-like-feature-v1.json'
CONTRACT_RAW = (ROOT/CONTRACT_PATH).read_bytes()
CONTRACT = json.loads(CONTRACT_RAW)
DESCRIPTOR_CONTRACT = json.loads((ROOT/'coverage/contracts/draw-haar-like-feature-v2.json').read_bytes())


def test_original_identity_order_and_scope():
    recipe = runpy.run_path(str(ROOT/'tests/haar_feature_drawing_cases.py'))
    frozen = json.loads((ROOT/CONTRACT['reference']['fixture']).read_bytes())
    assert frozen['api'] == 'feature.draw_haar_like_feature'
    assert len(recipe['cases']()) == 76
    assert recipe['FAST_CASES'] == CONTRACT['timing']['case_ids']
    assert set(frozen['profiles']) == {'minimum','modern'}
    for profile in frozen['profiles'].values():
        assert profile['original_calls'] == len(profile['cases']) == 76
        assert profile['original_sha256'] == CONTRACT['reference']['original_sha256']
        assert len({row['id'] for row in profile['cases']}) == 76


def test_explicit_rng_and_layout_domain():
    source = ast.parse((ROOT/'python/skimagexx/_draw_haar_like_feature.py').read_bytes())
    calls = [ast.unparse(node.func) for node in ast.walk(source) if isinstance(node,ast.Call)]
    assert calls.count('np.random.default_rng') == calls.count('generator.choice') == 1
    assert calls.count('_core._prepare_haar_drawing') == calls.count('_core._draw_haar_prepared') == 1
    assert calls.count('fallback') == 1
    frozen = json.loads((ROOT/CONTRACT['reference']['fixture']).read_bytes())
    minimum = {row['id']:row for row in frozen['profiles']['minimum']['cases']}
    modern = {row['id']:row for row in frozen['profiles']['modern']['cases']}
    assert 'error' in minimum['randomstate-provider'] and 'output' in modern['randomstate-provider']
    assert all(profile['choice-bad-rectangle']['rng_before'] != profile['choice-bad-rectangle']['rng_after']
               for profile in (minimum,modern))


def test_exact_family_source_pins_and_unbound_measurement():
    assert DESCRIPTOR_CONTRACT['predecessor'] == {
        'path': CONTRACT_PATH,
        'sha256': '0e8836dc03d98e3619a5564faeee1dba09d56b5146c1c8c3a93ddfcdcc6d0009',
    }
    assert hashlib.sha256(CONTRACT_RAW).hexdigest() == DESCRIPTOR_CONTRACT['predecessor']['sha256']
    assert DESCRIPTOR_CONTRACT['stock_descriptor_admission_sha256'] == (
        'cda287b925d2152b5fdca1d00a4f3d4f09f1744bc7ebfbca876ee4747bd71727'
    )
    successor = DESCRIPTOR_CONTRACT['source_canonical_sha256']
    assert set(successor) == {'src/bind_draw_haar_like_feature.cpp', 'tests/test_draw_haar_like_feature_public.py'}
    for name,expected in {**CONTRACT['source_canonical_sha256'], **successor}.items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == expected
    assert DESCRIPTOR_CONTRACT['execution_authority'] is None
    assert CONTRACT['timing']['execution_authority'] is None
