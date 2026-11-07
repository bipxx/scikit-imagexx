"""Pure source/storage guards. No NumPy/runtime/helper/target requests."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest
from flood_fill_contract import bounds, recipes

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'coverage/contracts/flood-fill-v1.json').read_text())
SNAPSHOT_BYTES = (ROOT / CONTRACT['source_snapshot']).read_bytes()
SOURCES = json.loads(SNAPSHOT_BYTES)


def function(name, filename='morphology/_flood_fill.py'):
    return next(n for n in ast.parse(SOURCES[filename]['text']).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize('api,positionals,keywords', [
    ('flood', ['image', 'seed_point'], ['footprint', 'connectivity', 'tolerance']),
    ('flood_fill', ['image', 'seed_point', 'new_value'],
     ['footprint', 'connectivity', 'tolerance', 'in_place']),
])
def test_actual_signature(api, positionals, keywords):
    f = function(api)
    assert [x.arg for x in f.args.args] == positionals
    assert [x.arg for x in f.args.kwonlyargs] == keywords
    assert f.args.vararg is f.args.kwarg is None
    assert not f.args.defaults
    assert [ast.literal_eval(x) for x in f.args.kw_defaults] == [None] * 3 + ([False] if api == 'flood_fill' else [])


def test_source_bytes_and_snapshot_pin():
    assert hashlib.sha256(SNAPSHOT_BYTES).hexdigest() == CONTRACT['source_snapshot_sha256']
    for row in SOURCES.values():
        if 'text' in row:
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256']
    assert CONTRACT['source_intakes']['extra-source-intake-v1.json']['src/skimage/morphology/_flood_fill.py']['matches_installed_minimum']


def test_alias_chain_has_two_operations_four_public_names():
    assert CONTRACT['canonical_operations'] == 2 and CONTRACT['public_names'] == 4
    assert 'from ._flood_fill import flood, flood_fill' in SOURCES['morphology/__init__.py']['text']
    assert 'from ._watershed import flood, flood_fill' in SOURCES['segmentation/__init__.pyi']['text']
    for p in ('minimum', 'modern'):
        assert SOURCES[p + '/skimage/segmentation/_watershed.py']['segments'] == ['from ..morphology import flood, flood_fill']


def test_source_order_and_fill_original_object_boundary():
    text = ast.unparse(function('flood'))
    tokens = ['image = np.asarray(image)', 'image.flags.f_contiguous', 'image.flags.c_contiguous',
              'if 0 in image.shape', 'iter(seed_point)', 'seed_value = image[seed_point]',
              'seed_point = tuple(np.asarray(seed_point) % image.shape)', 'footprint = _resolve_neighborhood',
              'working_image = np.pad', 'neighbor_offsets = _offsets_to_raveled_neighbors',
              'flags = np.zeros', '_set_border_values', 'tolerance = abs(tolerance)',
              'low_tol = max', 'high_tol = min', '_flood_fill_tolerance(',
              '_flood_fill_equal(', 'if working_image.dtype == np.float16:', 'return crop(flags']
    indexes = [text.index(t) for t in tokens]
    assert indexes == sorted(indexes)
    fill = ast.unparse(function('flood_fill'))
    assert fill.index('mask = flood(') < fill.index('if not in_place:') < fill.index('image = image.copy()') < fill.index('image[mask] = new_value') < fill.index('return image')
    assert 'np.asarray' not in fill


def test_actual_neighbor_and_fifo_source():
    helper = SOURCES['morphology/_util.py']['text']
    assert 'kind="stable"' in helper and 'sorted_raveled_offsets[1:]' in helper
    assert 'enforce_adjacency' in helper and 'footprint = np.asarray(footprint, dtype=bool)' in helper
    kernel = SOURCES['morphology/_flood_fill_cy.pyx']['text']
    assert '#cython: boundscheck=False' in kernel
    assert kernel.count('flags[start_index] = FILL') == 2
    assert kernel.count('flags[neighbor] = FILL\n                            queue_push') == 2
    assert 'image[neighbor] == seed_value' in kernel
    assert 'low_tol <= image[neighbor] <= high_tol' in kernel
    queue = SOURCES['morphology/_queue_with_history.pxi']['text']
    assert 'self._index_consumed += 1' in queue and 'self._buffer_size *= 2' in queue


def test_recipes_and_derived_bounds():
    rows = recipes()
    assert rows == CONTRACT['recipes']
    assert len({r['id'] for r in rows}) == len(rows) == 96
    assert {a: sum(r['api'] == a for r in rows) for a in ('flood', 'flood_fill')} == CONTRACT['recipe_counts']
    assert {k: max(bounds(r)[k] for r in rows) for k in CONTRACT['caps']} == CONTRACT['caps']
    assert CONTRACT['caps'] == dict(logical=24, physical_input=48, padded=120,
                                    queue_entries=24, neighbor_checks=648, storage_bytes=4400)


@pytest.mark.parametrize('shape,radius', [((5,), 1), ((3, 4), 1), ((3, 4), 2), ((2, 3, 4), 1)])
def test_padding_address_proof_without_array_or_target(shape, radius):
    padded = tuple(n + radius * 2 for n in shape)
    for point in itertools.product(*(range(radius, radius + n) for n in shape)):
        for offset in itertools.product(range(-radius, radius + 1), repeat=len(shape)):
            neighbor = tuple(i + d for i, d in zip(point, offset))
            assert all(0 <= i < n for i, n in zip(neighbor, padded))


def test_authority_and_full_scope_remain_open():
    assert CONTRACT['state'] == 'UNBOUND_SOURCE_ONLY'
    assert all(x == 0 if k.endswith('budget') or k == 'runtime_tests' else x is None
               for k, x in CONTRACT['authority'].items())
    assert 'QUARANTINED' in CONTRACT['safety']['scalar_0d']
    assert len(CONTRACT['next_gates']) == 6
