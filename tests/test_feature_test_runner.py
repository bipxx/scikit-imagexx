"""Bounded runner selection controls; no image packages or kernels are imported."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_feature_tests import FeatureSelection, select_entries, smoke_case


DTYPES = {name: type(name, (), {}) for name in ('uint8', 'float32', 'float64', 'uint16')}
SUPPORTED = ('uint8', 'float32', 'float64')


def test_api_patterns_deduplicate_in_inventory_order():
    entries = [{'api': name} for name in ('filters.sobel', 'morphology.label', 'filters.gaussian')]
    selected = select_entries(entries, ['filters.gaussian', 'filters.*', 'filters.sobel'])
    assert selected == [entries[0], entries[2]]
    assert selected[0] is entries[0]
    assert selected[1] is entries[2]


@pytest.mark.parametrize('patterns', [['missing.*'], ['filters.*', 'missing.*']])
def test_unmatched_pattern_is_an_error_even_after_a_valid_pattern(patterns):
    with pytest.raises(ValueError, match="No inventoried API matches 'missing"):
        select_entries([{'api': 'filters.sobel'}], patterns)


@pytest.mark.parametrize('dtype', SUPPORTED)
@pytest.mark.parametrize('shape', [(3, 19), (7, 19)])
def test_smoke_keeps_each_supported_dtype_and_existing_safe_geometry(dtype, shape):
    parameters = dict(configuration=1, dtype=DTYPES[dtype], shape=shape)
    assert smoke_case(parameters, SUPPORTED)
    assert parameters['shape'] == shape


@pytest.mark.parametrize('shape', [(1, 1), (4, 5), (519, 521)])
def test_smoke_rejects_tiny_alternative_and_large_cases(shape):
    assert not smoke_case(dict(configuration=1, dtype=DTYPES['uint8'], shape=shape), SUPPORTED)


def test_smoke_uses_the_individual_api_dtype_contract():
    parameters = dict(configuration=1, dtype=DTYPES['float32'], shape=(3, 19))
    assert not smoke_case(parameters, ('uint8',))
    parameters['dtype'] = DTYPES['uint16']
    assert not smoke_case(parameters, SUPPORTED)


@pytest.mark.parametrize('configuration', [4, None])
def test_smoke_never_selects_worker_four_or_an_undeclared_worker(configuration):
    assert not smoke_case(dict(configuration=configuration, dtype=DTYPES['uint8'], shape=(7, 19)), SUPPORTED)
    assert not smoke_case(dict(configuration=configuration), SUPPORTED)


def test_signature_selection_needs_no_image_dtype():
    assert smoke_case(dict(configuration=1), ())


def item(nodeid, originalname, parameters, filename='test_api.py'):
    return SimpleNamespace(nodeid=nodeid, originalname=originalname,
                           callspec=SimpleNamespace(params=parameters), path=Path(filename))


def test_collection_selects_exact_namespaces_supported_cases_and_worker_one():
    entry = dict(api='morphology.label', native_default_case_dtypes=list(SUPPORTED))
    plugin = FeatureSelection([entry], 'smoke')
    native = SimpleNamespace(__name__='skimagexx.morphology')
    other = SimpleNamespace(__name__='skimagexx.measure')
    base = dict(native=native, name='label', configuration=1)
    wanted = [
        item('registry-1', 'test_registry_matches_per_function_tests', {'configuration': 1}),
        item('signature-1', 'test_signatures', base),
        *[item('value-' + dtype, 'test_every_native_function',
               dict(base, dtype=DTYPES[dtype], shape=(3, 19))) for dtype in SUPPORTED],
        item('surface-selected', 'test_api_surface', {'entry': entry}, 'test_feature_api_surface.py'),
    ]
    rejected = [
        item('registry-4', 'test_registry_matches_per_function_tests', {'configuration': 4}),
        item('signature-4', 'test_signatures', dict(base, configuration=4)),
        item('same-short-name-other-module', 'test_signatures', dict(base, native=other)),
        item('other-module-value', 'test_every_native_function',
             dict(base, native=other, dtype=DTYPES['uint8'], shape=(3, 19))),
        item('value-4', 'test_every_native_function',
             dict(base, configuration=4, dtype=DTYPES['uint8'], shape=(3, 19))),
        item('tiny-value', 'test_every_native_function',
             dict(base, dtype=DTYPES['uint8'], shape=(1, 1))),
        item('unsupported-value', 'test_every_native_function',
             dict(base, dtype=DTYPES['uint16'], shape=(3, 19))),
        item('unrelated-test', 'test_pipeline_can_swap_backend', {'configuration': 1}),
        item('surface-other', 'test_api_surface', {'entry': {'api': 'measure.label'}},
             'test_feature_api_surface.py'),
    ]
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(
        pytest_deselected=lambda items: deselected.extend(items)))
    items = rejected[:3] + wanted + rejected[3:]
    plugin.pytest_collection_modifyitems(None, config, items)
    assert items == wanted
    assert deselected == rejected
    assert plugin.collected == {'morphology.label': 5}
    assert plugin.item_apis == {
        node.nodeid: None if node.nodeid == 'registry-1' else 'morphology.label'
        for node in wanted
    }
