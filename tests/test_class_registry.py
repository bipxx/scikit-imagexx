"""Registry mechanics only, not native class or performance evidence."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('class_registry_unit',Path(__file__).parents[1]/'python/skimagexx/coverage.py')
registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)


def test_class_registry_never_inflates_image_or_metadata_counts():
    functions=registry.native_functions()
    assert len(functions)==136
    assert sum(registry.execution_kind(api)=='metadata' for api in functions)==3
    assert sum(registry.execution_kind(api)=='native' for api in functions)==133
    assert not set(functions)&set(registry.native_classes())


@pytest.mark.parametrize('name',registry._CLASS_EXTRA_MEMBERS)
def test_class_contract_keeps_members_state_providers_and_evidence_distinct(name):
    api='transform.'+name;contract=registry.class_contract(api)
    assert contract['registered']==(api in registry.native_classes())
    assert set(contract['members'])>=set(registry._CLASS_BASE_MEMBERS)
    assert {'params','user_attributes','inverse','estimation'}==set(contract['state'])
    assert 'NumPy' in contract['providers']['fitting']
    assert 'external' in contract['workers']['math_providers']
    assert contract['parity_evidence'] is contract['performance_evidence'] is None
    contract['members'].clear()
    assert registry.class_contract(api)['members'], 'caller mutated registry facts'


def test_unknown_class_contract_is_explicit():
    with pytest.raises(KeyError):registry.class_contract('transform.UnportedClass')


def test_exception_metadata_provider_does_not_inflate_native_classes():
    assert registry.class_metadata_providers()==('util.FailedEstimationAccessError',)
    assert len(registry.native_classes())==4
    assert not set(registry.class_metadata_providers())&set(registry.native_classes())
    contract=registry.class_contract('util.FailedEstimationAccessError')
    assert contract['implementation']=='exception-metadata-provider'
    assert 'Canonical' in contract['providers']['exception']
    assert contract['parity_evidence'] is contract['performance_evidence'] is None
