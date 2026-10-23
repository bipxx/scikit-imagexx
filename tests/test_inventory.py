"""Availability/signature checks for the inventory, NOT numerical parity tests.

Unported APIs remain explicitly pending native parity and benchmark coverage.
"""
import json
from pathlib import Path
import warnings
import pytest
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions, native_dtypes, native_classes, class_contract
from tools.inventory import signature_of

INVENTORY = json.loads((Path(__file__).resolve().parents[1]/'coverage/api.json').read_text(encoding='utf-8'))


def test_inventory_reference_version():
    assert INVENTORY['reference_version'] == skimage.__version__ == '0.26.0'
    declared = {entry['api'] for entry in INVENTORY['entries'] if entry['implementation'] in ('native-partial', 'metadata')}
    assert declared == set(native_functions())
    classes={entry['api'] for entry in INVENTORY['entries'] if entry['implementation']=='native-class-partial'}
    assert classes==set(native_classes()) and not classes&declared


@pytest.mark.parametrize('entry', INVENTORY['entries'], ids=lambda entry: entry['api'])
def test_api_availability_and_signature(entry):
    previous = xx.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', xx.FallbackWarning)
            target = xx
            for name in entry['api'].split('.'):
                target = getattr(target, name)
        assert callable(target)
        if entry['signature'] is not None:
            assert signature_of(target) == entry['signature']
        if entry['implementation'] == 'upstream-only':
            assert entry['parity_test'] is None
            assert entry['benchmark'] is None
            assert entry['native_default_case_dtypes'] == []
        elif entry['implementation']=='native-class-partial':
            assert entry['kind']=='class' and entry['native_default_case_dtypes']==[]
            assert entry['class_contract']==class_contract(entry['api'])
            assert entry['parity_test']=='tests/test_transform_classes_public.py'
        else:
            assert entry['native_default_case_dtypes'] == list(native_dtypes(entry['api']))
    finally:
        xx.set_fallback_enabled(previous)
