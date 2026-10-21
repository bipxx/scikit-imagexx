"""Availability/signature checks for the inventory, NOT numerical parity tests.

Unported APIs remain explicitly pending native parity and benchmark coverage.
"""
import inspect
import json
import re
from pathlib import Path
import warnings
import pytest
import skimage
import skimagexx as xx

INVENTORY = json.loads((Path(__file__).resolve().parents[1]/'coverage/api.json').read_text(encoding='utf-8'))


def test_inventory_reference_version():
    assert INVENTORY['reference_version'] == skimage.__version__ == '0.26.0'


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
            assert re.sub(r' at 0x[0-9a-fA-F]+', '', str(inspect.signature(target))) == entry['signature']
        if entry['implementation'] == 'upstream-only':
            assert entry['parity_test'] is None
            assert entry['benchmark'] is None
    finally:
        xx.set_fallback_enabled(previous)
