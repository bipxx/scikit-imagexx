"""Exact bounded pyramid stages and dynamic original compatibility routes."""

import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from pyramid_spatial_cases import FIXED, array_state, cases

FIXTURE = Path(__file__).with_name('fixtures') / 'pyramid_spatial_originals_v1.json'
RAW = FIXTURE.read_bytes()
assert hashlib.sha256(RAW).hexdigest() == '26863806d71754887ed9536a1f909fad75eee2144fe2278159475c2ee268e665'
BUNDLE = json.loads(RAW)
ROWS = {row['case_id']: row for row in BUNDLE['minimum']}
if sys.platform == 'win32':
    ROWS.update(BUNDLE['windows_overrides'])

NATIVE = {case['id'] for case in FIXED} | {
    f'{api}-{name}' for api in ('reduce', 'expand')
    for name in ('scaled-u8', 'axis-modulo', 'wrap-zero-sigma-nearest-order')
}


@pytest.mark.parametrize('case,api,image,options', list(cases({'fixed': FIXED}, np)), ids=list(ROWS))
def test_frozen_pyramid_stages(case, api, image, options, monkeypatch):
    from skimage import transform as skimage_transform
    from skimagexx import _compat, _core, _pyramid_spatial, transform

    public = getattr(transform, api)
    stock = getattr(skimage_transform, api)
    signature = inspect.signature(stock)
    assert public is getattr(_pyramid_spatial, api)
    assert inspect.signature(public) == signature
    row = ROWS[case]
    assert array_state(image) == row['input']
    entries = (_core.gaussian_nd, _core.resize_zoom_typed)
    assert all(inspect.isbuiltin(entry) for entry in entries)
    functions = {
        _pyramid_spatial._convert.__code__: 'convert',
        _pyramid_spatial.gaussian.__code__: 'gaussian',
        _pyramid_spatial.resize.__code__: 'resize',
    }
    visited, stage_results = [], []

    def trace(frame, event, result):
        if event == 'call' and frame.f_code is stock.__code__:
            raise AssertionError('the strict route entered the stock public function')
        if event == 'c_return' and any(entry is result for entry in entries):
            visited.append(result.__name__)
        if event == 'return' and frame.f_code in functions and result is not None:
            stage_results.append((functions[frame.f_code], array_state(result)))

    previous = _compat.set_fallback_enabled(False)
    profile = sys.getprofile()
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                sys.setprofile(trace)
                if case in NATIVE:
                    result = public(image, **options)
                else:
                    with pytest.raises(NotImplementedError, match='fallback is disabled'):
                        public(image, **options)
            finally:
                sys.setprofile(profile)
        if case in NATIVE:
            expected_stages = [(stage['stage'], stage['output']) for stage in row['stages']
                               if not stage['stage'].startswith('scipy-')]
            assert stage_results == expected_stages
            assert [type(notice.message).__name__ for notice in notices] == [x[0] for x in row['warnings']]
            expected_calls = ['gaussian_nd', 'resize_zoom_typed'] if api == 'pyramid_reduce' else [
                'resize_zoom_typed', 'gaussian_nd']
            if image.dtype.type is np.uint8:
                expected_calls.insert(0, 'gaussian_nd')
            assert visited == expected_calls
            assert array_state(result) == row['output'] and not np.shares_memory(image, result)
        else:
            assert not stage_results and not visited and not notices
            received = []
            marker = object()

            def replacement(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                received.append(bound.arguments)
                return marker

            monkeypatch.setattr(skimage_transform, api, replacement)
            _compat.set_fallback_enabled(True)
            with pytest.warns(_compat.FallbackWarning, match='skimage.transform.' + api):
                assert public(image, **options) is marker
            expected = signature.bind(image, **options)
            expected.apply_defaults()
            assert len(received) == 1
            assert received[0]['image'] is image
            assert all(received[0][key] is value or received[0][key] == value
                       for key, value in expected.arguments.items() if key != 'image')
        assert array_state(image) == row['input']
    finally:
        _compat.set_fallback_enabled(previous)
