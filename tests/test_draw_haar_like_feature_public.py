"""Pinned Haar drawing results, state transitions and installed route checks."""
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest

from haar_feature_drawing_cases import cases, image_fingerprint, materialize, state, summarize

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/draw_haar_like_feature_originals_v1.json').read_bytes())
PROFILE = 'minimum' if np.__version__.startswith('1.') else 'modern'
ROWS = {row['id']:row for row in FIXTURE['profiles'][PROFILE]['cases']}
OWNED = {
    'gray-u8-all','rgb-f64-overlap','gray-f32-offset','seeded-u8-subset','large-overlap',
    'dtype-bool','dtype-float64','rgb-readonly','rgb-bytes','gray-readonly','roi-width-ignored',
    'roi-nonsensical-size','roi-wrap','negative-duplicates','increasing-reversed','relative-overlap',
    'zero-alpha','opaque-alpha','empty-features','empty-feature','feature-list','seed-int-subset',
    'seed-np-int-subset','seed-mt-subset','same-size-permutation','no-choice-same-generator',
    'zero-choice','negative-choice','oversized-choice','np-int-choice','seedsequence-provider',
    'numeric-coordinate-array','width-height-prov',
}


@pytest.fixture(scope='module')
def product():
    extension = os.environ.get('HAAR_DRAWING_CORE')
    if extension:
        name = '_haar_drawing_private'
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / 'python/skimagexx')]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name+'._core',extension)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        package._core = core
        sys.modules[spec.name] = core
        return importlib.import_module(name+'._draw_haar_like_feature'), core, importlib.import_module(name+'._compat')
    import skimagexx as xx

    owner = importlib.import_module('skimagexx._draw_haar_like_feature')
    compat = importlib.import_module('skimagexx._compat')
    assert xx.feature.draw_haar_like_feature is owner.draw_haar_like_feature
    assert xx.feature.__all__.count('draw_haar_like_feature') == 1
    assert xx.get_num_threads() == 1
    return owner, xx._core, compat


@pytest.mark.parametrize('recipe', cases(), ids=lambda obj:obj['id'])
def test_saved_result_and_rng(recipe, product):
    owner, _, compat = product
    expected = ROWS[recipe['id']]
    args, kwargs, protocol = materialize(recipe,np)
    image, rng = args[0], kwargs['rng']
    digest=image_fingerprint(image,np)
    assert state(rng,np) == expected['rng_before']
    prior=compat.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings(record=True) as logged:
            warnings.simplefilter('always')
            target=owner.draw_haar_like_feature
            if 'error' in expected:
                exception={'IndexError':IndexError,'TypeError':TypeError,'ValueError':ValueError,
                           'OverflowError':OverflowError}[expected['error']['type']]
                with pytest.raises(exception) as observed:
                    target(*args,**kwargs)
                assert str(observed.value)==expected['error']['text']
            else:
                result=target(*args,**kwargs)
                assert summarize(result,image,np)==expected['output']
        delegated=[warning for warning in logged if warning.category is compat.FallbackWarning]
        assert len(delegated)==(recipe['id'] not in OWNED)
        assert [{'type':item.category.__name__,'text':str(item.message)}
                for item in logged if item.category is not compat.FallbackWarning] == expected['warnings']
        assert state(rng,np) == expected['rng_after']
        assert image_fingerprint(image,np)==digest and expected['input_preserved']
        assert protocol==expected['protocol']
    finally:
        compat.set_fallback_enabled(prior)


def test_owned_calls_and_disabled_protocol_fallback(product):
    owner, core, compat=product
    entries={core._prepare_haar_drawing,core._draw_haar_prepared}
    assert all(inspect.isbuiltin(entry) and entry.__module__==core.__name__ for entry in entries)
    assert str(inspect.signature(owner.draw_haar_like_feature))==FIXTURE['profiles'][PROFILE]['signature']
    def collector(frame,event,value):
        if event=='c_return' and value in entries:
            events.append(value.__name__)
    events=[]
    previous=compat.set_fallback_enabled(False)
    prior=sys.getprofile()
    try:
        sys.setprofile(collector)
        args,kwargs,_=materialize(cases()[0],np)
        owner.draw_haar_like_feature(*args,**kwargs)
        assert events==['_prepare_haar_drawing','_draw_haar_prepared']
        bad=next(case for case in cases() if case['id']=='palette-provider')
        args,kwargs,trace=materialize(bad,np)
        with pytest.raises(NotImplementedError,match='fallback is disabled'):
            owner.draw_haar_like_feature(*args,**kwargs)
        assert not trace
    finally:
        compat.set_fallback_enabled(previous)
        sys.setprofile(prior)


@pytest.mark.parametrize('replacement', ['int8', 'equivalent'])
def test_prepared_storage_revalidated(product, replacement):
    _,core,_=product
    args,_,_=materialize(cases()[0],np)
    image, r, c, _, _, coords=args
    drawing=core._prepare_haar_drawing(image,r,c,coords)
    assert drawing is not None
    old_dtype=image.dtype
    image.dtype=(np.dtype('int8') if replacement=='int8'
                 else old_dtype.newbyteorder('>').newbyteorder('='))
    assert image.dtype is not old_dtype
    with pytest.raises(ValueError,match='prepared Haar input changed'):
        core._draw_haar_prepared(drawing,None,(1.,0.,0.),(0.,1.,0.),.5)


def test_hot_swap_same_pipeline_against_frozen_signature(product):
    from skimage import feature

    owner,_,compat=product
    def pipeline(backend):
        args,kwargs,_=materialize(cases()[1],np)
        return backend.draw_haar_like_feature(*args,**kwargs)
    previous=compat.set_fallback_enabled(False)
    try:
        result=pipeline(owner)
        np.testing.assert_array_equal(result,pipeline(feature))
    finally:
        compat.set_fallback_enabled(previous)


@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'float32', 'float64'])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('alpha', [0., .5])
def test_equivalent_and_tagged_image_descriptors(product, dtype, rgb, alpha):
    owner, _, compat = product
    args, options, _ = materialize(dict(cases()[0], dtype=dtype, rgb=rgb, alpha=alpha), np)
    image = args[0]
    digest = image_fingerprint(image, np)
    canonical = image.dtype
    equivalent = canonical.newbyteorder('>').newbyteorder('=')
    tags = {'units': 'linear', 'source': 'parity'}
    tagged = np.dtype(canonical, metadata=tags)
    assert equivalent is not canonical and equivalent == canonical
    previous = compat.set_fallback_enabled(False)
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            expected = owner.draw_haar_like_feature(*args, **options)
            assert expected.dtype.metadata is None
            for descriptor in (equivalent, tagged):
                view = image.view(descriptor)
                actual = owner.draw_haar_like_feature(view, *args[1:], **options)
                assert summarize(actual, view, np) == summarize(expected, image, np)
                retained = tags if descriptor is tagged and rgb and canonical.kind == 'f' else None
                assert actual.dtype.metadata == retained
                assert actual.base is None and actual.flags.aligned and actual.flags.c_contiguous
        assert not observed
        assert image_fingerprint(image, np) == digest
    finally:
        compat.set_fallback_enabled(previous)
