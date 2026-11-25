"""Public warning protocol and documentation-search output."""

import inspect
import pickle
import pydoc
import sys
import warnings

import pytest
from skimage import util as original

import skimagexx as xx
from skimagexx import _util_misc as owned


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize(
    'query', ('regular_grid', '  REGIONPROPS   LABELS ', 'array', '', None, 823995271),
)
def test_lookfor_output_and_public_hot_swap(query, capsys, monkeypatch, strict):
    pages = []
    monkeypatch.setattr(pydoc, 'getpager', lambda: pages.append)
    expected = original.lookfor(query)
    stdout = capsys.readouterr().out
    expected_pages = pages[:]
    pages.clear()

    def forbidden(*args, **kwargs):
        pytest.fail('unexpected upstream lookfor dispatch')

    target = xx.util.lookfor
    assert target is owned.lookfor
    assert inspect.signature(target) == inspect.signature(original.lookfor)
    monkeypatch.setattr(original, 'lookfor', forbidden)
    assert target(query) is expected is None
    assert capsys.readouterr().out == stdout
    assert pages == expected_pages


def test_pending_skimage2_change_warning_protocol(strict):
    from skimagexx.util import PendingSkimage2Change

    target = PendingSkimage2Change
    reference = original.PendingSkimage2Change
    assert target is xx.util.PendingSkimage2Change is owned._canonical_pending_skimage2_change()
    assert target is reference
    assert target.__module__ == 'skimage.util'
    assert target.__name__ == reference.__name__
    assert target.__doc__ == reference.__doc__
    assert target.__bases__ == reference.__bases__ == (PendingDeprecationWarning,)
    for name in ('__getattribute__', '__init__', '__new__', '__reduce__', '__repr__',
                 '__setstate__', '__str__', 'add_note', 'args', 'with_traceback'):
        assert getattr(target, name, None) is getattr(reference, name, None)

    for category in (reference, target):
        instance = category('fixture', 13)
        assert instance.args == ('fixture', 13)
        assert str(instance) == "('fixture', 13)"
        assert repr(instance) == "PendingSkimage2Change('fixture', 13)"
        instance.args = ('changed',)
        instance.__setstate__({'source': 26})
        assert instance.source == 26
        if sys.version_info >= (3, 11):
            with pytest.raises(TypeError):
                instance.add_note(2)
            instance.add_note('diagnostic')
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
            assert type(restored) is category
            assert restored.args == instance.args
            assert restored.__dict__ == instance.__dict__
        with pytest.raises(category) as caught:
            raise instance
        assert instance.with_traceback(caught.tb) is instance
        assert instance.__traceback__ is caught.tb
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('ignore', PendingDeprecationWarning)
            warnings.warn(instance)
            assert caught_warnings == []
            warnings.simplefilter('always', target)
            warnings.warn(instance)
            assert caught_warnings[0].category is category
            assert caught_warnings[0].message is instance
