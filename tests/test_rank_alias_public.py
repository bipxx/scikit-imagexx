"""Qualified public overlap contract; requires the genuine production core.

Never install a development backend or bypass admission in this module.
Unsafe mixed-word cases are native-only rejection tests, not upstream probes.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import skimagexx as xx
from skimagexx import _rank
from skimagexx.filters import rank
from skimagexx.filters.rank import generic
from skimage.filters import rank as original
from skimagexx.coverage import rank_contract

spec=importlib.util.spec_from_file_location('public_rank_alias_contracts',Path(__file__).with_name('test_rank_alias_contracts.py'))
contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)
MATRIX=json.loads(contracts.MATRIX.read_text())


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);count=xx.get_num_threads()
    xx.set_num_threads(request.param)
    yield request.param
    xx.set_num_threads(count);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',contracts.NAMES)
def test_public_identity_signature_and_separate_capability(name):
    assert _rank._core is xx._core
    assert xx._core.rank_alias_interface=='rank-alias-v1'
    assert xx._core.rank_alias_capabilities==480
    assert xx._core.rank_histogram_capabilities & 8191==8191
    assert getattr(rank,name) is getattr(generic,name) is getattr(_rank,name)
    assert getattr(rank,name) is not getattr(original,name)
    assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(original,name))
    facts=rank_contract('filters.rank.'+name)
    assert facts['alias_interface']=='rank-alias-v1'
    assert facts['alias_capability']==1<<(5+contracts.NAMES.index(name))
    assert facts['alias_performance_evidence'] is None


@pytest.mark.parametrize('namespace',[rank,generic],ids=['canonical','generic'])
@pytest.mark.parametrize('case',contracts.CASES,ids=[row[0] for row in contracts.CASES])
def test_public_matches_every_frozen_owner_and_observation(namespace,case,monkeypatch):
    key,name,case_spec=case
    expected=MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]]
    def forbidden(*args,**kwargs):raise AssertionError('upstream rank forwarding')
    monkeypatch.setattr(_rank,'fallback',forbidden)
    monkeypatch.setattr(original,name,forbidden)
    assert contracts.observe(getattr(namespace,name),case_spec)==expected,key


@pytest.mark.parametrize('name',['gradient','median','modal'])
@pytest.mark.parametrize('odd_base,odd_stride',[(True,False),(False,True),(True,True)])
def test_mixed_word_partial_histogram_rejects_before_store(name,odd_base,odd_stride):
    # Defined own rejection only. Original Cython may mutate then index outside
    # its histogram, and must never be invoked for these fabricated views.
    owner=np.zeros(64,np.uint8)
    image=np.ndarray((1,4),np.uint16,buffer=owner);image[:]=[1,2,3,4]
    output=np.ndarray((1,4),np.uint16,buffer=owner,offset=int(odd_base),strides=(12,3 if odd_stride else 2))
    before=owner.copy()
    error,match=(ValueError,'not C-contiguous') if odd_stride else (NotImplementedError,'mixed-byte feedback')
    with pytest.raises(error,match=match):
        getattr(rank,name)(image,np.ones((1,1),np.uint8),out=output)
    np.testing.assert_array_equal(owner,before)


def test_active_integral_entropy_feedback_stays_excluded():
    image=np.arange(35,dtype=np.uint8).reshape(5,7);before=image.copy()
    with pytest.raises(NotImplementedError,match='integral entropy'):
        rank.entropy(image,np.ones((3,3),np.uint8),out=image.view())
    np.testing.assert_array_equal(image,before)


def test_r3_overlap_is_still_excluded():
    for name in ('autolevel','equalize','enhance_contrast','threshold'):
        image=np.arange(35,dtype=np.uint8).reshape(5,7);before=image.copy()
        with pytest.raises(NotImplementedError,match='Overlapping buffers'):
            getattr(rank,name)(image,np.ones((3,3),np.uint8),out=image.view())
        np.testing.assert_array_equal(image,before)
