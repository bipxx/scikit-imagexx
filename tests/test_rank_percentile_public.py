"""R5 source-wrapper prototype and future actual-public gate.

Explicit PROTOTYPE_CORE binds a frozen private extension to the source wrapper
only. It never replaces public namespaces/registry and is not public proof.
Without that flag, numerical tests require actual installed public admission.
"""
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import rank as reference
from skimagexx.coverage import native_functions


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ROOT=Path(__file__).parents[1]
contracts=load('rank_percentile_wrapper_contracts',Path(__file__).with_name('test_rank_percentile_contracts.py'))
MATRIX=json.loads(contracts.MATRIX.read_text())
PROTOTYPE=os.environ.get('SKIMAGEXX_RANK_PERCENTILE_PROTOTYPE_CORE')
source_wrapper=load('skimagexx._rank_percentile_source_prototype',ROOT/'python/skimagexx/_rank.py')
if PROTOTYPE:source_wrapper._core=importlib.import_module(PROTOTYPE)
NAMES=contracts.NAMES
PATHS={'filters.rank.'+name for name in NAMES}


@pytest.fixture
def target():
    if PROTOTYPE:return source_wrapper
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Actual whole8 public wiring/package gate pending; no alternate backend substituted')
    assert PATHS<=registered,'partial percentile registration is not public acceptance'
    return importlib.import_module('skimagexx._rank')


@pytest.fixture(params=(1,4))
def active(request,target):
    previous=xx.set_fallback_enabled(False);count=target._core.get_num_threads();target._core.set_num_threads(request.param)
    yield target
    target._core.set_num_threads(count);xx.set_fallback_enabled(previous)


def test_source_signatures_canonical_exports_and_generic_handoff():
    assert len(source_wrapper._GENERIC_FUNCTIONS)==17
    assert source_wrapper._PERCENTILE_FUNCTIONS==NAMES
    assert not set(NAMES)&set(source_wrapper._GENERIC_FUNCTIONS)
    assert source_wrapper.__all__==list(source_wrapper._GENERIC_FUNCTIONS+NAMES+source_wrapper._BILATERAL_FUNCTIONS+source_wrapper._DECISIONS_FUNCTIONS)
    for name in NAMES:assert inspect.signature(getattr(source_wrapper,name))==inspect.signature(getattr(reference,name))
    assert not hasattr(source_wrapper,'equalize_percentile')
    assert reference.percentile is contracts.percentile_module.percentile
    assert 'sum_percentile' not in contracts.percentile_module.__all__


def test_actual_public_namespace_or_explicit_pending():
    if PROTOTYPE:pytest.skip('DIRECT PRIVATE-CORE WRAPPER PROTOTYPE: not public namespace/registry/sole-core/SDK acceptance')
    registered=set(native_functions())
    if not registered&PATHS:pytest.skip('Pending root shared canonical-only wiring and genuine whole8 package')
    assert PATHS<=registered
    wrapper=importlib.import_module('skimagexx._rank');rank=importlib.import_module('skimagexx.filters.rank');generic=importlib.import_module('skimagexx.filters.rank.generic')
    assert wrapper._core is xx._core
    assert wrapper._core.set_num_threads is xx.set_num_threads and wrapper._core.get_num_threads is xx.get_num_threads
    assert xx.get_num_threads()==1
    for name in NAMES:
        assert getattr(rank,name) is getattr(wrapper,name) and name in rank.__all__
        assert name not in generic.__dict__ and name not in generic.__all__
        assert 'filters.rank.generic.'+name not in registered
        assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_source_wrapper_frozen_values_errors_warnings_owners(case,active,monkeypatch):
    key,name,parameters=case
    def forbidden(*a,**k):raise AssertionError('upstream percentile algorithm/fallback forwarding')
    monkeypatch.setattr(active,'fallback',forbidden)
    for provider in (reference,contracts.percentile_module):
        for operation in NAMES:monkeypatch.setattr(provider,operation,forbidden)
    assert contracts.observe(getattr(active,name),name,parameters)==MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]],key


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('failure',('none','p0','p1','bad-output','bad-shift','bad-footprint'))
def test_source_wrapper_parameter_conversion_traces(name,failure,active,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    assert contracts.parameter_observe(name,failure)==MATRIX['parameter_events'][contracts.profile()][name+'/'+failure]


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('interval',((0.,1.),(.25,.75),(.5,.5),(1.,1.),(0.,0.)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_source_wrapper_independent_arithmetic(name,dtype,kind,interval,output,active,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_independent_bin_scan_replay(name,dtype,kind,interval,output)


@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','masked','empty'))
@pytest.mark.parametrize('q',(contracts.Fraction(0),contracts.Fraction(1,8),contracts.Fraction(1,4),contracts.Fraction(1,2),contracts.Fraction(3,4),contracts.Fraction(7,8),contracts.Fraction(1)))
def test_source_wrapper_independent_sorted_statistics(dtype,kind,q,active,monkeypatch):
    monkeypatch.setattr(contracts,'reference',active)
    contracts.test_sorted_order_statistic_oracle_without_histogram_scan(dtype,kind,q)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('mode',('missing-interface','wrong-interface','missing-cap','wrong-cap','float-cap','missing-bit','missing-entry','non-callable','python-forwarder','wrong-builtin','wrong-owned-builtin'))
def test_source_guard_harness_only_precedes_all_image_metadata(name,mode,monkeypatch):
    # These are deliberately metadata guard stubs, never native parity proof.
    core=source_wrapper._core;entry=getattr(core,'rank_percentile_run',None)
    fields=dict(__name__=getattr(core,'__name__','harness_only'),rank_percentile_interface='rank-percentile-v1',rank_percentile_capabilities=0x3fe0000,rank_percentile_run=entry)
    if mode=='missing-interface':fields.pop('rank_percentile_interface')
    elif mode=='wrong-interface':fields['rank_percentile_interface']='rank-percentile-v0'
    elif mode=='missing-cap':fields.pop('rank_percentile_capabilities')
    elif mode=='wrong-cap':fields['rank_percentile_capabilities']=8191
    elif mode=='float-cap':fields['rank_percentile_capabilities']=float(0x3fe0000)
    elif mode=='missing-bit':fields['rank_percentile_capabilities'] &= ~(1<<(17+NAMES.index(name)))
    elif mode=='missing-entry':fields.pop('rank_percentile_run')
    elif mode=='non-callable':fields['rank_percentile_run']=7
    elif mode=='python-forwarder':fields['rank_percentile_run']=lambda *a:entry(*a)
    elif mode=='wrong-builtin':fields['rank_percentile_run']=len
    else:fields['rank_percentile_run']=getattr(core,'rank_run',len)
    monkeypatch.setattr(source_wrapper,'_core',SimpleNamespace(**fields))
    class Bomb:
        def __array__(self,*a,**k):raise AssertionError('fixture/array work before capability admission')
    with pytest.raises(NotImplementedError,match='rank-percentile-v1 compiled backend required'):getattr(source_wrapper,name)(Bomb(),None)


@pytest.mark.parametrize('kind',('p0-negative','p1-infinite','unordered','overlap','zero-depth','short-output','short-mask','undefined-unsigned'))
def test_unsupported_scope_is_explicit_prewrite_without_original_probe(kind,active,monkeypatch):
    image=np.arange(15,dtype=np.uint8).reshape(3,5);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.uint8);opts=dict(out=out);name='mean_percentile'
    if kind=='p0-negative':opts['p0']=-.1
    elif kind=='p1-infinite':opts['p1']=float('inf')
    elif kind=='unordered':opts.update(p0=.8,p1=.2)
    elif kind=='overlap':out=image.view();opts['out']=out
    elif kind=='zero-depth':out=np.empty(image.shape+(0,),np.uint8);opts['out']=out
    elif kind=='short-output':out=np.full((2,5),99,np.uint8);opts['out']=out
    elif kind=='short-mask':opts['mask']=np.ones((1,1),np.uint8)
    else:image=np.full(image.shape,65535,np.uint16);name='subtract_mean_percentile'
    before=image.copy();previous=out.copy()
    def forbidden(*a,**k):raise AssertionError('unsupported source scope must not forward/probe original')
    monkeypatch.setattr(active,'fallback',forbidden)
    for provider in (reference,contracts.percentile_module):
        for operation in NAMES:monkeypatch.setattr(provider,operation,forbidden)
    with pytest.raises(NotImplementedError,match='Outside qualified rank-percentile-v1 coverage'):getattr(active,name)(image,fp,**opts)
    assert image.tobytes()==before.tobytes() and out.tobytes()==previous.tobytes()


def test_compiled_entry_and_old_interfaces(active):
    core=active._core
    assert core.rank_percentile_interface=='rank-percentile-v1' and core.rank_percentile_capabilities==0x3fe0000
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_statistics_capabilities,core.rank_vector_capabilities)==(8191,480,0xe000,0x10000)
    for operation in range(17,26):assert active._require_percentile(operation) is core.rank_percentile_run


@pytest.mark.parametrize('name',('mean_percentile','threshold_percentile'))
@pytest.mark.parametrize('message',('rank percentile parameters outside defined coverage','overlapping buffers are outside rank-statistics-v1 parity'))
def test_user_provider_exception_text_is_not_reclassified(name,message,active):
    # Source-safe: __float__ raises before either original/private kernel.
    class Provider:
        def __float__(self):raise ValueError(message)
    image=np.arange(12,dtype=np.uint8).reshape(3,4);fp=np.ones((3,3),np.uint8)
    for function in (getattr(reference,name),getattr(active,name)):
        out=np.full(image.shape,99,np.uint8)
        with pytest.raises(ValueError,match=message):function(image,fp,out=out,p0=Provider())
        assert np.all(out==99)
