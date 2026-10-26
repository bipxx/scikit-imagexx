"""Public colocalization536 parity; provider stages are not full native credit.

Uses actual imported public modules. Component testing does not qualify a whole
wheel/SDK. Lossless public expectations retain the frozen original journal
provenance; no private-core replacement occurs here.
"""
import hashlib
import gzip
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback
import warnings

import pytest

ROOT=Path(__file__).resolve().parents[1]
APIS=('intersection_coeff','manders_coloc_coeff','manders_overlap_coeff')
SEMANTIC_FIELDS=("result","error","warnings","events","setup_events",
                 "input_before","input_after","result_aliases","state")
PROVIDERS=("sum","square","multiply","count_nonzero","logical_and","any","bitwise_and")
STAGES=("coloc_binary","coloc_count","coloc_joint_count","coloc_min_negative",
        "coloc_gather","coloc_sum","coloc_product")


def _helper(name,pin):
    path=ROOT/'tests'/(name+'.py');raw=path.read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()==pin
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

FACTORY=_helper('colocalization_cases','adbee581546e2031377853393b354be0dff41731eac920eb839c37b6c9f9db59')
OBSERVER=_helper('colocalization_reference','90361227ea1c79238ad14ffaae8315e49007bfd411bec13ca051f15d5e92227e')
CASES=FACTORY.cases()
WORKER_FOUR_IDS=('intersection_coeff/dtype/bool','manders_coloc_coeff/dtype/float32',
                 'manders_overlap_coeff/dtype/float64')


@pytest.fixture(scope='module')
def coloc_runtime():
    import numpy as np
    import skimage
    import skimage.measure._colocalization as original
    import skimagexx as xx
    from skimagexx import _colocalization as shell
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert skimage.__version__=='0.26.0'
    assert shell._core is xx._core
    core=xx._core
    assert xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads
    assert core.colocalization_interface=='colocalization-stages-v1' and core.get_num_threads()==1
    for name in STAGES:
        entry=getattr(core,name)
        assert inspect.isbuiltin(entry) and entry.__module__==core.__name__
    assert inspect.isclass(core.ColocalizationGuards)
    registered=set(native_functions())
    for name in APIS:
        assert 'measure.'+name in registered
        assert getattr(xx.measure,name) is getattr(shell,name)
        assert inspect.signature(getattr(xx.measure,name))==inspect.signature(getattr(original,name))
    try:
        core.set_num_threads(4)
    except ValueError as error:
        assert str(error)=='this build has no OpenMP support' and core.get_num_threads()==1
        parallel=False
    else:
        assert core.get_num_threads()==4
        parallel=True
    finally:core.set_num_threads(1)
    profile={('win32','2.5.4'):'windows',('linux','2.5.4'):'modern',
             ('linux','1.24.4'):'minimum'}[sys.platform,np.__version__]
    path=ROOT/'coverage/contracts/colocalization.json';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='5c391fadf3af48db242bf9f172990590b1188fcf207085aac301af03f3619668'
    matrix=json.loads(raw);assert len(CASES)==536 and [s['id'] for s in CASES]==matrix['case_ids']
    data=json.loads(gzip.decompress((ROOT/'tests/data/colocalization_public_semantics.json.gz').read_bytes()))
    assert data['matrix_sha256']==hashlib.sha256(raw).hexdigest()
    assert set(data['profiles'])=={'windows','modern','minimum'}
    for name,record in data['profiles'].items():
        assert record['journal']==matrix['profiles'][name]['journal']
        assert len(record['observations'])==536 and set(record['observations'])==set(matrix['case_ids'])
        for ident,row in record['observations'].items():
            encoded=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            assert hashlib.sha256(encoded).hexdigest()==matrix['profiles'][name]['observations'][ident]['semantic_sha256']
    observations=data['profiles'][profile]['observations']
    assert set(observations)==set(matrix['case_ids'])
    OBSERVER.authenticate(original.__file__,original.as_binary_ndarray.__code__.co_filename)
    previous=xx.set_fallback_enabled(False)
    assert _compat._fallback_enabled is False
    try:
        yield dict(np=np,xx=xx,core=core,shell=shell,original=original,factory=FACTORY,
                   observer=OBSERVER,specs=CASES,expected=observations,parallel=parallel)
    finally:
        xx.set_fallback_enabled(previous);core.set_num_threads(1)


def _observe(runtime, spec, root):
    np, core, shell = runtime["np"], runtime["core"], runtime["shell"]
    factory, observer, original = runtime["factory"], runtime["observer"], runtime["original"]
    assets = observer.RawAssets(root / "assets")
    journal = observer.RawJournal(root / "actual.jsonl")
    row = dict(id=spec["id"], phase="fixture", result=None, error=None,
               observer_error=None, warnings=[], events=[], setup_events=[],
               input_before=None, input_after=None, result_aliases=None, state=None,
               c_events=[], stage_returns=[], array_providers=[], forbidden_calls=[])
    events = row["events"]
    old_policy, old_handler, old_buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    old_profile = sys.getprofile()
    original_functions = {name: getattr(original, name) for name in factory.APIS}
    original_providers = {name: getattr(np, name) for name in PROVIDERS}
    original_stages = {name: getattr(core, name) for name in STAGES}
    entry = getattr(shell, spec["api"])
    arrays, result = [], None

    def handler(message, flag):
        events.append(dict(event="FP-handler", message=str(message), flag=int(flag)))
        if spec["group"] == "protocol" and spec["name"] == "warning_mutation" and arrays[0].size:
            arrays[0].flat[-1] = arrays[0].flat[0]
            events.append(dict(event="finite-inallocation-mutation", index=-1))

    def forbid(name):
        def call(*args, **kwargs):
            row["forbidden_calls"].append(name)
            raise AssertionError("original colocalization forwarding: " + name)
        return call

    def provider(name, function):
        def call(*args, **kwargs):
            row["array_providers"].append(dict(name=name, inputs=[dict(dtype=x.dtype.str, shape=list(x.shape))
                                                                 for x in args if isinstance(x, np.ndarray)]))
            return function(*args, **kwargs)
        return call

    def stage(name, function):
        def call(*args, **kwargs):
            value = function(*args, **kwargs)
            row["stage_returns"].append(dict(name=name, admitted=value is not None,
                validation_only=name == "coloc_binary" and value is not None and value[0] is False))
            return value
        return call

    def profile(frame, event, argument):
        if event in ("c_call", "c_return", "c_exception"):
            name = getattr(argument, "__name__", "")
            owner = getattr(argument, "__self__", None)
            module = getattr(argument, "__module__", None)
            if name in STAGES and (module == core.__name__ or getattr(type(owner), "__module__", None) == core.__name__):
                row["c_events"].append(dict(event=event, name=name, module=module,
                                             owner_module=getattr(type(owner), "__module__", None)))

    try:
        assert old_profile is None and sys.flags.optimize == 0
        args, kwargs, arrays = factory.fixture(spec, np, events)
        row["setup_events"] = list(events)
        events.clear()
        row["input_before"] = observer.snapshots(arrays, np, assets)
        if spec["policy"] == "default":
            assert old_policy == dict(divide="warn", over="warn", under="ignore", invalid="warn") and old_handler is None
        else:
            np.seterr(all=spec["policy"])
            np.seterrcall(handler if spec["policy"] == "call" else old_handler)
        policy_before, handler_before = np.geterr(), np.geterrcall()
        traps = {name: forbid(name) for name in factory.APIS}
        providers = {name: provider(name, function) for name, function in original_providers.items()}
        for name, function in traps.items():
            setattr(original, name, function)
        for name, function in providers.items():
            setattr(np, name, function)
        for name, function in original_stages.items():
            setattr(core, name, stage(name, function))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sys.setprofile(profile)
            try:
                row["phase"] = "candidate-call"
                result = entry(*args, **kwargs)
            except Exception as error:
                row["error"] = dict(module=type(error).__module__, type=type(error).__name__, message=str(error))
            finally:
                sys.setprofile(old_profile)
            row["warnings"] = [dict(module=type(w.message).__module__, category=w.category.__name__, message=str(w.message)) for w in caught]
        # Restore spies before serializers/NumPy snapshots so their metadata
        # work cannot inflate any image-provider or C-completion count.
        providers_unchanged = all(getattr(np, name) is function for name, function in providers.items())
        for name, function in original_providers.items():
            setattr(np, name, function)
        for name, function in original_stages.items():
            setattr(core, name, function)
        row["result"] = observer.value(result, np, assets) if row["error"] is None else None
        row["result_aliases"] = [result is array for array in arrays]
        row["state"] = dict(policy_before=policy_before, policy_after=np.geterr(),
            handler_identity_unchanged=np.geterrcall() is handler_before,
            buffer_before=old_buffer, buffer_after=np.getbufsize(),
            entry_identity_unchanged=getattr(shell, spec["api"]) is entry,
            providers_unchanged=providers_unchanged)
        row["input_after"] = observer.snapshots(arrays, np, assets)
        row["phase"] = "complete"
    except Exception as error:
        row["observer_error"] = dict(module=type(error).__module__, type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        sys.setprofile(old_profile)
        for name, function in original_functions.items():
            setattr(original, name, function)
        for name, function in original_providers.items():
            setattr(np, name, function)
        for name, function in original_stages.items():
            setattr(core, name, function)
        if spec["policy"] != "default":
            np.seterr(**old_policy)
            np.seterrcall(old_handler)
        row["restored"] = (np.geterr() == old_policy and np.geterrcall() is old_handler
                           and np.getbufsize() == old_buffer and sys.getprofile() is old_profile)
        journal.append(dict(kind="candidate-observation", actual=row))
        journal.close()
    return row


def _check_case(runtime,spec,tmp_path,workers):
    core=runtime['core'];assert core.get_num_threads()==1
    core.set_num_threads(workers)
    try:actual=_observe(runtime,spec,tmp_path)
    finally:core.set_num_threads(1)
    expected = runtime["expected"][spec["id"]]
    assert actual["observer_error"] is None and actual["phase"] == "complete"
    assert actual["restored"] and not actual["forbidden_calls"]
    for key in SEMANTIC_FIELDS:
        assert actual[key] == expected[key], (spec["id"], key)
    returns = [event["name"] for event in actual["c_events"] if event["event"] == "c_return"]
    assert len(returns) == len(actual["stage_returns"])
    assert sorted(returns) == sorted(row["name"] for row in actual["stage_returns"])
    # Source-derived ordinary finite cohort: 12 dtype controls and all eight
    # layout controls per API except MCC's reversed, noncoalesced sum frontier.
    ordinary = spec["group"] == "dtype" or (spec["group"] == "layout" and
                not (spec["api"] == "manders_coloc_coeff" and spec["name"] == "reverse"))
    if ordinary:
        assert actual["array_providers"] == [], (spec["id"], actual["array_providers"])
        assert any(row["admitted"] for row in actual["stage_returns"])
    if actual["error"] is None and not any(row["admitted"] for row in actual["stage_returns"]):
        # Object/complex and the declared nonfinite MCC source frontier can
        # complete through literal primitives. ZERO native completion remains
        # ZERO; public semantic parity does not manufacture native credit.
        assert actual["array_providers"]
        nonfinite_mcc = (spec["api"] == "manders_coloc_coeff" and
                         spec["group"] in ("special", "fp") and
                         spec.get("special", spec["name"]) in ("nan0", "inf0"))
        assert ((spec["group"] == "protocol" and spec["name"] == "object_numeric") or
                (spec["group"] == "binary_control" and spec["name"] == "complex_binary") or
                nonfinite_mcc), spec["id"]


@pytest.mark.parametrize('index',range(536),ids=[s['id'] for s in CASES])
def test_frozen_public_semantics(coloc_runtime,tmp_path,index):
    _check_case(coloc_runtime,CASES[index],tmp_path,1)


@pytest.mark.parametrize('name',APIS)
def test_public_registration(coloc_runtime,name):
    runtime=coloc_runtime;entry=getattr(runtime['xx'].measure,name)
    assert entry is getattr(runtime['shell'],name)
    assert entry.__module__=='skimagexx._colocalization'
    assert runtime['core'].get_num_threads()==1


@pytest.mark.parametrize('ident',WORKER_FOUR_IDS)
def test_representative_worker_four(coloc_runtime,tmp_path,ident):
    runtime=coloc_runtime
    if not runtime['parallel']:
        with pytest.raises(ValueError,match='this build has no OpenMP support'):
            runtime['core'].set_num_threads(4)
        assert runtime['core'].get_num_threads()==1
        return
    spec=next(s for s in CASES if s['id']==ident)
    _check_case(runtime,spec,tmp_path,4)


@pytest.mark.parametrize('workers',(1,4))
def test_large_public_intersection_parallel_count(coloc_runtime,workers,monkeypatch):
    runtime=coloc_runtime;np=runtime['np'];xx=runtime['xx']
    assert xx.get_num_threads()==1
    if workers==4 and not runtime['parallel']:
        with pytest.raises(ValueError,match='this build has no OpenMP support'):
            xx.set_num_threads(4)
        return
    # Exceed the shared partition threshold and exercise the AVX2/scalar count
    # paths on nonuniform masks rather than only selecting a worker budget.
    indices=np.arange(515*513).reshape(515,513)
    first=indices%3==0;second=indices%5!=0
    assert first.size>256*1024
    expected=runtime['original'].intersection_coeff(first,second)
    try:
        xx.set_num_threads(workers)
        def denied(*args,**kwargs):
            raise AssertionError('upstream intersection forwarding')
        with monkeypatch.context() as patch:
            patch.setattr(runtime['original'],'intersection_coeff',denied)
            actual=xx.measure.intersection_coeff(first,second)
        assert type(actual) is type(expected) and actual==expected
    finally:xx.set_num_threads(1)


def test_owner_metadata_only(coloc_runtime):
    np, core = coloc_runtime["np"], coloc_runtime["core"]
    root = np.arange(8, dtype=np.float32)
    view = root[1:4]
    guard = core.ColocalizationGuards()
    guard.add(view)
    root.shape = (2, 4)
    with pytest.raises(ValueError, match="metadata changed"):
        guard.check()
    root.shape = (8,)
    # All subsequent fixtures reach only the metadata guard. No NumPy numeric
    # primitive, native pixel stage, or original is given a forged view.
    root = np.arange(8, dtype=np.float32)
    view = root[1:3]
    root.resize(0, refcheck=False)
    with pytest.raises(ValueError, match="escapes"):
        core.ColocalizationGuards().add(view)
    safe = np.arange(4, dtype=np.uint8)
    dummy = np.lib.stride_tricks.as_strided(safe, shape=(2,), strides=(1,))
    with pytest.raises(ValueError, match="bounded"):
        core.ColocalizationGuards().add(dummy)
    class Trap(np.ndarray):
        @property
        def base(self):
            raise AssertionError("Python base property must never run")
    core.ColocalizationGuards().add(safe.view(Trap))
    for owner in (bytes(9), bytearray(9), memoryview(bytes(9))):
        array = np.ndarray((2,), dtype=np.float32, buffer=owner, offset=1)
        guard = core.ColocalizationGuards()
        guard.add(array)
        guard.check()
    # A builtin memoryview may hide an ndarray allocation owner. Shape changes
    # do not read freed memory, but must still invalidate the complete stamp.
    root = np.arange(8, dtype=np.uint8)
    exported = memoryview(root)
    array = np.frombuffer(exported, dtype=np.uint8)
    guard = core.ColocalizationGuards()
    guard.add(array)
    root.shape = (2, 4)
    with pytest.raises(ValueError, match="metadata changed"):
        guard.check()
