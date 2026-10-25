"""Prospective private gates. Import is metadata-only; execution needs approval.

The future bounded driver supplies an authenticated build/runtime configuration.
No environment creation, original algorithm call, benchmark, or product claim
occurs here. All 536 observations are retained before any comparison/assertion.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import types
import warnings

import pytest


FACTORY_SHA = "adbee581546e2031377853393b354be0dff41731eac920eb839c37b6c9f9db59"
OBSERVER_SHA = "90361227ea1c79238ad14ffaae8315e49007bfd411bec13ca051f15d5e92227e"
MATRIX_SHA = "5c391fadf3af48db242bf9f172990590b1188fcf207085aac301af03f3619668"
SEMANTIC_FIELDS = ("result", "error", "warnings", "events", "setup_events",
                   "input_before", "input_after", "result_aliases", "state")
PROVIDERS = ("sum", "square", "multiply", "count_nonzero", "logical_and", "any", "bitwise_and")
STAGES = ("coloc_binary", "coloc_count", "coloc_joint_count", "coloc_min_negative",
          "coloc_gather", "coloc_sum", "coloc_product")

# Collection policy only; binary/runtime authentication stays in the fixture.
_collection_path = os.environ.get("COLOCALIZATION_TEST_CONFIG")
_collection_portable = False
if _collection_path:
    _collection_config = json.loads(Path(_collection_path).read_bytes())
    assert _collection_config["profile"] in ("windows", "modern", "minimum", "portable", "sanitizer")
    assert type(_collection_config["portable"]) is bool
    assert _collection_config["portable"] == (_collection_config["profile"] == "portable")
    _collection_portable = _collection_config["portable"]
DECLARED_WORKERS = (1,) if _collection_portable else (1, 4)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(path, expected):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected
    return raw


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def coloc_runtime():
    assert sys.flags.optimize == 0
    config_path = Path(os.environ["COLOCALIZATION_TEST_CONFIG"]).resolve()
    config = json.loads(config_path.read_text())
    assert config["schema"] == "colocalization-private-runtime-v1"
    assert config["scientific_execution_authorized"] is True
    assert config["profile"] in ("windows", "modern", "minimum", "portable", "sanitizer")
    # The actual-link receipt must close every imported binary and input, not
    # merely a source path. A future driver cannot replace a compiled module
    # after the independently reviewed build receipt and still enter this gate.
    receipt = config["actual_link_receipt"]
    assert _hash(receipt["path"]) == receipt["sha256"]
    actual = json.loads(Path(receipt["path"]).read_text())
    assert actual["schema"] == "colocalization-actual-link-v1"
    assert actual["sole_worker_owner_count"] == 1
    assert actual["colocalization_kernel_count"] == 1
    for record in actual["records"] + config["runtime_records"]:
        path = Path(record["path"])
        assert path.stat().st_size == record["bytes"] and _hash(path) == record["sha256"]
    assert any(Path(row["path"]).resolve() == Path(config["core"]["path"]).resolve()
               and row["sha256"] == config["core"]["sha256"] for row in actual["records"])
    assert _hash(sys.executable) == config["python_sha256"]
    reference = Path(config["reference_root"])
    factory_path = reference / "tests/colocalization_cases.py"
    observer_path = reference / "tests/colocalization_reference.py"
    matrix_path = reference / "coverage/contracts/colocalization.json"
    _source(factory_path, FACTORY_SHA)
    _source(observer_path, OBSERVER_SHA)
    assert _hash(matrix_path) == MATRIX_SHA
    factory = _load("colocalization_cases", factory_path)
    observer = _load("colocalization_reference", observer_path)
    matrix = json.loads(matrix_path.read_text())
    specs = factory.cases()
    assert len(specs) == 536 and [r["id"] for r in specs] == matrix["case_ids"]
    assert sum(r["policy"] == "default" for r in specs) == 250
    # Import the original distribution first, solely for dependency/bootstrap
    # identity and no-forward traps. None of its three functions is executed.
    import numpy as np
    import skimage.measure._colocalization as original
    assert np.__version__ == config["numpy_version"]
    observer.authenticate(original.__file__, original.as_binary_ndarray.__code__.co_filename)
    core = _load(config["core"]["module_name"], config["core"]["path"])
    assert Path(core.__file__).resolve() == Path(config["core"]["path"]).resolve()
    assert _hash(core.__file__) == config["core"]["sha256"]
    assert core.colocalization_interface == "colocalization-stages-v1"
    assert core.get_num_threads() == 1
    package = types.ModuleType("_colocalization_candidate")
    package.__path__ = []
    package._core = core
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + "._core"] = core
    assert _hash(config["shell"]["path"]) == config["shell"]["sha256"]
    shell = _load(package.__name__ + "._colocalization", config["shell"]["path"])
    assert shell._core is core and shell.np is np
    assert original.np is np
    runtime_pins = {str(Path(row["path"]).resolve()): row for row in config["runtime_records"]}
    # Authenticate the provider files ACTUALLY imported by this bootstrap, not
    # merely a same-version distribution elsewhere on disk.
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.split(".")[0] in ("numpy", "scipy", "skimage"):
            actual_path = str(Path(path).resolve())
            assert actual_path in runtime_pins, (name, actual_path)
            assert _hash(actual_path) == runtime_pins[actual_path]["sha256"]
    original_profile = "minimum" if config["profile"] == "minimum" else "windows" if config["profile"] == "windows" else "modern"
    journal_pin = matrix["profiles"][original_profile]["journal"]
    journal_path = reference / Path(journal_pin["path"].replace("\\", "/"))
    assert journal_path.stat().st_size == journal_pin["bytes"] and _hash(journal_path) == journal_pin["sha256"]
    observations = {}
    for line in journal_path.read_text().splitlines():
        envelope = json.loads(line)
        if envelope.get("kind") == "observation":
            row = observer.semantic_projection(envelope["actual"])
            observations.setdefault(row["id"], []).append(row)
    assert set(observations) == set(matrix["case_ids"])
    for rows in observations.values():
        assert len(rows) == 2 and rows[0] == rows[1]
    yield dict(config=config, np=np, original=original, core=core, shell=shell,
               factory=factory, observer=observer, specs=specs,
               expected={key: rows[1] for key, rows in observations.items()})
    core.set_num_threads(1)


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


@pytest.mark.parametrize("workers", DECLARED_WORKERS)
@pytest.mark.parametrize("index", range(536))
def test_frozen_public_semantics(coloc_runtime, tmp_path, index, workers):
    runtime = coloc_runtime
    spec = runtime["specs"][index]
    core = runtime["core"]
    if runtime["config"]["profile"] == "portable" and workers == 4:
        # An explicit refused-worker record is not an image call or native API
        # completion. Keep every logical ID in the closed projection count.
        error = None
        try:
            core.set_num_threads(4)
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        (tmp_path / "portable-refusal.json").write_text(json.dumps(dict(id=spec["id"], worker=4, error=error, image_calls=0)) + "\n")
        assert error is not None and error["type"] == "ValueError" and core.get_num_threads() == 1
        return
    core.set_num_threads(workers)
    try:
        actual = _observe(runtime, spec, tmp_path)
    finally:
        core.set_num_threads(1)
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


def test_declared_global_worker_budget(coloc_runtime, tmp_path):
    core = coloc_runtime["core"]
    portable = coloc_runtime["config"]["portable"]
    before = core.get_num_threads()
    error = observed = None
    try:
        core.set_num_threads(4)
        observed = core.get_num_threads()
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        core.set_num_threads(1)
    after = core.get_num_threads()
    with (tmp_path / "worker-budget.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(portable=portable, before=before, requested=4, observed=observed,
                       error=error, after=after, image_calls=0), stream)
        stream.write("\n")
        stream.flush()
    assert before == after == 1
    if portable:
        assert error == dict(type="ValueError", message="this build has no OpenMP support")
        assert observed is None and core.has_avx2() is False
    else:
        assert error is None and observed == 4
