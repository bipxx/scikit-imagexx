"""Frozen whole-public Medial Axis comparison, adapted from the admitted retained runner."""
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib import util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import warnings

CARRIER = Path(__file__).resolve().parent
SAMPLER = CARRIER / "sampler.py"
POOL = CARRIER / "threadpoolctl.py"
PROFILE = "linux-modern-hosted"

def native_public(identity):
    sys.path.insert(0,identity['installed'])
    import inspect
    import skimagexx as xx
    from skimagexx import _compat,_core,morphology

    assert xx.get_num_threads()==1 and xx.has_avx2()
    assert Path(_core.__file__).resolve()==Path(identity['haar_core'])
    assert digest(_core.__file__)==identity['haar_core_sha256']
    binary=Path(identity['private_core'])
    core=load('_medial_axis_private',binary)
    assert Path(core.__file__).resolve()==binary and digest(binary)==identity['private_core_sha256']
    entries=(core.medial_axis_prepare,core.medial_axis_ordered)
    for f in entries:
        assert f.__name__ not in vars(_core) and inspect.isbuiltin(f) and f.__module__==core.__name__
        setattr(_core,f.__name__,f)
    wrapper=load('skimagexx._medial_axis',SOURCE/'python/skimagexx/_medial_axis.py')
    assert wrapper._core is _core
    assert Path(inspect.getsourcefile(wrapper.medial_axis)).resolve()==SOURCE/'python/skimagexx/_medial_axis.py'
    assert Path(morphology.__file__).resolve()==Path(identity['installed'])/'skimagexx/morphology.py'
    morphology.medial_axis=wrapper.medial_axis
    return morphology.medial_axis,entries,_compat


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prove_case(stock_call, native_call, providers, snapshot, row, image, mask, distances, seed, np):
    stock, native, entries, edt, label = providers
    expected_generator = np.random.default_rng(seed)
    before = expected_generator.bit_generator.state
    priority = expected_generator.permutation(np.arange(row['loop'][0]['foreground']))
    after = expected_generator.bit_generator.state
    stock_events = {'public': 0, 'label': 0, 'edt': 0}
    stock_locals = {}
    labels = {stock.__code__: 'public', edt.__code__: 'edt', label.__code__: 'label'}
    native_calls = []
    native_priority = []

    def profile_stock(frame, event, arg):
        if event == 'call' and frame.f_code in labels:
            stock_events[labels[frame.f_code]] += 1
        if event == 'return' and frame.f_code is stock.__code__ and arg is not None:
            for name in ('tiebreaker', 'generator', 'order'):
                stock_locals[name] = frame.f_locals[name]

    def profile_native(frame, event, arg):
        if event == 'call' and frame.f_code is stock.__code__:
            raise AssertionError('fallback into stock during native proof')
        if event == 'call' and frame.f_code is label.__code__:
            raise AssertionError('native candidate rebuilt the stock topology table')
        if event == 'call' and frame.f_code is edt.__code__:
            native_calls.append('edt')
        if event == 'c_return' and any(arg is entry for entry in entries):
            native_calls.append(arg.__name__)
            if arg is entries[1]:
                native_priority.append(frame.f_locals['priority'])

    def check(output):
        if distances:
            assert type(output) is tuple and len(output) == 2
            skeleton, distance = output
            assert snapshot(distance, small=True) == row['distance']
            assert not np.shares_memory(skeleton, distance) and not np.shares_memory(distance, image)
        else:
            skeleton = output
        assert snapshot(skeleton, small=True) == row['skeleton']
        assert not np.shares_memory(skeleton, image)
        if mask is not None:
            assert np.array_equal(skeleton[~mask], image[~mask])
            if distances:
                assert (distance[~mask] == 0).all()

    prior = sys.getprofile()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        try:
            sys.setprofile(profile_stock)
            reference = stock_call()
            sys.setprofile(profile_native)
            actual = native_call()
        finally:
            sys.setprofile(prior)
        generator = np.random.default_rng(seed)
        assert generator.bit_generator.state == before
        explicit = native(image, mask, distances, rng=generator)
        assert generator.bit_generator.state == after
        check(explicit)
    assert not notes and row['warnings'] == []
    assert stock_events == {'public': 1, 'label': 1024, 'edt': 1}
    assert native_calls == ['medial_axis_prepare', 'edt', 'medial_axis_ordered']
    assert snapshot(stock_locals['order'], small=True) == row['loop'][0]['order']
    assert stock_locals['generator'].bit_generator.state == after
    assert len(native_priority) == 1
    assert snapshot(stock_locals['tiebreaker']) == snapshot(native_priority[0]) == snapshot(priority)
    check(reference)
    check(actual)
    assert snapshot(image, small=True) == row['input']
    assert mask is None or snapshot(mask, small=True) == row['mask']
    return dict(stock_calls=stock_events, native_calls=native_calls, input=row['input'], mask=row['mask'],
                skeleton=row['skeleton'], distance=row.get('distance'), stock_order=row['loop'][0]['order'],
                priority_sha256=sha256(priority.tobytes()).hexdigest(), pcg64_before=before, pcg64_after=after)


def execute(protocol, identity, affinity, deadline, go_sha, entry):
    journal, proof, result_path = (HERE / f'{PROFILE}-{name}-v1.{suffix}' for name, suffix in
                                   [('raw-batches', 'jsonl'), ('correctness', 'jsonl'), ('result', 'json')])
    assert not any(path.exists() for path in (journal, proof, result_path))
    import numpy as np
    from importlib import import_module
    pcg64_module = import_module("numpy.random._pcg64")
    import scipy
    import skimage
    from scipy import ndimage
    from scipy.ndimage import _measurements, _morphology, _nd_image, _ni_label
    from skimage import morphology
    from skimage.morphology import _skeletonize, _skeletonize_various_cy

    libs = identity['original']
    assert (np.__version__, scipy.__version__, skimage.__version__) == tuple(libs[name]['version'] for name in
                                                                           ('numpy', 'scipy', 'scikit-image'))
    members = {name: info for group in libs.values() for name, info in group['members'].items()}
    modules = dict(skeletonize_python=_skeletonize, skeletonize_compiled=_skeletonize_various_cy,
                   morphology_namespace=morphology, edt_python=_morphology, edt_compiled=_nd_image,
                   label_python=_measurements, label_compiled=_ni_label, rng_compiled=np.random._generator,
                   pcg64_compiled=pcg64_module, seed_compiled=np.random.bit_generator)
    for name, module in modules.items():
        pin = members[identity['provider_members'][name]]
        assert Path(module.__file__).resolve() == Path(pin['path']).resolve()
        assert digest(module.__file__) == pin['sha256']
    stock = morphology.medial_axis
    assert stock is _skeletonize.medial_axis and _skeletonize.ndi is ndimage
    assert ndimage.distance_transform_edt is _morphology.distance_transform_edt and _morphology._nd_image is _nd_image
    assert ndimage.label is _measurements.label and _measurements._ni_label is _ni_label
    assert _skeletonize._skeletonize_loop is _skeletonize_various_cy._skeletonize_loop
    assert np.random.default_rng is np.random._generator.default_rng
    edt, label = ndimage.distance_transform_edt, ndimage.label
    native, entries, compat = native_public(identity)
    assert inspect.signature(stock) == inspect.signature(native)
    adapter = load('_medial_fixed_adapter', SOURCE / 'benchmarks/medial_axis.py')
    snapshot = runpy.run_path(str(SOURCE / 'tests/medial_axis_cases.py'))['array_snapshot']
    rows = identity['fixed_stock_rows']
    sampler = load('_medial_sustained', SAMPLER)
    pools = load('threadpoolctl', POOL)
    previous = compat.set_fallback_enabled(False)
    policy = np.geterr(), np.geterrcall(), np.getbufsize()
    results = []
    try:
        with pools.threadpool_limits(limits=1):
            actual_pools = pools.threadpool_info()
            assert actual_pools and all(pool['num_threads'] == 1 for pool in actual_pools)
            for index, ((name, image, mask, distances, seed), row) in enumerate(zip(adapter.fixed_cases(np), rows, strict=True)):
                assert name == row['case_id'] == CASE_IDS[index]
                assert type(image) is np.ndarray and image.ndim == 2 and image.flags.c_contiguous
                assert image.dtype == np.dtype(bool) and all(1 <= dim <= 192 for dim in image.shape)
                assert (image.view(np.uint8) <= 1).all()
                assert mask is None or (mask.shape == image.shape and mask.flags.c_contiguous
                                        and mask.dtype == np.dtype(bool) and (mask.view(np.uint8) <= 1).all())
                assert type(seed) is int and str(seed) == row['rng'] and distances is row['return_distance']
                assert snapshot(image, small=True) == row['input']
                assert mask is None or snapshot(mask, small=True) == row['mask']
                assert affinity() and datetime.now(timezone.utc) < datetime.fromisoformat(deadline)
                stock_call = partial(adapter.public_call, stock, image, mask, distances, seed)
                native_call = partial(adapter.public_call, native, image, mask, distances, seed)
                observed = prove_case(stock_call, native_call, (stock, native, entries, edt, label),
                                      snapshot, row, image, mask, distances, seed, np)
                with proof.open('a', encoding='utf-8', newline='\n') as out:
                    out.write(json.dumps(dict(case_id=name, seed=seed, fallback_enabled=False, **observed)) + '\n')
                with journal.open('a', encoding='utf-8', newline='\n') as out:
                    out.write(json.dumps({"event": "paired_exact_before_samples", "case_index": index,
                                          "case_id": name, "elf": identity["private_core_sha256"]}) + '\n')
                result = sampler.measure(stock_call, native_call, 7, 100, journal=journal, case_index=index,
                                         case_id=name, pass_index=1, deadline=deadline)
                assert result['sample_order'] == [
                    [0, 1] if (index + turn + 1) % 2 == 0 else [1, 0] for turn in range(7)]
                for side in (0, 1):
                    accepted = [event for event in result['batch_records']
                                if event['function'] == side and event['accepted']]
                    assert len(accepted) == len(result['samples_ms'][side]) == 7
                    assert all(event['complete'] and event['elapsed_ns'] >= 100_000_000 for event in accepted)
                    assert result['warmup_elapsed_ns'][side] >= 250_000_000
                assert snapshot(image, small=True) == row['input']
                assert mask is None or snapshot(mask, small=True) == row['mask']
                assert np.random.default_rng is np.random._generator.default_rng
                assert ndimage.distance_transform_edt is edt and ndimage.label is label
                assert (np.geterr(), np.geterrcall(), np.getbufsize()) == policy
                results.append(dict(case_id=name, **result))
                print(name, format(result['speedup'], '.9f'), flush=True)
            assert affinity() and pools.threadpool_info() == actual_pools
    finally:
        compat.set_fallback_enabled(previous)
    check_source()
    assert digest(identity['private_core']) == identity['private_core_sha256']
    assert [row['case_id'] for row in results] == CASE_IDS
    passed = all(row['speedup'] >= 1.0 for row in results)
    protocol.write(result_path, dict(identity=identity, status='parity_speed_pass' if passed else 'below_parity',
                                     private_component_only=True, cases=results, accepted_samples=56,
                                     pools=actual_pools, entry_sha256=digest(entry), root_go_sha256=go_sha,
                                     correctness_sha256=digest(proof), raw_batches_sha256=digest(journal), pid=os.getpid()))
    return 0 if passed else 1


def main():
    global HERE, SOURCE, CASE_IDS, check_source
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    stage = runpy.run_path(str(CARRIER / "speed-stage.py"))
    grant, raw = stage["authority"](os.environ, datetime.now(timezone.utc))
    packet, plan, prior, admission = stage["inputs"]()
    read, write, seal = (stage[name] for name in ("read", "write", "seal"))
    binding_path, pin, HERE = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    assert digest(binding_path) == pin and "GH_TOKEN" not in os.environ
    binding = read(binding_path)
    assert binding["authority_sha256"] == sha256(raw).hexdigest()
    quiet = Path(binding["quiet"]["path"])
    assert seal(quiet) == {key: binding["quiet"][key] for key in ("bytes", "sha256")}
    checked = read(quiet)
    assert checked["active_competing_local_jobs"] == [] and checked["stage_exit"] == 0 and checked["cpu"] == 0
    assert checked["authority_sha256"] == binding["authority_sha256"]
    work = Path(binding["work"])
    assert work == binding_path.parent
    assert all(Path(binding[key]).resolve(strict=True).is_relative_to(work)
               for key in ("interpreter", "source", "stock", "pybind", "donor", "elf"))
    assert Path(sys.executable).resolve() == Path(binding["interpreter"]).resolve()
    stock, pybind, donor, SOURCE, private = [Path(binding[key]) for key in
                                            ("stock", "pybind", "donor", "source", "elf")]
    stage["DRIVER"]["runtime"](packet, plan, pybind, donor, scalar=False)
    assert seal(private) == {key: admission["elf"][key] for key in ("bytes", "sha256")}
    assert private.name == admission["elf"]["name"]
    assert seal(Path(binding["portable_path"])) == binding["portable"]
    assert seal(Path(binding["transfer_path"])) == binding["transfer"]
    assert read(Path(binding["transfer_path"]))["elf"] == admission["elf"]
    authority_path = work / "authority.json"
    assert digest(authority_path) == binding["authority_sha256"]
    providers = read(CARRIER / "timing-providers.json")
    records = plan["selected_retained_live_files_sha256"]
    installed_members = read(Path(binding["portable_path"]))["member_pins"]
    assert len(installed_members) == 3981
    original = {name: {"version": plan["stock_distributions"][name]["version"], "members": {}}
                for name in ("numpy", "scipy", "scikit-image")}
    for path in providers.values():
        name = ("scikit-image" if path.startswith("skimage/") else path.partition("/")[0])
        assert digest(stock / path) == records[path]
        original[name]["members"][path] = {"sha256": records[path], "path": str(stock / path)}
    CASE_IDS = stage["CASES"]
    rows = [json.loads(line) for line in (CARRIER / "linux-modern-local-cases-v1.jsonl").read_bytes().splitlines()[:4]]
    assert [row["case_id"] for row in rows] == CASE_IDS
    check_source = lambda: stage["DRIVER"]["exact_source"](SOURCE, plan)
    source_proof = check_source()
    sys.path.append(str(stock))
    installed_core = donor / packet["donor"]["core"]["member"]
    host_cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name"))
    identity = dict(scope=grant["purpose"], profile=PROFILE, **source_proof, cpu=0, cpu_model=host_cpu,
                    workers=1, fixed_case_ids=CASE_IDS, warmup_ms=250, minimum_batch_ms=100,
                    accepted_per_side=7, minimum_speed_ratio_each=1., installed=str(donor),
                    runtime=sys.version, original=original, provider_members=providers, fixed_stock_rows=rows,
                    haar_core=str(installed_core), haar_core_sha256=packet["donor"]["core"]["sha256"],
                    private_core=str(private), private_core_sha256=admission["elf"]["sha256"],
                    root_scalar_admission_sha256=stage["ADMISSION"], quiet_sha256=digest(quiet),
                    payload_sha256=grant["payload"], transfer_sha256=binding["transfer"]["sha256"])
    HERE.mkdir()
    entry = HERE / "entry.json"
    write(entry, identity)
    return execute(SimpleNamespace(read=read, write=write), identity, lambda: os.sched_getaffinity(0) == {0},
                   grant["deadline_utc"], binding["authority_sha256"], entry)


if __name__ == "__main__":
    raise SystemExit(main())
