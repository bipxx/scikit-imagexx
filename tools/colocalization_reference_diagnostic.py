"""Untimed original/frozen/installed comparison; mismatches are diagnostic data."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

PUBLIC_SHA256 = '17072b2d0a5fbf6d43ce91f8f6f9056f4caf9e39d5651e759429a470d40498b4'


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def machine_metadata(np):
    cpu = Path('/proc/cpuinfo')
    lines = cpu.read_text().splitlines() if cpu.is_file() else []
    maps = Path('/proc/self/maps')
    libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if '/libm.so' in line}) if maps.is_file() else []
    extension = next(module for name, module in sys.modules.items()
                     if name.startswith('numpy.') and name.endswith('._multiarray_umath')
                     and hasattr(module, '__cpu_features__'))
    return dict(platform=platform.uname()._asdict(), python=sys.version,
                executable=sys.executable, numpy=np.__version__,
                cpu_model=next((line.split(':', 1)[1].strip() for line in lines
                                if line.startswith('model name')), None),
                numpy_cpu_features={key: bool(value) for key, value in extension.__cpu_features__.items()},
                libm_maps_available=maps.is_file(), loaded_libm=[dict(path=name,
                    sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest()) for name in libraries])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--role', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules):
        raise RuntimeError('Require optimize=0 and no skimagexx import before original capture')
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = root / 'tests/test_colocalization_public.py'
    if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != PUBLIC_SHA256:
        raise RuntimeError('Reviewed public observer differs')
    spec = importlib.util.spec_from_file_location('colocalization_diagnostic_public', path)
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    observer, factory = public.OBSERVER, public.FACTORY
    import numpy as np
    import skimage
    import skimage.measure._colocalization as original
    if skimage.__version__ != '0.26.0' or len(public.CASES) != 536:
        raise RuntimeError('Require pinned original 0.26.0 and complete 536-case factory')
    auth = observer.authenticate(original.__file__, original.as_binary_ndarray.__code__.co_filename)
    save(output / 'runtime-before.json', dict(role=args.role, root=str(root), machine=machine_metadata(np),
         original_authentication=auth, public_helper_canonical_sha256=PUBLIC_SHA256,
         disable_simd=os.environ.get('SKIMAGEXX_DISABLE_SIMD'),
         frozen_sha256=hashlib.sha256((root / 'tests/data/colocalization_public_semantics.json.gz').read_bytes()).hexdigest()))
    fresh, native = {}, {}
    journal = observer.RawJournal(output / 'original.jsonl')
    assets = observer.RawAssets(output / 'original-assets')
    try:
        for case in public.CASES:
            fresh[case['id']] = observer.observe(case, factory, original, np, assets, journal)['observation']
    finally:
        journal.close()
    if any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules):
        raise RuntimeError('Original capture imported skimagexx')
    save(output / 'original-complete.json', dict(observations=len(fresh), skimagexx_imported=False))
    runtime_generator = public.coloc_runtime.__wrapped__()
    try:
        runtime = next(runtime_generator)
        core = runtime['core']
        save(output / 'native-runtime.json', dict(core=str(core.__file__),
             core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
             workers=core.get_num_threads(), has_avx2=bool(core.has_avx2())))
        if core.get_num_threads() != 1 or (os.environ.get('SKIMAGEXX_DISABLE_SIMD') and core.has_avx2()):
            raise RuntimeError('Installed worker or forced-scalar policy differs')
        for index, case in enumerate(public.CASES):
            native[case['id']] = public._observe(runtime, case, output / 'native' / str(index))
            if not native[case['id']]['restored']:
                raise RuntimeError('Observer did not restore state; raw row retained')
        frozen = runtime['expected']
    finally:
        runtime_generator.close()
    labels = ('original_vs_frozen', 'native_vs_fresh', 'native_vs_frozen')
    totals = {label: dict(matching_cases=0, differing_cases=[],
                         field_mismatches={key: 0 for key in public.SEMANTIC_FIELDS}) for label in labels}
    failures = [ident for ident, row in native.items() if row['observer_error'] is not None
                or row['phase'] != 'complete' or row['forbidden_calls']]
    comparisons = observer.RawJournal(output / 'comparison.jsonl')
    try:
        for case in public.CASES:
            ident = case['id']
            actual, expected, old = (observer.semantic_projection(rows[ident]) for rows in (native, fresh, frozen))
            results = {}
            for label, left, right in zip(labels, (expected, actual, actual), (old, expected, old)):
                equal = {key: left[key] == right[key] for key in public.SEMANTIC_FIELDS}
                results[label] = equal
                totals[label]['matching_cases'] += int(all(equal.values()))
                if not all(equal.values()):
                    totals[label]['differing_cases'].append(ident)
                for key, same in equal.items():
                    totals[label]['field_mismatches'][key] += int(not same)
            comparisons.append(dict(id=ident, comparisons=results))
    finally:
        comparisons.close()
    save(output / 'summary.json', dict(schema='colocalization-reference-diagnostic-v1', role=args.role,
         completed=not failures, cases=len(public.CASES), original_observations=len(fresh),
         native_observations=len(native), semantic_fields=list(public.SEMANTIC_FIELDS),
         observer_failures=failures, comparisons=totals, timings=False, acceptance=False))
    if failures:
        raise RuntimeError('Native observer failure; complete raw capture and summary retained')


if __name__ == '__main__':
    main()
