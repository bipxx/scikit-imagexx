"""Inventory the pinned upstream API, with honest implementation/test status."""
import importlib
import inspect
import json
import re
from pathlib import Path
import skimage
from skimagexx.coverage import native_functions, native_dtypes, execution_kind

MODULES = ('color', 'data', 'draw', 'exposure', 'feature', 'filters', 'filters.rank',
           'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
           'restoration', 'segmentation', 'transform', 'util')


def inventory():
    native = set(native_functions())
    entries = []
    for module in MODULES:
        upstream = importlib.import_module('skimage.'+module)
        for name in sorted(n for n in getattr(upstream, '__all__', dir(upstream)) if not n.startswith('_')):
            value = getattr(upstream, name)
            if not callable(value):
                continue
            path = module+'.'+name
            try:
                signature = re.sub(r' at 0x[0-9a-fA-F]+', '', str(inspect.signature(value)))
            except (TypeError, ValueError):
                signature = None
            entries.append({'api': path, 'kind': 'class' if inspect.isclass(value) else 'callable',
                            'signature': signature,
                            'implementation': ('metadata' if execution_kind(path) == 'metadata' else 'native-partial') if path in native else 'upstream-only',
                            'native_default_case_dtypes': list(native_dtypes(path)) if path in native else [],
                            'execution_kind': execution_kind(path) if path in native else 'upstream',
                            'parity_test': 'tests/test_api.py::test_every_native_function' if path in native else None,
                            'benchmark': 'benchmarks/coverage.py' if path in native else None})
    inventoried = {entry['api'] for entry in entries}
    missing = native-inventoried
    if missing:
        raise ValueError(f'Native registry entries absent from upstream inventory: {sorted(missing)}')
    return {'reference_version': skimage.__version__, 'modules': list(MODULES), 'entries': entries}


def main():
    result = inventory()
    root = Path(__file__).resolve().parents[1]
    path = root/'coverage/api.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    entries = result['entries']
    implemented = [e for e in entries if e['implementation'] in ('native-partial', 'metadata')]
    kernels = sum(e['implementation'] == 'native-partial' for e in entries)
    metadata = sum(e['implementation'] == 'metadata' for e in entries)
    lines = ['# API implementation inventory', '', f'Reference: scikit-image {skimage.__version__}.', '',
             f'{len(entries)} public callable names inventoried in {len(MODULES)} modules; '
             f'{kernels} have partially native image implementations and {metadata} has metadata-only execution. '
             'All covered entries have per-function tests and benchmarks.', '',
             'An inventoried name is not evidence of parity. `upstream-only` entries have no native implementation, '
             'no dedicated parity test, and no native performance result. Modules outside this inventory remain unassessed. '
             'Native entries support a documented subset of input/options; unsupported calls can use explicit upstream fallback. '
             'Metadata entries inspect dtype without running an image kernel. Identity-preserving conversion calls '
             'can return the input without processing pixels.', '',
             '| API | Implementation | Dedicated parity test | Benchmark |', '|---|---|---|---|']
    for entry in entries:
        lines.append(f"| `{entry['api']}` | {entry['implementation']} | {'yes (partial input contract)' if entry['parity_test'] else 'pending'} | {'yes' if entry['benchmark'] else 'pending'} |")
    (path.parent/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(entries)} API names; {kernels} native-partial, {metadata} metadata; {len(entries)-len(implemented)} pending')


if __name__ == '__main__':
    main()
