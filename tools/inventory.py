"""Inventory the pinned upstream API, with honest implementation/test status."""
import importlib
import functools
import inspect
import json
import re
from pathlib import Path
import skimage
import numpy as np
from skimagexx.coverage import native_functions, native_dtypes, execution_kind

MODULES = ('color', 'data', 'draw', 'exposure', 'feature', 'filters', 'filters.rank', 'future',
           'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
           'restoration', 'segmentation', 'transform', 'util')


def signature_of(value):
    try:
        signature=inspect.signature(value)
        text=re.sub(r' at 0x[0-9a-fA-F]+', '', str(signature))
        # NumPy 1.24 exposes np.max as amax; later releases display max.
        # Normalize only the identical default object, not arbitrary callables.
        if any(parameter.default is np.max for parameter in signature.parameters.values()):
            text=text.replace('<function amax>', '<function max>')
        return text
    except (TypeError,ValueError):return None


def class_members(value,path):
    """Static descriptors only; never construct an instance or evaluate getters."""
    owners={}
    for base in value.__mro__:
        for name in vars(base):owners.setdefault(name,base)
    members=[]
    for name,owner in sorted(owners.items()):
        if owner is object:continue
        special=name.startswith('__') and name.endswith('__')
        if name.startswith('_') and not special:continue
        descriptor=inspect.getattr_static(value,name)
        if isinstance(descriptor,(property,functools.cached_property)):
            kind='property';signature=None
        elif isinstance(descriptor,classmethod):
            kind='classmethod';signature=signature_of(classmethod.__get__(descriptor,None,value))
        elif isinstance(descriptor,staticmethod):
            kind='staticmethod';signature=signature_of(descriptor.__func__)
        elif inspect.isroutine(descriptor):
            kind='constructor' if name=='__init__' else 'method';signature=signature_of(descriptor)
        elif not special and inspect.isdatadescriptor(descriptor):
            kind='attribute';signature=None
        else:continue
        members.append({'api':path+'.'+name,'kind':kind,'signature':signature,
                        'defined_by':owner.__module__+'.'+owner.__qualname__,
                        'inherited':owner is not value,'implementation':'upstream-only',
                        'parity_test':None,'benchmark':None})
    return members


def inventory():
    native = set(native_functions())
    entries = []
    for module in MODULES:
        upstream = importlib.import_module('skimage.'+module)
        names=set(getattr(upstream, '__all__', dir(upstream)))
        # Still callable in the pinned release, but removed from its export list.
        if module=='morphology':
            names.update(('binary_erosion','binary_dilation','binary_opening','binary_closing'))
        for name in sorted(n for n in names if not n.startswith('_')):
            value = getattr(upstream, name)
            if not callable(value):
                continue
            path = module+'.'+name
            entries.append({'api': path, 'kind': 'class' if inspect.isclass(value) else 'callable',
                            'signature': signature_of(value),
                            'implementation': ('metadata' if execution_kind(path) == 'metadata' else 'native-partial') if path in native else 'upstream-only',
                            'native_default_case_dtypes': list(native_dtypes(path)) if path in native else [],
                            'execution_kind': execution_kind(path) if path in native else 'upstream',
                            'parity_test': 'tests/test_api.py::test_every_native_function' if path in native else None,
                            'benchmark': 'benchmarks/coverage.py' if path in native else None})
            if inspect.isclass(value):entries[-1]['members']=class_members(value,path)
    inventoried = {entry['api'] for entry in entries}
    missing = native-inventoried
    if missing:
        raise ValueError(f'Native registry entries absent from upstream inventory: {sorted(missing)}')
    return {'reference_version': skimage.__version__, 'modules': list(MODULES),
            'class_member_scope':'Static public descriptors and custom data-model methods, including inherited members. Instance fields and runtime-generated attributes remain unassessed. Availability and signatures are not numerical parity.',
            'entries': entries}


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
             f'{kernels} have partially native image implementations and {metadata} '
             f'{"has" if metadata == 1 else "have"} metadata-only execution. '
             'All covered entries have per-function tests and benchmarks.', '',
             'An inventoried name is not evidence of parity. `upstream-only` entries have no native implementation, '
             'no dedicated parity test, and no native performance result. Modules outside this inventory remain unassessed. '
             'Native entries support a documented subset of input/options; unsupported calls can use explicit upstream fallback. '
             'Metadata entries return small values/arrays without running an image kernel. Identity-preserving conversion calls '
             'can return the input without processing pixels.', '',
             '| API | Implementation | Dedicated parity test | Benchmark |', '|---|---|---|---|']
    for entry in entries:
        lines.append(f"| `{entry['api']}` | {entry['implementation']} | {'yes (partial input contract)' if entry['parity_test'] else 'pending'} | {'yes' if entry['benchmark'] else 'pending'} |")
    classes=[entry for entry in entries if entry['kind']=='class']
    members=[member for entry in classes for member in entry['members']]
    lines+=['','## Class member inventory','',
            f'{len(classes)} exported class names expose {len(members)} statically discoverable member paths, including inherited methods and properties. '
            'Their signatures and defining classes are recorded in `api.json`. '
            'These members still require native implementations, numerical/state parity cases and benchmarks; they do not increase native coverage. '
            'Constructors are not executed and property getters are not evaluated during inventory. '
            'Instance fields and runtime-generated attributes remain unassessed.','',
            '| Class | Static member paths | Native member parity |','|---|---:|---|']
    for entry in classes:lines.append(f"| `{entry['api']}` | {len(entry['members'])} | pending |")
    (path.parent/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(entries)} API names; {kernels} native-partial, {metadata} metadata; {len(entries)-len(implemented)} pending')
    print(f'{len(classes)} class names; {len(members)} static member paths pending implementation/parity')


if __name__ == '__main__':
    main()
